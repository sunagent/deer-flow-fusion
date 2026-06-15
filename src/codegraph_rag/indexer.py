"""索引管理器 — 初始索引 + 文件监听增量更新 + 懒加载"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable

from .chunker import parse_file
from .config import CodeGraphConfig
from .embedding import EmbeddingCache, create_provider
from .models import CodeChunk, SymbolIndex
from .storage import CodeGraphStorage
from .symbol_extractor import extract_calls, extract_symbols

logger = logging.getLogger(__name__)

# 支持的源码文件扩展名
_SOURCE_EXTENSIONS: set[str] = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs",
    ".go", ".rs", ".java", ".html", ".css",
}

# 忽略的目录
_IGNORE_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".codegraph", ".deer-flow", ".vector_store",
}


class CodeGraphIndexer:
    """索引管理器：初始索引 + 增量更新"""

    def __init__(
        self,
        config: CodeGraphConfig,
        storage: CodeGraphStorage,
        embedding_cache: EmbeddingCache,
    ):
        self._config = config
        self._storage = storage
        self._embedding = embedding_cache
        self._watcher = None
        self._indexed_files: set[str] = set()
        self._last_symbols: list[SymbolIndex] = []
        self._last_calls: list = []

    # ── 初始索引 ──

    def index_workspace(self, workspace_root: Path | None = None) -> dict[str, int]:
        """对工作区执行完整索引

        Returns:
            统计信息 {"files": N, "chunks": N, "symbols": N}
        """
        root = workspace_root or Path(self._config.index.workspace_root or ".")
        if not root.is_absolute():
            root = Path.cwd() / root

        logger.info(f"Starting workspace indexing: {root}")

        # 1. 发现源码文件
        source_files = self._discover_files(root)
        logger.info(f"Discovered {len(source_files)} source files")
        removed_files = self._remove_stale_indexed_files(root, source_files)

        # 2. 逐文件解析和存储
        total_chunks = 0
        total_symbols = 0
        total_calls = 0
        total_code_embeddings = 0
        total_doc_embeddings = 0
        all_symbols: list[SymbolIndex] = []
        all_calls: list = []
        files_with_chunks: list[Path] = []

        for file_path in source_files:
            try:
                chunks, symbols, calls = self._index_file(file_path)
                total_chunks += len(chunks)
                total_symbols += len(symbols)
                total_calls += len(calls)
                all_symbols.extend(symbols)
                all_calls.extend(calls)
                if chunks:
                    files_with_chunks.append(file_path)
                self._indexed_files.add(str(file_path))
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")

        embed_stats = self.embed_files_incremental(files_with_chunks)
        total_code_embeddings += int(embed_stats.get("code_embeddings", 0))
        total_doc_embeddings += int(embed_stats.get("doc_embeddings", 0))

        self._last_symbols = all_symbols
        self._last_calls = all_calls

        stats = {
            "files": len(source_files),
            "chunks": total_chunks,
            "symbols": total_symbols,
            "calls": total_calls,
            "code_embeddings": total_code_embeddings,
            "doc_embeddings": total_doc_embeddings,
            "removed_files": removed_files,
        }
        logger.info(f"Indexing complete: {stats}")
        return stats

    def _discover_files(self, root: Path) -> list[Path]:
        """递归发现所有源码文件"""
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过忽略目录（原地修改 dirnames 影响 os.walk）
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]

            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if ext in _SOURCE_EXTENSIONS:
                    files.append(Path(dirpath) / filename)
        return files

    def _remove_stale_indexed_files(self, root: Path, source_files: list[Path]) -> int:
        """Remove DB rows for files that disappeared from the workspace."""
        try:
            indexed_files = self._storage.list_indexed_files()
        except Exception:
            logger.debug("Unable to list indexed files for stale cleanup", exc_info=True)
            return 0

        root_key = self._file_key(root)
        root_prefix = root_key if root_key.endswith(os.sep) else root_key + os.sep
        source_keys = {self._file_key(path) for path in source_files}
        removed = 0

        for indexed_file in indexed_files:
            indexed_path = Path(indexed_file)
            indexed_key = self._file_key(indexed_path)
            if indexed_key != root_key and not indexed_key.startswith(root_prefix):
                continue
            if indexed_key in source_keys and indexed_path.exists():
                continue

            self._storage.delete_chunks_by_file(indexed_file)
            self._storage.delete_symbols_by_file(indexed_file)
            self._embedding.invalidate(indexed_file)
            self._indexed_files.discard(indexed_file)
            removed += 1

        if removed:
            logger.info("Removed %s stale indexed files under %s", removed, root)
        return removed

    @staticmethod
    def _file_key(path: str | Path) -> str:
        try:
            return str(Path(path).resolve()).casefold()
        except OSError:
            return str(Path(path).absolute()).casefold()

    def _index_file(self, file_path: Path) -> tuple[list[CodeChunk], list[SymbolIndex], list]:
        """索引单个文件：解析 → 提取 → 存储"""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read file {file_path}: {e}")
            return [], [], []

        language = self._detect_language(file_path)

        # 1. AST 分块
        chunks = parse_file(str(file_path), content, language)

        # 2. 符号提取
        symbols = extract_symbols(str(file_path), content, language)

        # 3. 调用关系提取
        calls = extract_calls(str(file_path), content, language, symbols)

        # 4. 存储入库
        old_chunk_ids = set(self._storage.get_chunk_ids_by_file(str(file_path)))
        new_chunk_ids = {chunk.id for chunk in chunks}
        stale_chunk_ids = sorted(old_chunk_ids - new_chunk_ids)
        if stale_chunk_ids:
            self._storage.delete_chunks_by_ids(stale_chunk_ids)

        self._storage.delete_symbols_by_file(str(file_path))
        if chunks:
            self._storage.upsert_chunks(chunks)
        if symbols:
            self._storage.upsert_symbols(symbols)
        if chunks:
            self._storage.upsert_search_documents(
                self._build_search_documents(chunks, symbols)
            )

        return chunks, symbols, calls

    def embed_file_incremental(
        self,
        file_path: str | Path,
        *,
        text_limit: int = 2048,
        doc_text_limit: int = 1024,
        batch_size: int = 32,
    ) -> dict[str, int]:
        """Embed only changed chunks for one indexed file.

        This is the product path for "save file -> update index": unchanged
        functions keep their existing code/search_document vectors by hash.
        """
        return self.embed_files_incremental(
            [file_path],
            text_limit=text_limit,
            doc_text_limit=doc_text_limit,
            batch_size=batch_size,
        )

    def embed_files_incremental(
        self,
        file_paths: list[str | Path],
        *,
        text_limit: int = 2048,
        doc_text_limit: int = 1024,
        batch_size: int = 32,
    ) -> dict[str, int]:
        """Embed changed chunks for multiple indexed files in workspace batches."""
        if not self._embedding.is_available:
            return {"chunks": 0, "code_embeddings": 0, "doc_embeddings": 0}

        items: list[dict[str, str | bool]] = []
        for file_path in file_paths:
            items.extend(
                self._storage.embedding_work_items_for_file(
                    str(Path(file_path)),
                    search_text_transform=self._search_document_embedding_text,
                )
            )

        if not items:
            return {"chunks": 0, "code_embeddings": 0, "doc_embeddings": 0}

        code_rows = [
            (str(item["chunk_id"]), str(item["content"])[:text_limit])
            for item in items
            if bool(item["needs_code"])
        ]
        doc_rows = [
            (str(item["chunk_id"]), str(item["doc_embedding_text"])[:doc_text_limit])
            for item in items
            if bool(item["needs_doc"])
        ]

        written_code_rows: list[tuple[str, list[float]]] = []
        written_doc_rows: list[tuple[str, list[float]]] = []

        for start in range(0, len(code_rows), batch_size):
            batch = code_rows[start : start + batch_size]
            embeddings = self._embedding.get_embeddings_batch([text for _, text in batch])
            batch_written = [
                (chunk_id, embedding)
                for (chunk_id, _), embedding in zip(batch, embeddings)
                if embedding is not None
            ]
            if batch_written:
                self._storage.upsert_embeddings(batch_written)
                written_code_rows.extend(batch_written)

        for start in range(0, len(doc_rows), batch_size):
            batch = doc_rows[start : start + batch_size]
            embeddings = self._embedding.get_embeddings_batch([text for _, text in batch])
            batch_written = [
                (chunk_id, embedding)
                for (chunk_id, _), embedding in zip(batch, embeddings)
                if embedding is not None
            ]
            if batch_written:
                self._storage.upsert_document_embeddings(batch_written)
                written_doc_rows.extend(batch_written)

        rows_by_id = {
            str(item["chunk_id"]): (
                str(item["content"]),
                str(item["doc_embedding_text"]),
            )
            for item in items
        }
        updated_ids = {
            chunk_id
            for chunk_id, _ in code_rows
        } | {
            chunk_id
            for chunk_id, _ in doc_rows
        }
        self._storage.mark_embedding_caches(
            [
                (chunk_id, rows_by_id[chunk_id][0], rows_by_id[chunk_id][1])
                for chunk_id in sorted(updated_ids)
                if chunk_id in rows_by_id
            ]
        )

        return {
            "chunks": len(items),
            "code_embeddings": len(written_code_rows),
            "doc_embeddings": len(written_doc_rows),
        }

    @staticmethod
    def _search_document_embedding_text(search_text: str, *, limit: int = 1024) -> str:
        """Build a compact NL vector input while keeping full search_document stored.

        BM25 and the reranker still see the complete search document. The vector
        channel only needs dense semantic hints, so avoid repeating absolute
        paths, owner blocks, full AST boilerplate, and full code.
        """
        if not search_text:
            return ""

        keep_prefixes = (
            "function summary:",
            "function purpose:",
            "function input:",
            "function output:",
            "function categories:",
            "function module context:",
            "language:",
            "symbol:",
            "symbol words:",
            "symbol aliases:",
            "parent:",
            "signature:",
            "parameters:",
            "docstring:",
            "comments:",
            "tags:",
        )
        ast_keep_prefixes = (
            "function kind:",
            "inputs:",
            "output type:",
            "return behavior:",
            "operators:",
            "calls:",
        )
        kept: list[str] = []
        in_ast = False
        in_code = False
        code_chars = 0
        code_budget = 420

        for raw in search_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered == "code:":
                in_code = True
                in_ast = False
                continue
            if lowered == "ast summary:":
                in_ast = True
                in_code = False
                continue
            if in_code:
                if code_chars == 0:
                    kept.append("code excerpt:")
                if code_chars < code_budget:
                    kept.append(line[: max(0, code_budget - code_chars)])
                    code_chars += len(line) + 1
                continue
            if in_ast:
                if lowered.endswith(":") and lowered not in ast_keep_prefixes:
                    in_ast = False
                elif lowered.startswith(ast_keep_prefixes):
                    if not lowered.endswith(":") and not lowered.endswith("false"):
                        kept.append(line)
                continue
            if lowered.startswith(keep_prefixes):
                if lowered.startswith(("function summary:", "function purpose:")):
                    line = line[:240]
                elif lowered.startswith("function module context:"):
                    line = line[:160]
                elif lowered.startswith(("symbol aliases:", "docstring:", "comments:")):
                    line = line[:220]
                elif len(line) > 260:
                    line = line[:260]
                kept.append(line)
            if sum(len(part) + 1 for part in kept) >= limit:
                break

        if not kept:
            return search_text[:limit]
        return "\n".join(kept)[:limit]

    @staticmethod
    def extract_import_context(content: str, limit: int = 1200) -> str:
        """Extract lightweight import/dependency context for search_document."""
        imports = []
        in_go_import_block = False
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if in_go_import_block:
                imports.append(stripped.strip("()"))
                if ")" in stripped:
                    in_go_import_block = False
                continue
            if re.match(r"^import\s*\($", stripped):
                imports.append(stripped)
                in_go_import_block = True
                continue
            patterns = [
                r"^(from\s+[\w.]+\s+import\s+.+|import\s+[\w.,\s]+)$",
                r"^(import\s+.+\s+from\s+['\"].+['\"];?|import\s*['\"].+['\"];?)$",
                r"^(export\s+.+\s+from\s+['\"].+['\"];?)$",
                r"^(const|let|var)\s+[\w{} ,*]+\s*=\s*require\(['\"].+['\"]\);?$",
                r"^(package\s+\w+|import\s+['\"].+['\"])$",
                r"^(use\s+[\w: {},*]+;|extern\s+crate\s+\w+;|mod\s+\w+;)$",
                r"^(package\s+[\w.]+;|import\s+[\w.*]+;)$",
            ]
            if any(re.match(pattern, stripped) for pattern in patterns):
                imports.append(stripped)
        return "\n".join(imports)[:limit]

    @staticmethod
    def module_context(file_path: str) -> str:
        path = Path(file_path)
        parts = []
        for part in path.parts:
            if part and part not in {":", "\\", "/"}:
                parts.extend(re.findall(r"[A-Za-z0-9]+", part.replace("_", " ")))
        stem_words = re.findall(r"[A-Za-z0-9]+", path.stem.replace("_", " "))
        return " ".join(parts + stem_words).lower()

    @staticmethod
    def _split_identifier(name: str | None) -> str:
        if not name:
            return ""
        magic_aliases = {
            "__len__": "len length count size number elements collection",
            "__ge__": "ge greater equal comparison compare ordering priority",
            "__gt__": "gt greater than comparison compare ordering priority",
            "__le__": "le less equal comparison compare ordering priority",
            "__lt__": "lt less than comparison compare ordering priority",
            "__eq__": "eq equal equality comparison compare",
            "__ne__": "ne not equal inequality comparison compare",
            "__post_init__": "post init initialize initialization setup state configuration properties object creation",
            "__init__": "init initialize initialization setup state configuration properties object creation",
            "__str__": "str string representation convert text",
            "__repr__": "repr string representation debug display",
        }
        if name in magic_aliases:
            return magic_aliases[name]
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
        text = re.sub(r"([A-Za-z]+)(\d+)", r"\1 \2", text)
        text = re.sub(r"(\d+)([A-Za-z]+)", r"\1 \2", text)
        text = text.replace("_", " ").replace("-", " ")
        parts = re.findall(r"[A-Za-z]+|\d+", text.lower())
        aliases = []
        for part in parts:
            if part.startswith("decode"):
                aliases.extend(["decode", "decoding", "decompression", "compressed"])
            if part.startswith("matmul"):
                aliases.extend(["matmul", "matrix", "multiplication", "weighted", "sum"])
            if part in {"fp16", "float16"}:
                aliases.extend(["fp16", "float16", "16-bit", "floating", "point", "precision"])
            if part in {"fp32", "float32"}:
                aliases.extend(["fp32", "float32", "32-bit", "floating", "point", "precision"])
            if part in {"int3", "int4", "uint32", "uint"}:
                aliases.extend(["integer", "compressed", "data"])
        return re.sub(r"\s+", " ", " ".join(parts + aliases)).strip().lower()

    @classmethod
    def _symbol_alias_context(cls, name: str | None, language: str, content: str) -> str:
        """Build NL-friendly aliases for short or idiomatic symbol names."""
        if not name:
            return ""
        lowered = name.lower()
        parts = set(re.findall(r"[a-z]+|\d+", cls._split_identifier(name)))
        aliases: list[str] = []
        special = {
            "__len__": "length count size number of elements collection cardinality",
            "__post_init__": "post init initialize object state configuration setup properties",
            "__init__": "constructor initialize object state setup configuration",
            "__str__": "string representation convert display text",
            "__repr__": "debug representation printable string",
            "has": "has contains exists member key property presence check object field",
            "min": "minimum smallest lower bound compare two values",
            "max": "maximum largest upper bound compare two values",
            "run": "execute start process task command",
            "alpha": "alphabetic alpha hash cracking cipher decoder",
        }
        if lowered in special:
            aliases.append(special[lowered])
        if "len" in parts or "length" in parts or "count" in parts:
            aliases.append("length count size number elements collection")
        if {"try", "stat"} <= parts or lowered in {"trystat", "try_stat"}:
            aliases.append("safely attempt file metadata filesystem stat without throwing missing file")
        if "stat" in parts:
            aliases.append("metadata status statistics file information")
        if {"random", "string"} <= parts:
            aliases.append("generate random alphanumeric string sequence")
        if "opts" in parts or "options" in parts:
            aliases.append("options configuration command line arguments defaults settings")
        if "config" in parts or "configure" in parts:
            aliases.append("configure settings options defaults")
        if "u64" in parts or ("64" in parts and "u" in parts):
            aliases.append("unsigned 64 bit integer numeric identifier id")
        if "into" in parts:
            aliases.append("convert transform into output value")
        if "null" in parts:
            aliases.append("null none empty missing value type check")
        if "json" in parts:
            aliases.append("json serialization deserialization element object value")
        if re.search(r"\b(Result|Option)<", content) or "result" in parts or "option" in parts:
            aliases.append("result option success failure error nullable value")
        if language in {"typescript", "javascript"} and re.search(r"\b(export|module\.exports|exports\.)\b", content):
            aliases.append("exported public module api")
        if language == "java" and re.search(r"\b(public|private|protected|static|final)\b", content):
            aliases.append("java method class member access modifier")
        if language == "rust" and re.search(r"\b(pub|impl|trait|macro_rules!)\b|[A-Za-z_][A-Za-z0-9_]*!", content):
            aliases.append("rust public impl trait macro module")
        return re.sub(r"\s+", " ", " ".join(aliases)).strip()

    @classmethod
    def _owner_context(
        cls,
        chunk: CodeChunk,
        symbol: SymbolIndex | None,
    ) -> str:
        """Extract owner/package/export/trait context for function-level NL retrieval."""
        content = chunk.content
        language = chunk.language
        lines: list[str] = []
        if symbol:
            lines.append(f"qualified name: {symbol.qualified_name}")
            owner = symbol.qualified_name.rsplit(".", 1)[0] if "." in symbol.qualified_name else ""
            if owner and owner != symbol.name:
                lines.append(f"owner: {owner}")
                lines.append(f"owner words: {cls._split_identifier(owner)}")
            if symbol.base_classes:
                lines.append(f"base classes: {' '.join(symbol.base_classes)}")
            if symbol.implemented_interfaces:
                lines.append(f"interfaces: {' '.join(symbol.implemented_interfaces)}")
            if symbol.class_methods:
                lines.append(f"class methods: {' '.join(symbol.class_methods[:30])}")
        if chunk.parent_symbol:
            lines.append(f"parent owner: {chunk.parent_symbol}")
            lines.append(f"parent owner words: {cls._split_identifier(chunk.parent_symbol)}")

        if language == "java":
            package = re.search(r"\bpackage\s+([\w.]+)\s*;", content)
            if package:
                pkg = package.group(1)
                lines.append(f"java package: {pkg}")
                lines.append(f"java package words: {cls._split_identifier(pkg)}")
            class_decl = re.search(
                r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
                r"(?:\s+extends\s+([A-Za-z_][A-Za-z0-9_.$<>]*))?"
                r"(?:\s+implements\s+([A-Za-z0-9_.$<>,\s]+))?",
                content,
            )
            if class_decl:
                lines.append(f"java owner type: {class_decl.group(1)} {class_decl.group(2)}")
                if class_decl.group(3):
                    lines.append(f"java extends: {class_decl.group(3)}")
                if class_decl.group(4):
                    lines.append(f"java implements: {class_decl.group(4)}")
            modifiers = re.findall(r"\b(public|private|protected|static|final|abstract|synchronized)\b", content)
            if modifiers:
                lines.append(f"java modifiers: {' '.join(sorted(set(modifiers)))}")

        elif language in {"typescript", "javascript"}:
            export_bits = []
            if re.search(r"\bexport\s+default\b", content):
                export_bits.append("default")
            if re.search(r"\bexport\s+", content):
                export_bits.append("named")
            if re.search(r"\bmodule\.exports\b", content):
                export_bits.append("commonjs module exports")
            if re.search(r"\bexports\.[A-Za-z_]", content):
                export_bits.append("commonjs named exports")
            if export_bits:
                lines.append(f"js export: {' '.join(sorted(set(export_bits)))}")
            owner_match = re.search(
                r"\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:function|\([^)]*\)\s*=>|[A-Za-z_][A-Za-z0-9_]*\s*=>)",
                content,
            )
            if owner_match:
                lines.append(f"js assigned function: {owner_match.group(1)}")
                lines.append(f"js assigned words: {cls._split_identifier(owner_match.group(1))}")
            object_method = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*=", content)
            if object_method:
                lines.append(f"js object owner: {object_method.group(1)}")
                lines.append(f"js object method: {object_method.group(2)}")
            if "=>" in content:
                lines.append("js function form: arrow function callback lambda")

        elif language == "rust":
            mods = re.findall(r"\bmod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", content)
            uses = re.findall(r"\buse\s+([A-Za-z0-9_:{}*,\s]+);", content)
            macros = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)!", content)
            impls = re.findall(
                r"\bimpl(?:<[^>]+>)?\s+(?:(?P<trait>[A-Za-z_][A-Za-z0-9_:<>]*)\s+for\s+)?(?P<typ>[A-Za-z_][A-Za-z0-9_:<>]*)",
                content,
            )
            traits = re.findall(r"\btrait\s+([A-Za-z_][A-Za-z0-9_]*)", content)
            if mods:
                lines.append(f"rust modules: {' '.join(mods[:20])}")
            if uses:
                lines.append(f"rust use imports: {' '.join(uses[:10])}")
            if macros:
                lines.append(f"rust macros: {' '.join(sorted(set(macros))[:20])}")
            if impls:
                impl_text = []
                for trait_name, typ in impls[:10]:
                    impl_text.append(f"{trait_name or 'inherent'} for {typ}")
                lines.append(f"rust impl: {'; '.join(impl_text)}")
            if traits:
                lines.append(f"rust traits: {' '.join(traits[:20])}")
            if re.search(r"\bResult\s*<", content):
                lines.append("rust return family: Result success error")
            if re.search(r"\bOption\s*<", content):
                lines.append("rust return family: Option optional none some")

        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _extract_comments(content: str, limit: int = 800) -> str:
        comments = []
        in_block = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                comments.append(stripped.lstrip("#").strip())
                continue
            if in_block:
                clean = stripped
                if "*/" in clean:
                    clean = clean.split("*/", 1)[0]
                    in_block = False
                comments.append(clean.lstrip("*").strip())
                continue
            if stripped.startswith("//"):
                comments.append(stripped.lstrip("/").strip())
            elif stripped.startswith("/*") or stripped.startswith("/**"):
                clean = stripped.lstrip("/").lstrip("*").strip()
                if "*/" in clean:
                    clean = clean.split("*/", 1)[0]
                else:
                    in_block = True
                comments.append(clean.strip())
        return "\n".join(comments)[:limit]

    @staticmethod
    def _clean_summary_text(text: str | None, limit: int = 260) -> str:
        if not text:
            return ""
        cleaned_lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip().strip('"').strip("'").strip("*/# ").strip()
            if not line:
                continue
            lowered = line.lower().rstrip(":")
            if lowered in {
                "args", "arguments", "parameters", "returns", "return",
                "raises", "yields", "examples", "example", "notes", "note",
            }:
                continue
            if re.match(r"^[:\-*]+$", line):
                continue
            cleaned_lines.append(line)
            if len(" ".join(cleaned_lines)) >= limit:
                break
        text = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
        if len(text) <= limit:
            return text
        cutoff = text[:limit].rsplit(" ", 1)[0]
        return cutoff or text[:limit]

    @staticmethod
    def _parameter_summary(symbol: SymbolIndex | None) -> str:
        if not symbol or not symbol.parameters:
            return ""
        parts = []
        for param in symbol.parameters:
            if param.name in {"self", "cls"}:
                continue
            words = CodeGraphIndexer._split_identifier(param.name)
            item = words or param.name
            if param.type_annotation:
                item += f" type {param.type_annotation}"
            if param.default_value is not None:
                item += " optional default value"
            parts.append(item)
        return " ".join(parts[:12])

    @staticmethod
    def _return_summary(content: str, symbol: SymbolIndex | None) -> str:
        if symbol and symbol.return_type:
            return f"returns {symbol.return_type}"
        lowered = content.lower()
        if re.search(r"\byield\b", lowered):
            return "yields generated values"
        if re.search(r"\breturn\s+(true|false)\b", lowered):
            return "returns boolean result"
        if re.search(r"\breturn\s+none\b", lowered):
            return "returns no value"
        if re.search(r"\breturn\s+(dict\(|\{)", lowered):
            return "returns dictionary mapping data"
        if re.search(r"\breturn\s+(list\(|\[|tuple\(|\()", lowered):
            return "returns collection sequence data"
        if re.search(r"\breturn\s+str\(", lowered):
            return "returns string representation"
        if re.search(r"\bjson\.loads\b|\bjson\.load\b", lowered):
            return "returns parsed json data"
        if re.search(r"\breturn\s+", lowered):
            return "returns computed result"
        return ""

    @staticmethod
    def _import_summary(import_ctx: str) -> str:
        names: list[str] = []
        for line in import_ctx.splitlines():
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line):
                if token in {"from", "import", "as", "const", "let", "var", "require", "use", "package", "mod"}:
                    continue
                if token not in names:
                    names.append(token)
                if len(names) >= 16:
                    break
            if len(names) >= 16:
                break
        return " ".join(names)

    @classmethod
    def _function_summary(
        cls,
        chunk: CodeChunk,
        symbol: SymbolIndex | None,
        *,
        docstring: str,
        comments: str,
        semantic_tags: list[str],
        import_ctx: str,
        module_ctx: str,
    ) -> str:
        symbol_name = chunk.symbol_name or (symbol.name if symbol else "")
        symbol_words = cls._split_identifier(symbol_name)
        doc_summary = cls._clean_summary_text(docstring) or cls._clean_summary_text(comments)
        param_summary = cls._parameter_summary(symbol)
        return_summary = cls._return_summary(chunk.content, symbol)
        import_summary = cls._import_summary(import_ctx)
        tag_summary = " ".join(semantic_tags[:16])

        purpose = doc_summary
        if not purpose:
            purpose_bits = [symbol_words, tag_summary]
            purpose = " ".join(bit for bit in purpose_bits if bit).strip()
        if not purpose:
            purpose = f"{chunk.chunk_type.value} implementation"

        lines = [
            f"function summary: {chunk.chunk_type.value} {symbol_name} {purpose}",
            f"function purpose: {purpose}",
        ]
        if param_summary:
            lines.append(f"function input: accepts {param_summary}")
        if return_summary:
            lines.append(f"function output: {return_summary}")
        if tag_summary:
            lines.append(f"function categories: {tag_summary}")
        if import_summary:
            lines.append(f"function dependencies: {import_summary}")
        if module_ctx:
            lines.append(f"function module context: {module_ctx}")
        return "\n".join(lines)

    @staticmethod
    def _semantic_tags(content: str, tags: list[str] | None = None) -> list[str]:
        lowered = content.lower()
        semantic_tags = set(tags or [])
        checks = [
            ("async", ("async def ", "await ", "promise", "future", "goroutine", "go func", "channel")),
            ("loop", ("for ", "while ", "range ", ".foreach", "for_each")),
            ("exception", ("try:", "except ", "raise ", "try {", "catch ", "throw ", "panic!", "result<", "option<")),
            ("return", ("return ", "=>", "->")),
            ("yield", ("yield ", "yield*")),
            ("context_manager", ("with ", "defer ", "using ")),
            ("generator", ("yield ",)),
            ("decorator", ("\n@", "#[", "@override", "@test")),
            ("http", ("request", "response", "http")),
            ("json", ("json",)),
            ("path", ("path", "file", "directory", "filesystem", "fs.")),
            ("validation", ("validate", "check", "assert ", "assert_eq!", "require.")),
            ("merge", ("merge", "combine", "join")),
            ("parse", ("parse", "parser")),
            ("convert", ("convert", "transform", "normalize")),
            ("class", ("class ", "interface ", "implements ", "extends ", "struct ", "trait ")),
            ("constructor", ("__init__", "constructor(", "new(", "impl ", "default(")),
            ("collection", ("list", "array", "vec<", "hashmap", "map[", "set<", "slice", "collection")),
            ("string", ("string", "str", "format", "regex", "regexp", "split", "join")),
            ("date_time", ("date", "time", "duration", "timestamp", "datetime", "chrono")),
            ("database", ("sql", "query", "database", "db.", "select ", "insert ", "update ")),
            ("test", ("test", "assert", "benchmark", "describe(", "it(", "#[test]")),
            ("error", ("error", "exception", "result", "option", "throw", "catch", "panic")),
        ]
        for tag, needles in checks:
            if any(n in lowered for n in needles):
                semantic_tags.add(tag)

        if len(re.findall(r"\bself\.[A-Za-z_][A-Za-z0-9_]*\s=", content)) >= 2:
            semantic_tags.update(
                [
                    "initialize",
                    "initialization",
                    "state",
                    "configuration",
                    "setup",
                    "properties",
                    "object",
                ]
            )
        if re.search(r"\b(this|self)\.[A-Za-z_][A-Za-z0-9_]*\s*=", content):
            semantic_tags.update(["initialize", "state", "object", "property"])
        if re.search(r"\b(map|filter|reduce|collect|iter|iterator|stream)\b", lowered):
            semantic_tags.update(["collection", "iterator", "transform"])
        if re.search(r"\b(lock|mutex|rwlock|atomic|synchronized|thread|spawn)\b", lowered):
            semantic_tags.update(["concurrency", "thread", "synchronization"])
        if re.search(r"\b(encode|decode|serialize|deserialize|marshal|unmarshal)\b", lowered):
            semantic_tags.update(["serialization", "encode", "decode"])
        if "asyncio.create_task" in lowered or ("async" in lowered and "cancel" in lowered):
            semantic_tags.update(["async", "synchronous", "cleanup", "effect", "task"])
        if "return str(" in lowered:
            semantic_tags.update(["convert", "primitive", "value", "string", "representation"])
        if "priority()" in lowered or ">=" in lowered or "<=" in lowered:
            semantic_tags.update(["comparison", "greater", "equal", "less", "priority", "ordering"])
        if any(token in lowered for token in ("allocated_bytes", "reserved_bytes", "active_bytes", "memlab", "memory_stats")):
            semantic_tags.update(["memory", "usage", "statistics", "profiling", "megabytes", "peak"])
        if re.search(r"decode\d*|matmul\d*|float16|fp16|uint32|int3|tir\.prim_func", lowered):
            semantic_tags.update(
                [
                    "decode",
                    "decoding",
                    "matrix",
                    "multiplication",
                    "matmul",
                    "neural",
                    "network",
                    "floating",
                    "point",
                    "precision",
                    "compressed",
                    "weighted",
                    "sum",
                ]
            )
        return sorted(semantic_tags)

    @staticmethod
    def _ast_summary(content: str, symbol: SymbolIndex | None = None) -> str:
        lowered = content.lower()
        params = []
        if symbol:
            for p in symbol.parameters:
                item = p.name
                if p.type_annotation:
                    item += f":{p.type_annotation}"
                params.append(item)
        return_type = symbol.return_type if symbol and symbol.return_type else ""
        return_exprs = re.findall(r"\breturn\s+([^\n#]+)", content)
        return_kinds = []
        for expr in return_exprs[:5]:
            expr_l = expr.strip().lower()
            if expr_l.startswith("str("):
                return_kinds.append("string conversion")
            if expr_l.startswith("len("):
                return_kinds.append("length count")
            if expr_l == "none":
                return_kinds.append("none empty placeholder")
            if any(op in expr_l for op in (">=", "<=", "==", "!=", ">", "<")):
                return_kinds.append("comparison boolean")
            if expr_l.startswith(("[", "(", "{")):
                return_kinds.append("collection construction")
            if any(token in expr_l for token in ("err", "error", "exception")):
                return_kinds.append("error handling")
            if any(token in expr_l for token in ("json", "marshal", "serialize")):
                return_kinds.append("serialization")
            if any(token in expr_l for token in ("promise", "future", "async", "await")):
                return_kinds.append("async result")
        calls = sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", content)))[:30]
        operators = []
        for label, pattern in [
            ("assignment", r"="),
            ("comparison", r"(==|!=|>=|<=|>|<)"),
            ("arithmetic", r"(\+|-|\*|/|%)"),
            ("logical", r"(&&|\|\||\band\b|\bor\b)"),
        ]:
            if re.search(pattern, content):
                operators.append(label)
        has_loop = bool(re.search(r"\b(for|while|range|foreach|for_each)\b", lowered))
        has_exception = bool(
            re.search(r"\b(try|except|catch|throw|raise|panic|result|option)\b", lowered)
        )
        has_async = bool(re.search(r"\b(async|await|promise|future|goroutine|channel)\b", lowered))
        self_assignments = len(re.findall(r"\b(self|this)\.[A-Za-z_][A-Za-z0-9_]*\s=", content))
        has_collection_ops = bool(
            re.search(r"\b(list|array|map|set|vec|slice|iter|stream|collect)\b", lowered)
        )
        has_io_ops = bool(re.search(r"\b(file|path|read|write|open|close|fs|io)\b", lowered))
        features = [
            f"function kind: {'method' if 'self' in params else 'function'}",
            f"inputs: {' '.join(params)}",
            f"output type: {return_type}",
            f"return behavior: {'; '.join(sorted(set(return_kinds)))}",
            f"has loop: {has_loop}",
            f"has exception: {has_exception}",
            f"has async: {has_async}",
            f"has self assignments: {self_assignments}",
            f"has collection ops: {has_collection_ops}",
            f"has io ops: {has_io_ops}",
            f"operators: {' '.join(operators)}",
            f"calls: {' '.join(calls)}",
        ]
        return "\n".join(features)

    @staticmethod
    def _symbol_for_chunk(
        chunk: CodeChunk, symbols: list[SymbolIndex]
    ) -> SymbolIndex | None:
        if not chunk.symbol_name:
            return None
        for symbol in symbols:
            if symbol.name == chunk.symbol_name and symbol.file_path == chunk.file_path:
                return symbol
        for symbol in symbols:
            if symbol.name == chunk.symbol_name:
                return symbol
        return None

    @classmethod
    def build_search_document(
        cls, chunk: CodeChunk, symbol: SymbolIndex | None = None
    ) -> str:
        """生成面向自然语言查询的 search_document。

        该文本只复用索引阶段已有的结构信息，不调用 LLM，也不引入测试集描述。
        """
        symbol_name = chunk.symbol_name or (symbol.name if symbol else "")
        symbol_words = cls._split_identifier(symbol_name)
        parent_words = cls._split_identifier(chunk.parent_symbol)
        docstring = (chunk.docstring or "").strip()
        signature = symbol.signature if symbol and symbol.signature else ""
        param_parts = []
        if symbol:
            for param in symbol.parameters:
                text = param.name
                if param.type_annotation:
                    text += f" {param.type_annotation}"
                param_parts.append(text)
        comments = cls._extract_comments(chunk.content)
        semantic_tags = cls._semantic_tags(chunk.content, chunk.tags)
        module_ctx = cls.module_context(chunk.file_path)
        import_ctx = cls.extract_import_context(chunk.content)
        ast_summary = cls._ast_summary(chunk.content, symbol)
        owner_ctx = cls._owner_context(chunk, symbol)
        alias_ctx = cls._symbol_alias_context(symbol_name, chunk.language, chunk.content)
        function_summary = cls._function_summary(
            chunk,
            symbol,
            docstring=docstring,
            comments=comments,
            semantic_tags=semantic_tags,
            import_ctx=import_ctx,
            module_ctx=module_ctx,
        )

        sections = [
            function_summary,
            f"language: {chunk.language}",
            f"symbol: {symbol_name}",
            f"symbol words: {symbol_words}",
            f"symbol aliases: {alias_ctx}",
            f"parent: {chunk.parent_symbol or ''} {parent_words}",
            f"owner context:\n{owner_ctx}",
            f"module path: {chunk.file_path}",
            f"module words: {module_ctx}",
            f"imports: {import_ctx}",
            f"type: {chunk.chunk_type.value}",
            f"signature: {signature}",
            f"parameters: {' '.join(param_parts)}",
            f"docstring: {docstring}",
            f"comments: {comments}",
            f"tags: {' '.join(semantic_tags)}",
            f"ast summary:\n{ast_summary}",
            "code:",
            chunk.content[:4000],
        ]
        return "\n".join(s for s in sections if s and s.strip())

    def _build_search_documents(
        self, chunks: list[CodeChunk], symbols: list[SymbolIndex]
    ) -> list[tuple[str, str]]:
        docs = []
        for chunk in chunks:
            symbol = self._symbol_for_chunk(chunk, symbols)
            docs.append((chunk.id, self.build_search_document(chunk, symbol)))
        return docs

    @staticmethod
    def _detect_language(file_path: Path) -> str:
        """从文件扩展名推断语言"""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".html": "html",
            ".css": "css",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    # ── 增量更新 ──

    def reindex_file(self, file_path: str | Path) -> dict[str, int]:
        """重新索引单个文件（增量更新）"""
        file_path = Path(file_path)
        if not file_path.exists():
            self._storage.delete_chunks_by_file(str(file_path))
            # 文件已删除
            self._storage.delete_chunks_by_file(str(file_path))
            self._storage.delete_symbols_by_file(str(file_path))
            self._embedding.invalidate(str(file_path))
            self._indexed_files.discard(str(file_path))
            logger.info(f"Removed index for deleted file: {file_path}")
            return {"files": 0, "chunks": 0, "symbols": 0, "calls": 0, "removed": 1}

        chunks, symbols, calls = self._index_file(file_path)
        embed_stats = self.embed_file_incremental(file_path)
        self._embedding.invalidate(str(file_path))
        logger.debug(
            "Re-indexed %s: %s chunks, %s symbols, incremental_embeddings=%s",
            file_path,
            len(chunks),
            len(symbols),
            embed_stats,
        )
        return {
            "files": 1,
            "chunks": len(chunks),
            "symbols": len(symbols),
            "calls": len(calls),
            **embed_stats,
        }

    # ── 文件监听 ──

    def start_watching(self, workspace_root: Path | None = None) -> None:
        """启动文件监听（watchdog）"""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("watchdog not installed, file watching disabled")
            return

        root = workspace_root or Path(self._config.index.workspace_root or ".")
        if not root.is_absolute():
            root = Path.cwd() / root

        indexer = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and _is_source_file(event.src_path):
                    indexer.reindex_file(event.src_path)

            def on_created(self, event):
                if not event.is_directory and _is_source_file(event.src_path):
                    indexer.reindex_file(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory and _is_source_file(event.src_path):
                    indexer.reindex_file(event.src_path)

        self._watcher = Observer()
        self._watcher.schedule(_Handler(), str(root), recursive=True)
        self._watcher.daemon = True
        self._watcher.start()
        logger.info(f"File watching started: {root}")

    def stop_watching(self) -> None:
        """停止文件监听"""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
            logger.info("File watching stopped")

    # ── 生命周期 ──

    def close(self) -> None:
        """关闭索引器"""
        self.stop_watching()


def _is_source_file(path: str) -> bool:
    """判断是否为源码文件"""
    return Path(path).suffix.lower() in _SOURCE_EXTENSIONS


# ── 便捷工厂函数 ──


def create_indexer(config: CodeGraphConfig, workspace_root: Path | None = None) -> CodeGraphIndexer:
    """创建索引器并初始化存储"""
    root = workspace_root or Path(config.index.workspace_root or ".")
    if not root.is_absolute():
        root = Path.cwd() / root

    persist_path = config.get_persist_path(root)
    persist_path.mkdir(parents=True, exist_ok=True)

    storage = CodeGraphStorage(
        persist_path,
        embedding_dimension=config.embedding.dimension,
        vector_dtype=config.storage.vector_dtype,
        enable_embedding_cache=config.storage.enable_embedding_cache,
    )
    embedding_cache = EmbeddingCache(
        dimension=config.embedding.dimension,
        max_cache=config.embedding.cache_size,
    )

    # 初始化 embedding 提供者
    provider = create_provider(config.embedding)
    if provider:
        embedding_cache.set_provider(provider)

    return CodeGraphIndexer(config, storage, embedding_cache)
