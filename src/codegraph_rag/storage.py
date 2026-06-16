"""CodeGraph 存储层 — SQLite + FTS5 + vec0"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Callable, Literal

from .models import (
    ChunkType,
    CodeChunk,
    Parameter,
    SearchResult,
    SymbolIndex,
    SymbolType,
)

logger = logging.getLogger(__name__)

_NL_FTS5_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "code", "designed", "do", "does", "for", "from", "function",
    "had", "has", "have", "in", "into", "is", "it", "its", "method",
    "not", "of", "on", "or", "purpose", "return", "that", "the", "this",
    "to", "used", "using", "was", "were", "which", "with",
}

# ── vec0 可选依赖 ──

_vec_available = False
_vec_module = None

try:
    import sqlite_vec as _vec_module  # type: ignore[import-untyped]

    _vec_available = True
except ImportError:
    logger.warning("sqlite-vec 未安装，向量搜索功能不可用。安装方式: pip install sqlite-vec")


# ── 建表 SQL ──

_CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    symbol_name TEXT,
    parent_symbol TEXT,
    start_line INTEGER,
    end_line INTEGER,
    tags TEXT,
    docstring TEXT
);
"""

_CREATE_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    symbol_name,
    docstring,
    tags,
    content='chunks',
    content_rowid='rowid'
);
"""

_CREATE_CHUNKS_FTS_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, symbol_name, docstring, tags)
    VALUES (new.rowid, new.content, new.symbol_name, new.docstring, new.tags);
END;
"""

_CREATE_CHUNKS_FTS_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, docstring, tags)
    VALUES ('delete', old.rowid, old.content, old.symbol_name, old.docstring, old.tags);
END;
"""

_CREATE_CHUNKS_FTS_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, docstring, tags)
    VALUES ('delete', old.rowid, old.content, old.symbol_name, old.docstring, old.tags);
    INSERT INTO chunks_fts(rowid, content, symbol_name, docstring, tags)
    VALUES (new.rowid, new.content, new.symbol_name, new.docstring, new.tags);
END;
"""

_CREATE_SYMBOLS_TABLE = """
CREATE TABLE IF NOT EXISTS symbols (
    qualified_name TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    signature TEXT,
    return_type TEXT,
    parameters TEXT,
    base_classes TEXT,
    implemented_interfaces TEXT,
    decorators TEXT,
    class_methods TEXT
);
"""

_CREATE_SYMBOLS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    signature,
    content='symbols',
    content_rowid='rowid'
);
"""

_CREATE_SYMBOLS_FTS_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS symbols_fts_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, signature)
    VALUES (new.rowid, new.name, new.signature);
END;
"""

_CREATE_SYMBOLS_FTS_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS symbols_fts_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, signature)
    VALUES ('delete', old.rowid, old.name, old.signature);
END;
"""

_CREATE_SYMBOLS_FTS_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS symbols_fts_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, signature)
    VALUES ('delete', old.rowid, old.name, old.signature);
    INSERT INTO symbols_fts(rowid, name, signature)
    VALUES (new.rowid, new.name, new.signature);
END;
"""

_CREATE_EMBEDDINGS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding {vector_type}[{dimension}]
);
"""

_CREATE_SEARCH_DOCS_TABLE = """
CREATE TABLE IF NOT EXISTS chunk_search_docs (
    chunk_id TEXT PRIMARY KEY,
    search_text TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);
"""

_CREATE_DOC_EMBEDDINGS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_doc_embeddings USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding {vector_type}[{dimension}]
);
"""

_CREATE_EMBEDDING_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS chunk_embedding_cache (
    chunk_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    search_doc_hash TEXT NOT NULL,
    vector_dtype TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class CodeGraphStorage:
    """CodeGraph 存储层 — 基于 SQLite 的混合检索"""

    def __init__(
        self,
        persist_path: Path,
        embedding_dimension: int = 1024,
        vector_dtype: Literal["float32", "int8"] = "float32",
        enable_embedding_cache: bool = True,
    ):
        self._persist_path = persist_path
        self._embedding_dimension = embedding_dimension
        self._vector_dtype = vector_dtype if vector_dtype in {"float32", "int8"} else "float32"
        self._vector_type = "int8" if self._vector_dtype == "int8" else "float"
        self._enable_embedding_cache = enable_embedding_cache
        self._vec_available = _vec_available

        persist_path.mkdir(parents=True, exist_ok=True)
        db_path = persist_path / "codegraph.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")

        if self._vec_available and _vec_module is not None:
            try:
                self._conn.enable_load_extension(True)
                _vec_module.load(self._conn)
                logger.info("sqlite-vec 扩展加载成功")
            except Exception:
                logger.warning("sqlite-vec 扩展加载失败，向量搜索不可用", exc_info=True)
                self._vec_available = False

        self._create_tables()

    # ── 内部工具 ──

    def _create_tables(self) -> None:
        cur = self._conn.executescript(
            _CREATE_CHUNKS_TABLE
            + _CREATE_CHUNKS_FTS
            + _CREATE_CHUNKS_FTS_INSERT_TRIGGER
            + _CREATE_CHUNKS_FTS_DELETE_TRIGGER
            + _CREATE_CHUNKS_FTS_UPDATE_TRIGGER
            + _CREATE_SYMBOLS_TABLE
            + _CREATE_SYMBOLS_FTS
            + _CREATE_SYMBOLS_FTS_INSERT_TRIGGER
            + _CREATE_SYMBOLS_FTS_DELETE_TRIGGER
            + _CREATE_SYMBOLS_FTS_UPDATE_TRIGGER
            + _CREATE_SEARCH_DOCS_TABLE
            + _CREATE_EMBEDDING_CACHE_TABLE
        )
        if self._vec_available:
            try:
                self._conn.execute(
                    _CREATE_EMBEDDINGS_TABLE.format(
                        dimension=self._embedding_dimension,
                        vector_type=self._vector_type,
                    )
                )
                self._conn.execute(
                    _CREATE_DOC_EMBEDDINGS_TABLE.format(
                        dimension=self._embedding_dimension,
                        vector_type=self._vector_type,
                    )
                )
                self._validate_vector_table_dtype("chunk_embeddings")
                self._validate_vector_table_dtype("chunk_doc_embeddings")
                self._conn.commit()
            except Exception:
                logger.warning("vec0 虚拟表创建失败，向量搜索不可用", exc_info=True)
                self._vec_available = False

    def _validate_vector_table_dtype(self, table_name: str) -> None:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if not row or not row[0]:
            return
        sql = str(row[0]).lower()
        expected = "int8[" if self._vector_dtype == "int8" else "float["
        if expected not in sql:
            raise RuntimeError(
                f"{table_name} was created with a different vector dtype; "
                f"rebuild this CodeGraph index for vector_dtype={self._vector_dtype}"
            )

    @staticmethod
    def _serialize_list(items: list | None) -> str | None:
        if items is None:
            return None
        return json.dumps(items, ensure_ascii=False)

    @staticmethod
    def _deserialize_list(text: str | None) -> list:
        if not text:
            return []
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _serialize_params(params: list[Parameter] | None) -> str | None:
        if params is None:
            return None
        return json.dumps(
            [
                {
                    "name": p.name,
                    "type_annotation": p.type_annotation,
                    "default_value": p.default_value,
                    "is_variadic": p.is_variadic,
                    "is_keyword_only": p.is_keyword_only,
                    "is_variadic_keyword": p.is_variadic_keyword,
                }
                for p in params
            ],
            ensure_ascii=False,
        )

    @staticmethod
    def _deserialize_params(text: str | None) -> list[Parameter]:
        if not text:
            return []
        try:
            raw = json.loads(text)
            return [
                Parameter(
                    name=p["name"],
                    type_annotation=p.get("type_annotation"),
                    default_value=p.get("default_value"),
                    is_variadic=p.get("is_variadic", False),
                    is_keyword_only=p.get("is_keyword_only", False),
                    is_variadic_keyword=p.get("is_variadic_keyword", False),
                )
                for p in raw
            ]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    @staticmethod
    def _content_hash(text: str | None) -> str:
        return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _serialize_float32_vector(embedding: list[float]) -> bytes:
        if _vec_module is not None and hasattr(_vec_module, "serialize_float32"):
            return _vec_module.serialize_float32(embedding)
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _vector_sql_expr(self) -> str:
        if self._vector_dtype == "int8":
            return "vec_quantize_int8(?, 'unit')"
        return "?"

    @staticmethod
    def _language_filter_values(language: str | None) -> list[str]:
        if not language:
            return []
        normalized = language.lower().strip().lstrip(".")
        aliases = {
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "jsx": "javascript",
            "mjs": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "rs": "rust",
            "golang": "go",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {"typescript", "javascript"}:
            return ["typescript", "javascript"]
        return [normalized]

    @staticmethod
    def _row_to_chunk(row: tuple) -> CodeChunk:
        return CodeChunk(
            id=row[0],
            content=row[1],
            file_path=row[2],
            language=row[3],
            chunk_type=ChunkType(row[4]),
            symbol_name=row[5],
            parent_symbol=row[6],
            start_line=row[7] or 0,
            end_line=row[8] or 0,
            tags=CodeGraphStorage._deserialize_list(row[9]),
            docstring=row[10],
        )

    @staticmethod
    def _row_to_symbol(row: tuple) -> SymbolIndex:
        return SymbolIndex(
            name=row[1],
            qualified_name=row[0],
            symbol_type=SymbolType(row[2]),
            file_path=row[3],
            line_number=row[4] or 0,
            signature=row[5],
            return_type=row[6],
            parameters=CodeGraphStorage._deserialize_params(row[7]),
            base_classes=CodeGraphStorage._deserialize_list(row[8]),
            implemented_interfaces=CodeGraphStorage._deserialize_list(row[9]),
            decorators=CodeGraphStorage._deserialize_list(row[10]),
            class_methods=CodeGraphStorage._deserialize_list(row[11]),
        )

    # ── Chunk 操作 ──

    def upsert_chunks(self, chunks: list[CodeChunk]) -> None:
        if not chunks:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO chunks (id, content, file_path, language, chunk_type,
                                    symbol_name, parent_symbol, start_line, end_line,
                                    tags, docstring)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    file_path=excluded.file_path,
                    language=excluded.language,
                    chunk_type=excluded.chunk_type,
                    symbol_name=excluded.symbol_name,
                    parent_symbol=excluded.parent_symbol,
                    start_line=excluded.start_line,
                    end_line=excluded.end_line,
                    tags=excluded.tags,
                    docstring=excluded.docstring
                """,
                [
                    (
                        c.id,
                        c.content,
                        c.file_path,
                        c.language,
                        c.chunk_type.value,
                        c.symbol_name,
                        c.parent_symbol,
                        c.start_line,
                        c.end_line,
                        self._serialize_list(c.tags),
                        c.docstring,
                    )
                    for c in chunks
                ],
            )

    def delete_chunks_by_file(self, file_path: str) -> None:
        cur = self._conn.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,))
        chunk_ids = [row[0] for row in cur.fetchall()]
        with self._conn:
            for chunk_id in chunk_ids:
                self._conn.execute(
                    "DELETE FROM chunk_search_docs WHERE chunk_id = ?",
                    (chunk_id,),
                )
                if self._vec_available:
                    self._conn.execute(
                        "DELETE FROM chunk_doc_embeddings WHERE chunk_id = ?",
                        (chunk_id,),
                    )
                    self._conn.execute(
                        "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
                        (chunk_id,),
                    )
                self._conn.execute(
                    "DELETE FROM chunk_embedding_cache WHERE chunk_id = ?",
                    (chunk_id,),
                )
            self._conn.execute(
                "DELETE FROM chunks WHERE file_path = ?", (file_path,)
            )

    def list_indexed_files(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT file_path FROM chunks ORDER BY file_path"
        )
        return [str(row[0]) for row in cur.fetchall()]

    def get_chunk_ids_by_file(self, file_path: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT id FROM chunks WHERE file_path = ? ORDER BY start_line",
            (file_path,),
        )
        return [row[0] for row in cur.fetchall()]

    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with self._conn:
            for chunk_id in chunk_ids:
                self._conn.execute(
                    "DELETE FROM chunk_search_docs WHERE chunk_id = ?",
                    (chunk_id,),
                )
                if self._vec_available:
                    self._conn.execute(
                        "DELETE FROM chunk_doc_embeddings WHERE chunk_id = ?",
                        (chunk_id,),
                    )
                    self._conn.execute(
                        "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
                        (chunk_id,),
                    )
                self._conn.execute(
                    "DELETE FROM chunk_embedding_cache WHERE chunk_id = ?",
                    (chunk_id,),
                )
                self._conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        cur = self._conn.execute(
            "SELECT id, content, file_path, language, chunk_type, "
            "symbol_name, parent_symbol, start_line, end_line, tags, docstring "
            "FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    # ── Symbol 操作 ──

    def upsert_symbols(self, symbols: list[SymbolIndex]) -> None:
        if not symbols:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO symbols (qualified_name, name, symbol_type, file_path,
                                     line_number, signature, return_type, parameters,
                                     base_classes, implemented_interfaces, decorators,
                                     class_methods)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(qualified_name) DO UPDATE SET
                    name=excluded.name,
                    symbol_type=excluded.symbol_type,
                    file_path=excluded.file_path,
                    line_number=excluded.line_number,
                    signature=excluded.signature,
                    return_type=excluded.return_type,
                    parameters=excluded.parameters,
                    base_classes=excluded.base_classes,
                    implemented_interfaces=excluded.implemented_interfaces,
                    decorators=excluded.decorators,
                    class_methods=excluded.class_methods
                """,
                [
                    (
                        s.qualified_name,
                        s.name,
                        s.symbol_type.value,
                        s.file_path,
                        s.line_number,
                        s.signature,
                        s.return_type,
                        self._serialize_params(s.parameters),
                        self._serialize_list(s.base_classes),
                        self._serialize_list(s.implemented_interfaces),
                        self._serialize_list(s.decorators),
                        self._serialize_list(s.class_methods),
                    )
                    for s in symbols
                ],
            )

    def delete_symbols_by_file(self, file_path: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM symbols WHERE file_path = ?", (file_path,)
            )

    def get_symbol(self, qualified_name: str) -> SymbolIndex | None:
        cur = self._conn.execute(
            "SELECT qualified_name, name, symbol_type, file_path, "
            "line_number, signature, return_type, parameters, "
            "base_classes, implemented_interfaces, decorators, class_methods "
            "FROM symbols WHERE qualified_name = ?",
            (qualified_name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_symbol(row)

    def search_symbols_by_name(self, name: str, top_k: int = 10) -> list[SymbolIndex]:
        # FTS5 不支持 . 和特殊字符，转义为短语查询
        safe_name = name.replace(".", " ").replace("(", "").replace(")", "")
        cur = self._conn.execute(
            """
            SELECT s.qualified_name, s.name, s.symbol_type, s.file_path,
                   s.line_number, s.signature, s.return_type, s.parameters,
                   s.base_classes, s.implemented_interfaces, s.decorators, s.class_methods
            FROM symbols_fts f
            JOIN symbols s ON s.rowid = f.rowid
            WHERE symbols_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (safe_name, top_k),
        )
        return [self._row_to_symbol(row) for row in cur.fetchall()]

    def search_symbols_fts(self, name: str, limit: int = 5) -> list[tuple]:
        safe_name = name.replace(".", " ").replace("(", "").replace(")", "")
        cur = self._conn.execute(
            """
            SELECT s.qualified_name, s.file_path, s.name, s.symbol_type
            FROM symbols_fts f
            JOIN symbols s ON s.rowid = f.rowid
            WHERE symbols_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (safe_name, limit),
        )
        return cur.fetchall()

    def get_chunk_by_symbol(self, symbol_name: str) -> CodeChunk | None:
        cur = self._conn.execute(
            "SELECT id, content, file_path, language, chunk_type, "
            "symbol_name, parent_symbol, start_line, end_line, tags, docstring "
            "FROM chunks WHERE symbol_name = ? LIMIT 1",
            (symbol_name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunk_by_qualified_symbol(self, qualified_name: str) -> CodeChunk | None:
        """Resolve a qualified symbol name to the most specific matching chunk."""
        sym = self.get_symbol(qualified_name)
        if sym is None:
            return self.get_chunk_by_symbol(qualified_name.split(".")[-1])

        simple_name = sym.name
        parent = None
        parts = qualified_name.split(".")
        if len(parts) >= 2:
            parent = parts[-2]

        if parent:
            cur = self._conn.execute(
                "SELECT id, content, file_path, language, chunk_type, "
                "symbol_name, parent_symbol, start_line, end_line, tags, docstring "
                "FROM chunks WHERE file_path = ? AND symbol_name = ? AND parent_symbol = ? "
                "ORDER BY start_line LIMIT 1",
                (sym.file_path, simple_name, parent),
            )
            row = cur.fetchone()
            if row is not None:
                return self._row_to_chunk(row)

        cur = self._conn.execute(
            "SELECT id, content, file_path, language, chunk_type, "
            "symbol_name, parent_symbol, start_line, end_line, tags, docstring "
            "FROM chunks WHERE file_path = ? AND symbol_name = ? "
            "ORDER BY start_line LIMIT 1",
            (sym.file_path, simple_name),
        )
        row = cur.fetchone()
        if row is not None:
            return self._row_to_chunk(row)
        return self.get_chunk_by_file(sym.file_path)

    def get_chunk_by_file(self, file_path: str) -> CodeChunk | None:
        cur = self._conn.execute(
            "SELECT id, content, file_path, language, chunk_type, "
            "symbol_name, parent_symbol, start_line, end_line, tags, docstring "
            "FROM chunks WHERE file_path = ? ORDER BY start_line LIMIT 1",
            (file_path,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    # ── Search document 操作（NL 专用双索引）──

    def upsert_search_documents(self, docs: list[tuple[str, str]]) -> None:
        """写入 chunk 的自然语言检索文档。

        search_text 是从符号名、签名、docstring、参数、标签和代码轻量生成的
        NL-friendly 表达，供纯自然语言查询使用；原代码向量索引不受影响。
        """
        if not docs:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO chunk_search_docs (chunk_id, search_text)
                VALUES (?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    search_text=excluded.search_text
                """,
                docs,
            )

    def upsert_search_document(self, chunk_id: str, search_text: str) -> None:
        self.upsert_search_documents([(chunk_id, search_text)])

    def get_search_document(self, chunk_id: str) -> str | None:
        cur = self._conn.execute(
            "SELECT search_text FROM chunk_search_docs WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    # ── Embedding cache metadata ──

    def mark_embedding_cache(
        self,
        chunk_id: str,
        content: str | None,
        search_text: str | None,
    ) -> None:
        self.mark_embedding_caches([(chunk_id, content, search_text)])

    def mark_embedding_caches(
        self,
        rows: list[tuple[str, str | None, str | None]],
    ) -> None:
        if not rows or not self._enable_embedding_cache:
            return
        payload = [
            (
                chunk_id,
                self._content_hash(content),
                self._content_hash(search_text),
                self._vector_dtype,
            )
            for chunk_id, content, search_text in rows
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO chunk_embedding_cache
                    (chunk_id, content_hash, search_doc_hash, vector_dtype, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    search_doc_hash=excluded.search_doc_hash,
                    vector_dtype=excluded.vector_dtype,
                    updated_at=CURRENT_TIMESTAMP
                """,
                payload,
            )

    def _mark_embedding_cache_for_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids or not self._enable_embedding_cache:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        cur = self._conn.execute(
            f"""
            SELECT c.id, c.content, COALESCE(d.search_text, c.content)
            FROM chunks c
            LEFT JOIN chunk_search_docs d ON d.chunk_id = c.id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        )
        self.mark_embedding_caches(
            [(chunk_id, content, search_text) for chunk_id, content, search_text in cur.fetchall()]
        )

    def _mark_code_embedding_cache_for_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids or not self._enable_embedding_cache:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        cur = self._conn.execute(
            f"""
            SELECT c.id, c.content, COALESCE(ec.search_doc_hash, '')
            FROM chunks c
            LEFT JOIN chunk_embedding_cache ec ON ec.chunk_id = c.id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        )
        rows = [
            (chunk_id, self._content_hash(content), search_doc_hash, self._vector_dtype)
            for chunk_id, content, search_doc_hash in cur.fetchall()
        ]
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO chunk_embedding_cache
                (chunk_id, content_hash, search_doc_hash, vector_dtype, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content_hash=excluded.content_hash,
                vector_dtype=excluded.vector_dtype,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )

    def _mark_doc_embedding_cache_for_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids or not self._enable_embedding_cache:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        cur = self._conn.execute(
            f"""
            SELECT c.id, COALESCE(d.search_text, c.content), COALESCE(ec.content_hash, '')
            FROM chunks c
            LEFT JOIN chunk_search_docs d ON d.chunk_id = c.id
            LEFT JOIN chunk_embedding_cache ec ON ec.chunk_id = c.id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        )
        rows = [
            (chunk_id, content_hash, self._content_hash(search_text), self._vector_dtype)
            for chunk_id, search_text, content_hash in cur.fetchall()
        ]
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO chunk_embedding_cache
                (chunk_id, content_hash, search_doc_hash, vector_dtype, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                search_doc_hash=excluded.search_doc_hash,
                vector_dtype=excluded.vector_dtype,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )

    def get_embedding_cache_status(
        self,
        chunk_id: str,
        content: str | None,
        search_text: str | None,
    ) -> tuple[bool, bool]:
        """Return (code_embedding_current, search_document_embedding_current)."""
        if not self._enable_embedding_cache or not self._vec_available:
            return False, False
        cur = self._conn.execute(
            """
            SELECT ec.content_hash, ec.search_doc_hash, ec.vector_dtype,
                   ce.chunk_id IS NOT NULL AS has_code_embedding,
                   de.chunk_id IS NOT NULL AS has_doc_embedding
            FROM chunk_embedding_cache ec
            LEFT JOIN chunk_embeddings ce ON ce.chunk_id = ec.chunk_id
            LEFT JOIN chunk_doc_embeddings de ON de.chunk_id = ec.chunk_id
            WHERE ec.chunk_id = ?
            """,
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return False, False
        content_hash, search_doc_hash, vector_dtype, has_code, has_doc = row
        code_current = (
            bool(has_code)
            and vector_dtype == self._vector_dtype
            and content_hash == self._content_hash(content)
        )
        doc_current = (
            bool(has_doc)
            and vector_dtype == self._vector_dtype
            and search_doc_hash == self._content_hash(search_text)
        )
        return code_current, doc_current

    def embedding_work_items_for_file(
        self,
        file_path: str,
        search_text_transform: Callable[[str], str] | None = None,
    ) -> list[dict[str, str | bool]]:
        """Return per-chunk embedding work needed for an indexed file.

        Unchanged functions keep their existing vectors. Changed code and changed
        search_document can be embedded independently by callers.
        """
        cur = self._conn.execute(
            """
            SELECT c.id, c.content, COALESCE(d.search_text, c.content)
            FROM chunks c
            LEFT JOIN chunk_search_docs d ON d.chunk_id = c.id
            WHERE c.file_path = ?
            ORDER BY c.start_line, c.id
            """,
            (file_path,),
        )
        items: list[dict[str, str | bool]] = []
        for chunk_id, content, search_text in cur.fetchall():
            full_search_text = search_text or content or ""
            doc_embedding_text = (
                search_text_transform(full_search_text)
                if search_text_transform
                else full_search_text
            )
            code_current, doc_current = self.get_embedding_cache_status(
                chunk_id, content, doc_embedding_text
            )
            if not code_current or not doc_current:
                items.append(
                    {
                        "chunk_id": chunk_id,
                        "content": content or "",
                        "search_text": full_search_text,
                        "doc_embedding_text": doc_embedding_text,
                        "needs_code": not code_current,
                        "needs_doc": not doc_current,
                    }
                )
        return items

    # ── FTS5 BM25 搜索 ──

    def cleanup_embedding_cache(
        self,
        *,
        remove_orphans: bool = True,
        max_age_days: int | None = None,
        vacuum: bool = False,
    ) -> dict[str, int | bool]:
        """Clean stale embedding/cache rows without touching current chunks.

        Default behavior is conservative: remove only rows whose chunk_id no
        longer exists in `chunks`. Hash-mismatch rows are refreshed by the
        normal embedding path, so this method does not delete them.
        """
        stats: dict[str, int | bool] = {
            "orphan_code_embeddings": 0,
            "orphan_doc_embeddings": 0,
            "orphan_cache_rows": 0,
            "expired_cache_rows": 0,
            "vacuum": False,
        }
        with self._conn:
            if remove_orphans:
                if self._vec_available:
                    cur = self._conn.execute(
                        """
                        DELETE FROM chunk_embeddings
                        WHERE chunk_id NOT IN (SELECT id FROM chunks)
                        """
                    )
                    stats["orphan_code_embeddings"] = int(
                        cur.rowcount if cur.rowcount != -1 else 0
                    )
                    cur = self._conn.execute(
                        """
                        DELETE FROM chunk_doc_embeddings
                        WHERE chunk_id NOT IN (SELECT id FROM chunks)
                        """
                    )
                    stats["orphan_doc_embeddings"] = int(
                        cur.rowcount if cur.rowcount != -1 else 0
                    )
                cur = self._conn.execute(
                    """
                    DELETE FROM chunk_embedding_cache
                    WHERE chunk_id NOT IN (SELECT id FROM chunks)
                    """
                )
                stats["orphan_cache_rows"] = int(
                    cur.rowcount if cur.rowcount != -1 else 0
                )

            if max_age_days is not None and max_age_days > 0:
                cur = self._conn.execute(
                    """
                    DELETE FROM chunk_embedding_cache
                    WHERE updated_at < datetime('now', ?)
                    """,
                    (f"-{int(max_age_days)} days",),
                )
                stats["expired_cache_rows"] = int(
                    cur.rowcount if cur.rowcount != -1 else 0
                )

        if vacuum:
            self._conn.execute("VACUUM")
            stats["vacuum"] = True
        return stats

    def bm25_search(
        self,
        query: str,
        top_k: int = 30,
        file_pattern: str | None = None,
        language: str | None = None,
        tags: list[str] | None = None,
        or_semantics: bool = False,
    ) -> list[SearchResult]:
        """BM25 全文搜索，支持文件路径和标签过滤"""
        # 构建附加过滤条件
        extra_where_parts: list[str] = []
        extra_params: list[str] = []

        if file_pattern is not None:
            extra_where_parts.append("c.file_path GLOB ?")
            extra_params.append(file_pattern)

        language_values = self._language_filter_values(language)
        if language_values:
            placeholders = ",".join("?" for _ in language_values)
            extra_where_parts.append(f"c.language IN ({placeholders})")
            extra_params.extend(language_values)

        if tags:
            for tag in tags:
                extra_where_parts.append("c.tags LIKE ?")
                extra_params.append(f'%"{tag}"%')

        extra_where = ""
        if extra_where_parts:
            extra_where = " AND " + " AND ".join(extra_where_parts)

        import re as _re

        tokens = [
            t
            for t in _re.findall(r"[A-Za-z0-9_]+", query)
            if len(t) >= 2 and t.lower() not in _NL_FTS5_STOP_WORDS
        ][:96]
        if not tokens:
            return []

        quoted_tokens = [f'"{t}"' for t in tokens]
        fts5_query = (
            " OR ".join(quoted_tokens)
            if or_semantics
            else " ".join(quoted_tokens)
        )
        sql = f"""
            SELECT c.id, c.symbol_name, NULL AS qualified_name, c.file_path,
                   c.start_line AS line_number, c.chunk_type, f.rank AS score,
                   c.content AS snippet, NULL AS signature, c.docstring, c.tags, c.language
            FROM chunks_fts f
            JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ?{extra_where}
            ORDER BY rank
            LIMIT ?
        """
        params: list = [fts5_query] + extra_params + [top_k]

        cur = self._conn.execute(sql, params)
        results: list[SearchResult] = []
        for row in cur.fetchall():
            results.append(
                SearchResult(
                    id=row[0],
                    symbol_name=row[1],
                    qualified_name=row[2],
                    file_path=row[3],
                    line_number=row[4] or 0,
                    chunk_type=ChunkType(row[5]),
                    score=row[6],
                    snippet=row[7],
                    signature=row[8],
                    docstring=row[9],
                    tags=self._deserialize_list(row[10]),
                    language=row[11],
                )
            )
        return results

    # ── 向量搜索 (vec0) ──

    def upsert_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        self.upsert_embeddings([(chunk_id, embedding)])

    def upsert_embeddings(self, rows: list[tuple[str, list[float]]]) -> None:
        if not rows:
            return
        if not self._vec_available:
            logger.warning("vec0 不可用，跳过向量写入: rows=%s", len(rows))
            return
        expr = self._vector_sql_expr()
        with self._conn:
            self._conn.executemany(
                "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id, _ in rows],
            )
            self._conn.executemany(
                f"INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, {expr})",
                [
                    (chunk_id, self._serialize_float32_vector(embedding))
                    for chunk_id, embedding in rows
                ],
            )
            self._mark_code_embedding_cache_for_chunk_ids([chunk_id for chunk_id, _ in rows])

    def upsert_document_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        self.upsert_document_embeddings([(chunk_id, embedding)])

    def upsert_document_embeddings(self, rows: list[tuple[str, list[float]]]) -> None:
        if not rows:
            return
        if not self._vec_available:
            logger.warning("vec0 不可用，跳过文档向量写入: rows=%s", len(rows))
            return
        expr = self._vector_sql_expr()
        with self._conn:
            self._conn.executemany(
                "DELETE FROM chunk_doc_embeddings WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id, _ in rows],
            )
            self._conn.executemany(
                f"INSERT INTO chunk_doc_embeddings (chunk_id, embedding) VALUES (?, {expr})",
                [
                    (chunk_id, self._serialize_float32_vector(embedding))
                    for chunk_id, embedding in rows
                ],
            )
            self._mark_doc_embedding_cache_for_chunk_ids([chunk_id for chunk_id, _ in rows])

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 30,
        file_pattern: str | None = None,
        language: str | None = None,
    ) -> list[SearchResult]:
        if not self._vec_available:
            logger.warning("vec0 不可用，无法执行向量搜索")
            return []
        vec_bytes = self._serialize_float32_vector(query_embedding)
        query_expr = self._vector_sql_expr()

        pattern_sql = " AND c.file_path GLOB ?" if file_pattern else ""
        language_values = self._language_filter_values(language)
        language_sql = (
            f" AND c.language IN ({','.join('?' for _ in language_values)})"
            if language_values
            else ""
        )
        sql = f"""
            SELECT e.chunk_id, e.distance,
                   c.symbol_name, NULL AS qualified_name, c.file_path,
                   c.start_line AS line_number, c.chunk_type,
                   c.content AS snippet, NULL AS signature, c.docstring, c.tags, c.language
            FROM chunk_embeddings e
            LEFT JOIN chunks c ON c.id = e.chunk_id
            WHERE e.embedding MATCH {query_expr} AND k = ?{pattern_sql}{language_sql}
            ORDER BY e.distance
        """
        params = [vec_bytes, top_k]
        if file_pattern:
            params.append(file_pattern)
        params.extend(language_values)
        cur = self._conn.execute(sql, params)
        results: list[SearchResult] = []
        for row in cur.fetchall():
            chunk_id = row[0]
            distance = row[1]
            symbol_name = row[2]
            file_path = row[4] or ""
            line_number = row[5] or 0
            chunk_type_str = row[6]
            snippet = row[7]
            docstring = row[9]
            tags_raw = row[10]
            language_value = row[11]

            try:
                chunk_type = ChunkType(chunk_type_str) if chunk_type_str else ChunkType.FILE
            except ValueError:
                chunk_type = ChunkType.FILE

            # 将距离转换为相似度分数（距离越小越相似，取负值使其与 BM25 分数方向一致）
            score = -distance if distance is not None else 0.0

            results.append(
                SearchResult(
                    id=chunk_id,
                    symbol_name=symbol_name,
                    qualified_name=None,
                    file_path=file_path,
                    line_number=line_number,
                    chunk_type=chunk_type,
                    score=score,
                    snippet=snippet,
                    signature=None,
                    docstring=docstring,
                    tags=self._deserialize_list(tags_raw),
                    language=language_value,
                )
            )
        return results

    def document_vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 30,
        file_pattern: str | None = None,
        language: str | None = None,
    ) -> list[SearchResult]:
        """NL search_document 向量检索。

        与 vector_search 不同，这里匹配的是面向自然语言生成的 search_text，
        但返回的 snippet 仍是原始代码，方便后续 LLM 使用。
        """
        if not self._vec_available:
            logger.warning("vec0 不可用，无法执行文档向量搜索")
            return []
        vec_bytes = self._serialize_float32_vector(query_embedding)
        query_expr = self._vector_sql_expr()

        pattern_sql = " AND c.file_path GLOB ?" if file_pattern else ""
        language_values = self._language_filter_values(language)
        language_sql = (
            f" AND c.language IN ({','.join('?' for _ in language_values)})"
            if language_values
            else ""
        )
        sql = f"""
            SELECT e.chunk_id, e.distance,
                   c.symbol_name, NULL AS qualified_name, c.file_path,
                   c.start_line AS line_number, c.chunk_type,
                   c.content AS snippet, NULL AS signature, c.docstring, c.tags, c.language
            FROM chunk_doc_embeddings e
            LEFT JOIN chunks c ON c.id = e.chunk_id
            WHERE e.embedding MATCH {query_expr} AND k = ?{pattern_sql}{language_sql}
            ORDER BY e.distance
        """
        params = [vec_bytes, top_k]
        if file_pattern:
            params.append(file_pattern)
        params.extend(language_values)
        cur = self._conn.execute(sql, params)
        results: list[SearchResult] = []
        for row in cur.fetchall():
            chunk_id = row[0]
            distance = row[1]
            symbol_name = row[2]
            file_path = row[4] or ""
            line_number = row[5] or 0
            chunk_type_str = row[6]
            snippet = row[7]
            docstring = row[9]
            tags_raw = row[10]
            language_value = row[11]

            try:
                chunk_type = ChunkType(chunk_type_str) if chunk_type_str else ChunkType.FILE
            except ValueError:
                chunk_type = ChunkType.FILE

            score = -distance if distance is not None else 0.0

            results.append(
                SearchResult(
                    id=chunk_id,
                    symbol_name=symbol_name,
                    qualified_name=None,
                    file_path=file_path,
                    line_number=line_number,
                    chunk_type=chunk_type,
                    score=score,
                    snippet=snippet,
                    signature=None,
                    docstring=docstring,
                    tags=self._deserialize_list(tags_raw),
                    language=language_value,
                )
            )
        return results

    # ── 统计 ──

    def get_stats(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM symbols")
        symbol_count = cur.fetchone()[0]

        embedding_count = 0
        if self._vec_available:
            try:
                cur = self._conn.execute("SELECT COUNT(*) FROM chunk_embeddings")
                embedding_count = cur.fetchone()[0]
            except Exception:
                embedding_count = 0
        search_doc_count = 0
        doc_embedding_count = 0
        try:
            cur = self._conn.execute("SELECT COUNT(*) FROM chunk_search_docs")
            search_doc_count = cur.fetchone()[0]
        except Exception:
            search_doc_count = 0
        if self._vec_available:
            try:
                cur = self._conn.execute("SELECT COUNT(*) FROM chunk_doc_embeddings")
                doc_embedding_count = cur.fetchone()[0]
            except Exception:
                doc_embedding_count = 0

        return {
            "chunks": chunk_count,
            "symbols": symbol_count,
            "embeddings": embedding_count,
            "search_docs": search_doc_count,
            "doc_embeddings": doc_embedding_count,
        }

    # ── 生命周期 ──

    def close(self) -> None:
        self._conn.close()

