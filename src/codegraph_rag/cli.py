"""Command line entry points for CodeGraph RAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import CodeGraphRAG
from .config import CodeGraphConfig


def _build_config(args: argparse.Namespace) -> CodeGraphConfig:
    config = CodeGraphConfig(enabled=True)
    config.embedding.provider = args.embedding_provider
    config.embedding.model_name = args.embedding_model
    config.storage.persist_directory = args.index_dir
    config.storage.vector_dtype = args.vector_dtype
    config.search.default_top_k = args.top_k
    return config


def _result_to_dict(result) -> dict:
    return {
        "id": result.id,
        "symbol_name": result.symbol_name,
        "qualified_name": result.qualified_name,
        "file_path": result.file_path,
        "line_number": result.line_number,
        "chunk_type": str(result.chunk_type.value if hasattr(result.chunk_type, "value") else result.chunk_type),
        "score": round(float(result.score), 6),
        "signature": result.signature,
        "docstring": result.docstring,
        "language": result.language,
        "snippet": result.snippet,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codegraph-rag")
    parser.add_argument("--index-dir", default=".codegraph", help="Index directory relative to the workspace")
    parser.add_argument("--embedding-provider", default="local", choices=["local", "jina", "jina-code", "openai", "ollama", "nomic", "none"])
    parser.add_argument("--embedding-model", default="jinaai/jina-embeddings-v2-base-code")
    parser.add_argument("--vector-dtype", default="float32", choices=["float32", "int8"])
    parser.add_argument("--top-k", type=int, default=10)

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build a workspace index")
    p_index.add_argument("workspace")

    p_search = sub.add_parser("search", help="Search a workspace")
    p_search.add_argument("workspace")
    p_search.add_argument("query")
    p_search.add_argument("--active-file")
    p_search.add_argument("--language")
    p_search.add_argument("--repo")
    p_search.add_argument("--allow-cross-language", action="store_true")
    p_search.add_argument("--json", action="store_true", help="Print JSON instead of a compact table")
    p_search.add_argument("--explain", action="store_true", help="Print deterministic routing/fusion trace")

    args = parser.parse_args(argv)
    if args.embedding_provider == "none":
        args.embedding_provider = None

    workspace = Path(getattr(args, "workspace")).resolve()
    config = _build_config(args)

    if args.command == "index":
        rag = CodeGraphRAG(workspace, config=config, auto_index=False)
        stats = rag.index()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    # Build the in-memory CallGraph for the current process. SQLite index files
    # are persisted, while CallGraph hydration is intentionally explicit here.
    rag = CodeGraphRAG(workspace, config=config, auto_index=True)
    if args.explain:
        payload = rag.search_explain(
            args.query,
            top_k=args.top_k,
            active_file=args.active_file,
            language=args.language,
            repo=args.repo,
            allow_cross_language=args.allow_cross_language,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    results = rag.search(
        args.query,
        top_k=args.top_k,
        active_file=args.active_file,
        language=args.language,
        repo=args.repo,
        allow_cross_language=args.allow_cross_language,
    )
    if args.json:
        print(json.dumps([_result_to_dict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        for i, result in enumerate(results, 1):
            name = result.qualified_name or result.symbol_name or result.id
            print(f"{i:>2}. {result.score:.4f} {name}  {result.file_path}:{result.line_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
