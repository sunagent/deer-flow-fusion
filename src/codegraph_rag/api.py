"""Public API for the standalone CodeGraph RAG engine."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .call_graph import CallGraph
from .config import CodeGraphConfig
from .data_flow import DataFlowAnalyzer
from .graph_search import GraphSearchEngine
from .indexer import CodeGraphIndexer, create_indexer
from .models import SearchResult
from .search import SearchEngine


class CodeGraphRAG:
    """Small standalone facade around the CodeGraph index and search engines.

    This class intentionally has no DeerFlow, LangChain, or LangGraph dependency.
    Product integrations can wrap it with their own tool/runtime layer.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        index_dir: str | Path | None = None,
        config: CodeGraphConfig | None = None,
        auto_index: bool = True,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.config = config or CodeGraphConfig(enabled=True)
        self.config.index.workspace_root = str(self.workspace)
        self._lock = threading.RLock()
        if index_dir is not None:
            self.config.storage.persist_directory = str(Path(index_dir))

        self.indexer: CodeGraphIndexer = create_indexer(self.config, self.workspace)
        self.call_graph = CallGraph()
        self.data_flow = DataFlowAnalyzer()
        self.graph_engine = GraphSearchEngine(
            call_graph=self.call_graph,
            data_flow=self.data_flow,
            storage=self.indexer._storage,
        )
        self.search_engine = SearchEngine(
            storage=self.indexer._storage,
            embedding_cache=self.indexer._embedding,
            config=self.config,
            graph_engine=self.graph_engine,
            call_graph=self.call_graph,
        )

        if auto_index:
            self.index()

    def index(self) -> dict[str, int]:
        """Build or refresh the workspace index and CallGraph."""
        with self._lock:
            stats = self.indexer.index_workspace(self.workspace)
            self.call_graph.add_symbols(self.indexer._last_symbols)
            self.call_graph.add_calls(self.indexer._last_calls)
            return stats

    def reindex_file(self, file_path: str | Path) -> dict[str, int]:
        """Reindex one file and refresh the in-memory CallGraph for that file."""
        with self._lock:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace / path
            self.call_graph.remove_file(str(path))
            stats = self.indexer.reindex_file(path)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                language = self.indexer._detect_language(path)
                self.call_graph.add_file(str(path), content, language)
            except Exception:
                pass
            return stats

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        active_file: str | None = None,
        language: str | None = None,
        repo: str | None = None,
        allow_cross_language: bool = False,
    ) -> list[SearchResult]:
        """Search with the frozen Graph + Vector + BM25 RRF pipeline."""
        context: dict[str, Any] | None = None
        if active_file or language or repo or allow_cross_language:
            context = {
                "active_file": active_file,
                "language": language,
                "repo": repo,
                "workspace": str(self.workspace),
                "allow_cross_language": allow_cross_language,
            }
        with self._lock:
            return self.search_engine.search(query, top_k=top_k, context=context)

    def search_explain(
        self,
        query: str,
        *,
        top_k: int = 10,
        active_file: str | None = None,
        language: str | None = None,
        repo: str | None = None,
        allow_cross_language: bool = False,
    ) -> dict:
        """Search and return deterministic trace metadata with the results."""
        context: dict[str, Any] | None = None
        if active_file or language or repo or allow_cross_language:
            context = {
                "active_file": active_file,
                "language": language,
                "repo": repo,
                "workspace": str(self.workspace),
                "allow_cross_language": allow_cross_language,
            }
        with self._lock:
            return self.search_engine.search_explain(query, top_k=top_k, context=context)

    def close(self) -> None:
        self.indexer.close()


__all__ = ["CodeGraphRAG"]
