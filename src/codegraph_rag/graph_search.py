"""图查询引擎 — 统一的图查询接口 + 依赖感知检索"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .call_graph import CallGraph
from .data_flow import DataFlowAnalyzer
from .models import SearchResult, SymbolIndex
from .storage import CodeGraphStorage

if TYPE_CHECKING:
    from .cache import MultiLevelCache
    from .dependency_layer import DependencyLayerManager

logger = logging.getLogger(__name__)


class GraphSearchEngine:
    """图查询引擎：整合调用图、数据流、依赖层的查询能力"""

    def __init__(
        self,
        call_graph: CallGraph,
        data_flow: DataFlowAnalyzer,
        storage: CodeGraphStorage,
        dependency_layer: DependencyLayerManager | None = None,
        cache: MultiLevelCache | None = None,
    ):
        self._call_graph = call_graph
        self._data_flow = data_flow
        self._storage = storage
        self._dependency_layer = dependency_layer
        self._cache = cache

    def set_dependency_layer(self, dl: DependencyLayerManager) -> None:
        self._dependency_layer = dl

    def set_cache(self, cache: MultiLevelCache) -> None:
        self._cache = cache

    # ── 依赖感知检索 ──

    def search_with_scope(
        self, symbol: str, query_type: str, top_k: int = 10,
    ) -> list[SearchResult]:
        """依赖感知的检索：根据查询类型自动确定检索范围

        query_type:
        - symbol_definition: 仅当前层
        - implementation: 当前层 + 直接依赖层
        - debugging: L-1 + L + L+1
        - impact_analysis: L + L+1 + L+2
        - refactoring: L-1 + L + L+1
        - architecture: 所有层
        """
        # 检查缓存
        cache_key = f"search_scope:{symbol}:{query_type}"
        if self._cache:
            cached = self._cache.get_query_cache(cache_key)
            if cached is not None:
                return [SearchResult(**r) for r in cached]

        if self._dependency_layer is None or self._dependency_layer.layer_count == 0:
            # 无依赖层信息，回退到普通搜索
            return self._fallback_search(symbol, top_k)

        # 获取检索范围
        scope_layers = self._dependency_layer.get_search_scope(symbol, query_type)
        scope_nodes = self._dependency_layer.get_nodes_in_scope(symbol, query_type)

        # 在范围内搜索
        results: list[SearchResult] = []
        for node_name in scope_nodes:
            sym = self._storage.get_symbol(node_name)
            if sym:
                # 计算相关性评分
                node_layer = self._dependency_layer.get_node_layer(node_name)
                target_layer = self._dependency_layer.get_node_layer(symbol) or 0
                layer_distance = abs((node_layer or 0) - target_layer)

                results.append(SearchResult(
                    id=f"{sym.file_path}#{sym.name}@{sym.line_number}",
                    symbol_name=sym.name,
                    qualified_name=sym.qualified_name,
                    file_path=sym.file_path,
                    line_number=sym.line_number,
                    chunk_type=sym.symbol_type,
                    score=self._calculate_relevance(
                        node_name, symbol, query_type, layer_distance
                    ),
                    signature=sym.signature,
                    docstring=None,
                    tags=[],
                ))

        # 按分数排序
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        # 写入缓存
        if self._cache and results:
            self._cache.set_query_cache(
                cache_key,
                [r.__dict__ for r in results],
            )

        return results

    def _calculate_relevance(
        self, node: str, query_node: str, query_type: str, layer_distance: int,
    ) -> float:
        """混合相关性评分：距离衰减 + 关系权重 + 查询类型权重"""
        import math

        # 1. 基础距离分（指数衰减）
        base_score = math.exp(-layer_distance * 0.5)

        # 2. 关系类型权重
        relation_weight = 1.0
        if node in self._call_graph.graph:
            if self._call_graph.graph.has_edge(node, query_node):
                relation_weight = 2.0  # 直接调用
            elif self._call_graph.graph.has_edge(query_node, node):
                relation_weight = 1.8  # 被调用

        # 3. 查询类型权重
        type_weights = {
            "implementation": {"function": 2.0, "class": 1.5},
            "debugging": {"function": 1.8, "error": 2.0},
            "impact_analysis": {"function": 1.5, "class": 1.8},
            "refactoring": {"class": 2.0, "function": 1.5},
        }
        # 简化：默认 1.0
        type_weight = 1.0

        return base_score * relation_weight * type_weight

    def _fallback_search(self, symbol: str, top_k: int) -> list[SearchResult]:
        """无依赖层时的回退搜索"""
        # 先尝试符号查找
        results = self._storage.bm25_search(symbol, top_k=top_k)
        return results

    # ── 调用图查询 ──

    def callers(self, symbol: str, depth: int = 1) -> list[SearchResult]:
        """查找谁调用了指定符号"""
        caller_list = self._call_graph.get_callers(symbol, depth)
        return self._enrich_results(caller_list)

    def callees(self, symbol: str, depth: int = 1) -> list[SearchResult]:
        """查找指定符号调用了谁"""
        callee_list = self._call_graph.get_callees(symbol, depth)
        return self._enrich_results(callee_list)

    def impact(self, symbol: str) -> dict:
        """Blast radius 影响分析"""
        impact_data = self._call_graph.get_impact(symbol)

        # 补充数据流信息
        unvalidated = self._data_flow.find_unvalidated_inputs()
        impact_data["unvalidated_inputs"] = [
            {"source": e.source, "sink": e.sink, "file_path": e.file_path}
            for e in unvalidated
            if e.file_path in impact_data.get("affected_files", [])
        ]

        # 补充依赖层信息
        if self._dependency_layer and self._dependency_layer.layer_count > 0:
            layer_id = self._dependency_layer.get_node_layer(symbol)
            if layer_id is not None:
                layer_info = self._dependency_layer.get_layer_info(layer_id)
                impact_data["dependency_layer"] = {
                    "layer_id": layer_id,
                    "layer_name": layer_info.name if layer_info else "unknown",
                    "layer_type": layer_info.layer_type.value if layer_info else "unknown",
                }

        return impact_data

    # ── 数据流查询 ──

    def trace_source(self, sink: str) -> list[dict]:
        """从 sink 回溯数据来源"""
        edges = self._data_flow.trace_source(sink)
        return [
            {
                "source": e.source, "sink": e.sink, "transform": e.transform,
                "file_path": e.file_path, "line_number": e.line_number,
                "nullable": e.nullable, "validation_level": e.validation_level.value,
                "sanitized": e.sanitized,
            }
            for e in edges
        ]

    def trace_sink(self, source: str) -> list[dict]:
        """从 source 追踪数据去向"""
        edges = self._data_flow.trace_sink(source)
        return [
            {
                "source": e.source, "sink": e.sink, "transform": e.transform,
                "file_path": e.file_path, "line_number": e.line_number,
                "nullable": e.nullable, "validation_level": e.validation_level.value,
                "sanitized": e.sanitized,
            }
            for e in edges
        ]

    # ── 图统计 ──

    def get_stats(self) -> dict:
        """获取图统计信息"""
        stats = {
            "call_graph_nodes": self._call_graph.node_count,
            "call_graph_edges": self._call_graph.edge_count,
            "data_flow_edges": self._data_flow.flow_count,
            "entry_points": len(self._call_graph.get_entry_points()),
            "isolated_symbols": len(self._call_graph.get_isolated_symbols()),
        }
        if self._dependency_layer:
            stats["dependency_layers"] = self._dependency_layer.layer_count
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        return stats

    # ── 内部方法 ──

    def _enrich_results(self, graph_results: list[dict]) -> list[SearchResult]:
        """将图查询结果转换为 SearchResult，补充存储层信息"""
        results: list[SearchResult] = []
        for item in graph_results:
            name = item.get("name", "")
            symbol = self._storage.get_symbol(name)
            if symbol:
                results.append(SearchResult(
                    id=f"{symbol.file_path}#{symbol.name}@{symbol.line_number}",
                    symbol_name=symbol.name,
                    qualified_name=symbol.qualified_name,
                    file_path=symbol.file_path,
                    line_number=symbol.line_number,
                    chunk_type=symbol.symbol_type,
                    score=1.0 / item.get("depth", 1),
                    signature=symbol.signature,
                    docstring=None,
                    tags=[],
                ))
            else:
                results.append(SearchResult(
                    id=name,
                    symbol_name=name.split(".")[-1] if "." in name else name,
                    qualified_name=name,
                    file_path=item.get("file_path", ""),
                    line_number=0,
                    chunk_type="external",
                    score=0.5 / item.get("depth", 1),
                    tags=[],
                ))
        return results
