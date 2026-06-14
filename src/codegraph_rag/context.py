"""上下文压缩 — 分层结构化上下文 + Token 预算管理"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .call_graph import CallGraph
from .data_flow import DataFlowAnalyzer
from .models import SearchResult, SymbolIndex
from .storage import CodeGraphStorage

if TYPE_CHECKING:
    from .cache import MultiLevelCache
    from .community import CommunityDetector
    from .dependency_layer import DependencyLayerManager

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token 预算管理"""
    total: int = 128000        # 总预算（默认 128K，适配 GPT-4）
    system_reserve: int = 8000  # 系统提示预留
    safety_margin: float = 0.1  # 安全余量比例

    @property
    def available(self) -> int:
        """可用 Token 预算"""
        usable = self.total - self.system_reserve
        return int(usable * (1 - self.safety_margin))

    def allocate(self, sections: dict[str, float]) -> dict[str, int]:
        """按比例分配 Token 预算

        Args:
            sections: {section_name: ratio}，ratio 之和应为 1.0

        Returns:
            {section_name: token_count}
        """
        available = self.available
        allocation = {}
        remaining = available

        sorted_sections = sorted(sections.items(), key=lambda x: x[1], reverse=True)
        for i, (name, ratio) in enumerate(sorted_sections):
            if i == len(sorted_sections) - 1:
                allocation[name] = remaining
            else:
                tokens = int(available * ratio)
                allocation[name] = tokens
                remaining -= tokens

        return allocation


class ContextComposer:
    """上下文组装器：分层结构化上下文 + Token 预算"""

    def __init__(
        self,
        storage: CodeGraphStorage,
        call_graph: CallGraph,
        data_flow: DataFlowAnalyzer,
        dependency_layer: DependencyLayerManager | None = None,
        community_detector: CommunityDetector | None = None,
        cache: MultiLevelCache | None = None,
    ) -> None:
        self._storage = storage
        self._call_graph = call_graph
        self._data_flow = data_flow
        self._dependency_layer = dependency_layer
        self._community_detector = community_detector
        self._cache = cache

    def set_dependency_layer(self, dl: DependencyLayerManager) -> None:
        self._dependency_layer = dl

    def set_community_detector(self, cd: CommunityDetector) -> None:
        self._community_detector = cd

    def set_cache(self, cache: MultiLevelCache) -> None:
        self._cache = cache

    # ── 分层结构化上下文 ──

    def compose_structured_context(
        self,
        symbol: str,
        query_type: str = "implementation",
        token_budget: TokenBudget | None = None,
    ) -> str:
        """组装分层结构化上下文（Token 预算感知）

        输出结构：
        1. 符号定义层（签名 + 类型 + 文档）
        2. 依赖层上下文（当前层 + 相邻层摘要）
        3. 调用关系层（callers + callees）
        4. 数据流层（source → sink）
        5. 社区层（所属功能模块）
        """
        # 检查缓存
        cache_key = f"context:{symbol}:{query_type}"
        if self._cache:
            cached = self._cache.get_query_cache(cache_key)
            if cached is not None:
                return cached

        if token_budget is None:
            token_budget = TokenBudget()

        # 分配 Token 预算
        budget_allocation = token_budget.allocate({
            "definition": 0.25,     # 符号定义
            "dependency": 0.20,     # 依赖层
            "relations": 0.25,      # 调用关系
            "data_flow": 0.15,      # 数据流
            "community": 0.15,      # 社区
        })

        parts: list[str] = []

        # 1. 符号定义层
        definition_ctx = self._compose_definition(symbol, budget_allocation["definition"])
        if definition_ctx:
            parts.append(definition_ctx)

        # 2. 依赖层上下文
        dependency_ctx = self._compose_dependency(symbol, query_type, budget_allocation["dependency"])
        if dependency_ctx:
            parts.append(dependency_ctx)

        # 3. 调用关系层
        relations_ctx = self._compose_relations(symbol, budget_allocation["relations"])
        if relations_ctx:
            parts.append(relations_ctx)

        # 4. 数据流层
        dataflow_ctx = self._compose_dataflow(symbol, budget_allocation["data_flow"])
        if dataflow_ctx:
            parts.append(dataflow_ctx)

        # 5. 社区层
        community_ctx = self._compose_community(symbol, budget_allocation["community"])
        if community_ctx:
            parts.append(community_ctx)

        result = "\n\n---\n\n".join(parts) if parts else f"Symbol not found: {symbol}"

        # 写入缓存
        if self._cache:
            self._cache.set_query_cache(cache_key, result)

        return result

    # ── 各层组装 ──

    def _compose_definition(self, symbol: str, token_limit: int) -> str:
        """符号定义层"""
        sym = self._storage.get_symbol(symbol)
        if sym is None:
            results = self._storage.search_symbols_by_name(symbol, top_k=1)
            if results:
                sym = results[0]

        if sym is None:
            return ""

        lines = [f"## Definition: `{sym.qualified_name}`"]
        if sym.signature:
            lines.append(f"```python\n{sym.signature}\n```")
        if sym.return_type:
            lines.append(f"Returns: `{sym.return_type}`")

        return "\n".join(lines)

    def _compose_dependency(self, symbol: str, query_type: str, token_limit: int) -> str:
        """依赖层上下文"""
        if self._dependency_layer is None or self._dependency_layer.layer_count == 0:
            return ""

        layer_id = self._dependency_layer.get_node_layer(symbol)
        if layer_id is None:
            return ""

        layer_info = self._dependency_layer.get_layer_info(layer_id)
        if layer_info is None:
            return ""

        lines = [f"## Dependency Layer: {layer_info.name} (Layer {layer_id})"]

        # 当前层摘要
        current_nodes = layer_info.nodes
        lines.append(f"Current layer ({len(current_nodes)} symbols): {', '.join(current_nodes[:8])}")
        if len(current_nodes) > 8:
            lines.append(f"  ... and {len(current_nodes) - 8} more")

        # 相邻层摘要
        scope = self._dependency_layer.get_search_scope(symbol, query_type)
        for scope_layer_id in scope:
            if scope_layer_id == layer_id:
                continue
            scope_info = self._dependency_layer.get_layer_info(scope_layer_id)
            if scope_info:
                lines.append(
                    f"Layer {scope_layer_id} ({scope_info.name}, {len(scope_info.nodes)} symbols): "
                    f"{', '.join(scope_info.nodes[:5])}"
                )

        return "\n".join(lines)

    def _compose_relations(self, symbol: str, token_limit: int) -> str:
        """调用关系层"""
        callers = self._call_graph.get_callers(symbol, depth=1)
        callees = self._call_graph.get_callees(symbol, depth=1)

        if not callers and not callees:
            return ""

        lines = ["## Call Relations"]

        if callers:
            max_callers = max(5, token_limit // 30)
            caller_names = [c["name"] for c in callers[:max_callers]]
            lines.append(f"Called by ({len(callers)}): {', '.join(caller_names)}")
            if len(callers) > max_callers:
                lines.append(f"  ... and {len(callers) - max_callers} more")

        if callees:
            max_callees = max(5, token_limit // 30)
            callee_names = [c["name"] for c in callees[:max_callees]]
            lines.append(f"Calls ({len(callees)}): {', '.join(callee_names)}")
            if len(callees) > max_callees:
                lines.append(f"  ... and {len(callees) - max_callees} more")

        return "\n".join(lines)

    def _compose_dataflow(self, symbol: str, token_limit: int) -> str:
        """数据流层"""
        source_flows = self._data_flow.trace_source(symbol)
        sink_flows = self._data_flow.trace_sink(symbol)

        if not source_flows and not sink_flows:
            return ""

        lines = ["## Data Flow"]

        if source_flows:
            max_sources = max(3, token_limit // 40)
            lines.append("Sources:")
            for f in source_flows[:max_sources]:
                validation = f.validation_level.value
                lines.append(f"  ← `{f.source}` (validation: {validation})")

        if sink_flows:
            max_sinks = max(3, token_limit // 40)
            lines.append("Sinks:")
            for f in sink_flows[:max_sinks]:
                lines.append(f"  → `{f.sink}` (sanitized: {f.sanitized})")

        return "\n".join(lines)

    def _compose_community(self, symbol: str, token_limit: int) -> str:
        """社区层"""
        if self._community_detector is None:
            return ""

        community = self._community_detector.get_community(symbol)
        if community is None:
            return ""

        lines = [f"## Community: {community.name or f'Community-{community.id}'}"]
        lines.append(f"Type: {community.community_type.value if community.community_type else 'unknown'}")
        lines.append(f"Cohesion: {community.cohesion:.3f}")
        lines.append(f"Members ({len(community.members)}): {', '.join(community.members[:6])}")
        if len(community.members) > 6:
            lines.append(f"  ... and {len(community.members) - 6} more")

        if community.god_nodes:
            lines.append(f"Core nodes: {', '.join(community.god_nodes[:3])}")

        return "\n".join(lines)

    # ── 简化接口（向后兼容） ──

    def compose_symbol_context(self, symbol: str, max_lines: int = 50) -> str:
        """组装符号全景上下文（简化版，向后兼容）"""
        return self.compose_structured_context(symbol, "implementation")

    def compose_impact_context(self, symbol: str) -> str:
        """组装影响分析上下文"""
        impact = self._call_graph.get_impact(symbol)

        parts: list[str] = []
        parts.append(f"## Impact Analysis: {symbol}")
        parts.append(f"**Affected symbols: {impact['affected_count']}**")
        parts.append(f"**Affected files: {len(impact['affected_files'])}**")

        # 补充依赖层信息
        if self._dependency_layer and self._dependency_layer.layer_count > 0:
            layer_id = self._dependency_layer.get_node_layer(symbol)
            if layer_id is not None:
                layer_info = self._dependency_layer.get_layer_info(layer_id)
                if layer_info:
                    parts.append(f"**Dependency layer: {layer_info.name} (Layer {layer_id})**")

        if impact["upstream"]:
            parts.append("### Upstream (who depends on this)")
            for s in impact["upstream"][:10]:
                parts.append(f"- `{s}`")

        if impact["downstream"]:
            parts.append("### Downstream (what this depends on)")
            for s in impact["downstream"][:10]:
                parts.append(f"- `{s}`")

        if impact.get("affected_files"):
            parts.append("### Affected Files")
            for f in impact["affected_files"]:
                parts.append(f"- `{f}`")

        return "\n\n".join(parts)

    def compose_search_context(
        self, results: list[SearchResult], max_snippet_lines: int = 15,
    ) -> str:
        """组装搜索结果上下文（压缩版）"""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            header = f"### {i}. {r.qualified_name or r.symbol_name or 'unknown'}"
            if r.file_path:
                header += f" (`{r.file_path}:{r.line_number}`)"
            parts.append(header)

            if r.signature:
                parts.append(f"```python\n{r.signature}\n```")

            if r.docstring:
                doc_lines = r.docstring.splitlines()
                if len(doc_lines) > 3:
                    doc = "\n".join(doc_lines[:3]) + " ..."
                else:
                    doc = r.docstring
                parts.append(f"> {doc}")

            if r.tags:
                parts.append(f"Tags: {', '.join(r.tags)}")

        return "\n\n".join(parts)
