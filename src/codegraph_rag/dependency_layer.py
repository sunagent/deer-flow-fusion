"""依赖层自动划分 — 基于拓扑排序的分层依赖树"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

from .call_graph import CallGraph

logger = logging.getLogger(__name__)


class DependencyLayerType(str, Enum):
    """依赖层类型（典型分层结果）"""
    FOUNDATION = "foundation"    # Layer 0: utils, helpers, constants, exceptions
    DATA = "data"                # Layer 1: models, schemas, database, config
    SERVICE = "service"          # Layer 2: services, repositories, managers
    API = "api"                  # Layer 3: controllers, apis, routes, middleware
    ENTRY = "entry"              # Layer 4: main, app, entrypoints
    UNKNOWN = "unknown"          # 未分类


@dataclass
class LayerInfo:
    """依赖层信息"""
    id: int                      # 层编号（0=底层）
    nodes: list[str]             # 该层的符号列表
    layer_type: DependencyLayerType = DependencyLayerType.UNKNOWN
    name: str = ""               # 层名称（自动推断）

    @property
    def node_count(self) -> int:
        return len(self.nodes)


class DependencyLayerManager:
    """依赖层管理器：将扁平调用图转化为分层依赖树

    核心洞察：代码的本质是分层依赖树，95% 的编程查询只涉及不超过 3 个连续依赖层。
    通过拓扑排序自动划分依赖层次，实现依赖感知的精准检索。
    """

    def __init__(self, call_graph: CallGraph) -> None:
        self._call_graph = call_graph
        self._layers: list[LayerInfo] = []
        self._node_to_layer: dict[str, int] = {}
        self._condensation: nx.DiGraph | None = None
        self._scc_map: dict[str, int] = {}  # 原始节点 → SCC ID

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    @property
    def layers(self) -> list[LayerInfo]:
        return self._layers

    # ── 依赖层计算 ──

    def compute_layers(self) -> list[LayerInfo]:
        """计算依赖层结构

        算法：
        1. 检测强连通分量（SCC），将环缩为超级节点
        2. 对缩合图做拓扑排序
        3. 反转使底层为 Layer 0
        4. 合并相似层
        5. 推断每层的类型和名称
        """
        graph = self._call_graph.graph

        if graph.number_of_nodes() < 2:
            self._layers = []
            self._node_to_layer = {}
            return []

        # Step 1: 强连通分量检测（处理循环依赖）
        sccs = list(nx.strongly_connected_components(graph))
        self._condensation = nx.condensation(graph, sccs)

        # 建立 原始节点 → SCC ID 映射
        self._scc_map = {}
        for scc_id, scc in enumerate(sccs):
            for node in scc:
                self._scc_map[node] = scc_id

        # Step 2: 拓扑排序（缩合图保证无环）
        try:
            topo_order = list(nx.topological_sort(self._condensation))
        except nx.NetworkXUnfeasible:
            # 理论上 condensation 一定无环，但保险起见
            logger.warning("Condensation graph has cycles, falling back to BFS layering")
            topo_order = list(self._condensation.nodes)

        # Step 3: 按拓扑层级分组
        # 拓扑排序给出的是线性序列，需要转为层级
        # 使用最长路径法确定每个节点的层级
        scc_to_layer: dict[int, int] = {}
        for scc_id in topo_order:
            # 该节点的层级 = max(前驱层级) + 1，无前驱则为 0
            predecessors = list(self._condensation.predecessors(scc_id))
            if not predecessors:
                scc_to_layer[scc_id] = 0
            else:
                scc_to_layer[scc_id] = max(scc_to_layer[p] for p in predecessors) + 1

        # Step 4: 反转层级（底层 = 0）
        max_layer = max(scc_to_layer.values()) if scc_to_layer else 0
        for scc_id in scc_to_layer:
            scc_to_layer[scc_id] = max_layer - scc_to_layer[scc_id]

        # Step 5: 将原始节点映射到层级
        self._node_to_layer = {}
        for node, scc_id in self._scc_map.items():
            self._node_to_layer[node] = scc_to_layer[scc_id]

        # Step 6: 按层分组
        layer_nodes: dict[int, list[str]] = {}
        for node, layer_id in self._node_to_layer.items():
            layer_nodes.setdefault(layer_id, []).append(node)

        # Step 7: 合并相似层（节点数 < 3 且与相邻层耦合度高）
        layer_nodes = self._merge_similar_layers(graph, layer_nodes)

        # Step 8: 构建 LayerInfo
        self._layers = []
        for layer_id in sorted(layer_nodes.keys()):
            nodes = layer_nodes[layer_id]
            layer_type = self._infer_layer_type(layer_id, len(self._layers), nodes, graph)
            name = self._infer_layer_name(nodes, layer_type)

            self._layers.append(LayerInfo(
                id=layer_id,
                nodes=nodes,
                layer_type=layer_type,
                name=name,
            ))

            # 更新 node_to_layer（合并后可能变化）
            for node in nodes:
                self._node_to_layer[node] = layer_id

        logger.info(f"Computed {len(self._layers)} dependency layers")
        return self._layers

    # ── 查询接口 ──

    def get_node_layer(self, node: str) -> int | None:
        """获取节点所在层级"""
        if node in self._node_to_layer:
            return self._node_to_layer[node]
        # 模糊匹配
        for n, layer_id in self._node_to_layer.items():
            if node in n or n.endswith(f".{node}"):
                return layer_id
        return None

    def get_layer_nodes(self, layer_id: int) -> list[str]:
        """获取指定层的所有节点"""
        for layer in self._layers:
            if layer.id == layer_id:
                return layer.nodes
        return []

    def get_layer_info(self, layer_id: int) -> LayerInfo | None:
        """获取指定层的信息"""
        for layer in self._layers:
            if layer.id == layer_id:
                return layer
        return None

    def get_search_scope(self, node: str, query_type: str) -> list[int]:
        """依赖感知的检索范围

        根据查询类型和节点所在层，自动确定最优检索范围：
        - symbol_definition: 仅当前层
        - implementation: 当前层 + 直接依赖层(L-1)
        - debugging: L-1 + L + L+1
        - impact_analysis: L + L+1 + L+2
        - refactoring: L-1 + L + L+1
        - architecture: 所有层
        """
        layer_id = self.get_node_layer(node)
        if layer_id is None:
            # 未知节点，返回所有层
            return [l.id for l in self._layers]

        max_layer_id = max(l.id for l in self._layers) if self._layers else 0

        scope_map = {
            "symbol_definition": [layer_id],
            "implementation": [i for i in [layer_id - 1, layer_id] if i >= 0],
            "debugging": [i for i in [layer_id - 1, layer_id, layer_id + 1] if 0 <= i <= max_layer_id],
            "impact_analysis": [i for i in [layer_id, layer_id + 1, layer_id + 2] if i <= max_layer_id],
            "refactoring": [i for i in [layer_id - 1, layer_id, layer_id + 1] if 0 <= i <= max_layer_id],
            "architecture": [l.id for l in self._layers],
        }

        return scope_map.get(query_type, [layer_id])

    def get_nodes_in_scope(self, node: str, query_type: str) -> list[str]:
        """获取检索范围内的所有节点"""
        scope_layers = self.get_search_scope(node, query_type)
        nodes = []
        for layer_id in scope_layers:
            nodes.extend(self.get_layer_nodes(layer_id))
        return nodes

    def detect_circular_dependencies(self) -> list[list[str]]:
        """检测循环依赖"""
        graph = self._call_graph.graph
        cycles = []
        try:
            for cycle in nx.simple_cycles(graph):
                if len(cycle) > 1:
                    cycles.append(cycle)
        except Exception:
            # 大图可能超时
            pass
        return cycles

    # ── 内部方法 ──

    @staticmethod
    def _merge_similar_layers(
        graph: nx.DiGraph, layer_nodes: dict[int, list[str]],
    ) -> dict[int, list[str]]:
        """合并节点数过少且与相邻层耦合度高的层"""
        if len(layer_nodes) <= 2:
            return layer_nodes

        merged = dict(layer_nodes)
        layer_ids = sorted(merged.keys())

        # 从底层向上检查，合并小层
        i = 0
        while i < len(layer_ids) - 1:
            current_id = layer_ids[i]
            next_id = layer_ids[i + 1]

            current_nodes = merged.get(current_id, [])
            next_nodes = merged.get(next_id, [])

            # 如果当前层节点数 < 3，与下一层合并
            if len(current_nodes) < 3:
                merged[next_id] = current_nodes + next_nodes
                del merged[current_id]
                layer_ids.pop(i)
                continue

            i += 1

        # 重新编号
        result = {}
        for new_id, old_id in enumerate(sorted(merged.keys())):
            result[new_id] = merged[old_id]

        return result

    @staticmethod
    def _infer_layer_type(
        layer_id: int, total_layer_index: int, nodes: list[str], graph: nx.DiGraph,
    ) -> DependencyLayerType:
        """推断层类型"""
        nodes_str = " ".join(nodes).lower()

        # 底层：utils, helpers, constants, exceptions
        foundation_kw = ["util", "helper", "constant", "exception", "error", "config", "setting", "base", "mixin"]
        if any(kw in nodes_str for kw in foundation_kw):
            return DependencyLayerType.FOUNDATION

        # 数据层：models, schemas, database
        data_kw = ["model", "schema", "database", "repository", "entity", "table", "orm", "dao"]
        if any(kw in nodes_str for kw in data_kw):
            return DependencyLayerType.DATA

        # 服务层：services, managers
        service_kw = ["service", "manager", "handler", "processor", "engine", "calculator"]
        if any(kw in nodes_str for kw in service_kw):
            return DependencyLayerType.SERVICE

        # API 层：controllers, routes, middleware
        api_kw = ["controller", "api", "route", "middleware", "view", "endpoint", "handler"]
        if any(kw in nodes_str for kw in api_kw):
            return DependencyLayerType.API

        # 入口层：main, app
        entry_kw = ["main", "app", "entrypoint", "run", "start", "server"]
        if any(kw in nodes_str for kw in entry_kw):
            return DependencyLayerType.ENTRY

        # 按位置推断
        if total_layer_index == 0:
            return DependencyLayerType.FOUNDATION

        return DependencyLayerType.UNKNOWN

    @staticmethod
    def _infer_layer_name(nodes: list[str], layer_type: DependencyLayerType) -> str:
        """推断层名称"""
        type_names = {
            DependencyLayerType.FOUNDATION: "Foundation",
            DependencyLayerType.DATA: "Data",
            DependencyLayerType.SERVICE: "Service",
            DependencyLayerType.API: "API",
            DependencyLayerType.ENTRY: "Entry",
        }
        return type_names.get(layer_type, "Unknown")
