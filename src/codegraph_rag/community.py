"""社区检测 — Leiden/Louvain 算法自动发现功能模块"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from .call_graph import CallGraph
from .models import CommunityIndex, CommunityType

logger = logging.getLogger(__name__)


class CommunityDetector:
    """社区检测器：基于图拓扑自动发现功能模块"""

    def __init__(self, call_graph: CallGraph) -> None:
        self._call_graph = call_graph
        self._communities: list[CommunityIndex] = []
        self._symbol_to_community: dict[str, int] = {}

    @property
    def community_count(self) -> int:
        return len(self._communities)

    @property
    def communities(self) -> list[CommunityIndex]:
        return self._communities

    # ── 检测 ──

    def detect(self) -> list[CommunityIndex]:
        """执行社区检测，返回社区列表"""
        graph = self._call_graph.graph

        if graph.number_of_nodes() < 2:
            logger.info("Graph too small for community detection")
            self._communities = []
            self._symbol_to_community = {}
            return []

        # 转为无向图进行社区检测
        undirected = graph.to_undirected()

        # 移除孤立节点（它们自成社区）
        isolates = list(nx.isolates(undirected))
        undirected_no_isolates = undirected.copy()
        undirected_no_isolates.remove_nodes_from(isolates)

        if undirected_no_isolates.number_of_nodes() < 2:
            # 所有节点都是孤立的
            self._communities = [
                CommunityIndex(id=i, members=[node], cohesion=0.0)
                for i, node in enumerate(isolates)
            ]
            self._symbol_to_community = {
                node: i for i, node in enumerate(isolates)
            }
            return self._communities

        # 使用 Louvain 算法（python-louvain）
        try:
            from community import community_louvain

            partition = community_louvain.best_partition(undirected_no_isolates)
        except ImportError:
            logger.warning("python-louvain not installed, using networkx label propagation")
            partition = self._label_propagation(undirected_no_isolates)

        # 孤立节点各自成社区
        max_id = max(partition.values()) + 1 if partition else 0
        for i, node in enumerate(isolates):
            partition[node] = max_id + i

        # 按社区 ID 分组
        communities_dict: dict[int, list[str]] = {}
        for symbol, comm_id in partition.items():
            communities_dict.setdefault(comm_id, []).append(symbol)

        # 构建 CommunityIndex 列表
        self._communities = []
        self._symbol_to_community = {}

        for comm_id, members in communities_dict.items():
            # 计算内聚度
            cohesion = self._compute_cohesion(graph, members)

            # 计算与其他社区的耦合度
            coupling = self._compute_coupling(graph, members, communities_dict)

            # 找核心节点（度数最高）
            god_nodes = self._find_god_nodes(graph, members)

            # 找入口点（入度为 0 的节点）
            entry_points = self._find_entry_points(graph, members)

            # 推断社区类型
            community_type = self._infer_community_type(members, graph)

            community = CommunityIndex(
                id=comm_id,
                members=members,
                name=None,  # 可由 LLM 后续生成
                community_type=community_type,
                cohesion=cohesion,
                coupling=coupling,
                god_nodes=god_nodes,
                entry_points=entry_points,
            )
            self._communities.append(community)

            # 更新符号→社区映射
            for member in members:
                self._symbol_to_community[member] = comm_id

        logger.info(f"Detected {len(self._communities)} communities")
        return self._communities

    # ── 查询 ──

    def get_community(self, symbol: str) -> CommunityIndex | None:
        """查找符号所属的社区"""
        comm_id = self._symbol_to_community.get(symbol)
        if comm_id is None:
            # 模糊匹配
            for s, cid in self._symbol_to_community.items():
                if symbol in s or s.endswith(f".{symbol}"):
                    comm_id = cid
                    break
        if comm_id is None:
            return None
        for c in self._communities:
            if c.id == comm_id:
                return c
        return None

    def search_communities(self, name_pattern: str) -> list[CommunityIndex]:
        """按名称模式搜索社区"""
        results = []
        for c in self._communities:
            if c.name and name_pattern.lower() in c.name.lower():
                results.append(c)
            # 也搜索成员名
            for m in c.members:
                if name_pattern.lower() in m.lower():
                    results.append(c)
                    break
        return results

    def get_coupled_communities(self, community_id: int, threshold: float = 0.3) -> list[tuple[int, float]]:
        """获取与指定社区耦合度超过阈值的其他社区"""
        for c in self._communities:
            if c.id == community_id:
                return [
                    (other_id, score)
                    for other_id, score in c.coupling.items()
                    if score >= threshold
                ]
        return []

    # ── 内部方法 ──

    @staticmethod
    def _compute_cohesion(graph: nx.DiGraph, members: list[str]) -> float:
        """计算社区内聚度（内部边数 / 最大可能边数）"""
        if len(members) < 2:
            return 0.0
        member_set = set(members)
        internal_edges = 0
        for u, v in graph.edges():
            if u in member_set and v in member_set:
                internal_edges += 1
        max_edges = len(members) * (len(members) - 1)
        return internal_edges / max_edges if max_edges > 0 else 0.0

    @staticmethod
    def _compute_coupling(
        graph: nx.DiGraph, members: list[str],
        all_communities: dict[int, list[str]],
    ) -> dict[int, float]:
        """计算与其他社区的耦合度"""
        member_set = set(members)
        coupling: dict[int, float] = {}

        # 统计到其他社区的边数
        external_edges: dict[int, int] = {}
        for u, v in graph.edges():
            if u in member_set and v not in member_set:
                # 找到 v 所属社区
                for cid, c_members in all_communities.items():
                    if v in c_members:
                        external_edges[cid] = external_edges.get(cid, 0) + 1
                        break

        # 归一化
        total_external = sum(external_edges.values())
        if total_external > 0:
            for cid, count in external_edges.items():
                coupling[cid] = count / total_external

        return coupling

    @staticmethod
    def _find_god_nodes(graph: nx.DiGraph, members: list[str]) -> list[str]:
        """找社区内度数最高的节点（架构重心）"""
        member_set = set(members)
        degrees = []
        for node in members:
            if node in graph:
                in_deg = sum(1 for _, v in graph.in_edges(node) if v in member_set or _ in member_set)
                out_deg = sum(1 for u, _ in graph.out_edges(node) if u in member_set or _ in member_set)
                degrees.append((node, in_deg + out_deg))
        degrees.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in degrees[:3]]

    @staticmethod
    def _find_entry_points(graph: nx.DiGraph, members: list[str]) -> list[str]:
        """找社区入口点（社区内无入边或入边来自社区外的节点）"""
        member_set = set(members)
        entry_points = []
        for node in members:
            if node not in graph:
                continue
            # 检查是否有来自社区内的入边
            has_internal_caller = False
            for predecessor in graph.predecessors(node):
                if predecessor in member_set:
                    has_internal_caller = True
                    break
            if not has_internal_caller:
                entry_points.append(node)
        return entry_points

    @staticmethod
    def _infer_community_type(members: list[str], graph: nx.DiGraph) -> CommunityType | None:
        """推断社区类型"""
        # 启发式规则
        member_str = " ".join(members).lower()

        # 横切关注点
        cross_cutting_keywords = ["log", "error", "exception", "config", "util", "helper", "middleware"]
        if any(kw in member_str for kw in cross_cutting_keywords):
            return CommunityType.CROSS_CUTTING

        # 架构层
        layer_keywords = ["api", "route", "controller", "handler", "service", "repository", "model", "schema", "dao"]
        if any(kw in member_str for kw in layer_keywords):
            return CommunityType.LAYER

        return CommunityType.FUNCTIONAL

    @staticmethod
    def _label_propagation(graph: nx.Graph) -> dict[str, int]:
        """使用 NetworkX 标签传播作为 Louvain 的后备"""
        communities = nx.algorithms.community.label_propagation_communities(graph)
        partition = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                partition[node] = comm_id
        return partition
