"""调用图构建 — 基于 NetworkX 的调用关系图"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx

from .models import CallEdge, CallType, SymbolIndex
from .symbol_extractor import extract_calls, extract_symbols

logger = logging.getLogger(__name__)


class CallGraph:
    """调用图：基于 NetworkX DiGraph，存储函数间的调用关系"""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._name_index: dict[str, list[str]] = {}
        self._fuzzy_cache: dict[str, list[str]] = {}

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ── 构建操作 ──

    def _remember_node(self, name: str) -> None:
        if not name:
            return
        simple_name = name.split(".")[-1]
        bucket = self._name_index.setdefault(simple_name, [])
        if name not in bucket:
            bucket.append(name)
        self._fuzzy_cache.clear()

    def add_symbols(self, symbols: list[SymbolIndex]) -> None:
        """添加符号节点到图中"""
        for sym in symbols:
            self._graph.add_node(
                sym.qualified_name,
                symbol_type=sym.symbol_type.value,
                file_path=sym.file_path,
                line_number=sym.line_number,
                signature=sym.signature,
                return_type=sym.return_type,
            )
            self._remember_node(sym.qualified_name)

    def add_calls(self, calls: list[CallEdge]) -> None:
        """添加调用边到图中，自动规范化名称"""
        # 构建名称索引：简单名 → 全限定名列表
        name_index = self._name_index

        for call in calls:
            # 规范化 caller 和 callee
            caller = self._normalize_name(call.caller, name_index)
            callee = self._normalize_name(call.callee, name_index)

            # 确保节点存在
            if caller not in self._graph:
                self._graph.add_node(caller, symbol_type="unknown")
                self._remember_node(caller)
            if callee not in self._graph:
                self._graph.add_node(callee, symbol_type="external")
                self._remember_node(callee)

            self._graph.add_edge(
                caller,
                callee,
                call_site=call.call_site,
                call_type=call.call_type.value,
                in_loop=call.in_loop,
            )

    def _normalize_name(self, name: str, name_index: dict[str, list[str]]) -> str:
        """将简单名或部分限定名规范化为图中的全限定名"""
        # 1. 精确匹配
        if name in self._graph:
            return name

        # 2. 后缀匹配（name 是全限定名的一部分）
        simple_name = name.split(".")[-1]
        candidates = name_index.get(simple_name, [])

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            # 多个候选：选择最匹配的
            for candidate in candidates:
                if name in candidate or candidate.endswith(f".{simple_name}"):
                    return candidate

        # 3. 模糊后缀匹配
        for node in self._graph.nodes:
            if node.endswith(f".{name}") or node.endswith(f".{simple_name}"):
                return node

        # 4. 无法规范化，返回原始名
        return name

    def add_file(self, file_path: str, content: str, language: str) -> None:
        """从单个文件提取符号和调用关系并添加到图中"""
        symbols = extract_symbols(file_path, content, language)
        calls = extract_calls(file_path, content, language, symbols)
        self.add_symbols(symbols)
        self.add_calls(calls)
        logger.debug(
            f"Added {len(symbols)} symbols, {len(calls)} calls from {file_path}"
        )

    def remove_file(self, file_path: str) -> None:
        """移除与某文件相关的所有节点和边"""
        nodes_to_remove = [
            node
            for node, data in self._graph.nodes(data=True)
            if data.get("file_path") == file_path
        ]
        self._graph.remove_nodes_from(nodes_to_remove)
        logger.debug(f"Removed {len(nodes_to_remove)} nodes from {file_path}")

    # ── 查询操作 ──

    def get_callers(self, symbol: str, depth: int = 1) -> list[dict]:
        """查找谁调用了指定符号（向上追溯）

        Args:
            symbol: 符号全限定名
            depth: 递归深度（1=直接调用方，2=调用方的调用方）

        Returns:
            调用方列表 [{"name": ..., "depth": N, "call_type": ..., "call_site": ...}]
        """
        if symbol not in self._graph:
            # 尝试模糊匹配
            matches = self._fuzzy_match(symbol)
            if matches:
                symbol = matches[0]
            else:
                return []

        result: list[dict] = []
        visited: set[str] = set()
        self._collect_callers(symbol, depth, 0, visited, result)
        return result

    def get_callees(self, symbol: str, depth: int = 1) -> list[dict]:
        """查找指定符号调用了谁（向下追溯）

        Args:
            symbol: 符号全限定名
            depth: 递归深度

        Returns:
            被调用方列表
        """
        if symbol not in self._graph:
            matches = self._fuzzy_match(symbol)
            if matches:
                symbol = matches[0]
            else:
                return []

        result: list[dict] = []
        visited: set[str] = set()
        self._collect_callees(symbol, depth, 0, visited, result)
        return result

    def get_impact(self, symbol: str) -> dict:
        """Blast radius 影响分析

        Returns:
            {
                "symbol": str,
                "upstream": [...],    # 所有上游调用方
                "downstream": [...],  # 所有下游被调用方
                "affected_files": list[str],
                "affected_symbols": list[str],
                "affected_count": int,
            }
        """
        if symbol not in self._graph:
            matches = self._fuzzy_match(symbol)
            if matches:
                symbol = matches[0]
            else:
                return {"symbol": symbol, "upstream": [], "downstream": [], "affected_files": [], "affected_symbols": [], "affected_count": 0}

        # 向上追溯所有调用方
        upstream = self._bfs_upstream(symbol)
        # 向下追溯所有被调用方
        downstream = self._bfs_downstream(symbol)

        # 收集受影响的文件和符号
        all_symbols = set(upstream + downstream + [symbol])
        affected_files: set[str] = set()
        for sym in all_symbols:
            data = self._graph.nodes.get(sym, {})
            fp = data.get("file_path")
            if fp:
                affected_files.add(fp)

        return {
            "symbol": symbol,
            "upstream": upstream,
            "downstream": downstream,
            "affected_files": sorted(affected_files),
            "affected_symbols": sorted(all_symbols),
            "affected_count": len(all_symbols),
        }

    def get_isolated_symbols(self) -> list[str]:
        """查找孤立符号（无入边也无出边的函数）"""
        return [
            node
            for node in self._graph.nodes
            if self._graph.in_degree(node) == 0 and self._graph.out_degree(node) == 0
        ]

    def get_entry_points(self) -> list[str]:
        """查找入口点（有出边但无入边的函数）"""
        return [
            node
            for node in self._graph.nodes
            if self._graph.in_degree(node) == 0 and self._graph.out_degree(node) > 0
        ]

    def get_leaf_symbols(self) -> list[str]:
        """查找叶子符号（有入边但无出边的函数）"""
        return [
            node
            for node in self._graph.nodes
            if self._graph.in_degree(node) > 0 and self._graph.out_degree(node) == 0
        ]

    # ── 内部方法 ──

    def _collect_callers(
        self, symbol: str, max_depth: int, current_depth: int,
        visited: set[str], result: list[dict],
    ) -> None:
        if current_depth >= max_depth or symbol in visited:
            return
        visited.add(symbol)

        for predecessor in self._graph.predecessors(symbol):
            edge_data = self._graph.edges[predecessor, symbol]
            node_data = self._graph.nodes[predecessor]
            result.append({
                "name": predecessor,
                "depth": current_depth + 1,
                "call_type": edge_data.get("call_type", "direct"),
                "call_site": edge_data.get("call_site", ""),
                "symbol_type": node_data.get("symbol_type", "unknown"),
                "file_path": node_data.get("file_path", ""),
            })
            self._collect_callers(predecessor, max_depth, current_depth + 1, visited, result)

    def _collect_callees(
        self, symbol: str, max_depth: int, current_depth: int,
        visited: set[str], result: list[dict],
    ) -> None:
        if current_depth >= max_depth or symbol in visited:
            return
        visited.add(symbol)

        for successor in self._graph.successors(symbol):
            edge_data = self._graph.edges[symbol, successor]
            node_data = self._graph.nodes[successor]
            result.append({
                "name": successor,
                "depth": current_depth + 1,
                "call_type": edge_data.get("call_type", "direct"),
                "call_site": edge_data.get("call_site", ""),
                "symbol_type": node_data.get("symbol_type", "unknown"),
                "file_path": node_data.get("file_path", ""),
            })
            self._collect_callees(successor, max_depth, current_depth + 1, visited, result)

    def _bfs_upstream(self, symbol: str) -> list[str]:
        """BFS 向上遍历所有调用方"""
        visited: set[str] = set()
        queue = [symbol]
        result = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for predecessor in self._graph.predecessors(current):
                if predecessor not in visited:
                    result.append(predecessor)
                    queue.append(predecessor)
        return result

    def _bfs_downstream(self, symbol: str) -> list[str]:
        """BFS 向下遍历所有被调用方"""
        visited: set[str] = set()
        queue = [symbol]
        result = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for successor in self._graph.successors(current):
                if successor not in visited:
                    result.append(successor)
                    queue.append(successor)
        return result

    def _fuzzy_match(self, symbol: str) -> list[str]:
        """模糊匹配符号名"""
        # 精确后缀匹配
        matches = [
            node for node in self._graph.nodes
            if node.endswith(f".{symbol}")
        ]
        if matches:
            return matches

        # 包含匹配
        matches = [
            node for node in self._graph.nodes
            if symbol in node
        ]
        return matches

    # ── 持久化 ──

    def save(self, path: Path) -> None:
        """保存图到文件"""
        graph = self._graph.copy()
        for _, data in graph.nodes(data=True):
            for key, value in list(data.items()):
                if value is None:
                    data[key] = ""
        for _, _, data in graph.edges(data=True):
            for key, value in list(data.items()):
                if value is None:
                    data[key] = ""
        nx.write_gexf(graph, str(path))
        logger.info(f"Call graph saved to {path}")

    def load(self, path: Path) -> None:
        """从文件加载图"""
        self._graph = nx.read_gexf(str(path))
        logger.info(f"Call graph loaded from {path} ({self.node_count} nodes, {self.edge_count} edges)")
