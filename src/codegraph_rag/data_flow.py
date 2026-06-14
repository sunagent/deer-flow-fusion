"""数据流推断 — 静态分析数据在符号间的流动"""

from __future__ import annotations

import logging

from .models import DataFlowEdge, SymbolIndex, ValidationLevel

logger = logging.getLogger(__name__)


class DataFlowAnalyzer:
    """数据流分析器：推断变量/参数在函数间的流动路径"""

    def __init__(self) -> None:
        # source/sink 符号 → 数据流边列表
        self._flows: list[DataFlowEdge] = []

    @property
    def flow_count(self) -> int:
        return len(self._flows)

    # ── 构建操作 ──

    def analyze_file(
        self,
        file_path: str,
        content: str,
        language: str,
        symbols: list[SymbolIndex],
    ) -> list[DataFlowEdge]:
        """分析单个文件的数据流"""
        if language == "python":
            edges = self._analyze_python(file_path, content, symbols)
        else:
            logger.debug(f"Data flow analysis not yet supported for {language}")
            return []

        self._flows.extend(edges)
        return edges

    def _analyze_python(
        self,
        file_path: str,
        content: str,
        symbols: list[SymbolIndex],
    ) -> list[DataFlowEdge]:
        """Python 数据流分析（基于启发式规则）"""
        edges: list[DataFlowEdge] = []
        lines = content.splitlines()

        # 构建符号查找表
        symbol_map: dict[str, SymbolIndex] = {}
        for sym in symbols:
            symbol_map[sym.name] = sym

        # 规则 1: 函数参数 → 返回值的数据流
        for sym in symbols:
            if sym.symbol_type.value != "function":
                continue
            if not sym.parameters:
                continue
            # 简单启发式：如果函数体中直接返回了参数表达式
            # 这里只做基础推断，完整的数据流分析需要更复杂的 SSA 分析
            for param in sym.parameters:
                edges.append(DataFlowEdge(
                    source=f"{sym.qualified_name}.{param.name}",
                    sink=sym.qualified_name,
                    transform="pass_through",
                    file_path=file_path,
                    line_number=sym.line_number,
                    type_at_source=param.type_annotation or "",
                    type_at_sink=sym.return_type or "",
                    nullable=_is_nullable_type(param.type_annotation),
                    validation_level=ValidationLevel.NONE,
                ))

        # 规则 2: 外部输入检测（常见的输入源）
        input_sources = {
            "request": "http_request",
            "input": "user_input",
            "sys.argv": "cli_args",
            "os.environ": "env_vars",
            "os.getenv": "env_vars",
            "environ.get": "env_vars",
        }

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            for source_pattern, source_type in input_sources.items():
                if source_pattern in stripped:
                    # 找到输入源，追踪赋值目标
                    assign_target = _extract_assign_target(stripped)
                    if assign_target:
                        edges.append(DataFlowEdge(
                            source=source_type,
                            sink=f"{file_path}:{assign_target}",
                            transform="input",
                            file_path=file_path,
                            line_number=i,
                            nullable=True,
                            validation_level=ValidationLevel.NONE,
                        ))

        # 规则 3: 验证检测
        validation_patterns = {
            "isinstance": ValidationLevel.TYPE_CHECK,
            "type(": ValidationLevel.TYPE_CHECK,
            ".validate": ValidationLevel.FULL,
            "pydantic": ValidationLevel.FULL,
            "marshmallow": ValidationLevel.FULL,
            "validator": ValidationLevel.FULL,
            "assert ": ValidationLevel.TYPE_CHECK,
        }

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            for pattern, level in validation_patterns.items():
                if pattern in stripped:
                    # 找到被验证的变量
                    validated_var = _extract_validated_var(stripped, pattern)
                    if validated_var:
                        # 更新相关数据流的验证级别
                        for edge in edges:
                            if validated_var in edge.sink or validated_var in edge.source:
                                edge.validation_level = level

        # 规则 4: 清洗检测
        sanitize_patterns = ["sanitize", "escape", "html.escape", "bleach", "encode"]
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            for pattern in sanitize_patterns:
                if pattern in stripped:
                    sanitized_var = _extract_assign_target(stripped)
                    if sanitized_var:
                        for edge in edges:
                            if sanitized_var in edge.sink or sanitized_var in edge.source:
                                edge.sanitized = True

        return edges

    # ── 查询操作 ──

    def trace_source(self, sink: str) -> list[DataFlowEdge]:
        """从 sink 回溯数据来源"""
        return [
            edge for edge in self._flows
            if sink in edge.sink
        ]

    def trace_sink(self, source: str) -> list[DataFlowEdge]:
        """从 source 追踪数据去向"""
        return [
            edge for edge in self._flows
            if source in edge.source
        ]

    def find_unvalidated_inputs(self) -> list[DataFlowEdge]:
        """查找未经验证的外部输入"""
        return [
            edge for edge in self._flows
            if edge.source in ("http_request", "user_input", "cli_args", "env_vars")
            and edge.validation_level == ValidationLevel.NONE
        ]

    def find_nullable_flows(self) -> list[DataFlowEdge]:
        """查找可能为 null 的数据流"""
        return [edge for edge in self._flows if edge.nullable]

    def get_all_flows(self) -> list[DataFlowEdge]:
        """获取所有数据流边"""
        return self._flows.copy()

    def remove_file(self, file_path: str) -> None:
        """移除与某文件相关的数据流"""
        self._flows = [e for e in self._flows if e.file_path != file_path]


# ── 辅助函数 ──


def _is_nullable_type(type_annotation: str | None) -> bool:
    """判断类型注解是否可能为 null"""
    if not type_annotation:
        return True  # 无类型注解，默认可能为 null
    nullable_indicators = ["None", "Optional", "| None", "?"]
    return any(ind in type_annotation for ind in nullable_indicators)


def _extract_assign_target(line: str) -> str | None:
    """从赋值语句中提取目标变量名"""
    if "=" not in line:
        return None
    # 处理简单赋值: x = ...
    target = line.split("=")[0].strip()
    # 处理解构赋值: x, y = ...
    if "," in target:
        target = target.split(",")[0].strip()
    # 去掉类型注解
    if ":" in target:
        target = target.split(":")[0].strip()
    # 只保留合法变量名
    if target and target[0].isalpha() or target[0] == "_":
        return target
    return None


def _extract_validated_var(line: str, pattern: str) -> str | None:
    """从验证语句中提取被验证的变量名"""
    # isinstance(x, ...) → x
    if pattern == "isinstance":
        idx = line.find("isinstance(")
        if idx >= 0:
            inner = line[idx + 11:]
            if "," in inner:
                return inner.split(",")[0].strip()
    # assert isinstance(x, ...) → x
    if pattern == "assert ":
        rest = line[len("assert "):]
        return _extract_validated_var(rest, "isinstance") if "isinstance" in rest else None
    # type(x) → x
    if pattern == "type(":
        idx = line.find("type(")
        if idx >= 0:
            inner = line[idx + 5:]
            if ")" in inner:
                return inner.split(")")[0].strip()
    # x.validate() → x
    if pattern == ".validate":
        idx = line.find(".validate")
        if idx > 0:
            return line[:idx].strip().split()[-1]
    return None
