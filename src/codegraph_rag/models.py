"""CodeGraph 数据模型"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Layer 1: 代码块层 ──


class ChunkType(str, Enum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"


@dataclass
class CodeChunk:
    """代码块 — AST 感知分块的最小单元"""

    id: str  # {file_path}#{symbol_name}@{start_line}-{end_line}
    content: str
    file_path: str
    language: str
    chunk_type: ChunkType
    symbol_name: str | None = None
    parent_symbol: str | None = None
    start_line: int = 0
    end_line: int = 0
    tags: list[str] = field(default_factory=list)
    docstring: str | None = None


# ── Layer 2: 符号层 ──


class SymbolType(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    DECORATOR = "decorator"
    MODULE = "module"


@dataclass
class Parameter:
    """函数/方法参数"""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_variadic: bool = False  # *args
    is_keyword_only: bool = False  # * 后参数
    is_variadic_keyword: bool = False  # **kwargs


@dataclass
class SymbolIndex:
    """符号索引 — 函数/类/变量等的结构化描述"""

    name: str
    qualified_name: str  # module.Class.method
    symbol_type: SymbolType
    file_path: str
    line_number: int
    signature: str | None = None
    return_type: str | None = None
    parameters: list[Parameter] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)
    implemented_interfaces: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    class_methods: list[str] = field(default_factory=list)


# ── Layer 3: 调用图层 ──


class CallType(str, Enum):
    DIRECT = "direct"
    CONDITIONAL = "conditional"
    ASYNC = "async"
    DYNAMIC = "dynamic"


@dataclass
class CallEdge:
    """调用边 — 函数间的调用关系"""

    caller: str  # 调用方全限定名
    callee: str  # 被调用方全限定名
    call_site: str  # file:line
    call_type: CallType = CallType.DIRECT
    context: str | None = None
    condition: str | None = None
    in_loop: bool = False


# ── Layer 4: 数据流层 ──


class ValidationLevel(str, Enum):
    NONE = "none"
    TYPE_CHECK = "type_check"
    FULL = "full"


@dataclass
class DataFlowEdge:
    """数据流边 — 数据在符号间的流动"""

    source: str
    sink: str
    transform: str
    file_path: str
    line_number: int
    type_at_source: str = ""
    type_at_sink: str = ""
    nullable: bool = False
    validation_level: ValidationLevel = ValidationLevel.NONE
    sanitized: bool = False


# ── Layer 5: 社区层 ──


class CommunityType(str, Enum):
    FUNCTIONAL = "functional"
    LAYER = "layer"
    CROSS_CUTTING = "cross_cutting"


@dataclass
class CommunityIndex:
    """社区 — Leiden 算法自动发现的功能模块"""

    id: int
    members: list[str] = field(default_factory=list)
    name: str | None = None
    community_type: CommunityType | None = None
    cohesion: float = 0.0
    coupling: dict[int, float] = field(default_factory=dict)
    god_nodes: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)


# ── 搜索结果 ──


@dataclass
class SearchResult:
    """统一搜索结果"""

    id: str
    symbol_name: str | None
    qualified_name: str | None
    file_path: str
    line_number: int
    chunk_type: ChunkType
    score: float = 0.0
    snippet: str | None = None
    signature: str | None = None
    docstring: str | None = None
    tags: list[str] = field(default_factory=list)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    language: str | None = None
    explain: dict = field(default_factory=dict)
