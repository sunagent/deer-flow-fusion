"""符号提取器 — 基于 tree-sitter AST 提取符号定义、签名、参数及调用关系"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

from .models import CallEdge, CallType, Parameter, SymbolIndex, SymbolType

logger = logging.getLogger(__name__)

# ── tree-sitter 可选依赖 ──

_TS_AVAILABLE = False
try:
    import tree_sitter_python as tspython  # type: ignore[import-untyped]
    from tree_sitter import Language, Parser  # type: ignore[import-untyped]

    _TS_AVAILABLE = True
except ImportError:
    pass


# ── 语言映射 ──

_PYTHON_LANG: object | None = None
_LANGUAGE_CACHE: dict[str, object | None] = {}


def _get_python_language() -> object | None:
    """获取 Python tree-sitter Language 对象（惰性初始化）"""
    global _PYTHON_LANG
    if not _TS_AVAILABLE:
        return None
    if _PYTHON_LANG is None:
        _PYTHON_LANG = Language(tspython.language())
    return _PYTHON_LANG


def _get_language(language: str) -> object | None:
    """Return a tree-sitter language object for all supported CodeGraph languages."""
    if language == "python":
        return _get_python_language()

    if language in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[language]

    module_name = {
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "go": "tree_sitter_go",
        "rust": "tree_sitter_rust",
        "java": "tree_sitter_java",
    }.get(language)
    if module_name is None:
        _LANGUAGE_CACHE[language] = None
        return None

    try:
        module = importlib.import_module(module_name)
        if language == "typescript":
            lang = Language(module.language_typescript())
        else:
            lang = Language(module.language())
    except Exception:
        logger.warning("tree-sitter language package unavailable: %s", language, exc_info=True)
        lang = None

    _LANGUAGE_CACHE[language] = lang
    return lang


def _parse_tree(content: str, language: str) -> tuple[object | None, bytes]:
    lang = _get_language(language)
    source_bytes = content.encode("utf-8")
    if lang is None:
        return None, source_bytes
    parser = Parser()
    parser.language = lang  # type: ignore[assignment]
    return parser.parse(source_bytes), source_bytes


# ── 辅助函数 ──


def _node_text(node: object, source_bytes: bytes) -> str:
    """从 AST 节点提取原始文本"""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _module_name_from_path(file_path: str) -> str:
    """从文件路径推导模块名（如 a/b/c.py → a.b.c）"""
    p = Path(file_path)
    parts = list(p.with_suffix("").parts)
    # 去掉前导的 . / 绝对根等无意义部分
    cleaned = [part for part in parts if part and part not in (".", "/")]
    return ".".join(cleaned) if cleaned else p.stem


def _build_qualified_name(node: object, module_name: str) -> str:
    """通过向上遍历父作用域构建全限定名（module.Class.method）"""
    segments: list[str] = []

    current = node
    while current is not None:
        node_type = current.type  # type: ignore[union-attr]
        if node_type == "function_definition":
            name_node = current.child_by_field_name("name")  # type: ignore[union-attr]
            if name_node is not None:
                segments.append(_node_text(name_node, b""))
        elif node_type == "class_definition":
            name_node = current.child_by_field_name("name")  # type: ignore[union-attr]
            if name_node is not None:
                segments.append(_node_text(name_node, b""))
        current = current.parent  # type: ignore[union-attr]

    segments.reverse()
    if not segments:
        return module_name
    return f"{module_name}.{'.'.join(segments)}"


def _build_qualified_name_with_source(
    node: object, module_name: str, source_bytes: bytes
) -> str:
    """同 _build_qualified_name，但使用 source_bytes 读取名称文本"""
    segments: list[str] = []

    current = node
    while current is not None:
        node_type = current.type  # type: ignore[union-attr]
        if node_type == "function_definition":
            name_node = current.child_by_field_name("name")  # type: ignore[union-attr]
            if name_node is not None:
                segments.append(_node_text(name_node, source_bytes))
        elif node_type == "class_definition":
            name_node = current.child_by_field_name("name")  # type: ignore[union-attr]
            if name_node is not None:
                segments.append(_node_text(name_node, source_bytes))
        current = current.parent  # type: ignore[union-attr]

    segments.reverse()
    if not segments:
        return module_name
    return f"{module_name}.{'.'.join(segments)}"


def _extract_decorators(node: object, source_bytes: bytes) -> list[str]:
    """提取装饰器列表 — 检查父节点是否为 decorated_definition"""
    parent = node.parent  # type: ignore[union-attr]
    if parent is None or parent.type != "decorated_definition":  # type: ignore[union-attr]
        return []

    decorators: list[str] = []
    for child in parent.children:  # type: ignore[union-attr]
        if child.type == "decorator":  # type: ignore[union-attr]
            decorators.append(_node_text(child, source_bytes))
    return decorators


def _extract_python_parameters(params_node: object, source_bytes: bytes) -> list[Parameter]:
    """从 Python parameters 节点提取参数列表"""
    parameters: list[Parameter] = []
    if params_node is None:
        return parameters

    for child in params_node.children:  # type: ignore[union-attr]
        child_type = child.type  # type: ignore[union-attr]

        if child_type == "identifier":
            # 普通参数
            parameters.append(
                Parameter(name=_node_text(child, source_bytes))
            )

        elif child_type == "typed_parameter":
            # name: type
            name_text = ""
            type_text = None
            for sub in child.children:  # type: ignore[union-attr]
                sub_type = sub.type  # type: ignore[union-attr]
                if sub_type == "identifier":
                    name_text = _node_text(sub, source_bytes)
                elif sub_type == "type":
                    type_text = _node_text(sub, source_bytes)
            if name_text:
                parameters.append(
                    Parameter(name=name_text, type_annotation=type_text)
                )

        elif child_type == "default_parameter":
            # name=value
            name_node = child.child_by_field_name("name")  # type: ignore[union-attr]
            value_node = child.child_by_field_name("value")  # type: ignore[union-attr]
            name_text = _node_text(name_node, source_bytes) if name_node else ""
            value_text = _node_text(value_node, source_bytes) if value_node else None
            if name_text:
                parameters.append(
                    Parameter(name=name_text, default_value=value_text)
                )

        elif child_type == "typed_default_parameter":
            # name: type = value
            name_text = ""
            type_text = None
            value_text = None
            for sub in child.children:  # type: ignore[union-attr]
                sub_type = sub.type  # type: ignore[union-attr]
                if sub_type == "identifier":
                    name_text = _node_text(sub, source_bytes)
                elif sub_type == "type":
                    type_text = _node_text(sub, source_bytes)
                elif sub_type not in ("identifier", "type", ":", "=", "(", ")"):
                    # 剩下的非标点节点视为默认值
                    if value_text is None:
                        value_text = _node_text(sub, source_bytes)
            if name_text:
                parameters.append(
                    Parameter(
                        name=name_text,
                        type_annotation=type_text,
                        default_value=value_text,
                    )
                )

        elif child_type == "list_splat_pattern":
            # *args
            inner = child.child_by_field_name("name")  # type: ignore[union-attr]
            if inner is None:
                # 可能直接包含 identifier
                for sub in child.children:  # type: ignore[union-attr]
                    if sub.type == "identifier":  # type: ignore[union-attr]
                        inner = sub
                        break
            name_text = _node_text(inner, source_bytes) if inner else "*args"
            parameters.append(
                Parameter(name=name_text, is_variadic=True)
            )

        elif child_type == "dictionary_splat_pattern":
            # **kwargs
            inner = child.child_by_field_name("name")  # type: ignore[union-attr]
            if inner is None:
                for sub in child.children:  # type: ignore[union-attr]
                    if sub.type == "identifier":  # type: ignore[union-attr]
                        inner = sub
                        break
            name_text = _node_text(inner, source_bytes) if inner else "**kwargs"
            parameters.append(
                Parameter(name=name_text, is_variadic_keyword=True)
            )

    return parameters


def _extract_python_signature(node: object, source_bytes: bytes) -> str:
    """重构函数签名字符串（def name(params) -> ret:）"""
    name_node = node.child_by_field_name("name")  # type: ignore[union-attr]
    params_node = node.child_by_field_name("parameters")  # type: ignore[union-attr]
    return_node = node.child_by_field_name("return_type")  # type: ignore[union-attr]

    name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"

    # 直接取参数的原始文本以保留完整格式
    params_text = ""
    if params_node is not None:
        params_text = _node_text(params_node, source_bytes)

    return_type_text = ""
    if return_node is not None:
        return_type_text = f" -> {_node_text(return_node, source_bytes)}"

    return f"def {name}{params_text}{return_type_text}"


def _extract_base_classes(node: object, source_bytes: bytes) -> list[str]:
    """从 class 定义中提取基类列表"""
    bases: list[str] = []
    arg_list = node.child_by_field_name("superclasses")  # type: ignore[union-attr]
    if arg_list is not None:
        for child in arg_list.children:  # type: ignore[union-attr]
            if child.type in ("identifier", "attribute", "subscript"):  # type: ignore[union-attr]
                bases.append(_node_text(child, source_bytes))
    return bases


def _get_enclosing_function(node: object, source_bytes: bytes) -> str | None:
    """向上查找最近的函数/类定义，返回其名称"""
    current = node.parent  # type: ignore[union-attr]
    while current is not None:
        node_type = current.type  # type: ignore[union-attr]
        if node_type in ("function_definition", "class_definition"):
            name_node = current.child_by_field_name("name")  # type: ignore[union-attr]
            if name_node is not None:
                return _node_text(name_node, source_bytes)
        current = current.parent  # type: ignore[union-attr]
    return None


def _get_enclosing_qualified_name(
    node: object, module_name: str, source_bytes: bytes
) -> str:
    """获取包含当前节点的最近函数/类的全限定名"""
    current = node.parent  # type: ignore[union-attr]
    while current is not None:
        node_type = current.type  # type: ignore[union-attr]
        if node_type in ("function_definition", "class_definition"):
            return _build_qualified_name_with_source(current, module_name, source_bytes)
        current = current.parent  # type: ignore[union-attr]
    return module_name


def _is_inside_if(node: object) -> bool:
    """判断节点是否位于 if 语句体内"""
    current = node.parent  # type: ignore[union-attr]
    while current is not None:
        if current.type in ("if_statement", "elif_clause"):  # type: ignore[union-attr]
            return True
        # 遇到函数/类边界就停止，不跨作用域
        if current.type in ("function_definition", "class_definition"):  # type: ignore[union-attr]
            return False
        current = current.parent  # type: ignore[union-attr]
    return False


def _is_inside_loop(node: object) -> bool:
    """判断节点是否位于循环体内"""
    current = node.parent  # type: ignore[union-attr]
    while current is not None:
        if current.type in ("for_statement", "while_statement"):  # type: ignore[union-attr]
            return True
        if current.type in ("function_definition", "class_definition"):  # type: ignore[union-attr]
            return False
        current = current.parent  # type: ignore[union-attr]
    return False


def _is_inside_await(node: object) -> bool:
    """判断 call 节点是否在 await 表达式内"""
    current = node.parent  # type: ignore[union-attr]
    while current is not None:
        if current.type == "await":  # type: ignore[union-attr]
            return True
        # await 只能直接包含 call，但保险起见多查一层
        if current.type in ("function_definition", "class_definition"):  # type: ignore[union-attr]
            return False
        current = current.parent  # type: ignore[union-attr]
    return False


def _extract_call_name(call_node: object, source_bytes: bytes) -> str | None:
    """从 call 节点提取被调用函数的名称"""
    func_node = call_node.child_by_field_name("function")  # type: ignore[union-attr]
    if func_node is None:
        return None

    func_type = func_node.type  # type: ignore[union-attr]

    if func_type == "identifier":
        return _node_text(func_node, source_bytes)
    elif func_type == "attribute":
        # obj.method → 提取完整属性链
        return _node_text(func_node, source_bytes)
    elif func_type == "call":
        # 高阶函数: f()() — 递归提取
        return _extract_call_name(func_node, source_bytes)
    else:
        return _node_text(func_node, source_bytes)


# ── Python 符号提取 ──


def _extract_python_symbols(
    tree: object, source_bytes: bytes, file_path: str
) -> list[SymbolIndex]:
    """从 Python AST 提取所有符号"""
    module_name = _module_name_from_path(file_path)
    symbols: list[SymbolIndex] = []

    def _walk(node: object) -> None:
        node_type = node.type  # type: ignore[union-attr]

        if node_type == "decorated_definition":
            # 跳过装饰定义本身，直接遍历子节点（真正的定义在内部）
            for child in node.children:  # type: ignore[union-attr]
                _walk(child)
            return

        if node_type == "function_definition":
            name_node = node.child_by_field_name("name")  # type: ignore[union-attr]
            params_node = node.child_by_field_name("parameters")  # type: ignore[union-attr]
            return_node = node.child_by_field_name("return_type")  # type: ignore[union-attr]

            name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
            qualified_name = _build_qualified_name_with_source(
                node, module_name, source_bytes
            )
            signature = _extract_python_signature(node, source_bytes)
            return_type = (
                _node_text(return_node, source_bytes) if return_node else None
            )
            parameters = (
                _extract_python_parameters(params_node, source_bytes)
                if params_node
                else []
            )
            decorators = _extract_decorators(node, source_bytes)
            line_number = node.start_point[0] + 1  # type: ignore[union-attr]

            # 判断是否为方法（在类内部）
            parent = node.parent  # type: ignore[union-attr]
            symbol_type = SymbolType.FUNCTION
            if parent is not None and parent.type == "class_definition":  # type: ignore[union-attr]
                # 记录到父类 class_methods 中（后面处理类时使用）
                pass

            symbols.append(
                SymbolIndex(
                    name=name,
                    qualified_name=qualified_name,
                    symbol_type=symbol_type,
                    file_path=file_path,
                    line_number=line_number,
                    signature=signature,
                    return_type=return_type,
                    parameters=parameters,
                    decorators=decorators,
                )
            )

        elif node_type == "class_definition":
            name_node = node.child_by_field_name("name")  # type: ignore[union-attr]
            name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
            qualified_name = _build_qualified_name_with_source(
                node, module_name, source_bytes
            )
            base_classes = _extract_base_classes(node, source_bytes)
            decorators = _extract_decorators(node, source_bytes)
            line_number = node.start_point[0] + 1  # type: ignore[union-attr]

            # 收集类方法列表
            class_methods: list[str] = []
            body_node = node.child_by_field_name("body")  # type: ignore[union-attr]
            if body_node is not None:
                for child in body_node.children:  # type: ignore[union-attr]
                    if child.type == "function_definition":  # type: ignore[union-attr]
                        method_name_node = child.child_by_field_name("name")  # type: ignore[union-attr]
                        if method_name_node:
                            method_name = _node_text(method_name_node, source_bytes)
                            class_methods.append(method_name)
                    elif child.type == "decorated_definition":  # type: ignore[union-attr]
                        # 装饰器包裹的方法
                        for sub in child.children:  # type: ignore[union-attr]
                            if sub.type == "function_definition":  # type: ignore[union-attr]
                                method_name_node = sub.child_by_field_name("name")  # type: ignore[union-attr]
                                if method_name_node:
                                    method_name = _node_text(
                                        method_name_node, source_bytes
                                    )
                                    class_methods.append(method_name)

            symbols.append(
                SymbolIndex(
                    name=name,
                    qualified_name=qualified_name,
                    symbol_type=SymbolType.CLASS,
                    file_path=file_path,
                    line_number=line_number,
                    base_classes=base_classes,
                    decorators=decorators,
                    class_methods=class_methods,
                )
            )

        # 递归遍历子节点
        for child in node.children:  # type: ignore[union-attr]
            _walk(child)

    root = tree.root_node  # type: ignore[union-attr]
    _walk(root)
    return symbols


# ── Python 调用提取 ──


def _extract_python_calls(
    tree: object, source_bytes: bytes, file_path: str, symbols: list[SymbolIndex]
) -> list[CallEdge]:
    """从 Python AST 提取调用关系"""
    module_name = _module_name_from_path(file_path)
    calls: list[CallEdge] = []

    # 建立符号名 → 全限定名 的查找表
    symbol_lookup: dict[str, str] = {}
    # 方法名 → 所属类全限定名 的映射（用于 self.xxx.method 解析）
    method_to_class: dict[str, list[str]] = {}

    for sym in symbols:
        symbol_lookup[sym.name] = sym.qualified_name
        symbol_lookup[sym.qualified_name] = sym.qualified_name

        # 记录方法名到类的映射
        if sym.symbol_type == SymbolType.CLASS and sym.class_methods:
            for method_name in sym.class_methods:
                method_to_class.setdefault(method_name, []).append(sym.qualified_name)

    # 收集类属性赋值（self.xxx = Yyy()）用于解析 self.xxx.method
    # 格式: {属性名: 类名}
    self_attr_types: dict[str, str] = {}

    def _collect_self_attrs(node: object) -> None:
        """收集 self.xxx = ClassName() 形式的属性赋值"""
        node_type = node.type  # type: ignore[union-attr]

        if node_type == "assignment":
            left_node = node.child_by_field_name("left")  # type: ignore[union-attr]
            right_node = node.child_by_field_name("right")  # type: ignore[union-attr]
            if left_node is not None and right_node is not None:
                left_text = _node_text(left_node, source_bytes)
                # 匹配 self.xxx = Yyy(...)
                if left_text.startswith("self."):
                    attr_name = left_text[5:]  # 去掉 "self."
                    # 检查右侧是否是构造调用
                    if right_node.type == "call":  # type: ignore[union-attr]
                        func_node = right_node.child_by_field_name("function")  # type: ignore[union-attr]
                        if func_node is not None:
                            class_name = _node_text(func_node, source_bytes)
                            self_attr_types[attr_name] = class_name

        for child in node.children:  # type: ignore[union-attr]
            _collect_self_attrs(child)

    root = tree.root_node  # type: ignore[union-attr]
    _collect_self_attrs(root)

    def _resolve_callee(callee_name: str) -> str:
        """解析被调用方全限定名

        处理以下模式：
        - simple_name → symbol_lookup
        - self.method → 当前类.method
        - self.attr.method → attr对应类.method
        - obj.method → 尝试匹配 ClassName.method
        - module.Class.method → 直接匹配
        """
        # 1. 直接匹配
        if callee_name in symbol_lookup:
            return symbol_lookup[callee_name]

        # 2. 属性链解析 (self.xxx.method, obj.method, module.Class.method)
        parts = callee_name.split(".")
        if len(parts) >= 2:
            method_name = parts[-1]

            # self.xxx.method → 查找 self_attr_types[xxx] 对应的类
            if parts[0] == "self" and len(parts) >= 3:
                attr_name = parts[1]
                class_name = self_attr_types.get(attr_name)
                if class_name:
                    # 查找 class_name.method_name
                    class_qname = symbol_lookup.get(class_name, class_name)
                    candidate = f"{class_qname}.{method_name}"
                    if candidate in symbol_lookup:
                        return candidate
                    # 回退：直接用方法名查找
                    if method_name in symbol_lookup:
                        return symbol_lookup[method_name]

            # self.method → 当前类的方法
            elif parts[0] == "self" and len(parts) == 2:
                if method_name in symbol_lookup:
                    return symbol_lookup[method_name]

            # obj.method 或 module.Class.method
            else:
                # 尝试匹配 ClassName.method
                for class_qname in method_to_class.get(method_name, []):
                    return f"{class_qname}.{method_name}"

                # 尝试逐级匹配
                for i in range(len(parts) - 1):
                    prefix = ".".join(parts[:i+1])
                    if prefix in symbol_lookup:
                        return f"{prefix}.{method_name}"

                # 回退：只匹配方法名
                if method_name in symbol_lookup:
                    return symbol_lookup[method_name]

        # 3. 无法解析，返回原始名
        return callee_name

    def _walk(node: object) -> None:
        node_type = node.type  # type: ignore[union-attr]

        if node_type == "call":
            callee_name = _extract_call_name(node, source_bytes)
            if callee_name is not None:
                # 确定调用方全限定名
                caller_qname = _get_enclosing_qualified_name(
                    node, module_name, source_bytes
                )

                # 解析被调用方全限定名（增强版）
                callee_qname = _resolve_callee(callee_name)

                # 确定调用类型
                if _is_inside_await(node):
                    call_type = CallType.ASYNC
                elif _is_inside_if(node):
                    call_type = CallType.CONDITIONAL
                elif callee_name in (
                    "getattr",
                    "setattr",
                    "eval",
                    "exec",
                    "globals",
                    "locals",
                ):
                    call_type = CallType.DYNAMIC
                else:
                    call_type = CallType.DIRECT

                line_number = node.start_point[0] + 1  # type: ignore[union-attr]
                in_loop = _is_inside_loop(node)

                calls.append(
                    CallEdge(
                        caller=caller_qname,
                        callee=callee_qname,
                        call_site=f"{file_path}:{line_number}",
                        call_type=call_type,
                        in_loop=in_loop,
                    )
                )

        # 递归遍历子节点
        for child in node.children:  # type: ignore[union-attr]
            _walk(child)

    _walk(root)
    return calls


# ── Multi-language symbol extraction (JS/TS/Go/Rust/Java) ──

_JS_TS_DEFS = {
    "function_declaration",
    "class_declaration",
    "method_definition",
    "interface_declaration",
    "enum_declaration",
}
_GO_DEFS = {"function_declaration", "method_declaration", "type_spec"}
_RUST_DEFS = {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"}
_JAVA_DEFS = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "method_declaration",
    "constructor_declaration",
}


def _point_row(point: object) -> int:
    row = getattr(point, "row", None)
    if row is not None:
        return int(row)
    return int(point[0])  # type: ignore[index]


def _child(node: object, field_name: str) -> object | None:
    try:
        return node.child_by_field_name(field_name)  # type: ignore[union-attr]
    except Exception:
        return None


def _first_child_text(node: object, source_bytes: bytes, types: set[str]) -> str | None:
    for child in node.children:  # type: ignore[union-attr]
        if child.type in types:  # type: ignore[union-attr]
            return _node_text(child, source_bytes)
    return None


def _first_descendant_text(node: object, source_bytes: bytes, types: set[str]) -> str | None:
    stack = list(node.children)  # type: ignore[union-attr]
    while stack:
        current = stack.pop(0)
        if current.type in types:  # type: ignore[union-attr]
            return _node_text(current, source_bytes)
        stack.extend(current.children)  # type: ignore[union-attr]
    return None


def _signature_before_body(node: object, source_bytes: bytes) -> str:
    body = (
        _child(node, "body")
        or _child(node, "block")
        or _child(node, "declarator")
    )
    if body is not None and body.start_byte > node.start_byte:  # type: ignore[union-attr]
        text = source_bytes[node.start_byte : body.start_byte].decode("utf-8", errors="replace")  # type: ignore[union-attr]
    else:
        text = _node_text(node, source_bytes).split("{", 1)[0]
    return re.sub(r"\s+", " ", text).strip()


def _split_params(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if not text.strip():
        return []
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
        else:
            current.append(ch)
    item = "".join(current).strip()
    if item:
        parts.append(item)
    return parts


def _parse_parameter_text(raw: str, language: str) -> Parameter | None:
    text = raw.strip()
    if not text:
        return None
    text = re.sub(r"^(public|private|protected|final|static|mut|ref|in|out)\s+", "", text)
    variadic = text.startswith("...") or text.startswith("*")
    text = text.lstrip(".")
    if text in {"self", "&self", "mut self", "&mut self"}:
        return Parameter(name="self")
    if ":" in text and language in {"typescript", "javascript", "rust"}:
        name, typ = text.split(":", 1)
        return Parameter(name=name.strip().lstrip("&*"), type_annotation=typ.strip(), is_variadic=variadic)
    tokens = text.replace("*", " * ").split()
    if not tokens:
        return None
    if language in {"go"}:
        name = tokens[0]
        typ = " ".join(tokens[1:]) if len(tokens) > 1 else None
        return Parameter(name=name.strip("&*"), type_annotation=typ, is_variadic=variadic)
    if language in {"java"} and len(tokens) >= 2:
        name = tokens[-1]
        typ = " ".join(tokens[:-1])
        return Parameter(name=name.strip("&*"), type_annotation=typ, is_variadic=variadic)
    return Parameter(name=tokens[0].strip("&*"), type_annotation=" ".join(tokens[1:]) or None, is_variadic=variadic)


def _extract_parameters_generic(params_node: object | None, source_bytes: bytes, language: str) -> list[Parameter]:
    if params_node is None:
        return []
    params = []
    for item in _split_params(_node_text(params_node, source_bytes)):
        param = _parse_parameter_text(item, language)
        if param:
            params.append(param)
    return params


def _go_receiver_type(node: object, source_bytes: bytes) -> str | None:
    receiver = _child(node, "receiver")
    if receiver is None:
        return None
    text = _node_text(receiver, source_bytes)
    text = text.strip("() ")
    parts = text.replace("*", " ").split()
    return parts[-1] if parts else None


def _rust_impl_type(node: object, source_bytes: bytes) -> str | None:
    typ = _child(node, "type")
    if typ is not None:
        return _node_text(typ, source_bytes)
    return _first_child_text(node, source_bytes, {"type_identifier", "generic_type"})


def _nearest_ancestor_name(node: object, source_bytes: bytes, language: str) -> str | None:
    current = node.parent  # type: ignore[union-attr]
    while current is not None:
        ctype = current.type  # type: ignore[union-attr]
        if language in {"javascript", "typescript", "java"} and ctype in {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        }:
            return _symbol_name_generic(current, source_bytes, language)
        if language == "rust" and ctype == "impl_item":
            return _rust_impl_type(current, source_bytes)
        current = current.parent  # type: ignore[union-attr]
    return None


def _symbol_name_generic(node: object, source_bytes: bytes, language: str) -> str | None:
    ntype = node.type  # type: ignore[union-attr]
    name_node = _child(node, "name")
    if name_node is not None:
        return _node_text(name_node, source_bytes)
    if language in {"javascript", "typescript"} and ntype == "variable_declarator":
        name_node = _child(node, "name")
        return _node_text(name_node, source_bytes) if name_node else None
    if language == "go" and ntype == "type_spec":
        return _first_child_text(node, source_bytes, {"type_identifier"})
    if language == "rust" and ntype == "impl_item":
        return _rust_impl_type(node, source_bytes)
    return _first_child_text(node, source_bytes, {"identifier", "type_identifier", "field_identifier", "property_identifier"})


def _symbol_type_generic(node: object, language: str) -> SymbolType:
    ntype = node.type  # type: ignore[union-attr]
    if ntype in {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "impl_item",
        "struct_item",
        "trait_item",
        "type_spec",
    }:
        return SymbolType.CLASS if ntype != "interface_declaration" else SymbolType.INTERFACE
    return SymbolType.FUNCTION


def _definition_nodes_for_language(language: str) -> set[str]:
    if language in {"javascript", "typescript"}:
        return _JS_TS_DEFS | {"variable_declarator"}
    if language == "go":
        return _GO_DEFS
    if language == "rust":
        return _RUST_DEFS
    if language == "java":
        return _JAVA_DEFS
    return set()


def _is_function_like_variable(node: object) -> bool:
    if node.type != "variable_declarator":  # type: ignore[union-attr]
        return False
    value = _child(node, "value")
    return value is not None and value.type in {"arrow_function", "function_expression"}  # type: ignore[union-attr]


def _qualified_name_generic(node: object, source_bytes: bytes, module_name: str, language: str) -> str | None:
    name = _symbol_name_generic(node, source_bytes, language)
    if not name:
        return None
    parent = None
    if language == "go" and node.type == "method_declaration":  # type: ignore[union-attr]
        parent = _go_receiver_type(node, source_bytes)
    else:
        parent = _nearest_ancestor_name(node, source_bytes, language)
    if parent and parent != name:
        return f"{module_name}.{parent}.{name}"
    return f"{module_name}.{name}"


def _collect_class_methods(node: object, source_bytes: bytes, language: str) -> list[str]:
    methods: list[str] = []
    method_types = {"method_definition", "method_declaration", "constructor_declaration", "function_item"}
    body = _child(node, "body") or _child(node, "declaration_list")
    if body is None:
        return methods
    stack = list(body.children)  # type: ignore[union-attr]
    while stack:
        current = stack.pop(0)
        if current.type in method_types:  # type: ignore[union-attr]
            name = _symbol_name_generic(current, source_bytes, language)
            if name:
                methods.append(name)
            continue
        stack.extend(current.children)  # type: ignore[union-attr]
    return methods


def _extract_generic_symbols(
    tree: object, source_bytes: bytes, file_path: str, language: str
) -> list[SymbolIndex]:
    module_name = _module_name_from_path(file_path)
    symbols: list[SymbolIndex] = []
    seen: set[str] = set()
    definition_types = _definition_nodes_for_language(language)

    def _walk(node: object) -> None:
        ntype = node.type  # type: ignore[union-attr]
        is_def = ntype in definition_types
        if ntype == "variable_declarator":
            is_def = is_def and _is_function_like_variable(node)
        if is_def:
            qname = _qualified_name_generic(node, source_bytes, module_name, language)
            name = _symbol_name_generic(node, source_bytes, language)
            if qname and name and qname not in seen:
                params_node = _child(node, "parameters")
                return_node = _child(node, "return_type") or _child(node, "result") or _child(node, "type")
                symbol_type = _symbol_type_generic(node, language)
                symbols.append(
                    SymbolIndex(
                        name=name,
                        qualified_name=qname,
                        symbol_type=symbol_type,
                        file_path=file_path,
                        line_number=_point_row(node.start_point) + 1,  # type: ignore[union-attr]
                        signature=_signature_before_body(node, source_bytes),
                        return_type=_node_text(return_node, source_bytes) if return_node is not None and symbol_type == SymbolType.FUNCTION else None,
                        parameters=_extract_parameters_generic(params_node, source_bytes, language),
                        class_methods=_collect_class_methods(node, source_bytes, language)
                        if symbol_type in {SymbolType.CLASS, SymbolType.INTERFACE}
                        else [],
                    )
                )
                seen.add(qname)

        for child in node.children:  # type: ignore[union-attr]
            _walk(child)

    _walk(tree.root_node)  # type: ignore[union-attr]
    return symbols


# ── Multi-language call extraction ──

def _extract_generic_call_name(node: object, source_bytes: bytes, language: str) -> str | None:
    ntype = node.type  # type: ignore[union-attr]
    if ntype == "call_expression":
        func = _child(node, "function")
        return _node_text(func, source_bytes) if func is not None else None
    if ntype in {"method_invocation", "method_call_expression"}:
        name = _child(node, "name")
        if name is not None:
            obj = _child(node, "object")
            if obj is not None:
                return f"{_node_text(obj, source_bytes)}.{_node_text(name, source_bytes)}"
            return _node_text(name, source_bytes)
    if ntype == "object_creation_expression":
        typ = _child(node, "type")
        return _node_text(typ, source_bytes) if typ is not None else None
    text = _node_text(node, source_bytes)
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_:.<>]*)(?:\s*\(|!|\s*\{)", text)
    return match.group(1) if match else None


def _get_enclosing_qualified_name_generic(
    node: object, source_bytes: bytes, module_name: str, language: str
) -> str:
    current = node.parent  # type: ignore[union-attr]
    definition_types = _definition_nodes_for_language(language)
    while current is not None:
        ctype = current.type  # type: ignore[union-attr]
        if ctype in definition_types:
            if ctype != "variable_declarator" or _is_function_like_variable(current):
                qname = _qualified_name_generic(current, source_bytes, module_name, language)
                if qname:
                    return qname
        current = current.parent  # type: ignore[union-attr]
    return module_name


def _is_inside_node_type(node: object, node_types: set[str]) -> bool:
    current = node.parent  # type: ignore[union-attr]
    while current is not None:
        ctype = current.type  # type: ignore[union-attr]
        if ctype in node_types:
            return True
        if ctype in {
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
            "function_item",
            "constructor_declaration",
        }:
            return False
        current = current.parent  # type: ignore[union-attr]
    return False


def _resolve_generic_callee(callee_name: str, symbol_lookup: dict[str, str], method_to_owner: dict[str, list[str]]) -> str:
    if callee_name in symbol_lookup:
        return symbol_lookup[callee_name]
    simple = callee_name.split(".")[-1].split("::")[-1]
    if simple in symbol_lookup:
        return symbol_lookup[simple]
    owners = method_to_owner.get(simple)
    if owners:
        return f"{owners[0]}.{simple}"
    return callee_name


def _extract_generic_calls(
    tree: object,
    source_bytes: bytes,
    file_path: str,
    language: str,
    symbols: list[SymbolIndex],
) -> list[CallEdge]:
    module_name = _module_name_from_path(file_path)
    calls: list[CallEdge] = []
    symbol_lookup: dict[str, str] = {}
    method_to_owner: dict[str, list[str]] = {}

    for sym in symbols:
        symbol_lookup[sym.name] = sym.qualified_name
        symbol_lookup[sym.qualified_name] = sym.qualified_name
        if sym.class_methods:
            for method in sym.class_methods:
                method_to_owner.setdefault(method, []).append(sym.qualified_name)
        if "." in sym.qualified_name:
            owner = sym.qualified_name.rsplit(".", 1)[0]
            method_to_owner.setdefault(sym.name, []).append(owner)

    call_node_types = {
        "call_expression",
        "method_invocation",
        "method_call_expression",
        "object_creation_expression",
    }

    def _walk(node: object) -> None:
        if node.type in call_node_types:  # type: ignore[union-attr]
            callee_name = _extract_generic_call_name(node, source_bytes, language)
            if callee_name:
                call_type = CallType.ASYNC if _is_inside_node_type(node, {"await_expression", "await"}) else CallType.DIRECT
                if _is_inside_node_type(node, {"if_statement", "switch_statement", "conditional_expression"}):
                    call_type = CallType.CONDITIONAL
                calls.append(
                    CallEdge(
                        caller=_get_enclosing_qualified_name_generic(node, source_bytes, module_name, language),
                        callee=_resolve_generic_callee(callee_name, symbol_lookup, method_to_owner),
                        call_site=f"{file_path}:{_point_row(node.start_point) + 1}",  # type: ignore[union-attr]
                        call_type=call_type,
                        in_loop=_is_inside_node_type(node, {"for_statement", "while_statement", "for_in_statement", "enhanced_for_statement"}),
                    )
                )
        for child in node.children:  # type: ignore[union-attr]
            _walk(child)

    _walk(tree.root_node)  # type: ignore[union-attr]
    return calls


# ── 公共接口 ──


def extract_symbols(file_path: str, content: str, language: str) -> list[SymbolIndex]:
    """从源代码文件提取所有符号定义

    Args:
        file_path: 文件路径（用于构建全限定名和记录位置）
        content: 源代码文本
        language: 语言标识（如 "python", "typescript"）

    Returns:
        符号索引列表
    """
    if not _TS_AVAILABLE:
        logger.warning("tree-sitter 未安装，无法提取符号；请安装 tree-sitter 和 tree-sitter-python")
        return []

    if language == "python":
        lang = _get_python_language()
        if lang is None:
            return []
        parser = Parser()
        parser.language = lang  # type: ignore[assignment]
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)
        return _extract_python_symbols(tree, source_bytes, file_path)

    # 其他语言暂为存根
    logger.debug("符号提取暂不支持语言: %s", language)
    return []


# Clean multi-language public entry points.  These intentionally shadow the
# legacy Python-only entry points above while keeping the existing helpers.
def extract_symbols(file_path: str, content: str, language: str) -> list[SymbolIndex]:
    """Extract symbol definitions from a source file."""
    if not _TS_AVAILABLE:
        logger.warning("tree-sitter is unavailable; cannot extract symbols")
        return []

    tree, source_bytes = _parse_tree(content, language)
    return extract_symbols_from_tree(file_path, language, tree, source_bytes)


def extract_symbols_from_tree(
    file_path: str,
    language: str,
    tree: object | None,
    source_bytes: bytes,
) -> list[SymbolIndex]:
    """Extract symbol definitions from an already parsed tree."""
    if tree is None:
        return []

    if language == "python":
        return _extract_python_symbols(tree, source_bytes, file_path)
    if language in {"javascript", "typescript", "go", "rust", "java"}:
        return _extract_generic_symbols(tree, source_bytes, file_path, language)

    logger.debug("Unsupported symbol extraction language: %s", language)
    return []


def extract_calls(
    file_path: str, content: str, language: str, symbols: list[SymbolIndex]
) -> list[CallEdge]:
    """Extract call edges from a source file."""
    if not _TS_AVAILABLE:
        logger.warning("tree-sitter is unavailable; cannot extract calls")
        return []

    tree, source_bytes = _parse_tree(content, language)
    return extract_calls_from_tree(file_path, language, tree, source_bytes, symbols)


def extract_calls_from_tree(
    file_path: str,
    language: str,
    tree: object | None,
    source_bytes: bytes,
    symbols: list[SymbolIndex],
) -> list[CallEdge]:
    """Extract call edges from an already parsed tree."""
    if tree is None:
        return []

    if language == "python":
        return _extract_python_calls(tree, source_bytes, file_path, symbols)
    if language in {"javascript", "typescript", "go", "rust", "java"}:
        return _extract_generic_calls(tree, source_bytes, file_path, language, symbols)

    logger.debug("Unsupported call extraction language: %s", language)
    return []


def _legacy_extract_calls_python_only(
    file_path: str, content: str, language: str, symbols: list[SymbolIndex]
) -> list[CallEdge]:
    """从源代码文件提取调用关系

    Args:
        file_path: 文件路径
        content: 源代码文本
        language: 语言标识
        symbols: 已提取的符号列表（用于解析被调用方全限定名）

    Returns:
        调用边列表
    """
    if not _TS_AVAILABLE:
        logger.warning("tree-sitter 未安装，无法提取调用关系；请安装 tree-sitter 和 tree-sitter-python")
        return []

    if language == "python":
        lang = _get_python_language()
        if lang is None:
            return []
        parser = Parser()
        parser.language = lang  # type: ignore[assignment]
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)
        return _extract_python_calls(tree, source_bytes, file_path, symbols)

    # 其他语言暂为存根
    logger.debug("调用提取暂不支持语言: %s", language)
    return []
