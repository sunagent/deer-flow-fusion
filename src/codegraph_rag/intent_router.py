"""意图路由器 — 轻量级规则分类器"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .models import SearchResult


class QueryIntent(str, Enum):
    SYMBOL = "symbol"
    RELATION = "relation"
    IMPACT = "impact"
    SEMANTIC = "semantic"


@dataclass
class IntentResult:
    """意图分类结果"""

    intent: QueryIntent
    confidence: float = 0.0
    extracted: dict[str, str] = field(default_factory=dict)


# 意图分类规则（按优先级排序，先匹配先命中）
_INTENT_RULES: list[tuple[QueryIntent, list[re.Pattern[str]]]] = []

# 影响查询：提到变更/影响（优先级最高，避免被 RELATION 抢先）
_INTENT_RULES.append(
    (
        QueryIntent.IMPACT,
        [
            re.compile(r"(?:impact|affect|break|blast\s+radius|ripple\s+effect)\s+(?:of\s+|for\s+)?(\w+)", re.IGNORECASE),
            re.compile(r"what\s+happens?\s+if\s+(?:I\s+)?(?:change|modify|remove|delete)\s+(\w+)", re.IGNORECASE),
            re.compile(r"(?:effect|consequence|result)\s+of\s+(?:changing|modifying|removing|deleting)\s+(\w+)", re.IGNORECASE),
            re.compile(r"(?:changing|modifying|removing|deleting)\s+(\w+)", re.IGNORECASE),
        ],
    )
)

# 关系查询：提到调用/依赖
_INTENT_RULES.append(
    (
        QueryIntent.RELATION,
        [
            re.compile(r"(?:who\s+calls?|callers?\s+of|callees?\s+of|depends?\s+on|referenced?\s+by)\s+(\w+)", re.IGNORECASE),
            re.compile(r"(\w+)\s*(?:calls?|invokes?|imports?|references?)\b", re.IGNORECASE),
            re.compile(r"(?:call\s+graph|dependency|dependencies)\s+(?:of\s+|for\s+)?(\w+)?", re.IGNORECASE),
            re.compile(r"what\s+(?:does|do)\s+(\w+)\s+call", re.IGNORECASE),
        ],
    )
)

# 符号查询：明确提到符号名/定义
_INTENT_RULES.append(
    (
        QueryIntent.SYMBOL,
        [
            re.compile(r"(?:where\s+is|locate|definition\s+of)\s+(\w+)", re.IGNORECASE),
            re.compile(r"(\w+)\s+(?:class|function|method|variable|interface|trait|struct)", re.IGNORECASE),
            re.compile(r"(?:find|show\s+me)\s+(\w+)", re.IGNORECASE),
            re.compile(r"(?:what\s+is)\s+(\w+)", re.IGNORECASE),
        ],
    )
)

# 语义查询：自然语言描述
_INTENT_RULES.append(
    (
        QueryIntent.SEMANTIC,
        [
            re.compile(r"(?:how\s+does|how\s+is|explain|what\s+is\s+the|describe)\s+(.+)", re.IGNORECASE),
            re.compile(r"(.+?)\s+(?:flow|process|pattern|patterns|architecture|design|implementation)", re.IGNORECASE),
            re.compile(r"(?:find|search|show)\s+(?:code\s+)?(?:similar\s+to|like)\s+(.+)", re.IGNORECASE),
        ],
    )
)


def classify_intent(query: str) -> IntentResult:
    """轻量级意图分类器，基于正则规则，不依赖 LLM

    优先级：SYMBOL > RELATION > IMPACT > SEMANTIC
    未匹配时默认走 SEMANTIC
    """
    if not query or not query.strip():
        return IntentResult(intent=QueryIntent.SEMANTIC, confidence=0.3, extracted={"query": query or ""})

    query = query.strip()

    for intent, patterns in _INTENT_RULES:
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                # 提取第一个非 None 的分组
                entity = ""
                for group in match.groups():
                    if group:
                        entity = group.strip()
                        break
                return IntentResult(
                    intent=intent,
                    confidence=0.8,
                    extracted={"entity": entity, "query": query},
                )

    # 默认走语义搜索
    return IntentResult(
        intent=QueryIntent.SEMANTIC,
        confidence=0.5,
        extracted={"query": query},
    )


def route_query(
    query: str,
    *,
    # 图查询回调
    symbol_lookup: object | None = None,
    graph_traversal: object | None = None,
    blast_radius: object | None = None,
    # 混合搜索回调
    hybrid_search_fn: object | None = None,
) -> list[SearchResult]:
    """根据意图路由到最优查询路径

    Args:
        query: 用户查询文本
        symbol_lookup: 符号查找回调 (entity: str) -> list[SearchResult]
        graph_traversal: 图遍历回调 (entity: str) -> list[SearchResult]
        blast_radius: 影响分析回调 (entity: str) -> list[SearchResult]
        hybrid_search_fn: 混合搜索回调 (query: str) -> list[SearchResult]

    Returns:
        搜索结果列表
    """
    intent_result = classify_intent(query)

    if intent_result.intent == QueryIntent.SYMBOL and symbol_lookup is not None:
        return symbol_lookup(intent_result.extracted.get("entity", query))

    if intent_result.intent == QueryIntent.RELATION and graph_traversal is not None:
        return graph_traversal(intent_result.extracted.get("entity", query))

    if intent_result.intent == QueryIntent.IMPACT and blast_radius is not None:
        return blast_radius(intent_result.extracted.get("entity", query))

    # 默认走混合搜索
    if hybrid_search_fn is not None:
        return hybrid_search_fn(query)

    return []
