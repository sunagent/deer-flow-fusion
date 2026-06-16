"""CodeGraph 配置模型"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Embedding 配置（可选，不配置时仅使用 BM25）"""

    provider: Literal["jina", "jina-code", "local", "nomic", "openai", "ollama"] | None = Field(
        default=None,
        description="Embedding 提供者，null 表示仅使用 BM25",
    )
    dimension: int = Field(default=768, description="向量维度（需与 provider 匹配）")
    cache_size: int = Field(default=10000, description="LRU 缓存条目数")
    api_base: str | None = Field(default=None, description="API 地址（null = 使用默认）")
    model_name: str | None = Field(default=None, description="模型名称覆盖")


class IndexConfig(BaseModel):
    """索引配置"""

    workspace_root: str | None = Field(
        default=None,
        description="工作区根目录，null = 使用运行时工作目录",
    )
    supported_languages: list[str] = Field(
        default=["python", "typescript", "javascript", "go", "rust", "java"],
        description="支持的语言",
    )
    chunk_max_lines: int = Field(default=200, description="单 chunk 最大行数")
    lazy_load_threshold: int = Field(default=1000, description="超过此文件数启用懒加载")
    class_method_split_threshold: int = Field(
        default=8,
        description="类方法数超过此值时拆分为类头+方法组",
    )


class StorageConfig(BaseModel):
    """存储配置"""

    persist_directory: str = Field(
        default=".codegraph",
        description="持久化目录（相对于工作区根目录）",
    )
    vector_dtype: Literal["float32", "int8"] = Field(
        default="float32",
        description="向量存储类型：float32=基线精度，int8=低内存量化后端",
    )
    enable_embedding_cache: bool = Field(
        default=True,
        description="启用函数级向量缓存，用于增量索引时复用未变化函数向量",
    )



class SearchConfig(BaseModel):
    """查询配置"""

    default_top_k: int = Field(default=50, description="默认返回结果数")
    max_call_depth: int = Field(default=3, description="最大调用图递归深度")
    rrf_k: int = Field(default=60, description="RRF 融合常数")
    enable_graph_search: bool = Field(default=True, description="启用图检索")
    max_graph_symbol_candidates: int = Field(default=12, description="单次图检索最多处理的候选符号数")
    max_graph_neighbors: int = Field(default=12, description="单个符号每个方向最多扩展的调用图邻居数")
    enable_vector_search: bool = Field(default=True, description="启用向量语义检索")
    enable_bm25: bool = Field(default=True, description="启用 BM25 全文检索")
    weight_graph: float = Field(default=0.7, description="图检索权重（代码结构）")
    weight_vector: float = Field(default=0.2, description="向量语义权重")
    weight_bm25: float = Field(default=0.1, description="BM25 关键词权重")
    merge_mode: Literal["rrf", "concat"] = Field(default="rrf", description="融合模式：rrf加权 / concat拼接")
    enable_rerank: bool = Field(default=True, description="Enable ReRanker v1 post-merge")
    rerank_top_k: int = Field(default=20, description="P1 ReRanker 精排置顶候选数")
    nl_rrf_k: int | None = Field(default=None, description="NL 查询专属 RRF k；null 表示沿用 rrf_k")
    enable_nl_query_enhance: bool = Field(default=True, description="启用纯英文 NL 查询规范化增强")
    enable_context_prior: bool = Field(default=True, description="启用语言/项目上下文先验；未传 context 时无影响")
    context_routing_mode: Literal["off", "language"] = Field(default="language", description="上下文路由模式：off=只软加分，language=同语言主池召回")
    context_rrf_boost: float = Field(default=0.003, description="RRF 阶段上下文最大加分，用于多语言候选池去噪")
    context_rerank_boost: float = Field(default=0.10, description="P1 精排阶段上下文最大加分")
    deterministic_sort: bool = Field(default=True, description="启用稳定 tie-breaker，保证同分候选排序可复现")


class CodeGraphConfig(BaseModel):
    """CodeGraph 完整配置"""

    enabled: bool = Field(default=False, description="是否启用 CodeGraph")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    def get_persist_path(self, workspace_root: Path | None = None) -> Path:
        """获取持久化目录的绝对路径"""
        root = Path(workspace_root) if workspace_root else Path.cwd()
        return root / self.storage.persist_directory

    def stable_hash(self) -> str:
        """Stable config hash for explain traces and reproducibility checks."""
        payload = self.model_dump(mode="json")
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


