"""Embedding 缓存 — 按需计算 + LRU 缓存"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


# ── Embedding 提供者抽象 ──


class EmbeddingProvider(ABC):
    """Embedding 提供者基类"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """计算单条文本的 embedding"""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量计算 embedding"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class JinaProvider(EmbeddingProvider):
    """jina-embeddings-v3 via API"""

    def __init__(self, config: EmbeddingConfig):
        self._dimension = config.dimension
        self._api_base = config.api_base or "https://api.jina.ai/v1/embeddings"
        self._model = config.model_name or "jina-embeddings-v3"
        self._api_key: str | None = None
        # 尝试从环境变量获取 API key
        import os

        self._api_key = os.getenv("JINA_API_KEY") or os.getenv("JINA_API_BASE")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        if not self._api_key:
            logger.warning("JINA_API_KEY not set, skipping embedding")
            return [[] for _ in texts]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimension,
        }
        try:
            resp = httpx.post(self._api_base, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Jina embedding API error: {e}")
            return [[] for _ in texts]




class JinaCodeProvider(JinaProvider):
    """jina-embeddings-v2-small-code - 768-dim code-optimized embeddings"""
    def __init__(self, config: EmbeddingConfig):
        config.dimension = 768
        config.model_name = config.model_name or "jina-embeddings-v2-small-code"
        super().__init__(config)

    def embed_batch(self, texts):
        import httpx
        if not self._api_key:
            logger.warning("JINA_API_KEY not set, skipping embedding")
            return [[] for _ in texts]
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self._model, "input": texts}
        try:
            resp = httpx.post(self._api_base, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Jina v2 embedding API error: {e}")
            return [[] for _ in texts]

class OpenAIProvider(EmbeddingProvider):
    """OpenAI text-embedding via API"""

    def __init__(self, config: EmbeddingConfig):
        self._dimension = config.dimension
        self._api_base = config.api_base or "https://api.openai.com/v1/embeddings"
        self._model = config.model_name or "text-embedding-3-small"
        import os

        self._api_key = os.getenv("OPENAI_API_KEY")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        if not self._api_key:
            logger.warning("OPENAI_API_KEY not set, skipping embedding")
            return [[] for _ in texts]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimension,
        }
        try:
            resp = httpx.post(self._api_base, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"OpenAI embedding API error: {e}")
            return [[] for _ in texts]


class OllamaProvider(EmbeddingProvider):
    """本地 Ollama embedding"""

    def __init__(self, config: EmbeddingConfig):
        self._dimension = config.dimension
        self._api_base = config.api_base or "http://localhost:11434"
        self._model = config.model_name or "nomic-embed-text"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        results = []
        for text in texts:
            try:
                resp = httpx.post(
                    f"{self._api_base}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                    timeout=60,
                )
                resp.raise_for_status()
                results.append(resp.json()["embedding"])
            except Exception as e:
                logger.error(f"Ollama embedding error: {e}")
                results.append([])
        return results


class NomicProvider(OllamaProvider):
    """nomic-embed-text via Ollama"""

    def __init__(self, config: EmbeddingConfig):
        config.model_name = config.model_name or "nomic-embed-text"
        super().__init__(config)


# ── 提供者工厂 ──

_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "jina": JinaProvider,
    "jina-code": JinaCodeProvider,
    "local": None,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "nomic": NomicProvider,
}

from .embedding_local import JinaCodeLocalProvider  # noqa: E402
_PROVIDERS["local"] = JinaCodeLocalProvider


def create_provider(config: EmbeddingConfig) -> EmbeddingProvider | None:
    """根据配置创建 Embedding 提供者"""
    if config.provider is None:
        return None
    cls = _PROVIDERS.get(config.provider)
    if cls is None:
        logger.warning(f"Unknown embedding provider: {config.provider}")
        return None
    try:
        return cls(config)
    except Exception as e:
        logger.error(f"Failed to create embedding provider '{config.provider}': {e}")
        return None


# ── Embedding 缓存 ──


class EmbeddingCache:
    """按需计算 + LRU 缓存的 Embedding 服务"""

    def __init__(self, dimension: int = 1024, max_cache: int = 10000):
        self.dimension = dimension
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.max_cache = max_cache
        self._provider: EmbeddingProvider | None = None

    def set_provider(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self.dimension = provider.dimension
        logger.info(f"Embedding provider set: {type(provider).__name__}, dimension={self.dimension}")

    def get_embedding(self, text: str) -> list[float] | None:
        """按需获取 embedding，命中缓存秒返回"""
        # 使用文本的 hash 作为缓存 key（避免超长 key）
        cache_key = str(hash(text))

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        if self._provider is None:
            return None

        try:
            embedding = self._provider.embed(text)
            if not embedding:
                return None
            self._cache[cache_key] = embedding
            if len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)  # LRU 淘汰
            return embedding
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return None

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float] | None]:
        """批量获取 embedding"""
        results: list[list[float] | None] = []
        uncached: list[tuple[int, str]] = []

        # 先查缓存
        for i, text in enumerate(texts):
            cache_key = str(hash(text))
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                results.append(self._cache[cache_key])
            else:
                results.append(None)
                uncached.append((i, text))

        # 批量计算未缓存的
        if uncached and self._provider is not None:
            try:
                batch_texts = [t for _, t in uncached]
                batch_embeddings = self._provider.embed_batch(batch_texts)
                for (idx, text), embedding in zip(uncached, batch_embeddings):
                    if embedding:
                        cache_key = str(hash(text))
                        self._cache[cache_key] = embedding
                        if len(self._cache) > self.max_cache:
                            self._cache.popitem(last=False)
                        results[idx] = embedding
            except Exception as e:
                logger.error(f"Batch embedding computation failed: {e}")

        return results

    def invalidate(self, file_path: str) -> None:
        """清除与某文件相关的缓存（简单实现：清空全部）"""
        # 精确清除需要维护 file_path → cache_key 映射，这里简化处理
        self._cache.clear()
        logger.debug(f"Embedding cache cleared (triggered by {file_path})")

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def is_available(self) -> bool:
        return self._provider is not None
