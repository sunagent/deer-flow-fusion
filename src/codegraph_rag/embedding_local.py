"""Local Jina code embeddings, zero API key."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class JinaCodeLocalProvider(EmbeddingProvider):
    """jina-embeddings-v2-base-code running locally.

    The preferred path uses sentence-transformers. Some dependency sets fail
    while loading Jina's remote tokenizer code, so a direct transformers loader
    is kept as an offline-safe fallback for the already cached local model.
    """

    def __init__(self, config):
        model_name = getattr(config, "model_name", None) or "jinaai/jina-embeddings-v2-base-code"
        self._dimension = 768
        self._model_name = model_name
        self._backend = "sentence-transformers"
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device = "cpu"

        logger.info("Loading local embedding model: %s ...", model_name)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, trust_remote_code=True)
            actual_dim = self._model.get_embedding_dimension()
            if actual_dim:
                self._dimension = actual_dim
            logger.info("Loaded %s dim=%s via sentence-transformers", model_name, self._dimension)
        except Exception as exc:
            logger.warning(
                "SentenceTransformer failed for %s: %s. Falling back to direct transformers loader.",
                model_name,
                exc,
            )
            self._load_transformers_fallback(model_name)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        if self._backend == "sentence-transformers":
            return [
                e.tolist()
                for e in self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            ]
        return self._embed_batch_transformers(texts)

    def _load_transformers_fallback(self, model_name: str) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        self._backend = "transformers"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        original_from_pretrained = transformers.AutoTokenizer.from_pretrained

        def slow_tokenizer_from_pretrained(*args, **kwargs):
            kwargs.setdefault("trust_remote_code", True)
            kwargs.setdefault("use_fast", False)
            return original_from_pretrained(*args, **kwargs)

        transformers.AutoTokenizer.from_pretrained = slow_tokenizer_from_pretrained
        try:
            self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        finally:
            transformers.AutoTokenizer.from_pretrained = original_from_pretrained

        self._model.eval()
        self._model.to(self._device)

        hidden_size = getattr(getattr(self._model, "config", None), "hidden_size", None)
        if hidden_size:
            self._dimension = int(hidden_size)
        logger.info(
            "Loaded %s dim=%s via transformers fallback on %s",
            model_name,
            self._dimension,
            self._device,
        )

    def _embed_batch_transformers(self, texts) -> list[list[float]]:
        import torch
        import torch.nn.functional as F

        if not texts:
            return []

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with torch.no_grad():
            output = self._model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embeddings = F.normalize(summed / counts, p=2, dim=1)

        return embeddings.cpu().tolist()
