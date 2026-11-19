"""Local BGE embeddings for review RAG and semantic user memory.

Embeddings are computed locally with ``BAAI/bge-small-en-v1.5``; DeepSeek is
never asked to generate embeddings.  The model is lazy-loaded so unit tests
can inject a deterministic fake embedding function.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class BGEEmbedding:
    """Thread-safe lazy wrapper around SentenceTransformer."""

    _model_cache: dict[tuple[str, str], Any] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 64,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.normalize = bool(normalize)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        key = (self.model_name, self.device)
        with self._cache_lock:
            if key not in self._model_cache:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "bge_model_loading model=%s device=%s",
                    self.model_name,
                    self.device,
                )
                self._model_cache[key] = SentenceTransformer(
                    self.model_name, device=self.device
                )
            self._model = self._model_cache[key]
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vector = self._load().encode(
            [BGE_QUERY_INSTRUCTION + str(text)],
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vector[0], dtype=np.float32)


def build_embedding_from_config(config: Any) -> BGEEmbedding:
    """Create the configured local embedding model."""
    embedding_config = config.model.embedding
    device = embedding_config.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    return BGEEmbedding(
        embedding_config.model_name,
        batch_size=embedding_config.batch_size,
        device=str(device),
        normalize=embedding_config.normalize,
    )
