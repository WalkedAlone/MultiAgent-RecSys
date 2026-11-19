"""Unified retrieval interface used by every recall channel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """Base class for ItemCF / Two-Tower / LightGCN recall channels."""

    source: str = "base"

    @abstractmethod
    def retrieve(self, user_id: str, top_k: int) -> list[dict[str, Any]]:
        """Return a ranked list of candidate items.

        Every result must contain ``item_id``, ``source`` and
        ``retrieval_score``.
        """
        raise NotImplementedError

    def _result(self, item_id: str, score: float) -> dict[str, Any]:
        return {
            "item_id": str(item_id),
            "source": self.source,
            "retrieval_score": float(score),
        }

    @staticmethod
    def validate_top_k(top_k: int) -> int:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return int(top_k)
