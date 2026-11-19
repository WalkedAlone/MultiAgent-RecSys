"""Chroma-backed semantic user memory."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.logger import get_logger
from src.rag.vector_store import ChromaCollection

logger = get_logger(__name__)


class UserSemanticMemory:
    """Semantic preference statements isolated per user_id."""

    def __init__(
        self,
        *,
        collection: ChromaCollection,
        embedder: Any,
    ) -> None:
        self.collection = collection
        self.embedder = embedder

    @classmethod
    def from_config(cls, config: Any, embedder: Any) -> "UserSemanticMemory":
        collection = ChromaCollection(
            path=config.resolve_path(config.agent.memory.chroma_dir),
            collection_name="user_semantic_memory",
        )
        return cls(collection=collection, embedder=embedder)

    @staticmethod
    def make_id(user_id: str, kind: str, text: str) -> str:
        return hashlib.sha1(
            f"{user_id}|{kind}|{str(text).strip()}".encode("utf-8")
        ).hexdigest()

    def add(
        self,
        *,
        user_id: str,
        text: str,
        kind: str = "other",
        source: str = "explicit_user",
        confidence: float = 0.9,
    ) -> str:
        text = str(text).strip()
        if not text:
            raise ValueError("semantic memory text must not be empty")
        user_id = str(user_id)
        memory_id = self.make_id(user_id, kind, text)
        embedding = self.embedder.encode_documents([text])
        self.collection.upsert(
            ids=[memory_id],
            documents=[text],
            metadatas=[
                {
                    "user_id": user_id,
                    "kind": kind,
                    "value": text,
                    "source": source,
                    "confidence": float(confidence),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            embeddings=embedding,
        )
        return memory_id

    def search(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not str(query).strip():
            return []
        if self.collection.count == 0:
            return []
        embedding = self.embedder.encode_query(str(query))
        rows = self.collection.query(
            embedding,
            n_results=top_k,
            where={"user_id": str(user_id)},
        )
        return [
            {
                "id": row["id"],
                "text": row["document"],
                "score": row["score"],
                "kind": row["metadata"].get("kind"),
                "source": row["metadata"].get("source"),
                "confidence": row["metadata"].get("confidence"),
                "updated_at": row["metadata"].get("updated_at"),
            }
            for row in rows
        ]

    def delete_user(self, user_id: str) -> None:
        self.collection.delete(where={"user_id": str(user_id)})

    def delete_ids(self, ids: Sequence[str]) -> None:
        self.collection.delete(ids=list(ids))

    @property
    def count(self) -> int:
        return self.collection.count
