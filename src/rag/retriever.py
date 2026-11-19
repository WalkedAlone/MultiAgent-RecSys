"""Product-scoped review retrieval over the real Chroma review index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.core.exceptions import ModelNotReadyError
from src.core.logger import get_logger
from src.rag.embedding import BGEEmbedding
from src.rag.vector_store import ChromaCollection

logger = get_logger(__name__)


class ReviewRetriever:
    """Retrieve evidence only from reviews of the requested parent_asin."""

    def __init__(
        self,
        *,
        collection: ChromaCollection,
        embedder: BGEEmbedding,
        top_k: int = 5,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.top_k = int(top_k)

    @classmethod
    def from_config(cls, config: Any, embedder: BGEEmbedding):
        base_dir = config.resolve_path(config.agent.rag.review_index_dir)
        shard_dirs = sorted(base_dir.glob("shard_*")) if base_dir.exists() else []
        if shard_dirs:
            return ShardedReviewRetriever.from_config(
                config, embedder, base_dir=base_dir, shard_dirs=shard_dirs
            )
        collection = ChromaCollection(
            path=base_dir,
            collection_name=config.agent.rag.collection_name,
        )
        return cls(
            collection=collection,
            embedder=embedder,
            top_k=config.agent.rag.top_k,
        )

    @property
    def count(self) -> int:
        return self.collection.count

    def search(
        self,
        *,
        item_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        item_id = str(item_id)
        if self.count == 0:
            raise ModelNotReadyError(
                "Review index is empty. Run scripts/build_review_index.py first."
            )
        embedding = self.embedder.encode_query(str(query))
        rows = self.collection.query(
            embedding,
            n_results=max(1, int(top_k or self.top_k)),
            where={"parent_asin": item_id},
        )
        evidence: list[dict[str, Any]] = []
        for row in rows:
            metadata = row.get("metadata", {})
            # Defensive product-scope check in addition to Chroma's where filter.
            if str(metadata.get("parent_asin")) != item_id:
                continue
            evidence.append(
                {
                    "review_id": str(metadata.get("review_id") or row["id"]),
                    "parent_asin": item_id,
                    "title": str(metadata.get("title") or ""),
                    "text": str(row.get("document") or ""),
                    "rating": metadata.get("rating"),
                    "timestamp": metadata.get("timestamp_iso"),
                    "helpful_vote": metadata.get("helpful_vote", 0),
                    "verified_purchase": metadata.get("verified_purchase", False),
                    "score": row.get("score", 0.0),
                }
            )
        return evidence

    def search_for_items(
        self,
        *,
        item_ids: Sequence[str],
        query: str,
        top_k: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            str(item_id): self.search(
                item_id=str(item_id), query=query, top_k=top_k
            )
            for item_id in item_ids
        }

    def close(self) -> None:
        self.collection.close()


class ShardedReviewRetriever:
    """Product-scoped retrieval across bounded Chroma shards."""

    def __init__(
        self,
        *,
        base_dir: Any,
        collection_name: str,
        embedder: BGEEmbedding,
        shard_dirs: list[Any],
        top_k: int = 5,
    ) -> None:
        self.base_dir = base_dir
        self.collection_name = collection_name
        self.embedder = embedder
        self.shard_dirs = list(shard_dirs)
        shard_numbers = []
        for path in self.shard_dirs:
            try:
                shard_numbers.append(int(Path(path).name.split("_")[-1]))
            except (ValueError, IndexError):
                continue
        self.num_shards = (
            max(shard_numbers) + 1 if shard_numbers else len(self.shard_dirs)
        )
        self.top_k = int(top_k)
        self._collections: dict[int, ChromaCollection] = {}

    @classmethod
    def from_config(
        cls,
        config: Any,
        embedder: BGEEmbedding,
        *,
        base_dir: Any,
        shard_dirs: list[Any],
    ) -> "ShardedReviewRetriever":
        return cls(
            base_dir=base_dir,
            collection_name=config.agent.rag.collection_name,
            embedder=embedder,
            shard_dirs=shard_dirs,
            top_k=config.agent.rag.top_k,
        )

    @staticmethod
    def shard_index(item_id: str, num_shards: int) -> int:
        import hashlib

        digest = hashlib.sha1(str(item_id).encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % int(num_shards)

    def _collection(self, shard: int) -> ChromaCollection:
        collection = self._collections.get(shard)
        if collection is None:
            path = self.base_dir / f"shard_{shard:02d}"
            collection = ChromaCollection(
                path=path,
                collection_name=self.collection_name,
            )
            self._collections[shard] = collection
        return collection

    @property
    def count(self) -> int:
        total = 0
        for path in self.shard_dirs:
            try:
                shard = int(Path(path).name.split("_")[-1])
            except (ValueError, IndexError):
                continue
            total += self._collection(shard).count
        return total

    def search(
        self,
        *,
        item_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        item_id = str(item_id)
        shard = self.shard_index(item_id, self.num_shards)
        collection = self._collection(shard)
        if collection.count == 0:
            return []
        embedding = self.embedder.encode_query(str(query))
        rows = collection.query(
            embedding,
            n_results=max(1, int(top_k or self.top_k)),
            where={"parent_asin": item_id},
        )
        evidence: list[dict[str, Any]] = []
        for row in rows:
            metadata = row.get("metadata", {})
            if str(metadata.get("parent_asin")) != item_id:
                continue
            evidence.append(
                {
                    "review_id": str(metadata.get("review_id") or row["id"]),
                    "parent_asin": item_id,
                    "title": str(metadata.get("title") or ""),
                    "text": str(row.get("document") or ""),
                    "rating": metadata.get("rating"),
                    "timestamp": metadata.get("timestamp_iso"),
                    "helpful_vote": metadata.get("helpful_vote", 0),
                    "verified_purchase": metadata.get("verified_purchase", False),
                    "score": row.get("score", 0.0),
                }
            )
        return evidence

    def close(self) -> None:
        for collection in self._collections.values():
            collection.close()
        self._collections.clear()
