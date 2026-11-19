"""Small persistent Chroma wrapper shared by review RAG and semantic memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.core.exceptions import ModelNotReadyError
from src.core.logger import get_logger

logger = get_logger(__name__)


class ChromaCollection:
    """Persistent Chroma collection with explicit embeddings."""

    def __init__(
        self,
        *,
        path: str | Path,
        collection_name: str,
        distance: str = "cosine",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guarded
            raise ModelNotReadyError(
                "chromadb is required for Phase 5 vector memory/RAG; "
                "install requirements.txt first"
            ) from exc
        import os

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.collection_name = collection_name
        settings = chromadb.Settings(anonymized_telemetry=False)
        # hnswlib's persistent writer in Chroma 0.5.23 fails silently when the
        # persistence path contains this project's non-ASCII absolute path.
        # A relative path from the project working directory is ASCII and has
        # been verified to write/reload the HNSW segment.
        try:
            relative = os.path.relpath(path, Path.cwd())
            chroma_path = relative if not relative.startswith("..") else str(path)
        except ValueError:
            chroma_path = str(path)
        self.client = chromadb.PersistentClient(
            path=chroma_path, settings=settings
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )
        logger.info(
            "chroma_collection_ready path=%s collection=%s count=%d",
            path,
            collection_name,
            self.collection.count(),
        )

    @property
    def count(self) -> int:
        return int(self.collection.count())

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 1000,
    ) -> None:
        if not ids:
            return
        if not (
            len(ids) == len(documents) == len(metadatas) == len(embeddings)
        ):
            raise ValueError("ids/documents/metadatas/embeddings lengths differ")
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            metadata_batch = [
                self._sanitize_metadata(metadata)
                for metadata in metadatas[start:stop]
            ]
            self.collection.upsert(
                ids=list(ids[start:stop]),
                documents=list(documents[start:stop]),
                metadatas=metadata_batch,
                embeddings=np.asarray(
                    embeddings[start:stop], dtype=np.float32
                ).tolist(),
            )

    @staticmethod
    def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[str(key)] = value
            else:
                cleaned[str(key)] = str(value)
        return cleaned

    def query(
        self,
        embedding: np.ndarray,
        *,
        n_results: int = 5,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=max(1, int(n_results)),
            where=dict(where) if where else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rows: list[dict[str, Any]] = []
        for index, item_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            rows.append(
                {
                    "id": item_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distance,
                    "score": max(0.0, 1.0 - distance),
                }
            )
        return rows

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> None:
        if ids is not None:
            if ids:
                self.collection.delete(ids=list(ids))
            return
        if where is not None:
            self.collection.delete(where=dict(where))

    def get(self, *, where: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.collection.get(where=dict(where) if where else None)

    def persist(self) -> None:
        """Force Chroma 0.5.x's local HNSW segment to disk.

        Chroma 0.5.x does not expose Client.close().  Its normal process
        teardown is not reliable when a large embedding model is also being
        torn down, so the build path asks the vector segment to persist
        explicitly.
        """
        manager = getattr(self.client, "_manager", None)
        if manager is None or self.collection is None:
            return
        try:
            import uuid

            from chromadb.segment import VectorReader

            segment = manager.get_segment(
                uuid.UUID(str(self.collection.id)), VectorReader
            )
            persist = getattr(segment, "_persist", None)
            if callable(persist):
                persist()
        except Exception as exc:  # pragma: no cover - backend version edge
            logger.warning(
                "chroma_persist_failed error=%s", type(exc).__name__
            )

    def close(self) -> None:
        self.persist()
        try:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
            # Chroma 0.5.x has no Client.close(); its PersistentClient writes
            # the local HNSW segment during normal process teardown.
        finally:
            self.collection = None  # type: ignore[assignment]
