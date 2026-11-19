"""Unified vector index with Faiss FastAPI and NumPy exact fallback.

Index vectors are L2-normalised, so inner product is cosine similarity.  The
same on-disk artifacts work with either backend: a ``.npy`` matrix is always
saved, and a Faiss index is written when the library is importable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# PyTorch and faiss-cpu on Windows load different OpenMP runtimes. This is the
# standard compatibility switch; without it some combined torch+faiss imports
# abort with OMP Error #15 before Python can catch anything.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:  # pragma: no cover - exercised through the fallback test branch
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # ImportError or DLL load failure
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


def _normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


class VectorIndex:
    """Vector index with the same interface for Faiss and NumPy."""

    def __init__(
        self,
        path: Path,
        vectors: np.ndarray,
        ids: list[str],
        *,
        backend: str,
        faiss_index: object | None = None,
    ) -> None:
        self.path = path
        self.vectors = vectors
        self.ids = ids
        self.backend = backend
        self._faiss_index = faiss_index

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        ids: list[str],
        path: str | Path,
        *,
        use_faiss: bool = True,
    ) -> "VectorIndex":
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        array = _normalize(vectors)
        if array.shape[0] != len(ids):
            raise ValueError("vectors and ids must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")

        np.save(base.with_suffix(".npy"), array)
        base.with_suffix(".ids.json").write_text(
            json.dumps(ids, ensure_ascii=False), encoding="utf-8"
        )
        backend = "numpy"
        faiss_index = None
        if use_faiss and FAISS_AVAILABLE:
            try:
                index = faiss.IndexFlatIP(array.shape[1])
                index.add(array)
                # Use in-memory serialisation: the project path contains
                # non-ASCII characters that Faiss's C++ FileIOWriter cannot open.
                serialized = faiss.serialize_index(index)
                base.with_suffix(".faiss").write_bytes(serialized.tobytes())
                backend = "faiss"
                faiss_index = index
            except Exception:
                backend = "numpy"
                faiss_index = None
        base.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "dim": int(array.shape[1]),
                    "count": int(array.shape[0]),
                    "normalized": True,
                    "created_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return cls(base, array, ids, backend=backend, faiss_index=faiss_index)

    @classmethod
    def load(
        cls, path: str | Path, *, force_numpy: bool = False
    ) -> "VectorIndex":
        base = Path(path)
        meta_path = base.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Index metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(base.with_suffix(".npy"))
        ids = json.loads(base.with_suffix(".ids.json").read_text(encoding="utf-8"))
        backend = "numpy"
        faiss_index = None
        if (
            not force_numpy
            and FAISS_AVAILABLE
            and meta.get("backend") == "faiss"
            and base.with_suffix(".faiss").is_file()
        ):
            try:
                raw = np.frombuffer(
                    base.with_suffix(".faiss").read_bytes(), dtype=np.uint8
                )
                faiss_index = faiss.deserialize_index(raw)
                backend = "faiss"
            except Exception:
                faiss_index = None
                backend = "numpy"
        return cls(base, vectors, ids, backend=backend, faiss_index=faiss_index)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self, query_vectors: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` with shape ``(n_queries, top_k)``."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        queries = _normalize(query_vectors)
        k = min(int(top_k), len(self.ids))
        if self.backend == "faiss" and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(queries, k)
            return scores.astype(np.float32), indices.astype(np.int64)

        scores = queries @ self.vectors.T
        if k == scores.shape[1]:
            indices = np.argsort(-scores, axis=1)[:, :k]
        else:
            indices = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, indices, axis=1), axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
        selected = np.take_along_axis(scores, indices, axis=1)
        return selected.astype(np.float32), indices.astype(np.int64)

    def search_with_ids(
        self, query_vectors: np.ndarray, top_k: int
    ) -> list[list[tuple[str, float]]]:
        scores, indices = self.search(query_vectors, top_k)
        output: list[list[tuple[str, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            output.append(
                [
                    (self.ids[int(index)], float(score))
                    for score, index in zip(row_scores, row_indices)
                    if int(index) >= 0
                ]
            )
        return output

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])
