"""Item-based Collaborative Filtering on positive train interactions.

Similarity is cosine over the sparse item-user matrix.  Only the top-N item
neighbours are retained per item, keeping the artifact small enough for a
72k-item sample catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse

from src.evaluation.retrieval_metrics import top_k_indices
from src.retrieval.base import BaseRetriever


class ItemCFRetriever(BaseRetriever):
    source = "itemcf"

    def __init__(
        self,
        similarity: sparse.csr_matrix,
        item_ids: list[str],
        user_history: dict[str, list[int]],
        *,
        neighbors: int,
        num_users: int,
        num_items: int,
        user_exclusions: dict[str, set[int]] | None = None,
    ) -> None:
        self.similarity = similarity.tocsr()
        self.item_ids = item_ids
        self.user_history = user_history
        self.neighbors = int(neighbors)
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        # Exclude every train interaction (positive/negative/neutral), while
        # using only positive history as ItemCF seed items.
        self.user_exclusions = (
            user_exclusions
            if user_exclusions is not None
            else {key: set(values) for key, values in user_history.items()}
        )
        self.item_id_to_idx = {
            item_id: index
            for index, item_id in enumerate(item_ids)
            if index > 0 and item_id
        }

    # ------------------------------------------------------------------
    # Build / persistence
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        train: pl.DataFrame,
        user_map: pl.DataFrame,
        item_map: pl.DataFrame,
        *,
        neighbors: int = 100,
    ) -> "ItemCFRetriever":
        positive = (
            train.filter(pl.col("split") == "train")
            .filter(pl.col("is_positive"))
            .select(["user_idx", "item_idx"])
            .unique()
        )
        users = user_map.sort("user_idx")
        items = item_map.sort("item_idx")
        num_users = users.height
        num_items = items.height
        if num_items < 2:
            raise ValueError("ItemCF needs at least two train items")

        rows = positive["item_idx"].to_numpy().astype(np.int64) - 1
        cols = positive["user_idx"].to_numpy().astype(np.int64) - 1
        data = np.ones(rows.shape[0], dtype=np.float32)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)), shape=(num_items, num_users)
        )
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).ravel()
        inverse = np.zeros_like(norms, dtype=np.float32)
        inverse[norms > 0] = 1.0 / norms[norms > 0]
        normalized = matrix.multiply(inverse[:, None]).tocsr()

        similarity = (normalized @ normalized.T).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()

        keep = min(int(neighbors), max(1, num_items - 1))
        row_ids: list[int] = []
        col_ids: list[int] = []
        values: list[float] = []
        indptr = similarity.indptr
        indices = similarity.indices
        data_values = similarity.data
        for row in range(num_items):
            start, end = int(indptr[row]), int(indptr[row + 1])
            row_indices = indices[start:end]
            row_values = data_values[start:end]
            if row_values.size > keep:
                chosen = np.argpartition(row_values, -keep)[-keep:]
                row_indices = row_indices[chosen]
                row_values = row_values[chosen]
            order = np.argsort(-row_values)
            row_ids.extend([row] * len(order))
            col_ids.extend(row_indices[order].tolist())
            values.extend(row_values[order].tolist())
        top_similarity = sparse.csr_matrix(
            (
                np.asarray(values, dtype=np.float32),
                (np.asarray(row_ids), np.asarray(col_ids)),
            ),
            shape=(num_items, num_items),
        )

        item_ids = [""] + items["parent_asin"].to_list()
        history_frame = (
            positive.join(
                users.select(["user_idx", "user_id"]), on="user_idx", how="left"
            )
            .group_by("user_id")
            .agg(pl.col("item_idx").sort().alias("items"))
        )
        user_history = {
            str(row["user_id"]): [int(value) for value in row["items"]]
            for row in history_frame.iter_rows(named=True)
        }
        all_train = (
            train.filter(pl.col("split") == "train")
            .select(["user_idx", "item_idx"])
            .unique()
            .join(
                users.select(["user_idx", "user_id"]),
                on="user_idx",
                how="left",
            )
            .group_by("user_id")
            .agg(pl.col("item_idx").sort().alias("items"))
        )
        user_exclusions = {
            str(row["user_id"]): {int(value) for value in row["items"]}
            for row in all_train.iter_rows(named=True)
        }
        return cls(
            top_similarity,
            item_ids,
            user_history,
            neighbors=keep,
            num_users=num_users,
            num_items=num_items,
            user_exclusions=user_exclusions,
        )

    def save(self, similarity_path: str | Path, meta_path: str | Path) -> None:
        similarity_path = Path(similarity_path)
        similarity_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(similarity_path, self.similarity, compressed=True)
        Path(meta_path).write_text(
            json.dumps(
                {
                    "item_ids": self.item_ids,
                    "user_history": self.user_history,
                    "user_exclusions": {
                        key: sorted(values)
                        for key, values in self.user_exclusions.items()
                    },
                    "neighbors": self.neighbors,
                    "num_users": self.num_users,
                    "num_items": self.num_items,
                    "source": self.source,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, similarity_path: str | Path, meta_path: str | Path
    ) -> "ItemCFRetriever":
        similarity = sparse.load_npz(similarity_path).tocsr()
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        user_history = {
            str(key): [int(value) for value in values]
            for key, values in meta["user_history"].items()
        }
        user_exclusions = {
            str(key): {int(value) for value in values}
            for key, values in meta.get(
                "user_exclusions", meta["user_history"]
            ).items()
        }
        return cls(
            similarity,
            meta["item_ids"],
            user_history,
            neighbors=int(meta["neighbors"]),
            num_users=int(meta["num_users"]),
            num_items=int(meta["num_items"]),
            user_exclusions=user_exclusions,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _scores_for_history(self, history: list[int]) -> np.ndarray:
        valid = [index for index in history if 1 <= index <= self.num_items]
        if not valid:
            return np.zeros(self.num_items, dtype=np.float32)
        rows = [value - 1 for value in valid]
        scores = np.asarray(
            self.similarity[rows].sum(axis=0)
        ).ravel().astype(np.float32)
        return scores

    def retrieve(self, user_id: str, top_k: int) -> list[dict[str, Any]]:
        top_k = self.validate_top_k(top_k)
        history = self.user_history.get(str(user_id), [])
        return self.retrieve_for_history(
            history,
            top_k,
            exclude=self.user_exclusions.get(str(user_id), set()),
        )

    def retrieve_for_history(
        self,
        history: list[int],
        top_k: int,
        *,
        exclude: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        top_k = self.validate_top_k(top_k)
        scores = self._scores_for_history(history)
        if scores.size == 0 or float(scores.max()) <= 0:
            return []
        for index in (exclude if exclude is not None else set()):
            if 1 <= index <= self.num_items:
                scores[index - 1] = -np.inf
        indices = top_k_indices(scores, top_k)
        results: list[dict[str, Any]] = []
        for index in indices:
            score = float(scores[int(index)])
            if not np.isfinite(score) or score <= 0:
                continue
            item_idx = int(index) + 1
            item_id = self.item_ids[item_idx]
            if item_id:
                results.append(self._result(item_id, score))
        return results

    def similar_items(self, item_id: str, top_k: int) -> list[dict[str, Any]]:
        top_k = self.validate_top_k(top_k)
        source_idx = self.item_id_to_idx.get(str(item_id))
        if source_idx is None:
            return []
        row = self.similarity[source_idx - 1]
        row = row.toarray().ravel()
        indices = top_k_indices(row, top_k)
        results: list[dict[str, Any]] = []
        for index in indices:
            score = float(row[int(index)])
            if not np.isfinite(score) or score <= 0:
                continue
            target_id = self.item_ids[int(index) + 1]
            if target_id:
                results.append(
                    {
                        "item_id": target_id,
                        "source": self.source,
                        "retrieval_score": score,
                    }
                )
        return results
