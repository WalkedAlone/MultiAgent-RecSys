"""Categorical/bucketised feature encoder for DeepFM and DIN.

All bucket edges and vocabularies are fitted on train artifacts only.  Missing
price/rating values use bucket 0; they are never treated as zero price.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.data.feature_engineering import stable_seed

FIELD_NAMES = [
    "user_idx",
    "item_idx",
    "category_id",
    "store_id",
    "price_bucket",
    "avg_rating_bucket",
    "rating_number_bucket",
    "item_count_bucket",
    "history_len_bucket",
    "user_count_bucket",
    "user_avg_rating_bucket",
    "user_category_diversity_bucket",
]


def _float_values(frame: pl.DataFrame, name: str) -> np.ndarray:
    return (
        frame.select(
            pl.col(name).cast(pl.Float64).fill_null(float("nan"))
        )
        .to_numpy()
        .ravel()
        .astype(np.float64)
    )


class RankingFeatureEncoder:
    """Fit categorical vocabularies and bucket boundaries on train data."""

    def __init__(
        self,
        *,
        item_category: np.ndarray,
        item_store: np.ndarray,
        item_price_bucket: np.ndarray,
        item_avg_rating_bucket: np.ndarray,
        item_rating_number_bucket: np.ndarray,
        item_count_bucket: np.ndarray,
        user_count_bucket: np.ndarray,
        user_avg_rating_bucket: np.ndarray,
        user_category_diversity_bucket: np.ndarray,
        category_vocab: dict[str, int],
        store_vocab: dict[str, int],
        price_edges: list[float],
        num_users: int,
        num_items: int,
    ) -> None:
        self.item_category = item_category
        self.item_store = item_store
        self.item_price_bucket = item_price_bucket
        self.item_avg_rating_bucket = item_avg_rating_bucket
        self.item_rating_number_bucket = item_rating_number_bucket
        self.item_count_bucket = item_count_bucket
        self.user_count_bucket = user_count_bucket
        self.user_avg_rating_bucket = user_avg_rating_bucket
        self.user_category_diversity_bucket = user_category_diversity_bucket
        self.category_vocab = category_vocab
        self.store_vocab = store_vocab
        self.price_edges = price_edges
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_categories = len(category_vocab)
        self.num_stores = len(store_vocab)
        self.field_cardinalities = [
            self.num_users + 1,
            self.num_items + 1,
            self.num_categories + 1,
            self.num_stores + 1,
            11,  # price bucket 0..10
            11,  # average rating bucket 0..10
            8,   # rating-number bucket 0..7
            12,  # item count 0..11
            12,  # history length 0..11
            12,  # user count 0..11
            11,  # user average rating 0..10
            12,  # user category diversity 0..11
        ]

    @classmethod
    def fit(
        cls,
        user_features: pl.DataFrame,
        item_features: pl.DataFrame,
        *,
        store_vocab_size: int = 5000,
    ) -> "RankingFeatureEncoder":
        users = user_features.sort("user_idx")
        items = item_features.filter(pl.col("item_idx") > 0).sort("item_idx")
        num_users = int(users["user_idx"].max())
        num_items = int(items["item_idx"].max())
        user_indices = users["user_idx"].to_numpy().astype(np.int64)
        item_indices = items["item_idx"].to_numpy().astype(np.int64)

        category_values = (
            items["main_category"].fill_null("").cast(pl.Utf8).to_list()
        )
        store_values = items["store"].fill_null("").cast(pl.Utf8).to_list()
        category_counts = Counter(
            value for value in category_values if value
        )
        store_counts = Counter(value for value in store_values if value)
        category_vocab = {
            value: index + 1
            for index, (value, _) in enumerate(
                sorted(category_counts.items(), key=lambda pair: (-pair[1], pair[0]))
            )
        }
        store_vocab = {
            value: index + 1
            for index, (value, _) in enumerate(
                sorted(store_counts.items(), key=lambda pair: (-pair[1], pair[0]))[
                    :store_vocab_size
                ]
            )
        }

        item_category = np.zeros(num_items + 1, dtype=np.int64)
        item_store = np.zeros(num_items + 1, dtype=np.int64)
        for item_index, category, store in zip(
            item_indices, category_values, store_values
        ):
            item_category[item_index] = category_vocab.get(category, 0)
            item_store[item_index] = store_vocab.get(store, 0)

        price = _float_values(items, "price")
        finite_prices = price[np.isfinite(price)]
        if finite_prices.size:
            price_edges = [
                float(value)
                for value in np.quantile(
                    finite_prices,
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                )
            ]
        else:
            price_edges = [float(value) for value in range(1, 10)]
        item_price_bucket = np.zeros(num_items + 1, dtype=np.int64)
        valid_price = np.isfinite(price)
        if valid_price.any():
            item_price_bucket[item_indices[valid_price]] = (
                np.searchsorted(
                    np.asarray(price_edges), price[valid_price], side="right"
                )
                + 1
            ).astype(np.int64)

        avg_rating = _float_values(items, "metadata_avg_rating")
        item_avg_rating_bucket = np.zeros(num_items + 1, dtype=np.int64)
        valid_avg = np.isfinite(avg_rating)
        if valid_avg.any():
            item_avg_rating_bucket[item_indices[valid_avg]] = np.clip(
                np.rint(avg_rating[valid_avg] * 2), 1, 10
            ).astype(np.int64)

        rating_number = _float_values(items, "metadata_rating_number")
        item_rating_number_bucket = np.zeros(num_items + 1, dtype=np.int64)
        valid_rating_number = np.isfinite(rating_number)
        if valid_rating_number.any():
            item_rating_number_bucket[item_indices[valid_rating_number]] = (
                np.clip(
                    np.floor(np.log10(1.0 + rating_number[valid_rating_number])),
                    1,
                    7,
                )
            ).astype(np.int64)

        item_counts = _float_values(items, "train_interactions")
        item_count_bucket = np.zeros(num_items + 1, dtype=np.int64)
        item_count_bucket[item_indices] = (
            np.clip(np.nan_to_num(item_counts, nan=0.0), 0, 10) + 1
        ).astype(np.int64)

        user_counts = _float_values(users, "train_interactions")
        user_count_bucket = np.zeros(num_users + 1, dtype=np.int64)
        user_count_bucket[user_indices] = (
            np.clip(np.nan_to_num(user_counts, nan=0.0), 0, 10) + 1
        ).astype(np.int64)

        user_avg = _float_values(users, "train_avg_rating")
        user_avg_rating_bucket = np.zeros(num_users + 1, dtype=np.int64)
        valid_user_avg = np.isfinite(user_avg)
        if valid_user_avg.any():
            user_avg_rating_bucket[user_indices[valid_user_avg]] = np.clip(
                np.rint(user_avg[valid_user_avg] * 2), 1, 10
            ).astype(np.int64)

        user_diversity = _float_values(users, "category_diversity")
        user_category_diversity_bucket = np.zeros(num_users + 1, dtype=np.int64)
        user_category_diversity_bucket[user_indices] = (
            np.clip(np.nan_to_num(user_diversity, nan=0.0), 0, 10) + 1
        ).astype(np.int64)

        return cls(
            item_category=item_category,
            item_store=item_store,
            item_price_bucket=item_price_bucket,
            item_avg_rating_bucket=item_avg_rating_bucket,
            item_rating_number_bucket=item_rating_number_bucket,
            item_count_bucket=item_count_bucket,
            user_count_bucket=user_count_bucket,
            user_avg_rating_bucket=user_avg_rating_bucket,
            user_category_diversity_bucket=user_category_diversity_bucket,
            category_vocab=category_vocab,
            store_vocab=store_vocab,
            price_edges=price_edges,
            num_users=num_users,
            num_items=num_items,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _fields(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        history_len: np.ndarray,
    ) -> np.ndarray:
        user_idx = np.asarray(user_idx, dtype=np.int64)
        item_idx = np.asarray(item_idx, dtype=np.int64)
        history_len = np.asarray(history_len, dtype=np.float64)
        fields = np.zeros((user_idx.shape[0], len(FIELD_NAMES)), dtype=np.int64)
        fields[:, 0] = user_idx
        fields[:, 1] = item_idx
        fields[:, 2] = self.item_category[item_idx]
        fields[:, 3] = self.item_store[item_idx]
        fields[:, 4] = self.item_price_bucket[item_idx]
        fields[:, 5] = self.item_avg_rating_bucket[item_idx]
        fields[:, 6] = self.item_rating_number_bucket[item_idx]
        fields[:, 7] = self.item_count_bucket[item_idx]
        fields[:, 8] = (
            np.clip(np.nan_to_num(history_len, nan=0.0), 0, 10) + 1
        ).astype(np.int64)
        fields[:, 9] = self.user_count_bucket[user_idx]
        fields[:, 10] = self.user_avg_rating_bucket[user_idx]
        fields[:, 11] = self.user_category_diversity_bucket[user_idx]
        return fields

    def encode_points(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        history_len: np.ndarray,
    ) -> np.ndarray:
        """Encode already-aligned arrays (used by the DIN inference ranker)."""
        return self._fields(user_idx, item_idx, history_len)

    def encode_samples(
        self,
        samples: pl.DataFrame,
        *,
        max_history_len: int = 50,
        include_history: bool = False,
    ) -> dict[str, Any]:
        user_idx = samples["user_idx"].to_numpy().astype(np.int64)
        item_idx = samples["item_idx"].to_numpy().astype(np.int64)
        history_len = (
            samples["history_len"].fill_null(0).to_numpy().astype(np.float64)
            if "history_len" in samples.columns
            else np.zeros(len(user_idx))
        )
        labels = samples["label"].to_numpy().astype(np.int64)
        fields = self._fields(user_idx, item_idx, history_len)
        history_matrix = None
        if include_history:
            history_matrix = np.zeros(
                (len(user_idx), max_history_len), dtype=np.int64
            )
            for row_index, values in enumerate(
                samples["history_item_idxs"].to_list()
            ):
                recent = [
                    int(value)
                    for value in (values or [])
                    if value and 1 <= int(value) <= self.num_items
                ][-max_history_len:]
                if recent:
                    history_matrix[row_index, -len(recent) :] = recent
        return {
            "fields": fields,
            "labels": labels,
            "history": history_matrix,
            "user_idx": user_idx,
            "item_idx": item_idx,
        }

    def build_validation_samples(
        self,
        valid_eval: pl.DataFrame,
        train: pl.DataFrame,
        *,
        num_negatives: int = 4,
        seed: int = 42,
        max_history_len: int = 50,
    ) -> dict[str, Any]:
        """One positive + sampled implicit negatives per scorable valid user."""
        user_items_frame = (
            train.select(["user_idx", "item_idx"])
            .unique()
            .group_by("user_idx")
            .agg(pl.col("item_idx").alias("items"))
        )
        user_items = {
            int(row["user_idx"]): {int(value) for value in row["items"]}
            for row in user_items_frame.iter_rows(named=True)
        }
        users: list[int] = []
        items: list[int] = []
        labels: list[int] = []
        histories: list[int] = []
        history_items: list[list[int]] = []

        for row in valid_eval.iter_rows(named=True):
            user_idx = int(row["user_idx"])
            target = int(row["item_idx"])
            if target <= 0:
                continue
            history = [
                int(value)
                for value in (row.get("history_item_idxs") or [])
                if value and 1 <= int(value) <= self.num_items
            ][-max_history_len:]
            excluded = set(user_items.get(user_idx, set()))
            excluded.add(target)
            rng = random.Random(stable_seed(seed, str(user_idx)))
            negatives: list[int] = []
            attempts = 0
            while len(negatives) < num_negatives and attempts < max(
                100, num_negatives * 100
            ):
                attempts += 1
                candidate = rng.randrange(1, self.num_items + 1)
                if candidate in excluded or candidate in negatives:
                    continue
                negatives.append(candidate)
            for item_idx, label in [(target, 1)] + [
                (value, 0) for value in negatives
            ]:
                users.append(user_idx)
                items.append(item_idx)
                labels.append(label)
                histories.append(len(history))
                history_items.append(history)

        user_array = np.asarray(users, dtype=np.int64)
        item_array = np.asarray(items, dtype=np.int64)
        history_len = np.asarray(histories, dtype=np.float64)
        fields = self._fields(user_array, item_array, history_len)
        history_matrix = np.zeros(
            (len(user_array), max_history_len), dtype=np.int64
        )
        for row_index, history in enumerate(history_items):
            if history:
                history_matrix[row_index, -len(history) :] = history
        return {
            "fields": fields,
            "labels": np.asarray(labels, dtype=np.int64),
            "history": history_matrix,
            "user_idx": user_array,
            "item_idx": item_array,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, state_path: str | Path, arrays_path: str | Path) -> None:
        state = {
            "field_names": FIELD_NAMES,
            "field_cardinalities": self.field_cardinalities,
            "category_vocab": self.category_vocab,
            "store_vocab": self.store_vocab,
            "price_edges": self.price_edges,
            "num_users": self.num_users,
            "num_items": self.num_items,
        }
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(state_path).write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            arrays_path,
            item_category=self.item_category,
            item_store=self.item_store,
            item_price_bucket=self.item_price_bucket,
            item_avg_rating_bucket=self.item_avg_rating_bucket,
            item_rating_number_bucket=self.item_rating_number_bucket,
            item_count_bucket=self.item_count_bucket,
            user_count_bucket=self.user_count_bucket,
            user_avg_rating_bucket=self.user_avg_rating_bucket,
            user_category_diversity_bucket=self.user_category_diversity_bucket,
        )

    @classmethod
    def load(
        cls, state_path: str | Path, arrays_path: str | Path
    ) -> "RankingFeatureEncoder":
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        arrays = np.load(arrays_path)
        return cls(
            item_category=arrays["item_category"],
            item_store=arrays["item_store"],
            item_price_bucket=arrays["item_price_bucket"],
            item_avg_rating_bucket=arrays["item_avg_rating_bucket"],
            item_rating_number_bucket=arrays["item_rating_number_bucket"],
            item_count_bucket=arrays["item_count_bucket"],
            user_count_bucket=arrays["user_count_bucket"],
            user_avg_rating_bucket=arrays["user_avg_rating_bucket"],
            user_category_diversity_bucket=arrays[
                "user_category_diversity_bucket"
            ],
            category_vocab={str(k): int(v) for k, v in state["category_vocab"].items()},
            store_vocab={str(k): int(v) for k, v in state["store_vocab"].items()},
            price_edges=[float(v) for v in state["price_edges"]],
            num_users=int(state["num_users"]),
            num_items=int(state["num_items"]),
        )
