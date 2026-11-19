"""Two-Tower retrieval model, feature encoder and retriever.

The user tower consumes a user-id embedding plus train-only user statistics.
The item tower consumes item-id/category/store embeddings plus catalog and
train-only interaction statistics.  Outputs are L2-normalised and trained with
BPR pairs generated from positive rows and marked implicit negatives.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.retrieval.base import BaseRetriever
from src.retrieval.faiss_index import VectorIndex

USER_FEATURE_NAMES = [
    "train_interactions",
    "train_positives",
    "train_negatives",
    "train_avg_rating",
    "train_unique_items",
    "avg_history_len",
    "category_diversity",
    "avg_item_price",
]

ITEM_FEATURE_NAMES = [
    "price",
    "price_missing",
    "metadata_avg_rating",
    "metadata_rating_number_log",
    "train_interactions_log",
    "image_count_log",
    "feature_count_log",
    "description_chars_log",
    "has_metadata",
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


def _standardise(
    values: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
    filled = np.where(np.isfinite(values), values, median)
    mean = float(filled.mean()) if filled.size else 0.0
    std = float(filled.std()) if filled.size else 1.0
    if std < 1e-8:
        std = 1.0
    return ((filled - mean) / std).astype(np.float32), mean, std, median


class TwoTowerFeatureEncoder:
    """Build fixed-shape user/item feature matrices from train-only artifacts."""

    def __init__(
        self,
        *,
        user_numeric: np.ndarray,
        user_category: np.ndarray,
        item_numeric: np.ndarray,
        item_category: np.ndarray,
        item_store: np.ndarray,
        category_vocab: dict[str, int],
        store_vocab: dict[str, int],
        user_mean: list[float],
        user_std: list[float],
        user_median: list[float],
        item_mean: list[float],
        item_std: list[float],
        item_median: list[float],
        num_users: int,
        num_items: int,
    ) -> None:
        self.user_numeric = user_numeric
        self.user_category = user_category
        self.item_numeric = item_numeric
        self.item_category = item_category
        self.item_store = item_store
        self.category_vocab = category_vocab
        self.store_vocab = store_vocab
        self.user_mean = user_mean
        self.user_std = user_std
        self.user_median = user_median
        self.item_mean = item_mean
        self.item_std = item_std
        self.item_median = item_median
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_categories = len(category_vocab)
        self.num_stores = len(store_vocab)

    # ------------------------------------------------------------------
    # Fit / persistence
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        user_features: pl.DataFrame,
        item_features: pl.DataFrame,
        *,
        store_vocab_size: int = 5000,
    ) -> "TwoTowerFeatureEncoder":
        users = user_features.sort("user_idx")
        items = item_features.filter(pl.col("item_idx") > 0).sort("item_idx")
        num_users = int(users["user_idx"].max())
        num_items = int(items["item_idx"].max())

        user_numeric = np.zeros((num_users + 1, len(USER_FEATURE_NAMES)), np.float32)
        user_mean: list[float] = []
        user_std: list[float] = []
        user_median: list[float] = []
        user_indices = users["user_idx"].to_numpy().astype(np.int64)
        for column_index, name in enumerate(USER_FEATURE_NAMES):
            raw = _float_values(users, name)
            scaled, mean, std, median = _standardise(raw)
            user_numeric[user_indices, column_index] = scaled
            user_mean.append(mean)
            user_std.append(std)
            user_median.append(median)

        item_numeric = np.zeros((num_items + 1, len(ITEM_FEATURE_NAMES)), np.float32)
        item_mean: list[float] = []
        item_std: list[float] = []
        item_median: list[float] = []
        item_indices = items["item_idx"].to_numpy().astype(np.int64)
        price_raw = _float_values(items, "price")
        price_missing = np.where(np.isfinite(price_raw), 0.0, 1.0)
        raw_features: dict[str, np.ndarray] = {
            "price": price_raw,
            "price_missing": price_missing,
            "metadata_avg_rating": _float_values(items, "metadata_avg_rating"),
            "metadata_rating_number_log": np.log1p(
                np.where(
                    np.isfinite(_float_values(items, "metadata_rating_number")),
                    _float_values(items, "metadata_rating_number"),
                    0.0,
                )
            ),
            "train_interactions_log": np.log1p(
                _float_values(items, "train_interactions")
            ),
            "image_count_log": np.log1p(_float_values(items, "image_count")),
            "feature_count_log": np.log1p(_float_values(items, "feature_count")),
            "description_chars_log": np.log1p(
                _float_values(items, "description_chars")
            ),
            "has_metadata": _float_values(items, "has_metadata"),
        }
        for column_index, name in enumerate(ITEM_FEATURE_NAMES):
            raw = raw_features[name]
            raw = np.where(np.isfinite(raw), raw, np.nan)
            scaled, mean, std, median = _standardise(raw)
            item_numeric[item_indices, column_index] = scaled
            item_mean.append(mean)
            item_std.append(std)
            item_median.append(median)

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
        user_category = np.zeros(num_users + 1, dtype=np.int64)
        favorite_values = (
            users["favorite_category"].fill_null("").cast(pl.Utf8).to_list()
        )
        for user_index, favorite in zip(user_indices, favorite_values):
            user_category[user_index] = category_vocab.get(favorite, 0)

        item_category = np.zeros(num_items + 1, dtype=np.int64)
        item_store = np.zeros(num_items + 1, dtype=np.int64)
        for item_index, category, store in zip(
            item_indices, category_values, store_values
        ):
            item_category[item_index] = category_vocab.get(category, 0)
            item_store[item_index] = store_vocab.get(store, 0)

        return cls(
            user_numeric=user_numeric,
            user_category=user_category,
            item_numeric=item_numeric,
            item_category=item_category,
            item_store=item_store,
            category_vocab=category_vocab,
            store_vocab=store_vocab,
            user_mean=user_mean,
            user_std=user_std,
            user_median=user_median,
            item_mean=item_mean,
            item_std=item_std,
            item_median=item_median,
            num_users=num_users,
            num_items=num_items,
        )

    def save(self, state_path: str | Path, arrays_path: str | Path) -> None:
        state = {
            "user_feature_names": USER_FEATURE_NAMES,
            "item_feature_names": ITEM_FEATURE_NAMES,
            "user_mean": self.user_mean,
            "user_std": self.user_std,
            "user_median": self.user_median,
            "item_mean": self.item_mean,
            "item_std": self.item_std,
            "item_median": self.item_median,
            "category_vocab": self.category_vocab,
            "store_vocab": self.store_vocab,
            "num_users": self.num_users,
            "num_items": self.num_items,
        }
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(state_path).write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            arrays_path,
            user_numeric=self.user_numeric,
            user_category=self.user_category,
            item_numeric=self.item_numeric,
            item_category=self.item_category,
            item_store=self.item_store,
        )

    @classmethod
    def load(
        cls, state_path: str | Path, arrays_path: str | Path
    ) -> "TwoTowerFeatureEncoder":
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        arrays = np.load(arrays_path)
        return cls(
            user_numeric=arrays["user_numeric"],
            user_category=arrays["user_category"],
            item_numeric=arrays["item_numeric"],
            item_category=arrays["item_category"],
            item_store=arrays["item_store"],
            category_vocab={str(k): int(v) for k, v in state["category_vocab"].items()},
            store_vocab={str(k): int(v) for k, v in state["store_vocab"].items()},
            user_mean=[float(v) for v in state["user_mean"]],
            user_std=[float(v) for v in state["user_std"]],
            user_median=[float(v) for v in state["user_median"]],
            item_mean=[float(v) for v in state["item_mean"]],
            item_std=[float(v) for v in state["item_std"]],
            item_median=[float(v) for v in state["item_median"]],
            num_users=int(state["num_users"]),
            num_items=int(state["num_items"]),
        )


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class TwoTowerModel(nn.Module):
    """User/item towers with a dot-product scoring function."""

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        user_numeric_dim: int,
        item_numeric_dim: int,
        num_categories: int,
        num_stores: int,
        embedding_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
        category_dim: int = 8,
        store_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.user_id_embedding = nn.Embedding(
            num_users + 1, embedding_dim, padding_idx=0
        )
        self.item_id_embedding = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=0
        )
        self.category_embedding = nn.Embedding(
            num_categories + 1, category_dim, padding_idx=0
        )
        self.user_category_embedding = nn.Embedding(
            num_categories + 1, category_dim, padding_idx=0
        )
        self.store_embedding = nn.Embedding(
            num_stores + 1, store_dim, padding_idx=0
        )
        self.user_tower = _mlp(
            embedding_dim + category_dim + user_numeric_dim,
            hidden_dims,
            embedding_dim,
            dropout,
        )
        self.item_tower = _mlp(
            embedding_dim + category_dim + store_dim + item_numeric_dim,
            hidden_dims,
            embedding_dim,
            dropout,
        )

    def user_vector(
        self,
        user_idx: torch.Tensor,
        user_numeric: torch.Tensor,
        user_category: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.user_id_embedding(user_idx)
        category_vector = self.user_category_embedding(user_category)
        vector = self.user_tower(
            torch.cat([embedding, category_vector, user_numeric], dim=-1)
        )
        return F.normalize(vector, p=2, dim=-1)

    def item_vector(
        self,
        item_idx: torch.Tensor,
        item_numeric: torch.Tensor,
        category: torch.Tensor,
        store: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.item_id_embedding(item_idx)
        category_vector = self.category_embedding(category)
        store_vector = self.store_embedding(store)
        vector = self.item_tower(
            torch.cat(
                [embedding, category_vector, store_vector, item_numeric],
                dim=-1,
            )
        )
        return F.normalize(vector, p=2, dim=-1)


def bpr_loss(
    user_vector: torch.Tensor,
    positive_vector: torch.Tensor,
    negative_vector: torch.Tensor,
) -> torch.Tensor:
    positive_score = (user_vector * positive_vector).sum(dim=-1)
    negative_score = (user_vector * negative_vector).sum(dim=-1)
    return -F.logsigmoid(positive_score - negative_score).mean()


def build_bpr_pairs(samples: pl.DataFrame) -> pl.DataFrame:
    """Pair each positive row with its marked sampled negatives."""
    positives = (
        samples.filter(pl.col("sample_type") == "positive")
        .select(["source_row", "user_idx", "item_idx"])
        .rename({"item_idx": "positive_item"})
    )
    negatives = (
        samples.filter(pl.col("sample_type") == "sampled_negative")
        .select(["source_row", "item_idx"])
        .rename({"item_idx": "negative_item"})
    )
    return positives.join(negatives, on="source_row", how="inner").select(
        ["user_idx", "positive_item", "negative_item"]
    )


def make_loader(
    pairs: pl.DataFrame, batch_size: int, shuffle: bool = True
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(pairs["user_idx"].to_numpy().astype(np.int64)),
        torch.from_numpy(pairs["positive_item"].to_numpy().astype(np.int64)),
        torch.from_numpy(pairs["negative_item"].to_numpy().astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def compute_two_tower_vectors(
    model: TwoTowerModel,
    encoder: TwoTowerFeatureEncoder,
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    num_users = encoder.num_users
    num_items = encoder.num_items
    dim = model.embedding_dim
    user_vectors = np.zeros((num_users + 1, dim), dtype=np.float32)
    item_vectors = np.zeros((num_items + 1, dim), dtype=np.float32)
    user_numeric = torch.from_numpy(encoder.user_numeric).to(device)
    user_category = torch.from_numpy(encoder.user_category).to(device)
    item_numeric = torch.from_numpy(encoder.item_numeric).to(device)
    item_category = torch.from_numpy(encoder.item_category).to(device)
    item_store = torch.from_numpy(encoder.item_store).to(device)

    for start in range(1, num_users + 1, batch_size):
        end = min(start + batch_size, num_users + 1)
        indices = torch.arange(start, end, device=device)
        vectors = model.user_vector(
            indices, user_numeric[indices], user_category[indices]
        )
        user_vectors[start:end] = vectors.cpu().numpy()
    for start in range(1, num_items + 1, batch_size):
        end = min(start + batch_size, num_items + 1)
        indices = torch.arange(start, end, device=device)
        vectors = model.item_vector(
            indices,
            item_numeric[indices],
            item_category[indices],
            item_store[indices],
        )
        item_vectors[start:end] = vectors.cpu().numpy()
    return user_vectors, item_vectors


def build_user_exclusions(train: pl.DataFrame) -> dict[int, set[int]]:
    grouped = (
        train.select(["user_idx", "item_idx"])
        .unique()
        .group_by("user_idx")
        .agg(pl.col("item_idx").alias("items"))
    )
    return {
        int(row["user_idx"]): {int(value) for value in row["items"]}
        for row in grouped.iter_rows(named=True)
    }


class TwoTowerRetriever(BaseRetriever):
    source = "two_tower"

    def __init__(
        self,
        user_vectors: np.ndarray,
        index: VectorIndex,
        user_map: dict[str, int],
        user_exclusions: dict[int, set[int]],
    ) -> None:
        self.user_vectors = user_vectors
        self.index = index
        self.user_map = user_map
        self.user_exclusions = user_exclusions
        self.id_to_idx = {
            item_id: position + 1
            for position, item_id in enumerate(index.ids)
        }

    @classmethod
    def from_artifacts(
        cls,
        *,
        index_path: str | Path,
        user_vectors_path: str | Path,
        user_map: pl.DataFrame,
        train: pl.DataFrame,
    ) -> "TwoTowerRetriever":
        user_vectors = np.load(user_vectors_path)
        index = VectorIndex.load(index_path)
        user_map_dict = {
            str(row["user_id"]): int(row["user_idx"])
            for row in user_map.iter_rows(named=True)
        }
        return cls(
            user_vectors,
            index,
            user_map_dict,
            build_user_exclusions(train),
        )

    def retrieve(self, user_id: str, top_k: int) -> list[dict[str, Any]]:
        top_k = self.validate_top_k(top_k)
        user_idx = self.user_map.get(str(user_id))
        if user_idx is None or user_idx <= 0:
            return []
        vector = self.user_vectors[user_idx]
        if not np.isfinite(vector).any() or float(np.linalg.norm(vector)) == 0:
            return []
        excluded = self.user_exclusions.get(user_idx, set())
        search_k = min(self.index.size, top_k + len(excluded) + 10)
        hits = self.index.search_with_ids(vector.reshape(1, -1), search_k)[0]
        results: list[dict[str, Any]] = []
        for item_id, score in hits:
            item_idx = self.id_to_idx.get(item_id)
            if item_idx is None or item_idx in excluded:
                continue
            results.append(self._result(item_id, score))
            if len(results) >= top_k:
                break
        return results
