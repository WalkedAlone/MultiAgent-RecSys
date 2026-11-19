"""LightGCN over the user-item bipartite graph.

LightGCN deliberately removes the feature transformation and nonlinear
activation used by ordinary GCNs.  It propagates only normalized embeddings and
combines all layers.  This implementation uses a symmetric-normalized sparse
adjacency and full-batch BPR training on positive train interactions with marked
sampled negatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from src.retrieval.base import BaseRetriever
from src.retrieval.faiss_index import VectorIndex
from src.retrieval.two_tower import build_user_exclusions


def build_normalized_adjacency(
    train: pl.DataFrame,
    user_map: pl.DataFrame,
    item_map: pl.DataFrame,
) -> tuple[torch.Tensor, int, int, int]:
    """Return symmetric-normalized sparse adjacency and node counts."""
    num_users = int(user_map.height)
    num_items = int(item_map.height)
    node_count = num_users + num_items + 1
    positive = (
        train.filter(pl.col("split") == "train")
        .filter(pl.col("is_positive"))
        .select(["user_idx", "item_idx"])
        .unique()
    )
    users = positive["user_idx"].to_numpy().astype(np.int64)
    items = positive["item_idx"].to_numpy().astype(np.int64)
    item_nodes = num_users + items
    rows = np.concatenate([users, item_nodes])
    cols = np.concatenate([item_nodes, users])
    data = np.ones(rows.shape[0], dtype=np.float32)
    degrees = np.bincount(rows, minlength=node_count).astype(np.float64)
    inv_sqrt = np.zeros_like(degrees)
    non_zero = degrees > 0
    inv_sqrt[non_zero] = 1.0 / np.sqrt(degrees[non_zero])
    values = (data * inv_sqrt[rows] * inv_sqrt[cols]).astype(np.float32)
    indices = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    adjacency = torch.sparse_coo_tensor(
        indices, torch.from_numpy(values), (node_count, node_count)
    ).coalesce()
    return adjacency, num_users, num_items, node_count


class LightGCNModel(nn.Module):
    def __init__(
        self,
        node_count: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(node_count, embedding_dim, padding_idx=0)
        nn.init.normal_(self.node_embedding.weight, std=0.1)
        with torch.no_grad():
            self.node_embedding.weight[0].zero_()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

    def propagate(self, adjacency: torch.Tensor) -> torch.Tensor:
        embeddings = self.node_embedding.weight
        layers = [embeddings]
        current = embeddings
        for _ in range(self.num_layers):
            current = torch.sparse.mm(adjacency, current)
            layers.append(current)
        return torch.stack(layers, dim=0).mean(dim=0)


@torch.no_grad()
def compute_lightgcn_vectors(
    model: LightGCNModel,
    adjacency: torch.Tensor,
    num_users: int,
    num_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    final = model.propagate(adjacency)
    user = final[1 : num_users + 1]
    item = final[num_users + 1 : num_users + num_items + 1]
    user = F.normalize(user, p=2, dim=-1)
    item = F.normalize(item, p=2, dim=-1)
    user_vectors = np.zeros((num_users + 1, user.shape[1]), dtype=np.float32)
    item_vectors = np.zeros((num_items + 1, item.shape[1]), dtype=np.float32)
    user_vectors[1:] = user.cpu().numpy()
    item_vectors[1:] = item.cpu().numpy()
    return user_vectors, item_vectors


class LightGCNRetriever(BaseRetriever):
    source = "lightgcn"

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
    ) -> "LightGCNRetriever":
        return cls(
            np.load(user_vectors_path),
            VectorIndex.load(index_path),
            {
                str(row["user_id"]): int(row["user_idx"])
                for row in user_map.iter_rows(named=True)
            },
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
