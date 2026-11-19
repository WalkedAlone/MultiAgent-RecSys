"""Full-catalog evaluation for user/item embedding retrievers."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from src.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    aggregate_metrics,
    coverage_at_k,
    metrics_from_rank,
    target_rank,
    top_k_indices,
)


def evaluate_embedding_retrieval(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    eval_frame: pl.DataFrame,
    user_exclusions: Mapping[int, set[int]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    batch_size: int = 512,
    coverage_k: int = 50,
) -> dict[str, float]:
    """Rank every scorable target against the train catalog.

    ``user_vectors`` has shape ``(num_users+1, dim)`` and ``item_vectors``
    ``(num_items+1, dim)``; row 0 is padding. Train items are excluded.
    """
    if user_vectors.ndim != 2 or item_vectors.ndim != 2:
        raise ValueError("vectors must be two-dimensional")
    if user_vectors.shape[1] != item_vectors.shape[1]:
        raise ValueError("user/item dimensions differ")
    item_matrix = item_vectors[1:].astype(np.float32)
    num_items = item_matrix.shape[0]
    per_user: list[dict[str, float]] = []
    recommended: list[list[int]] = []
    skipped = 0
    user_idx = eval_frame["user_idx"].to_numpy().astype(np.int64)
    targets = eval_frame["item_idx"].to_numpy().astype(np.int64)
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        batch_users = user_idx[start:stop]
        batch_targets = targets[start:stop]
        vectors = user_vectors[batch_users]
        scores_batch = vectors @ item_matrix.T
        for row_offset, target in enumerate(batch_targets):
            target = int(target)
            if target <= 0 or target > num_items:
                skipped += 1
                continue
            user = int(batch_users[row_offset])
            vector = user_vectors[user]
            if not np.isfinite(vector).any() or float(np.linalg.norm(vector)) == 0:
                skipped += 1
                continue
            scores = scores_batch[row_offset]
            excluded = user_exclusions.get(user, set())
            rank = target_rank(
                scores,
                target - 1,
                exclude_indices=[value - 1 for value in excluded],
            )
            per_user.append(metrics_from_rank(int(rank), k_values))
            masked = scores.copy()
            for value in excluded:
                if 1 <= value <= num_items:
                    masked[value - 1] = -np.inf
            recommended.append(
                [int(index) + 1 for index in top_k_indices(masked, coverage_k)]
            )
    metrics = aggregate_metrics(per_user)
    metrics["users"] = float(len(per_user))
    metrics["skipped"] = float(skipped)
    metrics["coverage@50"] = coverage_at_k(recommended, num_items, coverage_k)
    metrics["catalog_items"] = float(num_items)
    return metrics
