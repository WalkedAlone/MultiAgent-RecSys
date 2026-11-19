"""Retrieval evaluation metrics for full-catalog and sampled rankings.

This dataset is implicit preference data, not exposure logs.  Recall/HitRate/
NDCG/MRR here measure whether a held-out positive item ranks highly after the
user's train history is excluded.  They are not CTR estimates.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

DEFAULT_K_VALUES: tuple[int, ...] = (10, 20, 50, 100)


def target_rank(
    scores: np.ndarray,
    target_index: int,
    exclude_indices: Iterable[int] = (),
) -> int:
    """Return the 1-based rank of a target item among all catalog items.

    Excluded items (the user's train history) are set to ``-inf``.  Ties are
    broken optimistically by counting only strictly greater scores.

    Warning: for sparse scorers where ``0`` means "no evidence", do not apply
    this to the full score array.  If every score is zero, the target gets
    rank 1 under optimistic ties.  Evaluate the served positive-score top-k
    list instead (see ``scripts/train_itemcf.py``; the legacy Phase 2 ItemCF
    numbers were inflated by exactly this case).
    """
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0 <= target_index < scores.shape[0]:
        raise IndexError("target_index out of range")
    working = scores
    excluded = list(exclude_indices)
    if excluded:
        working = scores.copy()
        working[np.asarray(excluded, dtype=np.int64)] = -np.inf
    target_score = float(working[target_index])
    if not np.isfinite(target_score):
        return 10**9
    return 1 + int(np.sum(working > target_score))


def metrics_from_rank(
    rank: int, k_values: Sequence[int] = DEFAULT_K_VALUES
) -> dict[str, float]:
    """Convert one target rank into HitRate/Recall/NDCG/MRR@K."""
    result: dict[str, float] = {"mrr": 1.0 / rank if rank > 0 else 0.0}
    for k in k_values:
        hit = 1.0 if rank <= k else 0.0
        result[f"hit_rate@{k}"] = hit
        result[f"recall@{k}"] = hit
        result[f"precision@{k}"] = hit / k
        result[f"ndcg@{k}"] = (
            1.0 / math.log2(rank + 1) if rank <= k else 0.0
        )
    return result


def aggregate_metrics(
    per_user_metrics: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Average per-user metric dictionaries."""
    if not per_user_metrics:
        return {}
    keys = per_user_metrics[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in per_user_metrics]))
        for key in keys
    }


def coverage_at_k(
    recommended_items: Sequence[Sequence[int]], catalog_size: int, k: int
) -> float:
    """Fraction of catalog items appearing in users' top-k lists."""
    if catalog_size <= 0:
        return 0.0
    seen: set[int] = set()
    for items in recommended_items:
        seen.update(int(item) for item in items[:k])
    seen.discard(0)
    return len(seen) / catalog_size


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores, descending."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    k = min(max(int(k), 1), scores.shape[0])
    if k == scores.shape[0]:
        order = np.argsort(-scores)
        return order[:k]
    candidates = np.argpartition(-scores, k - 1)[:k]
    return candidates[np.argsort(-scores[candidates])]
