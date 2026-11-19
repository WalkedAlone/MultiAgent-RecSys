"""Shared utilities for Phase 7 ablation experiments.

The helpers here are deliberately deterministic.  They never invent a metric:
they either aggregate real per-user values or resample those values with a
fixed seed to report a paired bootstrap confidence interval.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def paired_bootstrap_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    seed: int = 42,
    iterations: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return ``treatment - baseline`` mean delta and a percentile 95% CI.

    Args:
        baseline: Per-user metric values for the baseline variant.
        treatment: Per-user metric values for the treatment variant, aligned by
            user with ``baseline``.
        seed: Fixed seed for reproducible resampling.
        iterations: Number of bootstrap resamples.
        alpha: Two-sided error level; 0.05 gives a 95% interval.

    Returns:
        A dictionary with ``delta``, ``ci95_low``, ``ci95_high``,
        ``probability_treatment_better`` and ``users``.  The probability is the
        fraction of paired resamples with a positive delta; it is not a p-value
        and is not a claim of production improvement.
    """
    base = np.asarray(baseline, dtype=np.float64)
    treat = np.asarray(treatment, dtype=np.float64)
    if base.ndim != 1 or treat.ndim != 1:
        raise ValueError("bootstrap inputs must be one-dimensional")
    if base.shape != treat.shape:
        raise ValueError("bootstrap inputs must be paired and equal length")
    if base.size == 0:
        raise ValueError("bootstrap inputs must not be empty")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    differences = treat - base
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    n = differences.shape[0]
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, n, size=n)
        deltas[index] = float(np.mean(differences[sample]))
    low = float(np.percentile(deltas, 100.0 * alpha / 2.0))
    high = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "delta": observed,
        "ci95_low": low,
        "ci95_high": high,
        "probability_treatment_better": float(np.mean(deltas > 0.0)),
        "users": float(n),
    }


def mean_metric(
    per_user: Sequence[Mapping[str, Any]], key: str
) -> float:
    """Mean a real metric key; missing values are an error, not zero-filled."""
    if not per_user:
        return 0.0
    values = [float(row[key]) for row in per_user]
    return float(np.mean(values))


def category_diversity(
    item_ids: Sequence[str],
    item_metadata: Mapping[str, Mapping[str, Any]],
) -> float:
    """Fraction of distinct non-empty categories in one recommendation list."""
    categories = {
        str(item_metadata.get(str(item_id), {}).get("main_category") or "")
        for item_id in item_ids
    }
    categories.discard("")
    return len(categories) / max(1, len(item_ids))


def label_rank(
    ids: Sequence[str], target_id: str, catalog_size: int
) -> int:
    """1-based rank if present, otherwise a deterministic absent rank."""
    target = str(target_id)
    for index, item_id in enumerate(ids, start=1):
        if str(item_id) == target:
            return index
    return int(catalog_size) + 1
