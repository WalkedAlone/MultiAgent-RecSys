"""Multi-channel candidate fusion.

Raw scores from ItemCF / Two-Tower / LightGCN are never added directly.
Supported strategies:

* ``quota``     — take a quota per channel and keep the best normalised score;
* ``rrf``       — Reciprocal Rank Fusion over all returned channel ranks;
* ``quota_rrf`` — quota per channel, then RRF over the retained candidates.

Every fused candidate keeps its per-channel scores, ranks and source list for
auditability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.retrieval.category_match import category_matches


@dataclass
class FusedCandidate:
    item_id: str
    fused_score: float = 0.0
    source_scores: dict[str, float] = field(default_factory=dict)
    raw_source_scores: dict[str, float] = field(default_factory=dict)
    source_ranks: dict[str, int] = field(default_factory=dict)


def _normalise_channel(
    results: Sequence[Mapping[str, Any]], method: str
) -> tuple[list[Mapping[str, Any]], list[float]]:
    ordered = sorted(
        results, key=lambda result: float(result.get("retrieval_score", 0.0)), reverse=True
    )
    # Deduplicate within a channel, keeping the highest score.
    seen: dict[str, Mapping[str, Any]] = {}
    for result in ordered:
        item_id = str(result["item_id"])
        if item_id not in seen:
            seen[item_id] = result
    ordered = list(seen.values())
    raw_scores = [float(result.get("retrieval_score", 0.0)) for result in ordered]
    if not raw_scores:
        return [], []
    if method == "none":
        return ordered, raw_scores
    if method == "zscore":
        mean = float(np.mean(raw_scores))
        std = float(np.std(raw_scores))
        if std < 1e-8:
            return ordered, [0.0 for _ in raw_scores]
        return ordered, [(value - mean) / std for value in raw_scores]
    # min_max
    minimum = float(np.min(raw_scores))
    maximum = float(np.max(raw_scores))
    if maximum - minimum < 1e-12:
        return ordered, [1.0 for _ in raw_scores]
    return ordered, [
        (value - minimum) / (maximum - minimum) for value in raw_scores
    ]


def fuse_candidates(
    channel_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    strategy: str = "quota_rrf",
    quota_per_channel: int = 100,
    total_candidates: int = 300,
    rrf_k: int = 60,
    score_normalization: str = "min_max",
    channel_weights: Mapping[str, float] | None = None,
) -> list[FusedCandidate]:
    """Fuse channel outputs without adding raw model scores.

    ``channel_weights`` calibrates channels with different ranking quality.
    Weights multiply normalised scores (quota) or reciprocal ranks (RRF);
    a zero/negative weight disables that channel entirely.
    """
    if strategy not in {"quota", "rrf", "quota_rrf"}:
        raise ValueError(f"Unknown fusion strategy: {strategy}")
    if quota_per_channel < 1 or total_candidates < 1 or rrf_k < 1:
        raise ValueError("quota/total/rrf_k must be positive")
    weights = dict(channel_weights or {})
    for source, weight in weights.items():
        if float(weight) < 0.0:
            raise ValueError(f"channel weight must be >= 0: {source}")

    fused: dict[str, FusedCandidate] = {}
    for source, results in channel_results.items():
        if not results:
            continue
        weight = float(weights.get(source, 1.0))
        if weight <= 0.0:
            continue
        ordered, normalised = _normalise_channel(results, score_normalization)
        if strategy in {"quota", "quota_rrf"}:
            ordered = ordered[:quota_per_channel]
            normalised = normalised[:quota_per_channel]
        for rank, (result, normalised_score) in enumerate(
            zip(ordered, normalised), start=1
        ):
            item_id = str(result["item_id"])
            raw_score = float(result.get("retrieval_score", 0.0))
            candidate = fused.get(item_id)
            if candidate is None:
                candidate = FusedCandidate(item_id=item_id)
                fused[item_id] = candidate
            candidate.raw_source_scores[source] = raw_score
            candidate.source_scores[source] = weight * float(normalised_score)
            candidate.source_ranks[source] = rank
            if strategy == "quota":
                candidate.fused_score = max(
                    candidate.fused_score, weight * float(normalised_score)
                )
            else:
                candidate.fused_score += weight / (rrf_k + rank)

    output = sorted(
        fused.values(), key=lambda candidate: candidate.fused_score, reverse=True
    )
    return output[:total_candidates]


def filter_fused_candidates(
    candidates: Sequence[FusedCandidate],
    *,
    exclude_item_ids: set[str] | None = None,
    item_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    category_filter: str | None = None,
    budget_min: float | None = None,
    budget_max: float | None = None,
    require_price_for_budget: bool = True,
) -> tuple[list[FusedCandidate], dict[str, int]]:
    """Apply deterministic hard filters before ranking/re-ranking."""
    exclude_item_ids = exclude_item_ids or set()
    item_metadata = item_metadata or {}
    category_text = (category_filter or "").strip()
    stats = {
        "input": len(candidates),
        "excluded_interacted": 0,
        "excluded_category": 0,
        "excluded_budget": 0,
        "excluded_budget_min": 0,
        "excluded_price_unknown": 0,
        "kept": 0,
    }
    output: list[FusedCandidate] = []
    for candidate in candidates:
        if candidate.item_id in exclude_item_ids:
            stats["excluded_interacted"] += 1
            continue
        metadata = item_metadata.get(candidate.item_id, {})
        if category_text and not category_matches(category_text, metadata):
            stats["excluded_category"] += 1
            continue
        if budget_min is not None or budget_max is not None:
            price = metadata.get("price")
            try:
                price_value = float(price) if price is not None else None
            except (TypeError, ValueError):
                price_value = None
            unknown_price = price_value is None or math.isnan(price_value)
            if unknown_price:
                if require_price_for_budget:
                    stats["excluded_price_unknown"] += 1
                    continue
            else:
                if budget_min is not None and price_value < float(budget_min):
                    stats["excluded_budget_min"] += 1
                    continue
                if budget_max is not None and price_value > float(budget_max):
                    stats["excluded_budget"] += 1
                    continue
        output.append(candidate)
    stats["kept"] = len(output)
    return output, stats
