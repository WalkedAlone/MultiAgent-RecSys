"""MMR re-ranking with category/brand diversity caps."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _category_similarity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return 1.0 if first == second else 0.0


def mmr_rerank(
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
    mmr_lambda: float = 0.7,
    max_per_category: int = 2,
    max_per_brand: int | None = None,
    similarity: str = "category",
    enable_mmr: bool = True,
) -> list[dict[str, Any]]:
    """Select a diverse top-k list.

    Each candidate must provide ``item_id``, ``ranking_score``, ``category``
    and ``store``.  ``ranking_score`` is a model preference score, not a CTR.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be in [0, 1]")
    if max_per_category < 1:
        raise ValueError("max_per_category must be positive")
    if similarity != "category":
        raise NotImplementedError(
            "Only category MMR similarity is available until Phase 5 item "
            "embeddings are built."
        )

    by_item: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        item_id = str(candidate["item_id"])
        existing = by_item.get(item_id)
        if existing is None or float(candidate["ranking_score"]) > float(
            existing["ranking_score"]
        ):
            by_item[item_id] = candidate
    remaining = list(by_item.values())

    selected: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    store_counts: dict[str, int] = {}
    selected_categories: list[str] = []

    if not enable_mmr:
        ordered = sorted(
            remaining, key=lambda item: float(item["ranking_score"]), reverse=True
        )
        for item in ordered:
            category = str(item.get("category", ""))
            store = str(item.get("store", ""))
            if category_counts.get(category, 0) >= max_per_category:
                continue
            if max_per_brand is not None and store_counts.get(store, 0) >= max_per_brand:
                continue
            category_counts[category] = category_counts.get(category, 0) + 1
            store_counts[store] = store_counts.get(store, 0) + 1
            selected.append(
                {
                    **dict(item),
                    "mmr_score": float(item["ranking_score"]),
                    "rank": len(selected) + 1,
                }
            )
            if len(selected) >= top_k:
                break
        return selected

    while remaining and len(selected) < top_k:
        best_index: int | None = None
        best_score = float("-inf")
        for index, item in enumerate(remaining):
            category = str(item.get("category", ""))
            store = str(item.get("store", ""))
            if category_counts.get(category, 0) >= max_per_category:
                continue
            if max_per_brand is not None and store_counts.get(store, 0) >= max_per_brand:
                continue
            similarity_max = 0.0
            if selected_categories:
                similarity_max = max(
                    _category_similarity(category, chosen)
                    for chosen in selected_categories
                )
            score = (
                mmr_lambda * float(item["ranking_score"])
                - (1.0 - mmr_lambda) * similarity_max
            )
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        item = remaining.pop(best_index)
        category = str(item.get("category", ""))
        store = str(item.get("store", ""))
        category_counts[category] = category_counts.get(category, 0) + 1
        store_counts[store] = store_counts.get(store, 0) + 1
        selected_categories.append(category)
        selected.append(
            {
                **dict(item),
                "mmr_score": best_score,
                "rank": len(selected) + 1,
            }
        )
    return selected
