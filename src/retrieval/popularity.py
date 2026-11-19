"""Non-personalised popularity recall channel.

The popularity baseline is intentionally simple: rank items by their number of
train interactions and exclude the target user's own train interactions.  It
is a standard sanity-check channel for sparse recommendation data and gives
cold-start users a real (non-fabricated) candidate list.

The channel is kept separate from ItemCF / Two-Tower / LightGCN so its effect
can be measured in ablation and reported honestly.
"""

from __future__ import annotations

from typing import Any, Iterable

import polars as pl

from src.retrieval.base import BaseRetriever


class PopularityRetriever(BaseRetriever):
    source = "popularity"

    def __init__(
        self,
        *,
        ranked_item_ids: list[str],
        interaction_counts: dict[str, float],
        user_exclusions: dict[str, set[str]],
        pool_size: int = 2000,
    ) -> None:
        self.ranked_item_ids = [str(item_id) for item_id in ranked_item_ids]
        self.interaction_counts = {
            str(item_id): float(count)
            for item_id, count in interaction_counts.items()
        }
        self.user_exclusions = {
            str(user_id): {str(item_id) for item_id in items}
            for user_id, items in user_exclusions.items()
        }
        self.pool_size = max(1, int(pool_size))

    @classmethod
    def from_artifacts(
        cls,
        train: pl.DataFrame,
        *,
        pool_size: int = 2000,
    ) -> "PopularityRetriever":
        counts = (
            train.group_by("parent_asin")
            .agg(pl.len().alias("interactions"))
            .sort(
                ["interactions", "parent_asin"],
                descending=[True, False],
            )
        )
        ranked_item_ids = [str(value) for value in counts["parent_asin"].to_list()]
        interaction_counts = {
            str(row["parent_asin"]): float(row["interactions"])
            for row in counts.iter_rows(named=True)
        }
        grouped = (
            train.group_by("user_id")
            .agg(pl.col("parent_asin").unique().alias("items"))
        )
        user_exclusions = {
            str(row["user_id"]): {str(item) for item in row["items"]}
            for row in grouped.iter_rows(named=True)
        }
        return cls(
            ranked_item_ids=ranked_item_ids,
            interaction_counts=interaction_counts,
            user_exclusions=user_exclusions,
            pool_size=pool_size,
        )

    def _iter_candidates(self) -> Iterable[str]:
        # The bounded pool is enough for normal users; the remainder is a
        # safety net for a user whose entire train history sits in the pool.
        yield from self.ranked_item_ids[: self.pool_size]
        yield from self.ranked_item_ids[self.pool_size :]

    def retrieve(self, user_id: str, top_k: int) -> list[dict[str, Any]]:
        top_k = self.validate_top_k(top_k)
        excluded = self.user_exclusions.get(str(user_id), set())
        results: list[dict[str, Any]] = []
        for item_id in self._iter_candidates():
            if item_id in excluded:
                continue
            results.append(
                self._result(
                    item_id, self.interaction_counts.get(item_id, 0.0)
                )
            )
            if len(results) >= top_k:
                break
        return results
