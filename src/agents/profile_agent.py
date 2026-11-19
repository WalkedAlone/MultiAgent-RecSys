"""Profile Agent and real user-feature store.

Phase 5 adds a SQLAlchemy/Chroma MemoryManager on top of the read-only train
profile.  Temporary query constraints are never written as durable memory.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from src.core.logger import get_logger

logger = get_logger(__name__)


def _iso_from_ms(value: Any) -> str | None:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(
            int(value) / 1000.0, tz=timezone.utc
        ).isoformat()
    except (TypeError, ValueError, OSError):
        return None


class UserProfileStore:
    """Read-only profile facts computed from the real sample data artifacts."""

    def __init__(
        self,
        *,
        user_rows: Mapping[str, Mapping[str, Any]],
        recent_by_user: Mapping[str, list[dict[str, Any]]],
        brand_counts: Mapping[str, Counter[str]],
        source: str,
    ) -> None:
        self._user_rows = dict(user_rows)
        self._recent_by_user = dict(recent_by_user)
        self._brand_counts = dict(brand_counts)
        self.source = source

    @classmethod
    def from_artifacts(
        cls,
        root: str | Path,
        *,
        item_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "UserProfileStore":
        root = Path(root)
        user_features = pl.read_parquet(
            root / "data/features/user_features.parquet"
        )
        train = pl.read_parquet(root / "data/splits/train.parquet").filter(
            pl.col("split") == "train"
        )
        if item_metadata is None:
            item_features = pl.read_parquet(
                root / "data/features/item_features.parquet",
                columns=[
                    "parent_asin",
                    "title",
                    "main_category",
                    "store",
                    "price",
                    "category_path",
                ],
            )
            item_metadata = {
                str(row["parent_asin"]): {
                    "title": row.get("title") or "",
                    "main_category": row.get("main_category") or "",
                    "store": row.get("store") or "",
                    "price": row.get("price"),
                    "category_path": row.get("category_path") or "",
                }
                for row in item_features.iter_rows(named=True)
            }

        user_rows = {
            str(row["user_id"]): dict(row)
            for row in user_features.iter_rows(named=True)
        }

        # Keep only the five latest train items per user; never look at
        # valid/test or at interactions after the current training boundary.
        recent_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in train.sort(["user_id", "timestamp_ms"]).iter_rows(named=True):
            user_id = str(row["user_id"])
            item_id = str(row["parent_asin"])
            meta = item_metadata.get(item_id, {})
            recent_by_user[user_id].append(
                {
                    "item_id": item_id,
                    "title": meta.get("title", ""),
                    "main_category": meta.get("main_category", ""),
                    "store": meta.get("store", ""),
                    "price": meta.get("price"),
                    "rating": row.get("rating"),
                    "timestamp": _iso_from_ms(row.get("timestamp_ms")),
                }
            )
            if len(recent_by_user[user_id]) > 5:
                recent_by_user[user_id].pop(0)

        brand_counts: dict[str, Counter[str]] = defaultdict(Counter)
        positive = train.filter(pl.col("is_positive"))
        for row in positive.iter_rows(named=True):
            store = str(
                item_metadata.get(str(row["parent_asin"]), {}).get("store") or ""
            ).strip()
            if store:
                brand_counts[str(row["user_id"])][store] += 1

        source = (
            "data/features/user_features.parquet; "
            "data/splits/train.parquet; data/features/item_features.parquet"
        )
        logger.info(
            "profile_store_loaded users=%d recent_users=%d source=%s",
            len(user_rows),
            len(recent_by_user),
            source,
        )
        return cls(
            user_rows=user_rows,
            recent_by_user=recent_by_user,
            brand_counts=brand_counts,
            source=source,
        )

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """Return real profile facts, or an explicit unknown-user record."""
        user_id = str(user_id)
        row = self._user_rows.get(user_id)
        if row is None:
            return {
                "user_id": user_id,
                "known_user": False,
                "source": self.source,
                "long_term": {},
                "recent_items": [],
                "message": (
                    "user_id is not present in the 20k sample user table; "
                    "no profile was fabricated"
                ),
            }

        recent = list(self._recent_by_user.get(user_id, []))
        recent.reverse()
        known_prices = [
            float(item["price"])
            for item in recent
            if item.get("price") is not None
        ]
        brand_preferences = [
            {"store": store, "positive_interactions": int(count)}
            for store, count in self._brand_counts.get(user_id, Counter()).most_common(3)
        ]
        return {
            "user_id": user_id,
            "known_user": True,
            "source": self.source,
            "long_term": {
                "favorite_category": row.get("favorite_category") or "",
                "interaction_count": int(row.get("train_interactions") or 0),
                "positive_interactions": int(row.get("train_positives") or 0),
                "negative_interactions": int(row.get("train_negatives") or 0),
                "neutral_interactions": int(row.get("train_neutrals") or 0),
                "train_avg_rating": row.get("train_avg_rating"),
                "unique_items": int(row.get("train_unique_items") or 0),
                "category_diversity": int(row.get("category_diversity") or 0),
                "avg_item_price": row.get("avg_item_price"),
                "price_range_recent": (
                    [min(known_prices), max(known_prices)]
                    if known_prices
                    else None
                ),
                "brand_preferences": brand_preferences,
                "first_interaction": _iso_from_ms(row.get("first_ts_ms")),
                "last_interaction": _iso_from_ms(row.get("last_ts_ms")),
            },
            "recent_items": recent,
            "memory_status": (
                "read_only_phase4; persistent SQLite/Chroma memory is Phase 5"
            ),
        }


class ProfileAgent:
    """Merge real train facts, durable memory and the current intent."""

    def __init__(
        self,
        store: UserProfileStore,
        memory_manager: Any | None = None,
        *,
        use_persistent_memory: bool = True,
    ) -> None:
        self.store = store
        self.memory_manager = memory_manager
        self.use_persistent_memory = bool(use_persistent_memory)

    def run(
        self,
        user_id: str,
        intent: Mapping[str, Any],
        conversation: list[dict[str, str]] | None = None,
        *,
        query: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        profile = self.store.get_profile(str(user_id))
        if self.memory_manager is not None:
            memory_profile = self.memory_manager.get_profile(
                str(user_id), query=query
            )
            profile["long_term"] = {
                **profile.get("long_term", {}),
                **memory_profile.get("long_term", {}),
            }
            profile["explicit_preferences"] = memory_profile.get(
                "explicit_preferences", []
            )
            profile["inferred_preferences"] = memory_profile.get(
                "inferred_preferences", []
            )
            profile["effective_constraints"] = memory_profile.get(
                "effective_constraints", {}
            )
            profile["semantic_memory"] = memory_profile.get("semantic_memory", [])
            profile["semantic_status"] = memory_profile.get("semantic_status")
            profile["memory_status"] = memory_profile.get(
                "memory_status", "sqlite_structured"
            )
            profile["recent_feedback"] = memory_profile.get("recent_feedback", [])
        elif not self.use_persistent_memory:
            # Phase 7 no-memory ablation: keep real train facts from
            # ``profile_store``, but remove every durable-memory field so the
            # recommendation agent cannot apply seeded/explicit preferences.
            long_term = dict(profile.get("long_term", {}))
            long_term["brand_preferences"] = []
            long_term["explicit_preference_count"] = 0
            long_term["inferred_preference_count"] = 0
            long_term["feedback_count"] = 0
            profile["long_term"] = long_term
            profile["explicit_preferences"] = []
            profile["inferred_preferences"] = []
            profile["effective_constraints"] = {}
            profile["recent_feedback"] = []
            profile["semantic_memory"] = []
            profile["semantic_status"] = "disabled_for_ablation"
            profile["memory_status"] = "disabled_for_ablation"
        profile["short_term"] = {
            "category": intent.get("category"),
            "budget_min": intent.get("budget_min"),
            "budget_max": intent.get("budget_max"),
            "use_case": intent.get("use_case"),
            "preferences": list(intent.get("preferences", [])),
            "exclude_brands": list(intent.get("exclude_brands", [])),
            "top_k": intent.get("top_k", 10),
            "conversation_turns": len(conversation or []),
        }
        if profile["known_user"]:
            profile["suggested_category"] = profile["long_term"].get(
                "favorite_category"
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        summary = (
            f"known_user={profile['known_user']}; "
            f"favorite_category={profile.get('long_term', {}).get('favorite_category', '')!r}; "
            f"recent_items={len(profile.get('recent_items', []))}; "
            f"explicit_preferences={len(profile.get('explicit_preferences', []))}; "
            f"semantic_memory={len(profile.get('semantic_memory', []))}; "
            f"memory_status={profile.get('memory_status', 'read_only')}"
        )
        return {
            "profile": profile,
            "trace": [
                {
                    "node": "profile_agent",
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "summary": summary,
                }
            ],
        }
