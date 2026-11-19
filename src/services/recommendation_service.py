"""Phase 3 recommendation engine.

Pipeline:
multi-recall -> quota/RRF fusion -> deterministic filters -> DIN ranking ->
MMR re-ranking -> auditable response.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from src.core.config import AppConfig, get_app_config
from src.core.logger import get_logger
from src.ranking.ranker import DINRanker
from src.ranking.reranker import mmr_rerank
from src.retrieval.base import BaseRetriever
from src.retrieval.category_match import category_matches
from src.retrieval.fusion import (
    FusedCandidate,
    filter_fused_candidates,
    fuse_candidates,
)
from src.retrieval.itemcf import ItemCFRetriever
from src.retrieval.lightgcn import LightGCNRetriever
from src.retrieval.popularity import PopularityRetriever
from src.retrieval.two_tower import TwoTowerRetriever

logger = get_logger(__name__)


class RecommendationEngine:
    """Load saved Phase 2 artifacts and serve explainable Top-K results."""

    def __init__(
        self,
        *,
        config: AppConfig,
        retrievers: Mapping[str, BaseRetriever],
        ranker: DINRanker,
        item_metadata: Mapping[str, Mapping[str, Any]],
        user_exclusions: Mapping[str, set[str]],
    ) -> None:
        self.config = config
        self.retrievers = dict(retrievers)
        self.ranker = ranker
        self.item_metadata = dict(item_metadata)
        self.user_exclusions = {
            str(key): {str(value) for value in values}
            for key, values in user_exclusions.items()
        }

    @classmethod
    def from_artifacts(
        cls, config: AppConfig | None = None, root: Path | None = None
    ) -> "RecommendationEngine":
        config = config or get_app_config()
        root = root or config.project_root
        train = pl.read_parquet(root / "data/splits/train.parquet")
        user_map = pl.read_parquet(
            root / "data/processed/users/user_id_map.parquet"
        )
        item_features = pl.read_parquet(
            root / "data/features/item_features.parquet"
        )
        itemcf = ItemCFRetriever.load(
            root / "checkpoints/itemcf_similarity.npz",
            root / "checkpoints/itemcf_meta.json",
        )
        two_tower = TwoTowerRetriever.from_artifacts(
            index_path=root / "indexes/two_tower.faiss",
            user_vectors_path=root / "indexes/two_tower_user_vectors.npy",
            user_map=user_map,
            train=train,
        )
        lightgcn = LightGCNRetriever.from_artifacts(
            index_path=root / "indexes/lightgcn.faiss",
            user_vectors_path=root / "indexes/lightgcn_user_vectors.npy",
            user_map=user_map,
            train=train,
        )
        popularity = PopularityRetriever.from_artifacts(
            train,
            pool_size=int(config.retrieval.popularity.pool_size),
        )
        ranker = DINRanker.from_artifacts(root, config, train)
        metadata: dict[str, dict[str, Any]] = {}
        for row in item_features.iter_rows(named=True):
            metadata[str(row["parent_asin"])] = {
                "title": row.get("title") or "",
                "price": row.get("price"),
                "main_category": row.get("main_category") or "",
                "category_path": row.get("category_path") or "",
                "store": row.get("store") or "",
                "metadata_avg_rating": row.get("metadata_avg_rating"),
                "metadata_rating_number": row.get("metadata_rating_number"),
                "image_count": row.get("image_count"),
                "description_chars": row.get("description_chars"),
            }
        exclusions_frame = (
            train.select(["user_id", "parent_asin"])
            .unique()
            .group_by("user_id")
            .agg(pl.col("parent_asin").alias("items"))
        )
        exclusions = {
            str(row["user_id"]): {str(value) for value in row["items"]}
            for row in exclusions_frame.iter_rows(named=True)
        }
        return cls(
            config=config,
            retrievers={
                "itemcf": itemcf,
                "two_tower": two_tower,
                "lightgcn": lightgcn,
                "popularity": popularity,
            },
            ranker=ranker,
            item_metadata=metadata,
            user_exclusions=exclusions,
        )

    def recommend(
        self,
        user_id: str,
        *,
        top_k: int | None = None,
        budget_min: float | None = None,
        budget_max: float | None = None,
        category: str | None = None,
        use_mmr: bool | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        top_k = (
            int(top_k)
            if top_k is not None
            else int(self.config.ranking.rerank.top_k)
        )
        if top_k < 1:
            raise ValueError("top_k must be positive")
        fusion_cfg = self.config.retrieval.fusion
        selected_strategy = strategy or fusion_cfg.strategy
        quota = fusion_cfg.quota_per_channel

        retrieval_trace: dict[str, Any] = {}
        channel_results: dict[str, list[dict[str, Any]]] = {}
        for name, retriever in self.retrievers.items():
            channel_started = time.perf_counter()
            try:
                results = retriever.retrieve(str(user_id), quota)
                channel_results[name] = results
                retrieval_trace[name] = {
                    "count": len(results),
                    "latency_ms": round(
                        (time.perf_counter() - channel_started) * 1000, 1
                    ),
                    "status": "ok",
                }
            except Exception as exc:  # defensive channel degradation
                logger.warning("retriever_failed source=%s error=%s", name, type(exc).__name__)
                channel_results[name] = []
                retrieval_trace[name] = {
                    "count": 0,
                    "latency_ms": round(
                        (time.perf_counter() - channel_started) * 1000, 1
                    ),
                    "status": f"error:{type(exc).__name__}",
                }

        fused = fuse_candidates(
            channel_results,
            strategy=selected_strategy,
            quota_per_channel=quota,
            total_candidates=fusion_cfg.total_candidates,
            rrf_k=fusion_cfg.rrf_k,
            score_normalization=fusion_cfg.score_normalization,
            channel_weights=fusion_cfg.channel_weights,
        )
        filtered, filter_stats = filter_fused_candidates(
            fused,
            exclude_item_ids=self.user_exclusions.get(str(user_id), set()),
            item_metadata=self.item_metadata,
            category_filter=category,
            budget_min=budget_min,
            budget_max=budget_max,
            require_price_for_budget=True,
        )

        # Collaborative recall only returns a few hundred candidates per user.
        # A hard category/budget filter can therefore leave fewer than the
        # requested top-k even when the real catalog contains valid products.
        # Fill the gap from real item metadata (never synthetic products) and
        # let the same DIN/MMR/Critic pipeline validate the fallback items.
        fallback_limit = min(
            1000, max(int(fusion_cfg.total_candidates), top_k * 20)
        )
        fallback_stats: dict[str, Any] = {
            "triggered": False,
            "reason": "not_needed",
            "source": "catalog_fallback",
            "scanned": 0,
            "matched": 0,
            "selected": 0,
            "limit": fallback_limit,
        }
        if category and len(filtered) < top_k:
            fallback_candidates, fallback_stats = (
                self._catalog_fallback_candidates(
                    category=category,
                    budget_min=budget_min,
                    budget_max=budget_max,
                    exclude_item_ids=(
                        set(self.user_exclusions.get(str(user_id), set()))
                        | {candidate.item_id for candidate in filtered}
                    ),
                    limit=fallback_limit,
                )
            )
            if fallback_candidates:
                filtered = [*filtered, *fallback_candidates]

        ranking_started = time.perf_counter()
        scores = self.ranker.score(
            str(user_id), [candidate.item_id for candidate in filtered]
        )
        score_by_item = {
            row["item_id"]: float(row["ranking_score"]) for row in scores
        }
        ranked: list[dict[str, Any]] = []
        for candidate in filtered:
            score = score_by_item.get(candidate.item_id)
            if score is None:
                continue
            metadata = self.item_metadata.get(candidate.item_id, {})
            ranked.append(
                {
                    "item_id": candidate.item_id,
                    "title": metadata.get("title", ""),
                    "price": metadata.get("price"),
                    "main_category": metadata.get("main_category", ""),
                    "category": metadata.get("main_category", ""),
                    "category_path": metadata.get("category_path", ""),
                    "store": metadata.get("store", ""),
                    "ranking_score": score,
                    "fusion_score": candidate.fused_score,
                    "retrieval_sources": sorted(candidate.source_scores),
                    "source_scores": candidate.source_scores,
                    "raw_source_scores": candidate.raw_source_scores,
                    "source_ranks": candidate.source_ranks,
                }
            )
        ranking_trace = {
            "model": "din",
            "input": len(filtered),
            "scored": len(ranked),
            "latency_ms": round(
                (time.perf_counter() - ranking_started) * 1000, 1
            ),
            "score_min": min(
                (row["ranking_score"] for row in ranked), default=None
            ),
            "score_max": max(
                (row["ranking_score"] for row in ranked), default=None
            ),
        }

        if use_mmr is None:
            use_mmr = self.config.ranking.rerank.enable_mmr
        selected = mmr_rerank(
            ranked,
            top_k=top_k,
            mmr_lambda=self.config.ranking.rerank.mmr_lambda,
            max_per_category=self.config.ranking.rerank.max_per_category,
            max_per_brand=self.config.ranking.rerank.max_per_brand,
            similarity=self.config.ranking.rerank.similarity,
            enable_mmr=bool(use_mmr),
        )
        for position, item in enumerate(selected, start=1):
            item["rank"] = position
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        response = {
            "user_id": str(user_id),
            "filters": {
                "budget_min": budget_min,
                "budget_max": budget_max,
                "category": category,
                "top_k": top_k,
                "use_mmr": bool(use_mmr),
            },
            "recommendations": selected,
            "trace": {
                "retrieval": retrieval_trace,
                "fusion": {
                    "strategy": selected_strategy,
                    "quota_per_channel": quota,
                    "total_candidates": fusion_cfg.total_candidates,
                    "input_channels": {
                        name: len(values)
                        for name, values in channel_results.items()
                    },
                    "channel_weights": {
                        name: float(fusion_cfg.channel_weights.get(name, 1.0))
                        for name in channel_results
                    },
                    "fused_candidates": len(fused),
                    "filter": filter_stats,
                    "fallback": fallback_stats,
                    "after_filter": len(filtered),
                },
                "ranking": ranking_trace,
                "rerank": {
                    "mmr": bool(use_mmr),
                    "lambda": self.config.ranking.rerank.mmr_lambda,
                    "max_per_category": self.config.ranking.rerank.max_per_category,
                    "similarity": self.config.ranking.rerank.similarity,
                    "selected": len(selected),
                },
                "latency_ms": latency_ms,
            },
        }
        return response

    def _catalog_fallback_candidates(
        self,
        *,
        category: str,
        budget_min: float | None,
        budget_max: float | None,
        exclude_item_ids: set[str],
        limit: int,
    ) -> tuple[list[FusedCandidate], dict[str, Any]]:
        """Scan real item metadata when collaborative recall is too narrow.

        This is intentionally a catalog scan, not a fabrication step: every
        returned item id and price comes from ``item_features.parquet``.  The
        candidates are then scored by the same DIN/MMR pipeline as the recall
        channels and re-checked by the Critic.
        """
        matches: list[tuple[float, str]] = []
        for item_id, metadata in self.item_metadata.items():
            item_id = str(item_id)
            if item_id in exclude_item_ids:
                continue
            if not category_matches(category, metadata):
                continue
            if budget_min is not None or budget_max is not None:
                price = self._price_value(metadata.get("price"))
                if price is None:
                    continue
                if budget_min is not None and price < float(budget_min):
                    continue
                if budget_max is not None and price > float(budget_max):
                    continue
            matches.append((self._popularity_score(metadata), item_id))

        # Deterministic popularity prior limits the DIN batch while keeping the
        # output auditable.  The model still decides the final order.
        matches.sort(key=lambda row: (-row[0], row[1]))
        selected = matches[: max(1, int(limit))]
        candidates = [
            FusedCandidate(
                item_id=item_id,
                fused_score=0.0,
                source_scores={"catalog_fallback": 1.0},
                raw_source_scores={"catalog_fallback": 0.0},
                source_ranks={"catalog_fallback": rank},
            )
            for rank, (_, item_id) in enumerate(selected, start=1)
        ]
        return candidates, {
            "triggered": True,
            "reason": "filtered_candidates_below_top_k",
            "source": "catalog_fallback",
            "category": category,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "scanned": len(self.item_metadata),
            "matched": len(matches),
            "selected": len(candidates),
            "limit": int(limit),
        }

    @staticmethod
    def _price_value(value: Any) -> float | None:
        try:
            price = float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        if price is None or math.isnan(price):
            return None
        return price

    @staticmethod
    def _popularity_score(metadata: Mapping[str, Any]) -> float:
        try:
            rating_number = float(metadata.get("metadata_rating_number") or 0.0)
        except (TypeError, ValueError):
            rating_number = 0.0
        try:
            avg_rating = float(metadata.get("metadata_avg_rating") or 0.0)
        except (TypeError, ValueError):
            avg_rating = 0.0
        if not math.isfinite(rating_number) or rating_number < 0:
            rating_number = 0.0
        if not math.isfinite(avg_rating) or avg_rating < 0:
            avg_rating = 0.0
        return math.log1p(rating_number) + 0.1 * avg_rating
