"""Real application tools exposed to the recommendation workflow.

Every handler calls a concrete Python service.  No model-generated code is
executed.  Review search uses only product-scoped Chroma evidence; memory writes
require an explicit durable preference and are never inferred from a query.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.tools import ToolExecutionError, ToolRegistry
from src.core.exceptions import ModelNotReadyError
from src.retrieval.category_match import category_matches
from src.retrieval.fusion import fuse_candidates
from src.services.recommendation_service import RecommendationEngine


class RecommendProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=120)
    top_k: int = Field(default=10, ge=1, le=50)
    budget_min: float | None = Field(default=None, ge=0)
    budget_max: float | None = Field(default=None, ge=0)
    category: str | None = Field(default=None, max_length=120)
    exclude_brands: list[str] = Field(default_factory=list, max_length=12)
    extra_exclude_item_ids: list[str] = Field(default_factory=list, max_length=50)
    use_mmr: bool = True
    strategy: Literal["quota", "rrf", "quota_rrf"] | None = None

    @model_validator(mode="after")
    def _budget_order(self) -> "RecommendProductsArgs":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min > self.budget_max
        ):
            raise ValueError("budget_min must not exceed budget_max")
        return self


class UserIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=120)


class ItemIdsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_ids: list[str] = Field(min_length=1, max_length=10)


class SearchProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=120)
    limit: int = Field(default=10, ge=1, le=50)


class SearchReviewsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1, max_length=80)
    query: str = Field(default="", max_length=300)
    top_k: int = Field(default=5, ge=1, le=20)


class UpdateMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=120)
    action: Literal["add", "delete"] = "add"
    kind: Literal[
        "like",
        "dislike",
        "purchased",
        "not_interested",
        "budget",
        "brand",
        "category",
        "other",
    ] = "other"
    value: str = Field(min_length=1, max_length=200)
    explicit: bool = True


class RankCandidatesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=120)
    item_ids: list[str] = Field(min_length=1, max_length=300)


class FilterProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_ids: list[str] = Field(min_length=1, max_length=300)
    budget_min: float | None = Field(default=None, ge=0)
    budget_max: float | None = Field(default=None, ge=0)
    category: str | None = Field(default=None, max_length=120)
    exclude_brands: list[str] = Field(default_factory=list, max_length=12)


class MergeCandidatesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel_results: dict[str, list[dict[str, Any]]]
    strategy: Literal["quota", "rrf", "quota_rrf"] = "quota"
    quota_per_channel: int = Field(default=100, ge=1)
    total_candidates: int = Field(default=300, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    channel_weights: dict[str, float] | None = None


class ProductToolkit:
    """Factory for the controlled tool registry used by the workflow."""

    def __init__(
        self,
        *,
        engine: RecommendationEngine,
        profile_store: Any,
        item_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        memory_manager: Any | None = None,
        review_retriever: Any | None = None,
    ) -> None:
        self.engine = engine
        self.profile_store = profile_store
        self.memory_manager = memory_manager
        self.review_retriever = review_retriever
        self.item_metadata = dict(item_metadata or engine.item_metadata)
        self.registry = ToolRegistry()
        self._register()

    def _register(self) -> None:
        self.registry.register(
            "recommend_products",
            "Run the real multi-recall, fusion, filtering, DIN ranking and MMR "
            "pipeline for a user; returns recommendations and an audit trace.",
            RecommendProductsArgs,
            self._recommend_products,
        )
        self.registry.register(
            "get_user_profile",
            "Read the real read-only user profile derived from the train split.",
            UserIdArgs,
            self._get_user_profile,
        )
        self.registry.register(
            "retrieve_itemcf",
            "Run the ItemCF retriever for a user and return the raw candidates.",
            RecommendProductsArgs,
            self._retrieve_itemcf,
        )
        self.registry.register(
            "retrieve_two_tower",
            "Run the Two-Tower Faiss retriever for a user.",
            RecommendProductsArgs,
            self._retrieve_two_tower,
        )
        self.registry.register(
            "retrieve_lightgcn",
            "Run the LightGCN Faiss retriever for a user.",
            RecommendProductsArgs,
            self._retrieve_lightgcn,
        )
        self.registry.register(
            "rank_candidates",
            "Score explicit item IDs with the trained DIN ranker.",
            RankCandidatesArgs,
            self._rank_candidates,
        )
        self.registry.register(
            "get_products",
            "Read real metadata for explicit parent_asin product IDs.",
            ItemIdsArgs,
            self._get_products,
        )
        self.registry.register(
            "search_product_metadata",
            "Search real product titles/categories/stores by keyword.",
            SearchProductsArgs,
            self._search_products,
        )
        self.registry.register(
            "compare_products",
            "Compare real metadata fields for 2-5 product IDs without inventing facts.",
            ItemIdsArgs,
            self._compare_products,
        )
        self.registry.register(
            "merge_candidates",
            "Fuse already retrieved channel candidates with quota/RRF; no raw model "
            "scores are added directly.",
            MergeCandidatesArgs,
            self._merge_candidates,
        )
        self.registry.register(
            "filter_products",
            "Apply deterministic budget/category/brand filters to explicit item IDs.",
            FilterProductsArgs,
            self._filter_products,
        )
        self.registry.register(
            "search_reviews",
            "Search product reviews. Phase 4 exposes the interface but review RAG "
            "is only implemented in Phase 5.",
            SearchReviewsArgs,
            self._search_reviews,
        )
        self.registry.register(
            "update_user_memory",
            "Persist or delete an explicit durable user preference in SQLite and "
            "semantic memory. Temporary query constraints are refused.",
            UpdateMemoryArgs,
            self._update_user_memory,
        )

    def _recommend_products(self, args: RecommendProductsArgs) -> dict[str, Any]:
        # Ask for a small buffer so critic/brand/min-budget exclusions can be
        # filled without inventing candidates.
        request_k = min(
            50,
            max(args.top_k + 10 + len(args.extra_exclude_item_ids), args.top_k),
        )
        response = self.engine.recommend(
            args.user_id,
            top_k=request_k,
            budget_min=args.budget_min,
            budget_max=args.budget_max,
            category=args.category,
            use_mmr=args.use_mmr,
            strategy=args.strategy,
        )
        excluded = set(args.extra_exclude_item_ids)
        brands = [value.casefold() for value in args.exclude_brands if value.strip()]
        stats = {
            "engine_candidates": len(response.get("recommendations", [])),
            "extra_excluded": 0,
            "brand_excluded": 0,
            "price_unknown": 0,
            "below_budget_min": 0,
            "kept": 0,
        }
        selected: list[dict[str, Any]] = []
        for item in response.get("recommendations", []):
            item_id = str(item["item_id"])
            if item_id in excluded:
                stats["extra_excluded"] += 1
                continue
            searchable = f"{item.get('store', '')} {item.get('title', '')}".casefold()
            if any(brand in searchable for brand in brands):
                stats["brand_excluded"] += 1
                continue
            if args.budget_min is not None:
                price = item.get("price")
                try:
                    price_value = float(price) if price is not None else None
                except (TypeError, ValueError):
                    price_value = None
                if price_value is None or math.isnan(price_value):
                    stats["price_unknown"] += 1
                    continue
                if price_value < args.budget_min:
                    stats["below_budget_min"] += 1
                    continue
            selected.append(item)
            if len(selected) >= args.top_k:
                break
        stats["kept"] = len(selected)
        result = dict(response)
        result["recommendations"] = selected
        result["constraint_filter"] = stats
        result["requested_top_k"] = args.top_k
        result["filters"] = {
            **response.get("filters", {}),
            "budget_min": args.budget_min,
            "exclude_brands": list(args.exclude_brands),
            "extra_exclude_item_ids": sorted(excluded),
        }
        return result

    def _get_user_profile(self, args: UserIdArgs) -> dict[str, Any]:
        return self.profile_store.get_profile(args.user_id)

    def _retrieve(self, source: str, args: RecommendProductsArgs) -> list[dict]:
        retriever = self.engine.retrievers.get(source)
        if retriever is None:
            raise ToolExecutionError(f"Retriever is not loaded: {source}")
        return retriever.retrieve(args.user_id, args.top_k)

    def _retrieve_itemcf(self, args: RecommendProductsArgs) -> list[dict]:
        return self._retrieve("itemcf", args)

    def _retrieve_two_tower(self, args: RecommendProductsArgs) -> list[dict]:
        return self._retrieve("two_tower", args)

    def _retrieve_lightgcn(self, args: RecommendProductsArgs) -> list[dict]:
        return self._retrieve("lightgcn", args)

    def _rank_candidates(self, args: RankCandidatesArgs) -> list[dict]:
        return self.engine.ranker.score(args.user_id, args.item_ids)

    def _get_products(self, args: ItemIdsArgs) -> dict[str, Any]:
        found = []
        missing = []
        for item_id in args.item_ids:
            metadata = self.item_metadata.get(str(item_id))
            if metadata is None:
                missing.append(str(item_id))
            else:
                found.append({"item_id": str(item_id), **dict(metadata)})
        return {"products": found, "missing_item_ids": missing}

    def _search_products(self, args: SearchProductsArgs) -> list[dict]:
        needle = args.query.casefold().strip()
        results: list[dict] = []
        for item_id, metadata in self.item_metadata.items():
            haystack = " ".join(
                [
                    str(metadata.get("title", "")),
                    str(metadata.get("main_category", "")),
                    str(metadata.get("category_path", "")),
                    str(metadata.get("store", "")),
                ]
            ).casefold()
            if needle in haystack:
                results.append({"item_id": item_id, **dict(metadata)})
                if len(results) >= args.limit:
                    break
        return results

    def _compare_products(self, args: ItemIdsArgs) -> dict[str, Any]:
        if len(args.item_ids) < 2:
            raise ToolExecutionError("compare_products requires at least 2 item IDs")
        return self._get_products(args)

    def _merge_candidates(self, args: MergeCandidatesArgs) -> list[dict[str, Any]]:
        fused = fuse_candidates(
            args.channel_results,
            strategy=args.strategy,
            quota_per_channel=args.quota_per_channel,
            total_candidates=args.total_candidates,
            rrf_k=args.rrf_k,
            channel_weights=args.channel_weights,
        )
        return [asdict(candidate) for candidate in fused]

    def _filter_products(self, args: FilterProductsArgs) -> dict[str, Any]:
        brands = [
            value.strip().casefold()
            for value in args.exclude_brands
            if value.strip()
        ]
        kept: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for item_id in args.item_ids:
            metadata = self.item_metadata.get(str(item_id))
            if metadata is None:
                excluded.append({"item_id": item_id, "reason": "missing_product"})
                continue
            if args.category and not category_matches(args.category, metadata):
                excluded.append({"item_id": item_id, "reason": "category"})
                continue
            searchable = (
                f"{metadata.get('store', '')} {metadata.get('title', '')}"
            ).casefold()
            if any(brand in searchable for brand in brands):
                excluded.append({"item_id": item_id, "reason": "excluded_brand"})
                continue
            price = metadata.get("price")
            try:
                price_value = float(price) if price is not None else None
            except (TypeError, ValueError):
                price_value = None
            if price_value is None or math.isnan(price_value):
                if args.budget_min is not None or args.budget_max is not None:
                    excluded.append({"item_id": item_id, "reason": "price_unknown"})
                    continue
            elif (
                args.budget_min is not None and price_value < args.budget_min
            ) or (
                args.budget_max is not None and price_value > args.budget_max
            ):
                excluded.append({"item_id": item_id, "reason": "budget"})
                continue
            kept.append({"item_id": str(item_id), **dict(metadata)})
        return {"kept": kept, "excluded": excluded}

    def _search_reviews(self, args: SearchReviewsArgs) -> dict[str, Any]:
        if self.review_retriever is None:
            raise ToolExecutionError(
                "Review RAG index is not configured. Build it with "
                "scripts/build_review_index.py; no review evidence was fabricated."
            )
        try:
            evidence = self.review_retriever.search(
                item_id=args.item_id,
                query=args.query or args.item_id,
                top_k=args.top_k,
            )
        except ModelNotReadyError as exc:
            raise ToolExecutionError(str(exc)) from exc
        return {
            "item_id": args.item_id,
            "query": args.query,
            "status": "ok" if evidence else "no_evidence",
            "evidence": evidence,
        }

    def _update_user_memory(self, args: UpdateMemoryArgs) -> dict[str, Any]:
        if self.memory_manager is None:
            raise ToolExecutionError(
                "Persistent memory is not configured; no memory was written."
            )
        if not args.explicit:
            raise ToolExecutionError(
                "Refusing to persist an implicit or temporary preference."
            )
        if args.action == "delete":
            deleted = self.memory_manager.delete_preference(
                user_id=args.user_id, kind=args.kind, value=args.value
            )
            return {"action": "delete", "deleted": deleted}
        preference = self.memory_manager.add_preference(
            user_id=args.user_id,
            kind=args.kind,
            value=args.value,
            source="explicit_user",
            confidence=0.9,
        )
        return {"action": "add", "preference": preference}
