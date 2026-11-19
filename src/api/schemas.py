"""Pydantic request/response schemas for the Phase 6 API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=120)


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_id: str
    title: str = ""
    price: float | None = None
    main_category: str = ""
    category_path: str = ""
    store: str = ""
    ranking_score: float | None = None
    mmr_score: float | None = None
    rank: int | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    recommendation_reason: str = ""
    reason_source: str = ""
    review_evidence: list[dict[str, Any]] = Field(default_factory=list)
    review_evidence_status: str = ""
    evidence_summary: str | None = None


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    profile: dict[str, Any] | None = None
    message: str | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    query: str = ""
    status: str = ""
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    intent: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(default_factory=dict)
    critic: dict[str, Any] = Field(default_factory=dict)
    review_evidence: list[dict[str, Any]] = Field(default_factory=list)
    review_status: str = ""
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    assistant_message: str = ""
    clarification: str | None = None
    memory_update: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    thread_id: str = ""
    checkpointing: dict[str, Any] = Field(default_factory=dict)


class UserRecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, ge=1, le=50)
    budget_min: float | None = Field(default=None, ge=0)
    budget_max: float | None = Field(default=None, ge=0)
    category: str | None = Field(default=None, max_length=120)
    use_mmr: bool = True
    use_memory: bool = True
    strategy: Literal["quota", "rrf", "quota_rrf"] | None = None

    @model_validator(mode="after")
    def _budget_order(self) -> "UserRecommendRequest":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min > self.budget_max
        ):
            raise ValueError("budget_min must not exceed budget_max")
        return self


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=120)
    item_id: str = Field(min_length=1, max_length=80)
    feedback_type: Literal["like", "dislike", "purchased", "not_interested"]
    query: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    feedback: dict[str, Any]
    preferences: list[dict[str, Any]] = Field(default_factory=list)


PreferenceKind = Literal[
    "like",
    "dislike",
    "purchased",
    "not_interested",
    "budget",
    "brand",
    "category",
    "other",
]


class PreferenceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PreferenceKind = "other"
    value: str = Field(min_length=1, max_length=200)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class PreferenceMutationResponse(BaseModel):
    preference: dict[str, Any] | None = None
    deleted: int = 0
    preferences: list[dict[str, Any]] = Field(default_factory=list)


class ProfileResponse(BaseModel):
    user_id: str
    profile: dict[str, Any]
    preferences: list[dict[str, Any]] = Field(default_factory=list)


class ProductResponse(BaseModel):
    item_id: str
    title: str = ""
    price: float | None = None
    main_category: str = ""
    category_path: str = ""
    store: str = ""
    metadata_avg_rating: float | None = None
    metadata_rating_number: int | None = None
    image_count: int | None = None
    description_chars: int | None = None


class HealthResponse(BaseModel):
    status: str
    api_version: str
    components: dict[str, Any]
    latency_stats: dict[str, Any]


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_count: int = 0
    error_count: int = 0
    latency_ms: dict[str, float] = Field(default_factory=dict)
    uptime_seconds: float = 0.0


class ModelMetricsResponse(BaseModel):
    metrics: dict[str, Any] = Field(default_factory=dict)
    note: str = (
        "Retrieval metrics are held-out positive metrics; DIN/DeepFM AUC is a "
        "sampled 1-positive/4-negative diagnostic, not CTR."
    )
