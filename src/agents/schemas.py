"""Pydantic schemas for the Phase 4 agent workflow.

The LLM is only allowed to return data matching these schemas.  Product IDs,
prices and hard constraints are checked again by Python in the Critic Agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryUpdate(BaseModel):
    """An explicit durable memory change requested by the user.

    Temporary request constraints (for example "this time under 200") must not
    be placed here.  ``explicit`` is a final safety guard against inferring a
    long-term preference from an ordinary shopping query.
    """

    model_config = ConfigDict(extra="forbid")

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


class IntentResult(BaseModel):
    """Structured conditions extracted from a natural-language request."""

    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    budget_min: float | None = Field(default=None, ge=0)
    budget_max: float | None = Field(default=None, ge=0)
    use_case: str | None = None
    preferences: list[str] = Field(default_factory=list, max_length=12)
    exclude_brands: list[str] = Field(default_factory=list, max_length=12)
    top_k: int = Field(default=10, ge=1, le=50)
    hard_constraints: list[str] = Field(default_factory=list, max_length=12)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=300)
    language: str = Field(default="zh", max_length=16)
    memory_updates: list[MemoryUpdate] = Field(default_factory=list, max_length=8)

    @field_validator("category", "use_case", "clarification_question", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("preferences", "exclude_brands", "hard_constraints")
    @classmethod
    def _clean_text_lists(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def _budget_range_is_valid(self) -> "IntentResult":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min > self.budget_max
        ):
            raise ValueError("budget_min must not exceed budget_max")
        return self


class ItemExplanation(BaseModel):
    """One fact-grounded recommendation reason."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=300)


class ExplanationBatch(BaseModel):
    """Reasons for the current candidate set."""

    model_config = ConfigDict(extra="forbid")

    explanations: list[ItemExplanation] = Field(default_factory=list, max_length=50)


class CriticIssue(BaseModel):
    """A single semantic issue found by the LLM critic."""

    model_config = ConfigDict(extra="forbid")

    item_id: str | None = Field(default=None, max_length=80)
    issue_type: Literal[
        "unsupported_claim",
        "irrelevant",
        "missing_evidence",
        "other",
    ] = "other"
    message: str = Field(min_length=1, max_length=300)


class CriticResult(BaseModel):
    """LLM semantic review result; deterministic checks run separately."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    issues: list[CriticIssue] = Field(default_factory=list, max_length=30)
    rejected_item_ids: list[str] = Field(default_factory=list, max_length=50)
    summary: str = Field(default="", max_length=500)
