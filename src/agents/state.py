"""Shared LangGraph state for the multi-agent recommendation workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    ``trace`` and ``errors`` use an additive reducer so retries do not erase
    the audit trail.  All other fields are replaced by the node that owns them.
    """

    user_id: str
    query: str
    thread_id: str
    conversation: list[dict[str, str]]

    intent: dict[str, Any]
    profile: dict[str, Any]
    recommendations: list[dict[str, Any]]
    recommendation_trace: dict[str, Any]
    review_evidence: list[dict[str, Any]]
    review_status: str
    critic: dict[str, Any]
    extra_exclude_item_ids: list[str]
    retry_count: int
    clarification: str | None
    status: str
    final: dict[str, Any]
    memory_update: dict[str, Any]

    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
