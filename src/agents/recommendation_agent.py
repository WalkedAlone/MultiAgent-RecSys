"""Recommendation Agent: real tool call + fact-grounded explanations."""

from __future__ import annotations

import time
from typing import Any, Mapping

from src.agents.prompts import (
    build_explanation_system_prompt,
    build_explanation_user_message,
    build_recommendation_facts,
)
from src.agents.schemas import ExplanationBatch
from src.core.config import AppConfig
from src.core.exceptions import AgentWorkflowError, DeepSeekError
from src.core.logger import get_logger

logger = get_logger(__name__)


class RecommendationAgent:
    """Call ``recommend_products`` and attach reasons based on real facts."""

    def __init__(
        self,
        *,
        client: Any,
        config: AppConfig,
        toolkit: Any,
    ) -> None:
        self.client = client
        self.config = config
        self.toolkit = toolkit

    def run(
        self,
        *,
        user_id: str,
        query: str,
        intent: Mapping[str, Any],
        profile: Mapping[str, Any],
        extra_exclude_item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        effective = profile.get("effective_constraints", {}) or {}
        budget_max = intent.get("budget_max")
        category = intent.get("category")
        memory_applied: list[str] = []
        if budget_max is None and effective.get("budget_max") is not None:
            budget_max = effective["budget_max"]
            memory_applied.append("budget_max")
        if not category and effective.get("category"):
            category = effective["category"]
            memory_applied.append("category")
        feedback_excludes = [
            str(pref.get("value") or "").strip()
            for pref in profile.get("explicit_preferences", [])
            if pref.get("kind") in {"dislike", "not_interested", "purchased"}
            and str(pref.get("value") or "").strip()
        ]
        if feedback_excludes:
            memory_applied.append("feedback_exclusions")
        all_exclusions = list(
            dict.fromkeys(
                list(extra_exclude_item_ids or []) + feedback_excludes
            )
        )[:50]
        args = {
            "user_id": str(user_id),
            "top_k": int(intent.get("top_k") or self.config.ranking.rerank.top_k),
            "budget_min": intent.get("budget_min"),
            "budget_max": budget_max,
            "category": category,
            "exclude_brands": list(intent.get("exclude_brands", [])),
            "extra_exclude_item_ids": all_exclusions,
            "use_mmr": bool(self.config.ranking.rerank.enable_mmr),
            "strategy": None,
        }
        tool_started = time.perf_counter()
        result = self.toolkit.registry.execute("recommend_products", args)
        tool_latency_ms = int((time.perf_counter() - tool_started) * 1000)
        if not result.ok:
            raise AgentWorkflowError(
                f"recommend_products failed: {result.error or 'unknown tool error'}"
            )
        payload = dict(result.result or {})
        recommendations = [dict(item) for item in payload.get("recommendations", [])]
        preferred_brands = {
            str(pref.get("value") or "").strip().casefold()
            for pref in profile.get("explicit_preferences", [])
            if pref.get("kind") == "brand"
            and str(pref.get("source") or "") == "explicit_user"
            and str(pref.get("value") or "").strip()
        }
        if preferred_brands and recommendations:
            # Explicit memory can be a soft ordering preference. It does not
            # change the model scores or invent a product; it only reorders the
            # already validated candidate window.
            recommendations.sort(
                key=lambda item: (
                    any(
                        brand
                        in f"{item.get('store', '')} {item.get('title', '')}".casefold()
                        for brand in preferred_brands
                    ),
                    float(item.get("ranking_score") or 0.0),
                ),
                reverse=True,
            )
            memory_applied.append("brand_preference")
        trace: list[dict[str, Any]] = [
            {
                "node": "recommendation_tool",
                "status": "ok",
                "tool": "recommend_products",
                "latency_ms": tool_latency_ms,
                "summary": (
                    f"returned={len(recommendations)}; "
                    f"memory_applied={memory_applied}; "
                    f"constraint_filter={payload.get('constraint_filter', {})}"
                ),
            }
        ]

        explanation_status = "not_needed"
        llm_usage: dict[str, Any] | None = None
        errors: list[str] = []
        if recommendations:
            (
                reasons,
                explanation_status,
                llm_usage,
                explanation_error,
            ) = self._explain(
                user_id=str(user_id),
                query=query,
                intent=dict(intent),
                profile=dict(profile),
                recommendations=recommendations,
            )
            if explanation_error:
                errors.append(explanation_error)
            for item in recommendations:
                reason = reasons.get(str(item["item_id"]))
                if reason:
                    item["recommendation_reason"] = reason
                    item["reason_source"] = "llm"
                else:
                    item["recommendation_reason"] = self._fallback_reason(
                        item, intent
                    )
                    item["reason_source"] = "template"
                item["review_evidence"] = []
                item["review_evidence_status"] = "pending_review_rag"
        else:
            explanation_status = "no_candidates"

        trace.append(
            {
                "node": "recommendation_explanation",
                "status": (
                    "ok"
                    if explanation_status in {"ok", "not_needed", "no_candidates"}
                    else explanation_status
                ),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "summary": (
                    f"status={explanation_status}; "
                    f"recommendations={len(recommendations)}; "
                    f"reasons_from_llm="
                    f"{sum(1 for item in recommendations if item.get('reason_source') == 'llm')}"
                ),
                "llm_usage": llm_usage,
            }
        )
        return {
            "recommendations": recommendations,
            "recommendation_trace": {
                "engine": payload.get("trace", {}),
                "constraint_filter": payload.get("constraint_filter", {}),
                "explanation_status": explanation_status,
            },
            "trace": trace,
            "errors": errors,
        }

    def _explain(
        self,
        *,
        user_id: str,
        query: str,
        intent: dict[str, Any],
        profile: dict[str, Any],
        recommendations: list[dict[str, Any]],
    ) -> tuple[dict[str, str], str, dict[str, Any] | None, str | None]:
        facts = build_recommendation_facts(
            query=query,
            intent=intent,
            profile=profile,
            recommendations=recommendations,
        )
        messages = [
            {"role": "system", "content": build_explanation_system_prompt()},
            {"role": "user", "content": build_explanation_user_message(facts)},
        ]
        try:
            parsed, response = self.client.chat_json(
                messages,
                ExplanationBatch,
                max_attempts=self.config.agent.llm.json_max_attempts,
                temperature=min(self.config.agent.llm.temperature, 0.4),
                call_name="recommendation_explanations",
            )
        except DeepSeekError as exc:
            logger.warning(
                "recommendation_explanation_failed user=%s error=%s",
                user_id,
                type(exc).__name__,
            )
            return (
                {},
                "llm_failed",
                None,
                f"recommendation_explanations: {type(exc).__name__}: {exc}",
            )

        valid_ids = {str(item["item_id"]) for item in recommendations}
        reasons: dict[str, str] = {}
        for explanation in parsed.explanations:
            item_id = str(explanation.item_id)
            if item_id in valid_ids and item_id not in reasons:
                reasons[item_id] = explanation.reason.strip()
        return reasons, "ok", response.usage.model_dump(), None

    @staticmethod
    def _fallback_reason(
        item: Mapping[str, Any], intent: Mapping[str, Any]
    ) -> str:
        """Deterministic, fact-only fallback when the LLM explanation fails."""
        title = str(item.get("title") or item.get("item_id") or "")
        category = str(item.get("main_category") or "")
        price = item.get("price")
        sources = ", ".join(item.get("retrieval_sources", []))
        language = str(intent.get("language") or "zh").lower()
        if language.startswith("en"):
            parts = [title]
            if category:
                parts.append(f"category: {category}")
            if price is not None:
                parts.append(f"price: {price}")
            if sources:
                parts.append(f"retrieved by: {sources}")
            return "; ".join(part for part in parts if part) + "."
        parts = [title]
        if category:
            parts.append(f"类别：{category}")
        if price is not None:
            parts.append(f"价格：{price}")
        if sources:
            parts.append(f"召回来源：{sources}")
        return "；".join(part for part in parts if part) + "。"
