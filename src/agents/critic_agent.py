"""Critic Agent: deterministic hard-rule checks plus LLM semantic review."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

from src.agents.prompts import (
    build_critic_system_prompt,
    build_critic_user_message,
    build_recommendation_facts,
)
from src.agents.schemas import CriticResult
from src.core.config import AppConfig
from src.core.exceptions import DeepSeekError
from src.core.logger import get_logger
from src.retrieval.category_match import category_matches

logger = get_logger(__name__)


class CriticAgent:
    """Never lets an unvalidated item through the final response."""

    def __init__(
        self,
        *,
        client: Any,
        config: AppConfig,
        item_metadata: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.client = client
        self.config = config
        self.item_metadata = dict(item_metadata)

    def run(
        self,
        *,
        query: str,
        intent: Mapping[str, Any],
        profile: Mapping[str, Any],
        recommendations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        issues: list[dict[str, Any]] = []
        rejected: set[str] = set()
        seen: set[str] = set()
        effective = profile.get("effective_constraints", {}) or {}
        budget_min = intent.get("budget_min")
        budget_max = intent.get("budget_max")
        if budget_max is None:
            budget_max = effective.get("budget_max")
        category = str(
            intent.get("category") or effective.get("category") or ""
        ).strip().casefold()
        brands = [
            str(value).strip().casefold()
            for value in intent.get("exclude_brands", [])
            if str(value).strip()
        ]

        if not recommendations:
            issues.append(
                {
                    "item_id": None,
                    "issue_type": "no_recommendations",
                    "message": (
                        "No product satisfied the request and filters; no fake "
                        "products were generated."
                    ),
                }
            )

        for item in recommendations:
            item_id = str(item.get("item_id") or "")
            if not item_id or item_id not in self.item_metadata:
                issues.append(
                    {
                        "item_id": item_id or None,
                        "issue_type": "missing_product",
                        "message": "item_id is absent from the real metadata table",
                    }
                )
                if item_id:
                    rejected.add(item_id)
                continue
            if item_id in seen:
                issues.append(
                    {
                        "item_id": item_id,
                        "issue_type": "duplicate",
                        "message": "duplicate item_id in recommendation list",
                    }
                )
                rejected.add(item_id)
                continue
            seen.add(item_id)

            metadata = self.item_metadata[item_id]
            if not str(metadata.get("title") or "").strip():
                issues.append(
                    {
                        "item_id": item_id,
                        "issue_type": "missing_product_info",
                        "message": "real metadata has no product title",
                    }
                )
                rejected.add(item_id)
            price = self._price_value(item.get("price", metadata.get("price")))
            if budget_min is not None or budget_max is not None:
                if price is None:
                    issues.append(
                        {
                            "item_id": item_id,
                            "issue_type": "price_unknown",
                            "message": (
                                "budget constraint cannot be verified because the "
                                "real price is missing"
                            ),
                        }
                    )
                    rejected.add(item_id)
                else:
                    if budget_min is not None and price < float(budget_min):
                        issues.append(
                            {
                                "item_id": item_id,
                                "issue_type": "budget_min",
                                "message": f"price {price} is below budget_min",
                            }
                        )
                        rejected.add(item_id)
                    if budget_max is not None and price > float(budget_max):
                        issues.append(
                            {
                                "item_id": item_id,
                                "issue_type": "budget_max",
                                "message": f"price {price} exceeds budget_max",
                            }
                        )
                        rejected.add(item_id)

            if category and not category_matches(category, metadata):
                issues.append(
                    {
                        "item_id": item_id,
                        "issue_type": "category",
                        "message": "item does not match the requested category",
                    }
                )
                rejected.add(item_id)

            if brands:
                searchable = (
                    f"{metadata.get('store', '')} {metadata.get('title', '')}"
                ).casefold()
                if any(brand in searchable for brand in brands):
                    issues.append(
                        {
                            "item_id": item_id,
                            "issue_type": "excluded_brand",
                            "message": "item matches an excluded brand",
                        }
                    )
                    rejected.add(item_id)

            if not str(item.get("recommendation_reason") or "").strip():
                issues.append(
                    {
                        "item_id": item_id,
                        "issue_type": "missing_evidence",
                        "message": "recommendation reason is empty",
                    }
                )
                rejected.add(item_id)
            if not item.get("retrieval_sources"):
                issues.append(
                    {
                        "item_id": item_id,
                        "issue_type": "missing_evidence",
                        "message": "retrieval source provenance is missing",
                    }
                )
                rejected.add(item_id)
            for review in item.get("review_evidence", []) or []:
                if str(review.get("parent_asin")) != item_id:
                    issues.append(
                        {
                            "item_id": item_id,
                            "issue_type": "review_product_mismatch",
                            "message": (
                                "review evidence belongs to a different product"
                            ),
                        }
                    )
                    rejected.add(item_id)
                    break
                if not str(review.get("text") or "").strip():
                    issues.append(
                        {
                            "item_id": item_id,
                            "issue_type": "empty_review_evidence",
                            "message": "review evidence text is empty",
                        }
                    )
                    rejected.add(item_id)
                    break

        llm_status = "skipped"
        llm_usage: dict[str, Any] | None = None
        fatal = not recommendations
        llm_approved: bool | None = None
        summary = ""
        if recommendations and not issues:
            llm_status = "ok"
            facts = build_recommendation_facts(
                query=query,
                intent=dict(intent),
                profile=dict(profile),
                recommendations=list(recommendations),
            )
            messages = [
                {"role": "system", "content": build_critic_system_prompt()},
                {"role": "user", "content": build_critic_user_message(facts)},
            ]
            try:
                parsed, response = self.client.chat_json(
                    messages,
                    CriticResult,
                    max_attempts=self.config.agent.llm.json_max_attempts,
                    temperature=min(self.config.agent.llm.temperature, 0.3),
                    call_name="critic_agent",
                )
            except DeepSeekError as exc:
                llm_status = "failed"
                fatal = True
                llm_approved = False
                issues.append(
                    {
                        "item_id": None,
                        "issue_type": "critic_llm_unavailable",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.warning(
                    "critic_llm_failed error=%s", type(exc).__name__
                )
            else:
                llm_usage = response.usage.model_dump()
                llm_approved = bool(parsed.approved)
                summary = parsed.summary
                valid_ids = {str(item["item_id"]) for item in recommendations}
                for issue in parsed.issues:
                    issue_item_id = str(issue.item_id) if issue.item_id else None
                    issues.append(
                        {
                            "item_id": issue_item_id,
                            "issue_type": issue.issue_type,
                            "message": issue.message,
                        }
                    )
                for item_id in parsed.rejected_item_ids:
                    if str(item_id) in valid_ids:
                        rejected.add(str(item_id))

        approved = bool(recommendations) and not issues and not rejected and (
            llm_approved is not False
        )
        if not summary:
            summary = (
                "deterministic checks passed"
                if approved
                else f"{len(issues)} issue(s) require attention"
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        evidence_statuses = {
            str(item.get("review_evidence_status") or "")
            for item in recommendations
        }
        if not recommendations:
            review_evidence_status = "no_recommendations"
        elif evidence_statuses == {"ok"}:
            review_evidence_status = "ok"
        elif "ok" in evidence_statuses:
            review_evidence_status = "partial"
        elif "error" in evidence_statuses:
            review_evidence_status = "error"
        elif "index_empty" in evidence_statuses:
            review_evidence_status = "index_empty"
        elif "unavailable" in evidence_statuses:
            review_evidence_status = "unavailable"
        elif "no_evidence" in evidence_statuses:
            review_evidence_status = "no_evidence"
        elif "not_requested" in evidence_statuses:
            review_evidence_status = "not_requested"
        else:
            review_evidence_status = "not_retrieved"
        critic = {
            "approved": approved,
            "fatal": bool(fatal),
            "issues": issues,
            "rejected_item_ids": sorted(rejected),
            "summary": summary,
            "llm_status": llm_status,
            "llm_approved": llm_approved,
            "review_evidence_status": review_evidence_status,
        }
        logger.info(
            "critic_completed approved=%s issues=%d rejected=%d llm_status=%s",
            approved,
            len(issues),
            len(rejected),
            llm_status,
        )
        return {
            "critic": critic,
            "trace": [
                {
                    "node": "critic_agent",
                    "status": "ok" if approved else ("fatal" if fatal else "retryable"),
                    "latency_ms": latency_ms,
                    "summary": (
                        f"approved={approved}; issues={len(issues)}; "
                        f"rejected={sorted(rejected)}; llm_status={llm_status}"
                    ),
                    "llm_usage": llm_usage,
                }
            ],
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
