"""Deterministic metrics for real Agent workflow runs.

These metrics do not judge prose quality.  They measure only verifiable facts:
returned item count, hard-constraint violations against real metadata, review
evidence coverage, Critic state, token usage and latency.  A run with no real
recommendation is reported as zero coverage, never filled with fake data.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def _price(item: Mapping[str, Any], metadata: Mapping[str, Any]) -> float | None:
    value = item.get("price", metadata.get("price"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def constraint_violations(
    recommendations: Sequence[Mapping[str, Any]],
    *,
    item_metadata: Mapping[str, Mapping[str, Any]],
    budget_max: float | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Count real hard-constraint violations; missing price is a violation."""
    budget_violations = 0
    category_violations = 0
    violating_items: list[str] = []
    category_text = (category or "").strip().casefold()
    for item in recommendations:
        item_id = str(item.get("item_id") or "")
        metadata = item_metadata.get(item_id, {})
        failed = False
        if budget_max is not None:
            price = _price(item, metadata)
            if price is None or price > float(budget_max) + 1e-6:
                budget_violations += 1
                failed = True
        if category_text:
            haystack = " ".join(
                [
                    str(metadata.get("main_category") or ""),
                    str(metadata.get("category_path") or ""),
                ]
            ).casefold()
            if category_text not in haystack:
                category_violations += 1
                failed = True
        if failed:
            violating_items.append(item_id)
    return {
        "budget_violations": budget_violations,
        "category_violations": category_violations,
        "violating_items": violating_items,
        "recommendations": len(recommendations),
    }


def evidence_stats(response: Mapping[str, Any]) -> dict[str, Any]:
    recommendations = list(response.get("recommendations") or [])
    with_evidence = 0
    chunks = 0
    for item in recommendations:
        evidence = item.get("review_evidence") or []
        if evidence:
            with_evidence += 1
            chunks += len(evidence)
    return {
        "recommendations": len(recommendations),
        "items_with_evidence": with_evidence,
        "evidence_chunks": chunks,
        "evidence_coverage": (
            with_evidence / len(recommendations) if recommendations else 0.0
        ),
    }


def token_usage(response: Mapping[str, Any]) -> dict[str, int]:
    prompt = 0
    completion = 0
    total = 0
    for event in response.get("agent_trace") or []:
        usage = event.get("llm_usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def summarize_run(
    response: Mapping[str, Any],
    *,
    variant: str,
    query_id: str,
    item_metadata: Mapping[str, Mapping[str, Any]],
    expected_budget_max: float | None = None,
    expected_category: str | None = None,
) -> dict[str, Any]:
    """Summarize one real run without inventing quality labels."""
    recommendations = list(response.get("recommendations") or [])
    violations = constraint_violations(
        recommendations,
        item_metadata=item_metadata,
        budget_max=expected_budget_max,
        category=expected_category,
    )
    profile = response.get("profile") or {}
    effective = profile.get("effective_constraints") or {}
    critic = response.get("critic") or {}
    usage = token_usage(response)
    reason_sources = Counter(
        str(item.get("reason_source") or "unknown") for item in recommendations
    )
    return {
        "variant": variant,
        "query_id": query_id,
        "status": str(response.get("status") or ""),
        "recommendations": len(recommendations),
        "top_item_ids": [
            str(item.get("item_id")) for item in recommendations[:10]
        ],
        "expected_budget_max": expected_budget_max,
        "expected_category": expected_category,
        "budget_violations": violations["budget_violations"],
        "category_violations": violations["category_violations"],
        "violating_items": violations["violating_items"],
        "items_with_evidence": evidence_stats(response)["items_with_evidence"],
        "evidence_chunks": evidence_stats(response)["evidence_chunks"],
        "evidence_coverage": evidence_stats(response)["evidence_coverage"],
        "review_status": str(response.get("review_status") or ""),
        "critic_approved": critic.get("approved"),
        "critic_skipped": bool(critic.get("skipped", False)),
        "critic_llm_status": critic.get("llm_status"),
        "memory_effective_constraints": effective,
        "memory_budget_max": effective.get("budget_max"),
        "memory_category": effective.get("category"),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_ms": float(response.get("latency_ms") or 0.0),
        "reason_sources": dict(reason_sources),
    }


def aggregate_agent_runs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate real per-run rows, grouped by variant."""
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    output: dict[str, Any] = {}
    for variant, variant_rows in sorted(by_variant.items()):
        run_count = len(variant_rows)
        recommendations = sum(int(row["recommendations"]) for row in variant_rows)
        items_with_evidence = sum(
            int(row["items_with_evidence"]) for row in variant_rows
        )
        budget_violations = sum(
            int(row["budget_violations"]) for row in variant_rows
        )
        category_violations = sum(
            int(row["category_violations"]) for row in variant_rows
        )
        output[variant] = {
            "runs": run_count,
            "status_counts": dict(
                Counter(str(row["status"]) for row in variant_rows)
            ),
            "no_recommendation_runs": sum(
                1 for row in variant_rows if int(row["recommendations"]) == 0
            ),
            "recommendations": recommendations,
            "items_with_evidence": items_with_evidence,
            "evidence_coverage": (
                items_with_evidence / recommendations if recommendations else 0.0
            ),
            "budget_violations": budget_violations,
            "category_violations": category_violations,
            "constraint_violation_rate": (
                (budget_violations + category_violations) / recommendations
                if recommendations
                else 0.0
            ),
            "runs_with_memory_constraints": sum(
                1
                for row in variant_rows
                if bool(row.get("memory_budget_max") is not None)
                or bool(row.get("memory_category"))
            ),
            "critic_approved_runs": sum(
                1 for row in variant_rows if row.get("critic_approved") is True
            ),
            "critic_skipped_runs": sum(
                1 for row in variant_rows if row.get("critic_skipped")
            ),
            "mean_latency_ms": (
                sum(float(row["latency_ms"]) for row in variant_rows) / run_count
            ),
            "mean_total_tokens": (
                sum(int(row["total_tokens"]) for row in variant_rows) / run_count
            ),
            "mean_recommendations": recommendations / run_count,
        }
    return output
