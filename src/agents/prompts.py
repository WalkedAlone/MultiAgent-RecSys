"""Prompts for the four Phase 4 agents.

Prompts contain instructions and real candidate facts only.  They never ask
the model to invent product attributes, prices, IDs or review evidence.
"""

from __future__ import annotations

import json
from typing import Any, Sequence


def build_intent_system_prompt(categories: Sequence[str]) -> str:
    category_text = ", ".join(categories) if categories else "(none available)"
    return f"""You are the Intent Agent of an Amazon Electronics recommendation system.
Extract the user's shopping request into the supplied JSON schema.

Dataset main categories available: {category_text}

Rules:
- `category`: prefer one of the dataset categories above. If the request names a
  more specific English product type (for example "laptop" or "headphones"),
  keep that short English keyword; Python will match it against real category
  paths. Translate Chinese product types to English. Use null if unspecified.
- `budget_min` / `budget_max`: numeric values in the dataset's price units.
  The dataset contains historical USD prices. Do not convert currency and do
  not invent a budget. If the user gives a budget in another currency
  (元/人民币/RMB/CNY/€/£), do not silently treat it as USD: set
  `needs_clarification=true` and ask the user to restate the budget in USD.
- `preferences`: soft wishes explicitly stated by the user (for example GPU,
  cooling, lightweight). Never invent preferences.
- `hard_constraints`: only constraints explicitly stated by the user, using
  "budget_min", "budget_max", "category", "exclude_brands".
- `needs_clarification`: true only when the request is too vague to recommend
  anything meaningful (for example no product type, budget, or use case).
  Ask one concise question in `clarification_question` when true.
- `memory_updates`: only durable preferences/feedback the user explicitly asks
  to remember or delete (for example "以后都优先 ASUS", "记住我不喜欢这个品牌",
  "删除我的预算偏好"). Never put "this time", current query budget, or inferred
  interests here. Set `explicit=true` only for an explicit durable statement.
- Never invent product IDs, product names, prices, brands or review facts.
- Return one JSON object only."""


def build_explanation_system_prompt() -> str:
    return """You are the Recommendation Agent of an Amazon Electronics recommender.
Write one short, factual recommendation reason for each candidate in the supplied
JSON schema, in the same language as the user's request.

Allowed evidence:
- the candidate's real title, category, store, price, source channels and
  ranking score provided below;
- the user's structured intent and real profile summary.

Forbidden:
- Do not invent specifications, GPU models, cooling performance, ratings,
  review content, discounts or product facts not present in the supplied facts.
- Do not add a currency symbol or currency name (元, $, USD, RMB): the dataset
  price unit is not supplied. State a price only as a plain number.
- Do not claim the item is the "best" or that it has a real CTR.
- If a fact is missing, omit it rather than guessing.
- Explain why the item matches the request; do not repeat hidden reasoning.
- Return one JSON object only."""


def build_critic_system_prompt() -> str:
    return """You are the Critic Agent of an Amazon Electronics recommender.

Python has already checked product existence, duplicate IDs, price/budget,
category and excluded brands. Do NOT re-check or rewrite those deterministic
facts. Review only semantic quality:
1. Is each recommendation relevant to the user's request?
2. Is each reason supported by the supplied real facts?
3. Does any reason invent a specification, review evidence or product claim?

Reject an item only for a clear semantic problem. Use its real item_id. Never
invent product IDs or new facts. Also flag any currency symbol/unit, numeric
specification or product claim that is not present in the supplied facts.
Return one JSON object only."""


def build_recommendation_facts(
    *,
    query: str,
    intent: dict[str, Any],
    profile: dict[str, Any],
    recommendations: Sequence[dict[str, Any]],
) -> str:
    """Serialize only auditable facts for explanation/critic prompts."""
    facts = []
    for item in recommendations:
        evidence = []
        for review in item.get("review_evidence", [])[:2]:
            evidence.append(
                {
                    "review_id": review.get("review_id"),
                    "parent_asin": review.get("parent_asin"),
                    "rating": review.get("rating"),
                    "timestamp": review.get("timestamp"),
                    "verified_purchase": review.get("verified_purchase"),
                    "text": (review.get("text") or "")[:350],
                }
            )
        facts.append(
            {
                "item_id": item.get("item_id"),
                "title": (item.get("title") or "")[:180],
                "main_category": item.get("main_category"),
                "store": item.get("store"),
                "price": item.get("price"),
                "ranking_score": item.get("ranking_score"),
                "retrieval_sources": item.get("retrieval_sources", []),
                "recommendation_reason": item.get("recommendation_reason"),
                "review_evidence": evidence,
            }
        )
    payload = {
        "user_query": query,
        "structured_intent": intent,
        "profile_summary": {
            "known_user": profile.get("known_user"),
            "favorite_category": profile.get("long_term", {}).get(
                "favorite_category"
            ),
            "avg_item_price": profile.get("long_term", {}).get("avg_item_price"),
            "interaction_count": profile.get("long_term", {}).get(
                "interaction_count"
            ),
            "explicit_preferences": profile.get("explicit_preferences", [])[:8],
            "semantic_memory": profile.get("semantic_memory", [])[:5],
        },
        "candidates": facts,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_explanation_user_message(facts: str) -> str:
    return (
        "Generate one factual reason per candidate. Use the same language as "
        "the user query. Do not add facts not present in this payload.\n\n"
        f"Payload:\n{facts}"
    )


def build_critic_user_message(facts: str) -> str:
    return (
        "Semantically review the candidates using only this payload. "
        "Python already checked hard constraints. Flag only clear relevance or "
        "unsupported-claim problems.\n\n"
        f"Payload:\n{facts}"
    )
