"""Intent Agent: natural language -> validated structured recommendation intent."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from src.agents.prompts import build_intent_system_prompt
from src.agents.schemas import IntentResult
from src.core.logger import get_logger

logger = get_logger(__name__)

_CATEGORY_ALIASES = {
    "笔记本电脑": "laptop",
    "笔记本": "laptop",
    "notebook": "laptop",
    "notebooks": "laptop",
    "电脑": "computer",
    "手机": "cell phones",
    "智能手机": "cell phones",
    "phone": "cell phones",
    "phones": "cell phones",
    "smartphone": "cell phones",
    "smartphones": "cell phones",
    "mobile phone": "cell phones",
    "cell phone": "cell phones",
    "耳机": "headphones",
    "earphone": "headphones",
    "earphones": "headphones",
    "earbud": "headphones",
    "earbuds": "headphones",
    "相机": "camera & photo",
    "摄影": "camera & photo",
    "音响": "home audio & theater",
    "电子": "all electronics",
    "ssd": "solid state",
    "固态硬盘": "solid state",
    "硬盘": "hard drive",
    "机械硬盘": "hard drive",
    "内存卡": "memory card",
    "存储卡": "memory card",
    "显示器": "monitor",
    "键盘": "keyboard",
    "鼠标": "mouse",
    "音箱": "speakers",
    "摄像头": "webcam",
}

_ALLOWED_HARD_CONSTRAINTS = {
    "budget_min",
    "budget_max",
    "category",
    "exclude_brands",
}


class IntentAgent:
    """Use the shared DeepSeek client with JSON Output and Pydantic validation."""

    def __init__(
        self,
        client: Any,
        *,
        categories: Sequence[str] = (),
        max_attempts: int = 2,
        temperature: float = 0.2,
        max_conversation_messages: int = 12,
    ) -> None:
        self.client = client
        self.categories = [str(value) for value in categories if str(value).strip()]
        self.max_attempts = int(max_attempts)
        self.temperature = float(temperature)
        self.max_conversation_messages = int(max_conversation_messages)

    def run(
        self,
        query: str,
        conversation: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query = str(query).strip()
        if not query:
            intent = IntentResult(
                needs_clarification=True,
                clarification_question="请描述想购买的商品类别、预算或主要用途。",
            ).model_dump()
            return {
                "intent": intent,
                "trace": [
                    {
                        "node": "intent_agent",
                        "status": "clarification",
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "summary": "empty query; clarification requested",
                    }
                ],
            }

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_intent_system_prompt(self.categories),
            }
        ]
        if conversation:
            messages.extend(dict(message) for message in conversation[-self.max_conversation_messages :])
        messages.append({"role": "user", "content": query})

        parsed, response = self.client.chat_json(
            messages,
            IntentResult,
            max_attempts=self.max_attempts,
            temperature=self.temperature,
            call_name="intent_agent",
        )
        intent = self._normalise(parsed)
        latency_ms = int((time.perf_counter() - started) * 1000)
        summary = (
            f"category={intent.get('category')!r}; "
            f"budget_min={intent.get('budget_min')}; "
            f"budget_max={intent.get('budget_max')}; "
            f"top_k={intent.get('top_k')}; "
            f"memory_updates={len(intent.get('memory_updates', []))}; "
            f"needs_clarification={intent.get('needs_clarification')}"
        )
        logger.info("intent_agent_completed %s", summary)
        return {
            "intent": intent,
            "trace": [
                {
                    "node": "intent_agent",
                    "status": (
                        "clarification"
                        if intent.get("needs_clarification")
                        else "ok"
                    ),
                    "latency_ms": latency_ms,
                    "summary": summary,
                    "llm_usage": response.usage.model_dump(),
                }
            ],
        }

    def _normalise(self, parsed: IntentResult) -> dict[str, Any]:
        intent = parsed.model_dump()
        category = intent.get("category")
        if category:
            intent["category"] = self._canonical_category(str(category))

        constraints = {
            str(value).strip()
            for value in intent.get("hard_constraints", [])
            if str(value).strip() in _ALLOWED_HARD_CONSTRAINTS
        }
        # These fields are enforced as hard filters by the Phase 4 engine/critic.
        # Keeping the list consistent avoids pretending a constraint is soft.
        if intent.get("budget_min") is not None:
            constraints.add("budget_min")
        if intent.get("budget_max") is not None:
            constraints.add("budget_max")
        if intent.get("category"):
            constraints.add("category")
        if intent.get("exclude_brands"):
            constraints.add("exclude_brands")
        intent["hard_constraints"] = sorted(constraints)
        intent["top_k"] = max(1, min(50, int(intent.get("top_k") or 10)))
        intent["memory_updates"] = [
            update
            for update in intent.get("memory_updates", [])
            if isinstance(update, dict) and update.get("explicit", False)
        ]
        if intent.get("needs_clarification") and not intent.get(
            "clarification_question"
        ):
            intent["clarification_question"] = (
                "请补充想购买的商品类别、预算或主要用途。"
            )
        return intent

    def _canonical_category(self, value: str) -> str:
        text = value.strip()
        lowered = text.lower()
        for category in self.categories:
            if category.lower() == lowered:
                return category
        if lowered in _CATEGORY_ALIASES:
            return _CATEGORY_ALIASES[lowered]
        # A short English product keyword is intentionally preserved so the
        # engine can match it against real category paths (e.g. "laptop").
        return text
