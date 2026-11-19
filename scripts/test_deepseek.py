#!/usr/bin/env python
"""Phase 0 live smoke tests for the DeepSeek integration.

Three checks are implemented separately:

1. ``chat``  - plain multi-turn chat completion.
2. ``json``  - JSON Output validated against a Pydantic schema.
3. ``tools`` - Function Calling followed by *application-side* argument
   validation and execution of a registered Python tool.

The tool handler returns deliberately labelled mock data.  It exists only to
prove the calling protocol and must never be presented as real catalog data.

Usage:
    python scripts/test_deepseek.py --mode all
    python scripts/test_deepseek.py --mode chat
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Literal

# Git Bash / Windows consoles may default to a non-UTF-8 code page.  The
# project path and LLM answers contain Chinese characters, so force UTF-8.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from src.agents.tools import ToolRegistry  # noqa: E402
from src.core.config import DeepSeekSettings, get_deepseek_settings  # noqa: E402
from src.core.exceptions import RecSysError  # noqa: E402
from src.core.logger import get_logger, setup_logging  # noqa: E402
from src.llm.deepseek_client import DeepSeekClient  # noqa: E402
from src.llm.schemas import LLMResponse  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas used by the smoke tests
# ---------------------------------------------------------------------------
class IntentExtraction(BaseModel):
    """Minimal intent schema used by the JSON Output check."""

    category: str = Field(min_length=1)
    budget_min: float | None = Field(default=None, ge=0)
    budget_max: float | None = Field(default=None, ge=0)
    use_case: str | None = None
    preferences: list[str] = Field(default_factory=list)
    exclude_brands: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    hard_constraints: list[str] = Field(default_factory=list)


class SearchCatalogArgs(BaseModel):
    """Arguments accepted by the mock catalog tool."""

    category: Literal["laptop", "gpu", "headphones", "monitor"]
    budget_max: float | None = Field(default=None, ge=0)
    limit: int = Field(default=5, ge=1, le=10)


# ---------------------------------------------------------------------------
# Mock tool (test fixture only)
# ---------------------------------------------------------------------------
_MOCK_CATALOG: dict[str, dict[str, Any]] = {
    "laptop": {
        "item_id": "MOCK-LAPTOP-001",
        "title": "Mock 14-inch laptop (test fixture)",
        "price": 7999.0,
    },
    "gpu": {
        "item_id": "MOCK-GPU-001",
        "title": "Mock desktop GPU (test fixture)",
        "price": 6999.0,
    },
    "headphones": {
        "item_id": "MOCK-HEADPHONES-001",
        "title": "Mock noise-cancelling headphones (test fixture)",
        "price": 1999.0,
    },
    "monitor": {
        "item_id": "MOCK-MONITOR-001",
        "title": "Mock 27-inch monitor (test fixture)",
        "price": 2499.0,
    },
}


def search_catalog(args: SearchCatalogArgs) -> dict[str, Any]:
    """Return deterministic mock data, with application-side budget filtering."""
    entry = _MOCK_CATALOG[args.category]
    items = []
    if args.budget_max is None or entry["price"] <= args.budget_max:
        items.append(entry)
    return {
        "source": "mock_catalog_for_phase0_test",
        "note": "Mock data only; not a real product catalog.",
        "category": args.category,
        "budget_max": args.budget_max,
        "items": items[: args.limit],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_usage(label: str, response: LLMResponse) -> None:
    usage = response.usage
    print(
        f"  [{label}] model={response.model} "
        f"prompt_tokens={usage.prompt_tokens} "
        f"completion_tokens={usage.completion_tokens} "
        f"total_tokens={usage.total_tokens} "
        f"latency_ms={usage.latency_ms} attempts={usage.attempts} "
        f"finish_reason={response.finish_reason}"
    )


def _report_checks(checks: list[tuple[bool, str]]) -> bool:
    ok = True
    for passed, message in checks:
        print(f"  {'PASS' if passed else 'FAIL'}: {message}")
        ok = ok and passed
    return ok


# ---------------------------------------------------------------------------
# Test 1: plain chat
# ---------------------------------------------------------------------------
def run_chat_test(client: DeepSeekClient) -> bool:
    response = client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "用一句话说明个性化推荐系统的核心目标。",
            },
        ],
        call_name="phase0_chat",
    )
    print(f"\n[chat] answer: {response.content}")
    _print_usage("chat", response)
    return _report_checks(
        [
            (bool(response.content and response.content.strip()), "non-empty answer"),
            (response.usage.total_tokens > 0, "token usage reported"),
            (response.usage.latency_ms >= 0, "latency reported"),
        ]
    )


# ---------------------------------------------------------------------------
# Test 2: JSON Output
# ---------------------------------------------------------------------------
def run_json_test(client: DeepSeekClient) -> bool:
    parsed, response = client.chat_json(
        messages=[
            {
                "role": "user",
                "content": (
                    "我想买一台适合深度学习的笔记本电脑，预算不超过 8000 元，"
                    "优先考虑显卡和散热。请抽取结构化意图，金额单位为元。"
                ),
            }
        ],
        schema=IntentExtraction,
        call_name="phase0_json",
    )
    print("\n[json] validated structured output:")
    print(parsed.model_dump_json(indent=2))
    _print_usage("json", response)

    budget = parsed.budget_max
    return _report_checks(
        [
            (bool(parsed.category.strip()), "category is non-empty"),
            (budget is not None, "budget_max was extracted"),
            (budget is not None and budget <= 8000, "budget_max respects the 8000 limit"),
            (bool(parsed.preferences), "preferences are non-empty"),
            (bool(parsed.hard_constraints), "hard_constraints are non-empty"),
        ]
    )


# ---------------------------------------------------------------------------
# Test 3: Function Calling with safe application-side execution
# ---------------------------------------------------------------------------
def run_tools_test(client: DeepSeekClient, max_rounds: int) -> bool:
    registry = ToolRegistry()
    executed_args: list[dict[str, Any]] = []

    def handler(args: SearchCatalogArgs) -> dict[str, Any]:
        executed_args.append(args.model_dump())
        return search_catalog(args)

    registry.register(
        name="search_catalog",
        description=(
            "Search the product catalog by category and optional budget. "
            "Returns mock items for the Phase 0 integration test."
        ),
        args_model=SearchCatalogArgs,
        handler=handler,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a shopping assistant. Call the search_catalog tool when a "
                "product search is requested. Never invent catalog items."
            ),
        },
        {
            "role": "user",
            "content": "帮我找预算不超过 8000 元的笔记本电脑，最多 5 个。",
        },
    ]

    final_response: LLMResponse | None = None
    for round_number in range(1, max_rounds + 1):
        response = client.chat(
            messages=messages,
            tools=registry.openai_tools(),
            tool_choice="auto",
            call_name=f"phase0_tools_round_{round_number}",
        )
        _print_usage(f"tools round {round_number}", response)
        messages.append(response.to_assistant_message())

        if not response.tool_calls:
            final_response = response
            break

        for tool_call in response.tool_calls:
            print(
                f"  model requested tool={tool_call.name} "
                f"args={json.dumps(tool_call.arguments, ensure_ascii=False)}"
            )
            result = registry.execute(tool_call.name, tool_call.arguments)
            status = f"ok={result.ok} latency_ms={result.latency_ms}"
            if not result.ok:
                status += f" error={result.error}"
            print(f"  application executed tool={result.tool_name} {status}")
            messages.append(result.as_tool_message(tool_call.id))
    else:
        print(f"  FAIL: exceeded the maximum of {max_rounds} tool rounds")
        return False

    if final_response is not None:
        print(f"\n[tools] final answer: {final_response.content}")

    checks: list[tuple[bool, str]] = [
        (bool(executed_args), "model requested a registered tool"),
        (
            final_response is not None
            and bool((final_response.content or "").strip()),
            "final natural-language answer returned after tool execution",
        ),
    ]
    if executed_args:
        first = executed_args[0]
        budget = first.get("budget_max")
        checks.extend(
            [
                (first.get("category") == "laptop", "tool argument category == 'laptop'"),
                (
                    budget is not None and float(budget) <= 8000,
                    "tool argument budget_max respects 8000",
                ),
                (
                    int(first.get("limit", 5)) <= 5,
                    "tool argument limit respects 'at most 5'",
                ),
            ]
        )
    return _report_checks(checks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_case(
    name: str,
    fn: Any,
    client: DeepSeekClient,
    verbose: bool,
    *args: Any,
) -> bool:
    print(f"\n{'=' * 72}\n=== {name} ===\n{'=' * 72}")
    try:
        ok = bool(fn(client, *args))
    except RecSysError as exc:
        print(f"[{name}] FAIL: {type(exc).__name__}: {exc}")
        logger.error("%s failed: %s", name, type(exc).__name__)
        ok = False
    except Exception as exc:  # pragma: no cover - defensive CLI handling
        print(f"[{name}] FAIL (unexpected): {type(exc).__name__}: {exc}")
        logger.error("%s crashed: %s", name, type(exc).__name__)
        if verbose:
            traceback.print_exc()
        ok = False
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek Phase 0 connectivity tests"
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "json", "tools", "all"],
        default="all",
        help="Which check to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override DEEPSEEK_MODEL for this run (must be a model your key can use)",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=5,
        help="Maximum tool-calling rounds for the tools test",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG logs and tracebacks")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    print("DeepSeek Phase 0 connectivity test")
    print(f"project_root={PROJECT_ROOT}")

    try:
        settings: DeepSeekSettings = get_deepseek_settings()
    except RecSysError as exc:
        print(f"Configuration error: {exc}")
        return 2
    if args.model:
        settings = settings.model_copy(update={"model": args.model})

    print(
        f"base_url={settings.base_url} model={settings.model} "
        f"timeout={settings.timeout}s max_retries={settings.max_retries}"
    )

    if not settings.has_api_key:
        print("\nDEEPSEEK_API_KEY is not configured.")
        print(f"Create {PROJECT_ROOT / '.env'} from .env.example and set the key.")
        print("Never commit the .env file.")
        return 2

    try:
        client = DeepSeekClient(settings)
    except RecSysError as exc:
        print(f"Client initialisation failed: {type(exc).__name__}: {exc}")
        return 2

    try:
        modes = ["chat", "json", "tools"] if args.mode == "all" else [args.mode]
        results: dict[str, bool] = {}
        for mode in modes:
            if mode == "chat":
                results[mode] = _run_case("chat", run_chat_test, client, args.verbose)
            elif mode == "json":
                results[mode] = _run_case("json", run_json_test, client, args.verbose)
            elif mode == "tools":
                results[mode] = _run_case(
                    "tools",
                    run_tools_test,
                    client,
                    args.verbose,
                    args.max_tool_rounds,
                )

        print(f"\n{'=' * 72}\n=== Summary ===\n{'=' * 72}")
        for mode, passed in results.items():
            print(f"{mode:6s} {'PASS' if passed else 'FAIL'}")
        print(f"client_stats={json.dumps(client.stats(), ensure_ascii=False)}")
        print(f"usage_log={PROJECT_ROOT / 'logs' / 'llm_usage.jsonl'}")
        return 0 if all(results.values()) else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
