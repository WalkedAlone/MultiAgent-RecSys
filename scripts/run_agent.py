#!/usr/bin/env python
"""Run the real Phase 4 LangGraph agent workflow for one user request."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from src.agents.workflow import AgentWorkflow  # noqa: E402
from src.core.config import get_app_config  # noqa: E402
from src.core.exceptions import (  # noqa: E402
    AgentWorkflowError,
    ConfigurationError,
    DeepSeekError,
    RecSysError,
)
from src.core.logger import setup_logging  # noqa: E402
from src.core.training import save_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the multi-agent recommender")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    workflow = None
    try:
        # CLI runs use an in-memory checkpointer; the FastAPI service owns the
        # persistent SQLite checkpoint file and must not share writes with a
        # second process.
        workflow = AgentWorkflow.from_artifacts(
            config, checkpointer=MemorySaver()
        )
        final = workflow.run(
            args.query,
            user_id=args.user_id,
            thread_id=args.thread_id,
        )
    except DeepSeekError as exc:
        print(f"DeepSeek error: {type(exc).__name__}: {exc}")
        return 2
    except (AgentWorkflowError, ConfigurationError, RecSysError) as exc:
        print(f"Agent workflow error: {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"Unexpected agent error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        if workflow is not None:
            workflow.close()

    save_json(final, config.project_root / "logs/phase4_agent_demo.json")
    save_json(final, config.project_root / "logs/phase5_agent_demo.json")
    if args.json:
        print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"status={final['status']} user_id={final['user_id']} "
          f"recommendations={len(final['recommendations'])} "
          f"latency_ms={final['latency_ms']}")
    if final.get("clarification"):
        print(f"clarification: {final['clarification']}")
    for index, item in enumerate(final["recommendations"], start=1):
        print(
            f"  {index:2d}. {item.get('title') or item.get('item_id')} "
            f"({item.get('item_id')}) "
            f"price={item.get('price')} "
            f"category={item.get('main_category')} "
            f"score={item.get('ranking_score'):.4f}"
        )
        print(f"      reason: {item.get('recommendation_reason')}")
        print(
            f"      sources={item.get('retrieval_sources')} "
            f"reason_source={item.get('reason_source')} "
            f"evidence_status={item.get('review_evidence_status')}"
        )
        if item.get("evidence_summary"):
            print(f"      evidence: {item['evidence_summary']}")
    print(
        f"\nreview_status={final.get('review_status')} "
        f"memory_update={final.get('memory_update', {}).get('status')}"
    )
    print("\nagent_trace:")
    for event in final.get("agent_trace", []):
        print(
            f"  {event.get('node'):24s} {event.get('status'):12s} "
            f"{event.get('latency_ms', 0):>6} ms  {event.get('summary', '')}"
        )
    total_tokens = 0
    for event in final.get("agent_trace", []):
        usage = event.get("llm_usage") or {}
        total_tokens += int(usage.get("total_tokens", 0))
    print(f"\nllm_tokens={total_tokens} checkpoint={final.get('checkpointing')}")
    print(f"saved={config.project_root / 'logs/phase4_agent_demo.json'}")
    print(f"saved={config.project_root / 'logs/phase5_agent_demo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
