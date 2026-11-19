#!/usr/bin/env python
"""Phase 7 Agent ablation: full vs no-RAG vs no-memory vs no-Critic.

Only deterministic, verifiable outcomes are reported: status, item count,
hard-constraint violations against real metadata, real review-evidence
coverage, Critic state, token usage and latency.  This script does not judge
prose quality and does not claim a CTR improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from src.agents.profile_agent import ProfileAgent  # noqa: E402
from src.agents.workflow import AgentWorkflow  # noqa: E402
from src.core.config import get_app_config  # noqa: E402
from src.core.exceptions import DeepSeekError, RecSysError  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.core.training import save_json  # noqa: E402
from src.evaluation.agent_eval import (  # noqa: E402
    aggregate_agent_runs,
    summarize_run,
)


DEFAULT_QUERIES = [
    {
        "id": "ssd_budget",
        "query": "我想买一个用于日常办公的 SSD，预算不超过 200 美元。",
        "category": "solid state",
        "budget_max": 200.0,
    },
    {
        "id": "headphones_budget",
        "query": "推荐一款头戴式降噪耳机，预算 150 美元以内。",
        "category": "headphones",
        "budget_max": 150.0,
    },
    {
        "id": "keyboard_memory",
        "query": "推荐一款适合办公的键盘。",
        "category": "keyboard",
        "budget_max": None,
    },
]


def _make_variant(
    full: AgentWorkflow,
    *,
    enable_review_rag: bool = True,
    use_memory: bool = True,
    enable_critic: bool = True,
) -> AgentWorkflow:
    memory_manager = full.memory_manager if use_memory else None
    if use_memory:
        profile_agent = full.profile_agent
    else:
        profile_agent = ProfileAgent(
            full.profile_agent.store,
            memory_manager=None,
            use_persistent_memory=False,
        )
    return AgentWorkflow(
        config=full.config,
        intent_agent=full.intent_agent,
        profile_agent=profile_agent,
        recommendation_agent=full.recommendation_agent,
        critic_agent=full.critic_agent,
        toolkit=full.toolkit,
        client=full.client,
        checkpointer=MemorySaver(),
        memory_manager=memory_manager,
        review_retriever=(
            full.review_retriever if enable_review_rag else None
        ),
        enable_review_rag=enable_review_rag,
        enable_critic=enable_critic,
    )


def _load_queries(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return [dict(query) for query in DEFAULT_QUERIES]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("queries file must be a non-empty JSON list")
    required = {"id", "query"}
    for query in payload:
        if not isinstance(query, dict) or not required.issubset(query):
            raise ValueError("each query needs string fields id/query")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7 Agent ablation")
    parser.add_argument(
        "--user-id",
        default="AE22JIZVWSRGCS3UUQUYDQ2HRN7Q",
        help="real user id; default is an existing train user",
    )
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--raw-output", default=None)
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    root = config.project_root
    queries = _load_queries(args.queries_file)
    # The API may be serving from the persistent SQLite checkpoint DB.  Local
    # ablation runs use an in-memory checkpointer so two processes never write
    # the same checkpoint file concurrently.
    full = AgentWorkflow.from_artifacts(config, checkpointer=MemorySaver())
    variants: dict[str, AgentWorkflow] = {
        "full": full,
        "no_rag": _make_variant(full, enable_review_rag=False),
        "no_memory": _make_variant(full, use_memory=False),
        "no_critic": _make_variant(full, enable_critic=False),
    }
    run_id = f"phase7-{uuid.uuid4().hex[:8]}"
    rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for variant, workflow in variants.items():
            for query in queries:
                query_id = str(query["id"])
                print(f"[{variant}] {query_id}: {query['query']}")
                response = workflow.run(
                    str(query["query"]),
                    user_id=str(args.user_id),
                    thread_id=f"{run_id}-{variant}-{query_id}",
                )
                row = summarize_run(
                    response,
                    variant=variant,
                    query_id=query_id,
                    item_metadata=full.toolkit.engine.item_metadata,
                    expected_budget_max=query.get("budget_max"),
                    expected_category=query.get("category"),
                )
                rows.append(row)
                raw_runs.append(
                    {
                        "variant": variant,
                        "query_id": query_id,
                        "response": response,
                    }
                )
                print(
                    f"  status={row['status']} recs={row['recommendations']} "
                    f"evidence={row['evidence_coverage']:.2f} "
                    f"budget_violations={row['budget_violations']} "
                    f"category_violations={row['category_violations']} "
                    f"metrics=({row['prompt_tokens']}+{row['completion_tokens']}) "
                    f"latency_ms={row['latency_ms']:.0f}"
                )
    except DeepSeekError as exc:
        print(f"DeepSeek error: {type(exc).__name__}: {exc}")
        return 2
    except (RecSysError, ValueError) as exc:
        print(f"Agent ablation error: {type(exc).__name__}: {exc}")
        return 1
    finally:
        full.close()

    aggregate = aggregate_agent_runs(rows)
    output = {
        "phase": 7,
        "experiment": "agent_ablation",
        "run_id": run_id,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_id": str(args.user_id),
        "model": full.client.model_name if full.client is not None else None,
        "queries": queries,
        "runs": rows,
        "aggregate": aggregate,
        "note": (
            "Single deterministic query set, temperature="
            f"{config.agent.llm.temperature}. No human/LLM prose-quality score "
            "is claimed. Evidence coverage, constraint violations, tokens and "
            "latency are factual measurements from the real runs. The no-critic "
            "variant is not a production mode."
        ),
    }
    output_path = Path(
        args.output or root / "logs/phase7_agent_ablation.json"
    )
    if not output_path.is_absolute():
        output_path = root / output_path
    save_json(output, output_path)
    raw_path = Path(
        args.raw_output or root / "logs/phase7_agent_ablation_raw.json"
    )
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    save_json({"run_id": run_id, "runs": raw_runs}, raw_path)
    print("\nAggregate by variant:")
    for variant, values in aggregate.items():
        print(
            f"{variant:10s} runs={values['runs']} "
            f"status={values['status_counts']} "
            f"evidence={values['evidence_coverage']:.3f} "
            f"budget_viol={values['budget_violations']} "
            f"category_viol={values['category_violations']} "
            f"memory_runs={values['runs_with_memory_constraints']} "
            f"tokens={values['mean_total_tokens']:.0f} "
            f"latency_ms={values['mean_latency_ms']:.0f}"
        )
    print(f"saved={output_path}")
    print(f"raw_saved={raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
