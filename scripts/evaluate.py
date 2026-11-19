#!/usr/bin/env python
"""Summarise Phase 2 model metrics from training logs.

This is a lightweight report reader. The comprehensive Phase 7 evaluator is
``scripts/run_ablation.py`` (retrieval/ranking) and
``scripts/run_agent_ablation.py`` (Agent/RAG/memory/Critic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_app_config  # noqa: E402
from src.core.training import save_json  # noqa: E402


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise Phase 2 metrics")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    config = get_app_config()
    root = config.project_root
    rows: list[dict] = []
    for model in ("itemcf", "two_tower", "lightgcn"):
        path = root / f"logs/metrics_{model}_{args.split}.json"
        metrics = _load(path)
        rows.append(
            {
                "model": model,
                "eval": f"full_catalog_{args.split}",
                "recall@10": metrics.get("recall@10"),
                "recall@50": metrics.get("recall@50"),
                "ndcg@10": metrics.get("ndcg@10"),
                "mrr": metrics.get("mrr"),
                "coverage@50": metrics.get("coverage@50"),
                "users": metrics.get("users"),
            }
        )
    for model in ("deepfm", "din"):
        path = root / f"logs/metrics_{model}_valid.json"
        metrics = _load(path)
        rows.append(
            {
                "model": model,
                "eval": "sampled_valid_1pos_4neg",
                "auc": metrics.get("auc"),
                "log_loss": metrics.get("log_loss"),
                "accuracy": metrics.get("accuracy"),
                "users": metrics.get("users"),
            }
        )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"Phase 2 metrics (split={args.split})")
        print(f"{'model':12s} {'eval':26s} {'Recall@10':>9s} {'Recall@50':>9s} "
              f"{'NDCG@10':>8s} {'MRR':>7s} {'Cov@50':>7s} {'AUC':>7s} {'LogLoss':>8s}")
        for row in rows:
            print(
                f"{row['model']:12s} {row['eval']:26s} "
                f"{_fmt(row.get('recall@10')):>9s} {_fmt(row.get('recall@50')):>9s} "
                f"{_fmt(row.get('ndcg@10')):>8s} {_fmt(row.get('mrr')):>7s} "
                f"{_fmt(row.get('coverage@50')):>7s} {_fmt(row.get('auc')):>7s} "
                f"{_fmt(row.get('log_loss')):>8s}"
            )
        print(
            "\nRetrieval: full train catalog, scorable positive targets, train "
            "history excluded. Ranking: 1 positive + 4 sampled implicit "
            "negatives; AUC is a sampled diagnostic, not CTR."
        )
    save_json(
        {"split": args.split, "models": rows},
        root / f"logs/phase2_summary_{args.split}.json",
    )
    return 0


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
