#!/usr/bin/env python
"""Compare quota / RRF / quota_rrf fusion on held-out positive targets.

This evaluates retrieval fusion (not DIN/MMR) on a deterministic user subset;
Phase 7's complete ablation is in ``scripts/run_ablation.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl  # noqa: E402

from src.core.config import get_app_config  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.core.training import save_json  # noqa: E402
from src.evaluation.retrieval_metrics import (  # noqa: E402
    aggregate_metrics,
    metrics_from_rank,
)
from src.retrieval.fusion import (  # noqa: E402
    filter_fused_candidates,
    fuse_candidates,
)
from src.services.recommendation_service import RecommendationEngine  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval fusion strategies")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--users", type=int, default=500, help="0 = all")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional channel weights, e.g. itemcf=1.0,two_tower=0.1,lightgcn=0.1",
    )
    args = parser.parse_args(argv)
    setup_logging(level="INFO")
    config = get_app_config()
    weights = None
    if args.weights:
        weights = {}
        for part in args.weights.split(","):
            name, _, value = part.partition("=")
            if not name or not value:
                parser.error("--weights must look like name=value,name=value")
            weights[name.strip()] = float(value)
    elif config.retrieval.fusion.channel_weights:
        weights = dict(config.retrieval.fusion.channel_weights)
    root = config.project_root
    engine = RecommendationEngine.from_artifacts(config)
    eval_frame = pl.read_parquet(root / f"data/splits/{args.split}_eval.parquet")
    if args.users > 0 and eval_frame.height > args.users:
        eval_frame = eval_frame.gather_every(
            max(1, eval_frame.height // args.users)
        )
    strategies = ["quota", "rrf", "quota_rrf"]
    catalog_size = int(engine.ranker.encoder.num_items)
    metrics: dict[str, list[dict[str, float]]] = {
        strategy: [] for strategy in strategies
    }
    recommended: dict[str, list[list[str]]] = {
        strategy: [] for strategy in strategies
    }
    channel_metrics: dict[str, list[dict[str, float]]] = {
        name: [] for name in engine.retrievers
    }
    channel_recommended: dict[str, list[list[str]]] = {
        name: [] for name in engine.retrievers
    }
    channel_counts: dict[str, list[int]] = {
        name: [] for name in engine.retrievers
    }
    started = time.perf_counter()
    for row in eval_frame.iter_rows(named=True):
        user_id = str(row["user_id"])
        target_id = str(row["parent_asin"])
        channel_results: dict[str, list[dict]] = {}
        for name, retriever in engine.retrievers.items():
            results = retriever.retrieve(
                user_id, config.retrieval.fusion.quota_per_channel
            )
            channel_results[name] = results
            channel_counts[name].append(len(results))
            channel_ids = [result["item_id"] for result in results]
            channel_rank = (
                channel_ids.index(target_id) + 1
                if target_id in channel_ids
                else catalog_size + 1
            )
            channel_metrics[name].append(metrics_from_rank(channel_rank))
            channel_recommended[name].append(channel_ids[:50])
        exclusions = engine.user_exclusions.get(user_id, set())
        for strategy in strategies:
            fused = fuse_candidates(
                channel_results,
                strategy=strategy,
                quota_per_channel=config.retrieval.fusion.quota_per_channel,
                total_candidates=config.retrieval.fusion.total_candidates,
                rrf_k=config.retrieval.fusion.rrf_k,
                score_normalization=config.retrieval.fusion.score_normalization,
                channel_weights=weights,
            )
            filtered, _ = filter_fused_candidates(
                fused, exclude_item_ids=exclusions
            )
            ranked_ids = [candidate.item_id for candidate in filtered]
            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
            else:
                rank = catalog_size + 1
            metrics[strategy].append(metrics_from_rank(rank))
            recommended[strategy].append(ranked_ids[:50])

    def _summarise(
        per_user: list[dict[str, float]], item_lists: list[list[str]]
    ) -> dict[str, float]:
        aggregate = aggregate_metrics(per_user)
        aggregate["users"] = float(len(per_user))
        seen: set[str] = set()
        for item_ids in item_lists:
            seen.update(item_ids[:50])
        aggregate["coverage@50"] = (
            len(seen) / catalog_size if catalog_size else 0.0
        )
        return aggregate

    output = {}
    for strategy in strategies:
        aggregate = _summarise(metrics[strategy], recommended[strategy])
        output[strategy] = aggregate
        print(
            f"{strategy:10s} users={aggregate['users']:.0f} "
            f"recall@10={aggregate.get('recall@10',0):.4f} "
            f"recall@50={aggregate.get('recall@50',0):.4f} "
            f"ndcg@10={aggregate.get('ndcg@10',0):.4f} "
            f"mrr={aggregate.get('mrr',0):.4f} "
            f"coverage@50={aggregate.get('coverage@50',0):.4f}"
        )
    output["channels"] = {}
    for name in engine.retrievers:
        aggregate = _summarise(channel_metrics[name], channel_recommended[name])
        output["channels"][name] = aggregate
        print(
            f"[{name:9s}] recall@10={aggregate.get('recall@10',0):.4f} "
            f"recall@50={aggregate.get('recall@50',0):.4f} "
            f"mrr={aggregate.get('mrr',0):.4f} "
            f"coverage@50={aggregate.get('coverage@50',0):.4f}"
        )
    output["diagnostics"] = {
        "split": args.split,
        "users": int(len(eval_frame)),
        "catalog_items": catalog_size,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "channel_avg_counts": {
            name: round(sum(values) / max(1, len(values)), 1)
            for name, values in channel_counts.items()
        },
        "channel_weights": weights,
        "rank_when_absent": "catalog_size + 1",
        "note": (
            "Fusion retrieval only. Targets are held-out scorable positives; "
            "train interactions excluded. DIN/MMR are evaluated separately."
        ),
    }
    save_json(output, root / f"logs/phase3_fusion_{args.split}.json")
    print(f"saved={root / f'logs/phase3_fusion_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
