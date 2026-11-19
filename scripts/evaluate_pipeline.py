#!/usr/bin/env python
"""Evaluate the end-to-end ranking/re-ranking pipeline on held-out targets.

Variants:
* ``fusion_top10``   -- retrieval fusion order;
* ``din_top10``      -- DIN scoring after fusion, no diversity re-rank;
* ``din_mmr_top10``  -- DIN scores + MMR category diversity.

Targets are scorable held-out positives.  ``ranking_score`` and MMR scores are
model preference scores, not CTR estimates.
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
from src.ranking.reranker import mmr_rerank  # noqa: E402
from src.retrieval.fusion import (  # noqa: E402
    filter_fused_candidates,
    fuse_candidates,
)
from src.services.recommendation_service import RecommendationEngine  # noqa: E402


def _parse_weights(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    weights: dict[str, float] = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if not name or not value:
            raise ValueError("--weights must look like name=value,name=value")
        weights[name.strip()] = float(value)
    return weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end ranking")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--users", type=int, default=1000, help="0 = all")
    parser.add_argument("--strategy", choices=["quota", "rrf", "quota_rrf"], default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no-mmr", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    root = config.project_root
    engine = RecommendationEngine.from_artifacts(config)
    eval_frame = pl.read_parquet(root / f"data/splits/{args.split}_eval.parquet")
    if args.users > 0 and eval_frame.height > args.users:
        eval_frame = eval_frame.gather_every(
            max(1, eval_frame.height // args.users)
        )
    strategy = args.strategy or config.retrieval.fusion.strategy
    weights = _parse_weights(args.weights) or dict(
        config.retrieval.fusion.channel_weights
    )
    top_k = 10
    catalog_size = int(engine.ranker.encoder.num_items)
    variants = ["fusion_top10", "din_top10", "din_mmr_top10"]
    metrics: dict[str, list[dict[str, float]]] = {
        variant: [] for variant in variants
    }
    selected_items: dict[str, list[list[str]]] = {
        variant: [] for variant in variants
    }
    diversity: dict[str, list[float]] = {variant: [] for variant in variants}
    retrieval_ms: list[float] = []
    ranking_ms: list[float] = []
    rerank_ms: list[float] = []
    started = time.perf_counter()
    for row in eval_frame.iter_rows(named=True):
        user_id = str(row["user_id"])
        target_id = str(row["parent_asin"])
        retrieval_started = time.perf_counter()
        channel_results = {
            name: retriever.retrieve(
                user_id, config.retrieval.fusion.quota_per_channel
            )
            for name, retriever in engine.retrievers.items()
        }
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
            fused,
            exclude_item_ids=engine.user_exclusions.get(user_id, set()),
            item_metadata=engine.item_metadata,
        )
        retrieval_ms.append((time.perf_counter() - retrieval_started) * 1000)

        fusion_ids = [candidate.item_id for candidate in filtered][:top_k]
        ranking_started = time.perf_counter()
        scores = engine.ranker.score(
            user_id, [candidate.item_id for candidate in filtered]
        )
        score_by_item = {
            row_["item_id"]: float(row_["ranking_score"]) for row_ in scores
        }
        scored_candidates = [
            {
                "item_id": candidate.item_id,
                "ranking_score": score_by_item[candidate.item_id],
                "category": engine.item_metadata.get(candidate.item_id, {}).get(
                    "main_category", ""
                ),
                "store": engine.item_metadata.get(candidate.item_id, {}).get(
                    "store", ""
                ),
                "title": engine.item_metadata.get(candidate.item_id, {}).get(
                    "title", ""
                ),
                "price": engine.item_metadata.get(candidate.item_id, {}).get(
                    "price"
                ),
            }
            for candidate in filtered
            if candidate.item_id in score_by_item
        ]
        scored_candidates.sort(
            key=lambda item: item["ranking_score"], reverse=True
        )
        din_ids = [item["item_id"] for item in scored_candidates][:top_k]
        ranking_ms.append((time.perf_counter() - ranking_started) * 1000)

        rerank_started = time.perf_counter()
        if args.no_mmr:
            mmr_items = [
                {**item, "mmr_score": item["ranking_score"]}
                for item in scored_candidates[:top_k]
            ]
        else:
            mmr_items = mmr_rerank(
                scored_candidates,
                top_k=top_k,
                mmr_lambda=config.ranking.rerank.mmr_lambda,
                max_per_category=config.ranking.rerank.max_per_category,
                max_per_brand=config.ranking.rerank.max_per_brand,
                similarity=config.ranking.rerank.similarity,
                enable_mmr=config.ranking.rerank.enable_mmr,
            )
        mmr_ids = [item["item_id"] for item in mmr_items]
        rerank_ms.append((time.perf_counter() - rerank_started) * 1000)

        for variant, ids in (
            ("fusion_top10", fusion_ids),
            ("din_top10", din_ids),
            ("din_mmr_top10", mmr_ids),
        ):
            rank = ids.index(target_id) + 1 if target_id in ids else catalog_size + 1
            metrics[variant].append(metrics_from_rank(rank, (top_k,)))
            selected_items[variant].append(ids)
            categories = {
                engine.item_metadata.get(item_id, {}).get("main_category", "")
                for item_id in ids
            }
            categories.discard("")
            diversity[variant].append(
                len(categories) / max(1, len(ids))
            )

    output = {}
    for variant in variants:
        aggregate = aggregate_metrics(metrics[variant])
        aggregate["users"] = float(len(metrics[variant]))
        seen: set[str] = set()
        for ids in selected_items[variant]:
            seen.update(ids)
        aggregate["coverage@10"] = (
            len(seen) / catalog_size if catalog_size else 0.0
        )
        aggregate["category_diversity@10"] = float(
            sum(diversity[variant]) / max(1, len(diversity[variant]))
        )
        output[variant] = aggregate
        print(
            f"{variant:14s} users={aggregate['users']:.0f} "
            f"recall@10={aggregate.get('recall@10',0):.5f} "
            f"ndcg@10={aggregate.get('ndcg@10',0):.5f} "
            f"mrr={aggregate.get('mrr',0):.5f} "
            f"coverage@10={aggregate.get('coverage@10',0):.4f} "
            f"diversity@10={aggregate.get('category_diversity@10',0):.4f}"
        )
    output["diagnostics"] = {
        "split": args.split,
        "users": int(len(eval_frame)),
        "strategy": strategy,
        "channel_weights": weights,
        "top_k": top_k,
        "mmr": not args.no_mmr,
        "catalog_items": catalog_size,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "avg_retrieval_ms": round(sum(retrieval_ms) / max(1, len(retrieval_ms)), 2),
        "avg_ranking_ms": round(sum(ranking_ms) / max(1, len(ranking_ms)), 2),
        "avg_rerank_ms": round(sum(rerank_ms) / max(1, len(rerank_ms)), 2),
        "note": (
            "Held-out scorable positives; train interactions excluded. "
            "MRR uses catalog_size+1 for targets outside the top-10 list. "
            "Scores are preference scores, not CTR."
        ),
    }
    save_json(output, root / f"logs/phase3_pipeline_{args.split}.json")
    print(f"saved={root / f'logs/phase3_pipeline_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
