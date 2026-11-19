#!/usr/bin/env python
"""Phase 7 retrieval/ranking ablation on real held-out positives.

This script evaluates:

* each recall channel alone;
* all-channel fusion with quota / RRF / quota-RRF;
* equal weights vs validation-calibrated weights;
* channel-subset fusions;
* fusion order, DeepFM, DIN, and their MMR variants on the same candidate pool.

Targets are held-out positive items.  This is not a CTR/exposure benchmark;
sampled ranking AUC is reported separately by the Phase 2 training scripts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
from src.evaluation.ablation import (  # noqa: E402
    category_diversity,
    label_rank,
    paired_bootstrap_ci,
)
from src.evaluation.retrieval_metrics import (  # noqa: E402
    aggregate_metrics,
    metrics_from_rank,
)
from src.ranking.deepfm_ranker import DeepFMRanker  # noqa: E402
from src.ranking.reranker import mmr_rerank  # noqa: E402
from src.retrieval.fusion import (  # noqa: E402
    filter_fused_candidates,
    fuse_candidates,
)
from src.services.recommendation_service import RecommendationEngine  # noqa: E402


def _subset_frame(frame: pl.DataFrame, users: int) -> pl.DataFrame:
    if users > 0 and frame.height > users:
        return frame.gather_every(max(1, frame.height // users))
    return frame


def _metric_block(
    per_user: Sequence[dict[str, float]],
    recommended: Sequence[Sequence[str]],
    catalog_size: int,
    *,
    coverage_k: int,
) -> dict[str, Any]:
    aggregate = aggregate_metrics(per_user)
    aggregate["users"] = float(len(per_user))
    seen: set[str] = set()
    for ids in recommended:
        seen.update(str(item_id) for item_id in ids[:coverage_k])
    aggregate[f"coverage@{coverage_k}"] = (
        len(seen) / catalog_size if catalog_size else 0.0
    )
    return aggregate


def _fused_ids(
    engine: RecommendationEngine,
    channel_results: dict[str, list[dict[str, Any]]],
    *,
    strategy: str,
    channels: Sequence[str],
    weights: dict[str, float],
    config: Any,
    user_id: str,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    selected = {
        name: channel_results[name]
        for name in channels
        if name in channel_results
    }
    fused = fuse_candidates(
        selected,
        strategy=strategy,
        quota_per_channel=config.retrieval.fusion.quota_per_channel,
        total_candidates=config.retrieval.fusion.total_candidates,
        rrf_k=config.retrieval.fusion.rrf_k,
        score_normalization=config.retrieval.fusion.score_normalization,
        channel_weights=weights,
    )
    filtered, stats = filter_fused_candidates(
        fused,
        exclude_item_ids=engine.user_exclusions.get(str(user_id), set()),
        item_metadata=engine.item_metadata,
    )
    return [candidate.item_id for candidate in filtered], filtered, stats


def evaluate_retrieval(
    engine: RecommendationEngine,
    frame: pl.DataFrame,
    config: Any,
    *,
    split: str,
) -> dict[str, Any]:
    all_channels = ["itemcf", "two_tower", "lightgcn", "popularity"]
    no_popularity_channels = ["itemcf", "two_tower", "lightgcn"]
    calibrated = {
        name: float(
            config.retrieval.fusion.channel_weights.get(name, 1.0)
        )
        for name in all_channels
    }
    fusion_specs: dict[str, tuple[str, list[str], dict[str, float]]] = {
        "fusion_quota_all": ("quota", all_channels, calibrated),
        "fusion_rrf_all": ("rrf", all_channels, calibrated),
        "fusion_quota_rrf_all": ("quota_rrf", all_channels, calibrated),
        "fusion_quota_no_popularity": (
            "quota",
            no_popularity_channels,
            calibrated,
        ),
        "fusion_quota_equal_weights": (
            "quota",
            all_channels,
            {name: 1.0 for name in all_channels},
        ),
        "fusion_quota_itemcf_two_tower": (
            "quota",
            ["itemcf", "two_tower"],
            calibrated,
        ),
        "fusion_quota_itemcf_lightgcn": (
            "quota",
            ["itemcf", "lightgcn"],
            calibrated,
        ),
        "fusion_quota_two_tower_lightgcn": (
            "quota",
            ["two_tower", "lightgcn"],
            calibrated,
        ),
    }
    channels = [name for name in all_channels if name in engine.retrievers]
    variants = [f"channel_{name}" for name in channels] + list(fusion_specs)
    per_user: dict[str, list[dict[str, float]]] = {
        variant: [] for variant in variants
    }
    recommended: dict[str, list[list[str]]] = {
        variant: [] for variant in variants
    }
    candidate_counts: dict[str, list[int]] = {
        variant: [] for variant in fusion_specs
    }
    catalog_size = int(engine.ranker.encoder.num_items)
    started = time.perf_counter()
    for row in frame.iter_rows(named=True):
        user_id = str(row["user_id"])
        target_id = str(row["parent_asin"])
        channel_results = {
            name: retriever.retrieve(
                user_id, config.retrieval.fusion.quota_per_channel
            )
            for name, retriever in engine.retrievers.items()
        }
        for name in channels:
            ids = [str(result["item_id"]) for result in channel_results[name]]
            rank = label_rank(ids, target_id, catalog_size)
            per_user[f"channel_{name}"].append(metrics_from_rank(rank))
            recommended[f"channel_{name}"].append(ids[:50])
        for variant, (strategy, selected_channels, weights) in fusion_specs.items():
            ids, filtered, _stats = _fused_ids(
                engine,
                channel_results,
                strategy=strategy,
                channels=selected_channels,
                weights=weights,
                config=config,
                user_id=user_id,
            )
            rank = label_rank(ids, target_id, catalog_size)
            per_user[variant].append(metrics_from_rank(rank))
            recommended[variant].append(ids[:50])
            candidate_counts[variant].append(len(filtered))
    output: dict[str, Any] = {}
    for variant in variants:
        block = _metric_block(
            per_user[variant],
            recommended[variant],
            catalog_size,
            coverage_k=50,
        )
        output[variant] = block
        print(
            f"{variant:34s} "
            f"R@10={block.get('recall@10', 0):.5f} "
            f"R@50={block.get('recall@50', 0):.5f} "
            f"NDCG@10={block.get('ndcg@10', 0):.5f} "
            f"MRR={block.get('mrr', 0):.5f} "
            f"Cov@50={block.get('coverage@50', 0):.4f}"
        )
    def _values(variant: str, key: str) -> list[float]:
        return [row[key] for row in per_user[variant]]

    comparisons = {
        "rrf_vs_quota_ndcg@10": paired_bootstrap_ci(
            _values("fusion_quota_all", "ndcg@10"),
            _values("fusion_rrf_all", "ndcg@10"),
        ),
        "rrf_vs_quota_recall@50": paired_bootstrap_ci(
            _values("fusion_quota_all", "recall@50"),
            _values("fusion_rrf_all", "recall@50"),
        ),
        "quota_rrf_vs_quota_ndcg@10": paired_bootstrap_ci(
            _values("fusion_quota_all", "ndcg@10"),
            _values("fusion_quota_rrf_all", "ndcg@10"),
        ),
        "equal_weights_vs_calibrated_ndcg@10": paired_bootstrap_ci(
            _values("fusion_quota_all", "ndcg@10"),
            _values("fusion_quota_equal_weights", "ndcg@10"),
        ),
        "two_tower_lightgcn_vs_all_channels_recall@50": paired_bootstrap_ci(
            _values("fusion_quota_all", "recall@50"),
            _values("fusion_quota_two_tower_lightgcn", "recall@50"),
        ),
        "two_tower_vs_fusion_ndcg@10": paired_bootstrap_ci(
            _values("channel_two_tower", "ndcg@10"),
            _values("fusion_quota_all", "ndcg@10"),
        ),
        "lightgcn_vs_fusion_ndcg@10": paired_bootstrap_ci(
            _values("channel_lightgcn", "ndcg@10"),
            _values("fusion_quota_all", "ndcg@10"),
        ),
        "itemcf_vs_fusion_ndcg@10": paired_bootstrap_ci(
            _values("channel_itemcf", "ndcg@10"),
            _values("fusion_quota_all", "ndcg@10"),
        ),
        "popularity_channel_vs_no_popularity_recall@10": paired_bootstrap_ci(
            _values("fusion_quota_no_popularity", "recall@10"),
            _values("channel_popularity", "recall@10"),
        ),
        "popularity_fusion_vs_no_popularity_recall@10": paired_bootstrap_ci(
            _values("fusion_quota_no_popularity", "recall@10"),
            _values("fusion_quota_all", "recall@10"),
        ),
        "popularity_fusion_vs_no_popularity_recall@50": paired_bootstrap_ci(
            _values("fusion_quota_no_popularity", "recall@50"),
            _values("fusion_quota_all", "recall@50"),
        ),
        "popularity_fusion_vs_no_popularity_ndcg@10": paired_bootstrap_ci(
            _values("fusion_quota_no_popularity", "ndcg@10"),
            _values("fusion_quota_all", "ndcg@10"),
        ),
    }
    output["comparisons"] = comparisons
    output["comparison_pairs"] = {
        "rrf_vs_quota_ndcg@10": [
            "fusion_quota_all",
            "fusion_rrf_all",
            "ndcg@10",
        ],
        "rrf_vs_quota_recall@50": [
            "fusion_quota_all",
            "fusion_rrf_all",
            "recall@50",
        ],
        "quota_rrf_vs_quota_ndcg@10": [
            "fusion_quota_all",
            "fusion_quota_rrf_all",
            "ndcg@10",
        ],
        "equal_weights_vs_calibrated_ndcg@10": [
            "fusion_quota_all",
            "fusion_quota_equal_weights",
            "ndcg@10",
        ],
        "two_tower_lightgcn_vs_all_channels_recall@50": [
            "fusion_quota_all",
            "fusion_quota_two_tower_lightgcn",
            "recall@50",
        ],
        "two_tower_vs_fusion_ndcg@10": [
            "channel_two_tower",
            "fusion_quota_all",
            "ndcg@10",
        ],
        "lightgcn_vs_fusion_ndcg@10": [
            "channel_lightgcn",
            "fusion_quota_all",
            "ndcg@10",
        ],
        "itemcf_vs_fusion_ndcg@10": [
            "channel_itemcf",
            "fusion_quota_all",
            "ndcg@10",
        ],
        "popularity_channel_vs_no_popularity_recall@10": [
            "fusion_quota_no_popularity",
            "channel_popularity",
            "recall@10",
        ],
        "popularity_fusion_vs_no_popularity_recall@10": [
            "fusion_quota_no_popularity",
            "fusion_quota_all",
            "recall@10",
        ],
        "popularity_fusion_vs_no_popularity_recall@50": [
            "fusion_quota_no_popularity",
            "fusion_quota_all",
            "recall@50",
        ],
        "popularity_fusion_vs_no_popularity_ndcg@10": [
            "fusion_quota_no_popularity",
            "fusion_quota_all",
            "ndcg@10",
        ],
    }
    output["comparison_definition"] = (
        "Each comparison is paired treatment - baseline; a positive delta "
        "favors the treatment. comparison_pairs lists [baseline, treatment, metric]."
    )
    output["diagnostics"] = {
        "split": split,
        "users": int(frame.height),
        "catalog_items": catalog_size,
        "channel_counts": {
            name: int(len(per_user[f"channel_{name}"])) for name in channels
        },
        "calibrated_weights": calibrated,
        "strategy_for_ranking": "quota",
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "candidate_counts": {
            name: round(sum(values) / max(1, len(values)), 1)
            for name, values in candidate_counts.items()
        },
        "note": (
            "Held-out positive targets; train history excluded. "
            "Rank absent from a 100-item channel/fusion list is catalog_size+1. "
            "Recall/NDCG/MRR are preference-ranking metrics, not CTR."
        ),
    }
    return output


def _scored_candidates(
    engine: RecommendationEngine,
    filtered: Sequence[Any],
    score_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_by_item = {
        str(row["item_id"]): float(row["ranking_score"]) for row in score_rows
    }
    candidates = []
    for candidate in filtered:
        if candidate.item_id not in score_by_item:
            continue
        metadata = engine.item_metadata.get(candidate.item_id, {})
        candidates.append(
            {
                "item_id": candidate.item_id,
                "ranking_score": score_by_item[candidate.item_id],
                "category": metadata.get("main_category", ""),
                "store": metadata.get("store", ""),
                "title": metadata.get("title", ""),
                "price": metadata.get("price"),
            }
        )
    candidates.sort(key=lambda item: item["ranking_score"], reverse=True)
    return candidates


def evaluate_ranking(
    engine: RecommendationEngine,
    frame: pl.DataFrame,
    config: Any,
    *,
    split: str,
    deepfm: DeepFMRanker,
) -> dict[str, Any]:
    variants = ["fusion", "deepfm", "din", "deepfm_mmr", "din_mmr"]
    per_user: dict[str, list[dict[str, float]]] = {
        variant: [] for variant in variants
    }
    recommended: dict[str, list[list[str]]] = {
        variant: [] for variant in variants
    }
    diversity: dict[str, list[float]] = {variant: [] for variant in variants}
    retrieval_ms: list[float] = []
    deepfm_ms: list[float] = []
    din_ms: list[float] = []
    mmr_ms: list[float] = []
    catalog_size = int(engine.ranker.encoder.num_items)
    top_k = int(config.ranking.rerank.top_k)
    all_channels = list(engine.retrievers)
    weights = {
        name: float(config.retrieval.fusion.channel_weights.get(name, 1.0))
        for name in all_channels
    }
    started = time.perf_counter()
    for row in frame.iter_rows(named=True):
        user_id = str(row["user_id"])
        target_id = str(row["parent_asin"])
        retrieval_started = time.perf_counter()
        channel_results = {
            name: retriever.retrieve(
                user_id, config.retrieval.fusion.quota_per_channel
            )
            for name, retriever in engine.retrievers.items()
        }
        ids, filtered, _stats = _fused_ids(
            engine,
            channel_results,
            strategy="quota",
            channels=all_channels,
            weights=weights,
            config=config,
            user_id=user_id,
        )
        retrieval_ms.append((time.perf_counter() - retrieval_started) * 1000)
        fusion_ids = ids[:top_k]

        din_started = time.perf_counter()
        din_rows = engine.ranker.score(
            user_id, [candidate.item_id for candidate in filtered]
        )
        din_candidates = _scored_candidates(engine, filtered, din_rows)
        din_ids = [item["item_id"] for item in din_candidates][:top_k]
        din_ms.append((time.perf_counter() - din_started) * 1000)

        deepfm_started = time.perf_counter()
        deepfm_rows = deepfm.score(
            user_id, [candidate.item_id for candidate in filtered]
        )
        deepfm_candidates = _scored_candidates(
            engine, filtered, deepfm_rows
        )
        deepfm_ids = [item["item_id"] for item in deepfm_candidates][:top_k]
        deepfm_ms.append((time.perf_counter() - deepfm_started) * 1000)

        mmr_started = time.perf_counter()
        din_mmr = mmr_rerank(
            din_candidates,
            top_k=top_k,
            mmr_lambda=config.ranking.rerank.mmr_lambda,
            max_per_category=config.ranking.rerank.max_per_category,
            max_per_brand=config.ranking.rerank.max_per_brand,
            similarity=config.ranking.rerank.similarity,
            enable_mmr=True,
        )
        deepfm_mmr = mmr_rerank(
            deepfm_candidates,
            top_k=top_k,
            mmr_lambda=config.ranking.rerank.mmr_lambda,
            max_per_category=config.ranking.rerank.max_per_category,
            max_per_brand=config.ranking.rerank.max_per_brand,
            similarity=config.ranking.rerank.similarity,
            enable_mmr=True,
        )
        mmr_ms.append((time.perf_counter() - mmr_started) * 1000)

        ids_by_variant = {
            "fusion": fusion_ids,
            "deepfm": deepfm_ids,
            "din": din_ids,
            "deepfm_mmr": [item["item_id"] for item in deepfm_mmr],
            "din_mmr": [item["item_id"] for item in din_mmr],
        }
        for variant, variant_ids in ids_by_variant.items():
            rank = label_rank(variant_ids, target_id, catalog_size)
            per_user[variant].append(metrics_from_rank(rank, (top_k,)))
            recommended[variant].append(variant_ids)
            diversity[variant].append(
                category_diversity(variant_ids, engine.item_metadata)
            )

    output: dict[str, Any] = {}
    for variant in variants:
        block = _metric_block(
            per_user[variant],
            recommended[variant],
            catalog_size,
            coverage_k=top_k,
        )
        block["category_diversity@10"] = float(
            sum(diversity[variant]) / max(1, len(diversity[variant]))
        )
        output[variant] = block
        print(
            f"{variant:12s} "
            f"R@10={block.get('recall@10', 0):.5f} "
            f"NDCG@10={block.get('ndcg@10', 0):.5f} "
            f"MRR={block.get('mrr', 0):.5f} "
            f"Cov@10={block.get('coverage@10', 0):.4f} "
            f"Div@10={block.get('category_diversity@10', 0):.4f}"
        )
    output["comparisons"] = {
        "deepfm_vs_din_recall@10": paired_bootstrap_ci(
            [row["recall@10"] for row in per_user["din"]],
            [row["recall@10"] for row in per_user["deepfm"]],
        ),
        "din_vs_fusion_recall@10": paired_bootstrap_ci(
            [row["recall@10"] for row in per_user["fusion"]],
            [row["recall@10"] for row in per_user["din"]],
        ),
        "din_mmr_vs_din_recall@10": paired_bootstrap_ci(
            [row["recall@10"] for row in per_user["din"]],
            [row["recall@10"] for row in per_user["din_mmr"]],
        ),
        "din_mmr_vs_din_diversity@10": paired_bootstrap_ci(
            diversity["din"],
            diversity["din_mmr"],
        ),
        "deepfm_mmr_vs_deepfm_recall@10": paired_bootstrap_ci(
            [row["recall@10"] for row in per_user["deepfm"]],
            [row["recall@10"] for row in per_user["deepfm_mmr"]],
        ),
    }
    output["comparison_pairs"] = {
        "deepfm_vs_din_recall@10": ["din", "deepfm", "recall@10"],
        "din_vs_fusion_recall@10": ["fusion", "din", "recall@10"],
        "din_mmr_vs_din_recall@10": ["din", "din_mmr", "recall@10"],
        "din_mmr_vs_din_diversity@10": [
            "din",
            "din_mmr",
            "category_diversity@10",
        ],
        "deepfm_mmr_vs_deepfm_recall@10": [
            "deepfm",
            "deepfm_mmr",
            "recall@10",
        ],
    }
    output["comparison_definition"] = (
        "Each comparison is paired treatment - baseline; a positive delta "
        "favors the treatment. comparison_pairs lists [baseline, treatment, metric]."
    )
    output["diagnostics"] = {
        "split": split,
        "users": int(frame.height),
        "catalog_items": catalog_size,
        "top_k": top_k,
        "strategy": "quota",
        "channel_weights": weights,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "avg_retrieval_ms": round(sum(retrieval_ms) / max(1, len(retrieval_ms)), 2),
        "avg_deepfm_ms": round(sum(deepfm_ms) / max(1, len(deepfm_ms)), 2),
        "avg_din_ms": round(sum(din_ms) / max(1, len(din_ms)), 2),
        "avg_mmr_ms": round(sum(mmr_ms) / max(1, len(mmr_ms)), 2),
        "note": (
            "Same candidate pool for fusion/DeepFM/DIN/MMR. "
            "Held-out positives; train history excluded. "
            "Ranking scores are model preference scores, not CTR."
        ),
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7 offline ablation")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument(
        "--users", type=int, default=0, help="retrieval users; 0 = all"
    )
    parser.add_argument(
        "--ranking-users",
        type=int,
        default=0,
        help="ranking ablation users; 0 = all",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    root = config.project_root
    engine = RecommendationEngine.from_artifacts(config)
    train = pl.read_parquet(root / "data/splits/train.parquet")
    deepfm = DeepFMRanker.from_artifacts(root, config, train)
    retrieval_frame = _subset_frame(
        pl.read_parquet(root / f"data/splits/{args.split}_eval.parquet"),
        args.users,
    )
    ranking_frame = _subset_frame(
        pl.read_parquet(root / f"data/splits/{args.split}_eval.parquet"),
        args.ranking_users,
    )
    started = time.perf_counter()
    print(
        f"Phase 7 ablation split={args.split} "
        f"retrieval_users={retrieval_frame.height} "
        f"ranking_users={ranking_frame.height}"
    )
    retrieval = evaluate_retrieval(
        engine, retrieval_frame, config, split=args.split
    )
    ranking = evaluate_ranking(
        engine, ranking_frame, config, split=args.split, deepfm=deepfm
    )
    output = {
        "phase": 7,
        "split": args.split,
        "retrieval": retrieval,
        "ranking": ranking,
        "diagnostics": {
            "retrieval_users": int(retrieval_frame.height),
            "ranking_users": int(ranking_frame.height),
            "runtime_seconds": round(time.perf_counter() - started, 1),
            "seed": int(config.model.seed),
            "note": (
                "All values are computed from real held-out positives and real "
                "saved model artifacts. No synthetic improvement or CTR claim."
            ),
        },
    }
    output_path = Path(
        args.output or root / f"logs/phase7_ablation_{args.split}.json"
    )
    if not output_path.is_absolute():
        output_path = root / output_path
    save_json(output, output_path)
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
