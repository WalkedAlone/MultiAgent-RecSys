#!/usr/bin/env python
"""Build ItemCF similarity and evaluate it on held-out positive targets."""

from __future__ import annotations

import argparse
import json
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
from src.core.training import save_json, set_seed  # noqa: E402
from src.evaluation.retrieval_metrics import (  # noqa: E402
    DEFAULT_K_VALUES,
    aggregate_metrics,
    coverage_at_k,
    metrics_from_rank,
)
from src.retrieval.itemcf import ItemCFRetriever  # noqa: E402


def evaluate_itemcf(
    retriever: ItemCFRetriever,
    eval_frame: pl.DataFrame,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict:
    """Evaluate the served ItemCF list, not optimistic zero-score ties.

    ``ItemCFRetriever.retrieve`` only returns positive-similarity items.  A
    target with zero similarity is therefore absent from the list and ranks
    after the catalog, rather than being counted as rank 1 by a full-array
    tie-count (the legacy Phase 2 bug, kept in ``*_legacy.json`` for audit).
    """
    started = time.perf_counter()
    per_user: list[dict[str, float]] = []
    recommended: list[list[int]] = []
    skipped = 0
    max_k = max(k_values)
    for row in eval_frame.iter_rows(named=True):
        user_id = str(row["user_id"])
        target_idx = int(row["item_idx"])
        target_id = str(row["parent_asin"])
        if target_idx <= 0 or target_idx > retriever.num_items:
            skipped += 1
            continue
        results = retriever.retrieve(user_id, max_k)
        item_ids = [result["item_id"] for result in results]
        if target_id in item_ids:
            rank = item_ids.index(target_id) + 1
        else:
            rank = retriever.num_items + 1
        per_user.append(metrics_from_rank(rank, k_values))
        recommended.append(
            [
                int(retriever.item_id_to_idx[item_id])
                for item_id in item_ids[:50]
                if item_id in retriever.item_id_to_idx
            ]
        )
    metrics = aggregate_metrics(per_user)
    metrics["users"] = float(len(per_user))
    metrics["skipped"] = float(skipped)
    metrics["coverage@50"] = coverage_at_k(
        recommended, retriever.num_items, 50
    )
    metrics["catalog_items"] = float(retriever.num_items)
    metrics["runtime_seconds"] = round(time.perf_counter() - started, 1)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate ItemCF")
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument(
        "--eval-split", choices=["valid", "test", "both"], default="both"
    )
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    set_seed(config.model.seed)
    root = config.project_root
    train = pl.read_parquet(root / "data/splits/train.parquet")
    user_map = pl.read_parquet(root / "data/processed/users/user_id_map.parquet")
    item_map = pl.read_parquet(root / "data/processed/items/item_id_map.parquet")
    print(
        f"ItemCF: train_rows={train.height} users={user_map.height} "
        f"items={item_map.height} neighbors={args.neighbors}"
    )
    started = time.perf_counter()
    retriever = ItemCFRetriever.fit(
        train, user_map, item_map, neighbors=args.neighbors
    )
    build_seconds = round(time.perf_counter() - started, 1)
    similarity_path = root / "checkpoints/itemcf_similarity.npz"
    meta_path = root / "checkpoints/itemcf_meta.json"
    retriever.save(similarity_path, meta_path)
    print(
        f"ItemCF built in {build_seconds}s; similarity_nnz="
        f"{retriever.similarity.nnz}; saved={similarity_path}"
    )

    split_names = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]
    all_metrics: dict[str, dict] = {}
    for split in split_names:
        eval_frame = pl.read_parquet(root / f"data/splits/{split}_eval.parquet")
        metrics = evaluate_itemcf(retriever, eval_frame)
        all_metrics[split] = metrics
        print(f"\n[{split}] ItemCF metrics:")
        for key in (
            "users",
            "recall@10",
            "recall@20",
            "recall@50",
            "recall@100",
            "ndcg@10",
            "ndcg@20",
            "ndcg@50",
            "hit_rate@10",
            "mrr",
            "coverage@50",
            "runtime_seconds",
        ):
            print(f"  {key}: {metrics.get(key)}")
        save_json(
            metrics,
            root / f"logs/metrics_itemcf_{split}.json",
        )
    save_json(
        {
            "model": "itemcf",
            "build_seconds": build_seconds,
            "neighbors": args.neighbors,
            "similarity_nnz": int(retriever.similarity.nnz),
            "metrics": all_metrics,
        },
        root / "logs/phase2_itemcf.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
