"""Evaluate a non-personalised popularity baseline on the held-out splits.

The recommender project previously reported Recall/NDCG for ItemCF,
Two-Tower, LightGCN and DIN, but not for a trivial popularity baseline.  This
script uses the same ``valid_eval.parquet`` / ``test_eval.parquet`` files and
the same rule (exclude every train interaction of the target user) so the
numbers are directly comparable with ``logs/metrics_*`` and
``logs/phase3_pipeline_*``.

Outputs:
    logs/metrics_popularity_valid.json
    logs/metrics_popularity_test.json

Run:
    python scripts/evaluate_popularity.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _popularity_order(train: pl.DataFrame) -> list[str]:
    counts = (
        train.group_by("parent_asin")
        .agg(pl.len().alias("interactions"))
        .sort(["interactions", "parent_asin"], descending=[True, False])
    )
    return counts["parent_asin"].to_list()


def _user_history(train: pl.DataFrame) -> dict[str, set[str]]:
    grouped = (
        train.group_by("user_id")
        .agg(pl.col("parent_asin").unique().alias("items"))
    )
    return {
        str(row["user_id"]): {str(item) for item in row["items"]}
        for row in grouped.iter_rows(named=True)
    }


def evaluate(
    *,
    train: pl.DataFrame,
    targets: pl.DataFrame,
    popularity: list[str],
    split: str,
) -> dict[str, float | int | str]:
    started = time.perf_counter()
    history = _user_history(train)
    catalog_items = int(train["parent_asin"].n_unique())
    hits_at_10 = 0
    hits_at_50 = 0
    ndcg_at_10 = 0.0
    recommended_at_50: set[str] = set()
    random_recall_10 = 0.0
    random_recall_50 = 0.0
    users = 0

    # Filtering the global order per user is enough for a top-50 list; no
    # user has more than a few hundred train interactions in this sample.
    for row in targets.iter_rows(named=True):
        user_id = str(row["user_id"])
        target = str(row["parent_asin"])
        seen = history.get(user_id, set())
        recommendations: list[str] = []
        for item_id in popularity:
            if item_id in seen:
                continue
            recommendations.append(item_id)
            if len(recommendations) >= 50:
                break
        users += 1
        recommended_at_50.update(recommendations)
        top_10 = recommendations[:10]
        if target in top_10:
            hits_at_10 += 1
            rank = top_10.index(target) + 1
            ndcg_at_10 += 1.0 / math.log2(rank + 1)
        if target in recommendations:
            hits_at_50 += 1

        available = max(1, catalog_items - len(seen))
        random_recall_10 += min(10, available) / available
        random_recall_50 += min(50, available) / available

    runtime = round(time.perf_counter() - started, 2)
    return {
        "method": "popularity",
        "split": split,
        "users": users,
        "catalog_items": catalog_items,
        "recall@10": round(hits_at_10 / users, 6) if users else 0.0,
        "ndcg@10": round(ndcg_at_10 / users, 6) if users else 0.0,
        "recall@50": round(hits_at_50 / users, 6) if users else 0.0,
        "coverage@50": round(len(recommended_at_50) / catalog_items, 6),
        "random_recall@10": round(random_recall_10 / users, 6) if users else 0.0,
        "random_recall@50": round(random_recall_50 / users, 6) if users else 0.0,
        "runtime_seconds": runtime,
        "note": (
            "Non-personalised train-popularity baseline, same held-out "
            "positive targets and train-interaction exclusions as the model "
            "metrics. Not CTR; one positive per user means Recall@K = HitRate@K."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        default=str(PROJECT_ROOT / "data/splits/train.parquet"),
    )
    parser.add_argument(
        "--valid",
        default=str(PROJECT_ROOT / "data/splits/valid_eval.parquet"),
    )
    parser.add_argument(
        "--test",
        default=str(PROJECT_ROOT / "data/splits/test_eval.parquet"),
    )
    parser.add_argument("--logs-dir", default=str(PROJECT_ROOT / "logs"))
    args = parser.parse_args()

    train = pl.read_parquet(args.train)
    popularity = _popularity_order(train)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for split, path in (("valid", args.valid), ("test", args.test)):
        targets = pl.read_parquet(path)
        result = evaluate(
            train=train,
            targets=targets,
            popularity=popularity,
            split=split,
        )
        output = logs_dir / f"metrics_popularity_{split}.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
