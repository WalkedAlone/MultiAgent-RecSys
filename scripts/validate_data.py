#!/usr/bin/env python
"""Standalone integrity checks for Phase 1 artifacts.

Examples:
    python scripts/validate_data.py
    python scripts/validate_data.py --allow-missing-raw-derived
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

from src.core.config import get_app_config  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.data.loader import configure_duckdb, sql_path  # noqa: E402


class Validator:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.checks.append((name, bool(condition), detail))

    def summary(self) -> int:
        passed = sum(1 for _, ok, _ in self.checks if ok)
        failed = len(self.checks) - passed
        print(f"checks: {len(self.checks)}  passed={passed}  failed={failed}")
        for name, ok, detail in self.checks:
            status = "PASS" if ok else "FAIL"
            suffix = f"  ({detail})" if detail else ""
            print(f"  {status}  {name}{suffix}")
        return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1 artifacts")
    parser.add_argument(
        "--allow-missing-raw-derived",
        action="store_true",
        help="Do not fail when metadata/reviews are not processed yet",
    )
    args = parser.parse_args(argv)
    setup_logging(level="INFO")

    config = get_app_config()
    root = config.project_root
    processed = config.resolve_path(config.data.paths.processed)
    splits_dir = config.resolve_path(config.data.paths.splits)
    report_path = processed / "dataset_report.json"
    validator = Validator()

    validator.check("dataset_report exists", report_path.is_file())
    if not report_path.is_file():
        return validator.summary()
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))

    split_frames: dict[str, pl.DataFrame] = {}
    for split in ("train", "valid", "test"):
        path = splits_dir / f"{split}.parquet"
        exists = path.is_file()
        validator.check(f"{split}.parquet exists", exists)
        if exists:
            frame = pl.read_parquet(path)
            split_frames[split] = frame
            expected = report.get("splits", {}).get(split, {}).get("rows")
            validator.check(
                f"{split} row count matches report",
                frame.height == expected,
                f"{frame.height} vs {expected}",
            )

    if "train" not in split_frames:
        return validator.summary()
    train = split_frames["train"]
    valid = split_frames.get("valid", pl.DataFrame())
    test = split_frames.get("test", pl.DataFrame())

    required_columns = {
        "user_id",
        "user_idx",
        "parent_asin",
        "item_idx",
        "split",
        "rating",
        "label",
        "is_positive",
        "is_negative",
        "is_neutral",
        "timestamp_ms",
        "history_items",
        "history_item_idxs",
        "can_score",
        "is_scorable_positive",
        "is_sampled_negative",
    }
    validator.check(
        "interaction columns present",
        required_columns.issubset(set(train.columns)),
        f"missing={sorted(required_columns - set(train.columns))}",
    )

    if required_columns.issubset(set(train.columns)):
        validator.check(
            "train items are all scorable",
            bool(train["can_score"].all()),
        )
        validator.check(
            "scorable flag matches item_idx",
            bool(
                (train["can_score"] == (train["item_idx"] > 0)).all()
            ),
        )
        neutral = train.filter(pl.col("is_neutral"))
        validator.check(
            "neutral labels are null",
            neutral["label"].null_count() == neutral.height,
            f"{neutral['label'].null_count()}/{neutral.height}",
        )
        non_null_labels = set(
            train.filter(pl.col("label").is_not_null())["label"].unique().to_list()
        )
        validator.check(
            "labels only 0/1",
            non_null_labels.issubset({0, 1}),
            f"values={non_null_labels}",
        )
        validator.check(
            "no target in train history",
            train.filter(
                pl.col("history_items").list.contains(pl.col("parent_asin"))
            ).height
            == 0,
        )
        history_max = train["history_item_idxs"].list.max().max()
        validator.check(
            "history indices within catalog",
            history_max is not None
            and int(history_max)
            <= int(pl.read_parquet(processed / "items/item_id_map.parquet")["item_idx"].max()),
            f"max_history_idx={history_max}",
        )

    user_map_path = processed / "users/user_id_map.parquet"
    item_map_path = processed / "items/item_id_map.parquet"
    validator.check("user_id_map exists", user_map_path.is_file())
    validator.check("item_id_map exists", item_map_path.is_file())
    if user_map_path.is_file() and item_map_path.is_file():
        user_map = pl.read_parquet(user_map_path)
        item_map = pl.read_parquet(item_map_path)
        validator.check(
            "user map unique",
            user_map["user_id"].n_unique() == user_map.height,
        )
        validator.check(
            "item map unique",
            item_map["parent_asin"].n_unique() == item_map.height,
        )
        validator.check(
            "user indices contiguous 1..N",
            user_map["user_idx"].min() == 1
            and user_map["user_idx"].max() == user_map.height,
        )
        validator.check(
            "item indices contiguous 1..M",
            item_map["item_idx"].min() == 1
            and item_map["item_idx"].max() == item_map.height,
        )

    for split_name, frame in (("valid", valid), ("test", test)):
        if frame.height == 0:
            continue
        eval_path = splits_dir / f"{split_name}_eval.parquet"
        validator.check(f"{split_name}_eval exists", eval_path.is_file())
        if eval_path.is_file():
            eval_frame = pl.read_parquet(eval_path)
            validator.check(
                f"{split_name}_eval all scorable positive",
                bool(eval_frame["is_scorable_positive"].all()),
                f"rows={eval_frame.height}",
            )
        validator.check(
            f"{split_name} no target in history",
            frame.filter(
                pl.col("history_items").list.contains(pl.col("parent_asin"))
            ).height
            == 0,
        )

    samples_path = processed / "interactions/train_samples.parquet"
    validator.check("train_samples exists", samples_path.is_file())
    if samples_path.is_file():
        samples = pl.read_parquet(samples_path)
        validator.check(
            "train_samples has no neutral rows",
            samples.filter(pl.col("sample_type") == "neutral").height == 0,
        )
        validator.check(
            "sampled negatives labeled 0",
            samples.filter(pl.col("is_sampled_negative"))
            .filter(pl.col("label") != 0)
            .height
            == 0,
        )
        validator.check(
            "sampled negatives not padding index",
            samples.filter(pl.col("is_sampled_negative"))
            .filter(pl.col("item_idx") <= 0)
            .height
            == 0,
        )
        user_items = {
            row["user_id"]: set(row["item_idx"])
            for row in train.group_by("user_id")
            .agg(pl.col("item_idx").unique().alias("item_idx"))
            .iter_rows(named=True)
        }
        overlap = sum(
            1
            for row in samples.filter(pl.col("is_sampled_negative")).iter_rows(
                named=True
            )
            if row["item_idx"] in user_items.get(row["user_id"], set())
        )
        validator.check(
            "sampled negatives exclude user train items",
            overlap == 0,
            f"overlap={overlap}",
        )

    sequences_path = processed / "sequences/user_sequences.parquet"
    validator.check("user_sequences exists", sequences_path.is_file())
    if sequences_path.is_file():
        sequences = pl.read_parquet(sequences_path)
        validator.check(
            "one sequence row per user",
            sequences["user_id"].n_unique() == sequences.height,
        )
        validator.check(
            "valid target not in valid context",
            sequences.filter(
                pl.col("valid_target_items").list.set_intersection(
                    pl.col("valid_history_items")
                ).list.len()
                > 0
            ).height
            == 0,
        )
        validator.check(
            "test target not in test context",
            sequences.filter(
                pl.col("test_target_items").list.set_intersection(
                    pl.col("test_history_items")
                ).list.len()
                > 0
            ).height
            == 0,
        )

    metadata_path = processed / "items/item_metadata.parquet"
    reviews_path = processed / "reviews/reviews.parquet"
    if metadata_path.is_file():
        metadata = pl.read_parquet(metadata_path)
        validator.check(
            "metadata keys unique",
            metadata["parent_asin"].n_unique() == metadata.height,
        )
        validator.check(
            "metadata has item text",
            "text_for_embedding" in metadata.columns,
        )
    else:
        validator.check(
            "metadata processed",
            args.allow_missing_raw_derived,
            "missing item_metadata.parquet",
        )

    if reviews_path.is_file():
        cap = None
        if report.get("mode") == "sample":
            cap = config.data.sample.max_reviews_per_item
        else:
            cap = config.data.full.max_reviews_per_item
        # Aggregate in DuckDB; never load millions of review texts into Polars.
        connection = duckdb.connect()
        try:
            configure_duckdb(
                connection,
                temp_dir=processed / ".duckdb_tmp",
                memory_limit="1GB",
                threads=2,
            )
            total_reviews = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({sql_path(reviews_path)})"
            ).fetchone()[0]
            null_parent = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({sql_path(reviews_path)}) "
                f"WHERE parent_asin IS NULL"
            ).fetchone()[0]
            max_per_item = connection.execute(
                f"SELECT MAX(c) FROM ("
                f"  SELECT COUNT(*) AS c FROM read_parquet({sql_path(reviews_path)}) "
                f"  GROUP BY parent_asin"
                f")"
            ).fetchone()[0]
        finally:
            connection.close()
        if cap is not None and total_reviews:
            validator.check(
                "review per-item cap respected",
                int(max_per_item) <= cap,
                f"max={max_per_item} cap={cap}",
            )
        validator.check(
            "reviews have source parent_asin",
            int(null_parent) == 0,
            f"null_parent={null_parent}",
        )
    else:
        validator.check(
            "reviews processed",
            args.allow_missing_raw_derived,
            "missing reviews.parquet",
        )

    manifest_path = root / "data/raw/manifest.json"
    validator.check("download manifest exists", manifest_path.is_file())
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        for name, record in files.items():
            path = root / record["path"]
            validator.check(
                f"manifest {name} size matches",
                bool(
                    path.is_file()
                    and path.stat().st_size == record["size_bytes"]
                ),
            )

    return validator.summary()


if __name__ == "__main__":
    raise SystemExit(main())
