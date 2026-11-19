#!/usr/bin/env python
"""Recover reviews.parquet from completed per-bucket part files.

Use when the bucketed finalizer was stopped during the final combine but all
``.finalize/reviews_part_*.parquet`` files have valid Parquet footers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.core.config import get_app_config  # noqa: E402
from src.core.exceptions import DataValidationError, RecSysError  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.data.loader import configure_duckdb, sql_path, write_json_report  # noqa: E402

_BUCKET_RE = re.compile(
    r"reviews_bucket_done index=(\d+) rows=(\d+) keep=(\d+) dedup=(\d+)"
)


def _log_sums() -> dict[str, int] | None:
    log_path = PROJECT_ROOT / "logs" / "app.log"
    if not log_path.is_file():
        return None
    matches = _BUCKET_RE.findall(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if not matches:
        return None
    # Use the last complete 16-bucket run.
    selected = matches[-16:]
    return {
        "bucket_rows": sum(int(row[1]) for row in selected),
        "keep_rows": sum(int(row[2]) for row in selected),
        "dedup_rows": sum(int(row[3]) for row in selected),
        "buckets": len(selected),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover reviews.parquet from bucket part files"
    )
    parser.add_argument(
        "--finalize-dir",
        default=None,
        help="Default: data/processed/reviews/.finalize",
    )
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    reviews_dir = config.resolve_path(config.data.paths.processed) / "reviews"
    finalize_dir = (
        Path(args.finalize_dir)
        if args.finalize_dir
        else reviews_dir / ".finalize"
    )
    parts = sorted(finalize_dir.glob("reviews_part_*.parquet"))
    if not parts:
        print(f"FAILED: no part files under {finalize_dir}")
        return 1
    part_rows = 0
    for part in parts:
        parquet = pq.ParquetFile(part)
        part_rows += int(parquet.metadata.num_rows)
    print(f"validated {len(parts)} parts, rows={part_rows}")

    scans = sorted(reviews_dir.glob(".reviews_filtered*.parquet"))
    temp_path = scans[-1] if scans else None
    matched_rows = (
        int(pq.ParquetFile(temp_path).metadata.num_rows) if temp_path else None
    )
    bucket_sums = _log_sums()

    reviews_path = reviews_dir / "reviews.parquet"
    item_stats_path = reviews_dir / "item_review_stats.parquet"
    reviews_path.unlink(missing_ok=True)
    item_stats_path.unlink(missing_ok=True)

    started = time.perf_counter()
    connection = duckdb.connect()
    try:
        configure_duckdb(
            connection,
            temp_dir=config.resolve_path(config.data.paths.processed)
            / ".duckdb_tmp",
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        connection.execute(
            f"COPY (SELECT * FROM read_parquet({sql_path(finalize_dir / 'reviews_part_*.parquet')})) "
            f"TO {sql_path(reviews_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        kept_rows, items_with_reviews = connection.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT parent_asin) "
            f"FROM read_parquet({sql_path(reviews_path)})"
        ).fetchone()
        connection.execute(
            f"COPY ("
            f"  SELECT parent_asin, COUNT(*) AS raw_review_count, "
            f"         AVG(rating) AS raw_avg_rating, "
            f"         AVG(CAST(verified_purchase AS INTEGER)) AS verified_ratio, "
            f"         AVG(helpful_vote) AS mean_helpful, "
            f"         MAX(timestamp_ms) AS latest_review_ts "
            f"  FROM read_parquet({sql_path(reviews_path)}) "
            f"  GROUP BY parent_asin"
            f") TO {sql_path(item_stats_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()

    report_path = (
        config.resolve_path(config.data.paths.processed) / "dataset_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    catalog_items = int(report.get("catalog", {}).get("unique_items", 0))
    scanned_records = None
    manifest_path = config.project_root / "data" / "raw" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest.get("files", {}).get("reviews_electronics")
        if record:
            scanned_records = record.get("line_count")

    duplicates = None
    capped_out = None
    if bucket_sums:
        duplicates = bucket_sums["keep_rows"] - bucket_sums["dedup_rows"]
        capped_out = bucket_sums["bucket_rows"] - bucket_sums["keep_rows"]

    stats: dict[str, Any] = {
        "status": "ok",
        "source": "recovered_bucket_parts",
        "scanned_records": scanned_records,
        "matched_records": matched_rows,
        "per_item_cap": (
            config.data.sample.max_reviews_per_item
            if config.data.mode == "sample"
            else config.data.full.max_reviews_per_item
        ),
        "bounded_stream": False,
        "catalog_items": catalog_items,
        "coverage": (
            int(items_with_reviews) / catalog_items if catalog_items else 0.0
        ),
        "duplicates_removed": duplicates,
        "capped_out": capped_out,
        "kept_reviews": int(kept_rows),
        "items_with_reviews": int(items_with_reviews),
        "bucket_count": len(parts),
        "note": (
            "Final file rebuilt from 16 valid bucket parts after the host was "
            "stopped during the final combine. Scan temp and bucket files were "
            "not rescanned."
        ),
    }
    report["reviews"] = stats
    report["reviews_only_run"] = {
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "recovered_from_parts": True,
    }
    write_json_report(report, report_path)

    cache_key = {
        "mode": config.data.mode,
        "raw_size": (config.project_root / "data/raw/reviews/Electronics.jsonl.gz").stat().st_size
        if (config.project_root / "data/raw/reviews/Electronics.jsonl.gz").is_file()
        else None,
        "raw_mtime_ns": (config.project_root / "data/raw/reviews/Electronics.jsonl.gz").stat().st_mtime_ns
        if (config.project_root / "data/raw/reviews/Electronics.jsonl.gz").is_file()
        else None,
        "item_set_sha256": report.get("catalog", {}).get("item_set_sha256"),
        "max_reviews_per_item": stats["per_item_cap"],
        "max_records": None,
    }
    (reviews_dir / ".reviews_cache.json").write_text(
        json.dumps({"cache_key": cache_key, "stats": stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
