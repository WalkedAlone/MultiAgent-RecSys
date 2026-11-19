#!/usr/bin/env python
"""Finalize reviews from a completed scan temp without re-reading 44M lines.

This is a recovery path for memory-constrained hosts. It requires:
* data/processed/reviews/.reviews_filtered.parquet (valid Parquet footer)
* data/processed/items/item_id_map.parquet
* data/processed/dataset_report.json

It writes data/processed/reviews/reviews.parquet and item_review_stats.parquet,
updates dataset_report.json, and writes the reviews cache marker.
"""

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

from src.core.config import get_app_config  # noqa: E402
from src.core.exceptions import RecSysError  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.data.preprocess import DataPreprocessor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize reviews from a completed scan temp"
    )
    parser.add_argument(
        "--temp",
        default=None,
        help="Scan temp path (default: data/processed/reviews/.reviews_filtered.parquet)",
    )
    parser.add_argument(
        "--memory-limit",
        default="2GB",
        help="DuckDB memory limit (default: 2GB)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="DuckDB threads (default: 1, lower memory)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    config = get_app_config()
    print("Finalize reviews from cached scan temp")
    print(f"project_root={PROJECT_ROOT}")
    print(f"memory_limit={args.memory_limit} threads={args.threads}")
    try:
        stats = DataPreprocessor(config).finalize_reviews_from_temp(
            args.temp,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
    except RecSysError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        if args.verbose:
            traceback.print_exc()
        return 1
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"FAILED (unexpected): {type(exc).__name__}: {exc}")
        if args.verbose:
            traceback.print_exc()
        return 1

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
