#!/usr/bin/env python
"""Run the Phase 1 data preprocessing pipeline.

Examples:
    python scripts/preprocess_data.py --mode sample
    python scripts/preprocess_data.py --mode sample --skip-reviews
    python scripts/preprocess_data.py --mode full --force
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig, get_app_config  # noqa: E402
from src.core.exceptions import RecSysError  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.data.preprocess import DataPreprocessor  # noqa: E402


def with_mode(config: AppConfig, mode: str) -> AppConfig:
    """Return a copy of the config with sample/full mode toggles applied."""
    if mode == "config":
        return config
    data = config.data.model_copy(
        update={
            "mode": mode,
            "sample": config.data.sample.model_copy(
                update={"enabled": mode == "sample"}
            ),
            "full": config.data.full.model_copy(
                update={"enabled": mode == "full"}
            ),
        }
    )
    return config.model_copy(update={"data": data})


def _print_summary(report: dict) -> None:
    print("\n=== Phase 1 preprocessing summary ===")
    print(f"mode={report.get('mode')}  runtime_s={report.get('runtime_seconds')}")
    for split in ("train", "valid", "test"):
        stats = report.get("splits", {}).get(split, {})
        print(
            f"{split:5s} rows={stats.get('rows')} users={stats.get('users')} "
            f"items={stats.get('items')} pos={stats.get('positives')} "
            f"neg={stats.get('negatives')} neutral={stats.get('neutrals')}"
        )
    oov = report.get("id_mapping", {}).get("oov", {})
    for split in ("valid", "test"):
        stats = oov.get(split, {})
        print(
            f"{split:5s} OOV: unique_item_rate="
            f"{stats.get('unique_item_oov_rate', 0):.4f} "
            f"interaction_item_rate="
            f"{stats.get('interaction_item_oov_rate', 0):.4f} "
            f"history_occurrence_rate={stats.get('history_oov_rate', 0):.4f}"
        )
    validation = report.get("split_validation", {})
    print(
        "split time violations: train->valid="
        f"{validation.get('train_valid_time_violations')} "
        f"valid->test={validation.get('valid_test_time_violations')}"
    )
    metadata = report.get("metadata", {})
    print(
        f"metadata: status={metadata.get('status')} "
        f"coverage={metadata.get('coverage')} "
        f"items={metadata.get('items_kept')} "
        f"price_missing={metadata.get('price_missing')}"
    )
    reviews = report.get("reviews", {})
    print(
        f"reviews: status={reviews.get('status')} "
        f"kept={reviews.get('kept_reviews')} "
        f"items={reviews.get('items_with_reviews')} "
        f"coverage={reviews.get('coverage')} "
        f"cap={reviews.get('per_item_cap')} "
        f"bounded={reviews.get('bounded_stream')}"
    )
    negative = report.get("negative_sampling", {})
    print(
        f"negative sampling: {negative.get('counts')} "
        f"exclusion={negative.get('exclusion_policy')}"
    )
    print(f"report={PROJECT_ROOT / 'data/processed/dataset_report.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 Amazon Electronics preprocessing"
    )
    parser.add_argument(
        "--mode",
        choices=["config", "sample", "full"],
        default="config",
        help="Override data.mode for this run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest splits and rebuild cached metadata/reviews",
    )
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-reviews", action="store_true")
    parser.add_argument(
        "--reviews-only",
        action="store_true",
        help=(
            "Low-memory recovery path: process reviews only and update the "
            "existing dataset report; does not rebuild interactions/features."
        ),
    )
    parser.add_argument(
        "--max-review-records",
        type=int,
        default=None,
        help="Development-only bounded review scan (partial output, reported)",
    )
    parser.add_argument(
        "--no-negative-samples",
        action="store_true",
        help="Do not materialise train_samples.parquet",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    config = with_mode(get_app_config(), args.mode)
    print("Amazon Reviews 2023 / Electronics preprocessing")
    print(f"project_root={PROJECT_ROOT}")
    print(f"mode={config.data.mode} seed={config.data.random_seed}")

    if config.data.mode == "full":
        print(
            "WARNING: full mode uses every 5-Core user and can produce large "
            "Parquet files and long runtimes."
        )

    if args.reviews_only and args.skip_reviews:
        print("FAILED: --reviews-only cannot be combined with --skip-reviews")
        return 2

    try:
        preprocessor = DataPreprocessor(config)
        if args.reviews_only:
            stats = preprocessor.run_reviews_only(
                force=True,
                max_review_records=args.max_review_records,
            )
            print("\n=== Reviews-only summary ===")
            print(f"status={stats.get('status')}")
            print(f"kept_reviews={stats.get('kept_reviews')}")
            print(f"items_with_reviews={stats.get('items_with_reviews')}")
            print(f"coverage={stats.get('coverage')}")
            print(f"per_item_cap={stats.get('per_item_cap')}")
            print(f"duplicates_removed={stats.get('duplicates_removed')}")
            print(f"capped_out={stats.get('capped_out')}")
            print(f"report={PROJECT_ROOT / 'data/processed/dataset_report.json'}")
            return 0
        report = preprocessor.run(
            force=args.force,
            skip_metadata=args.skip_metadata,
            skip_reviews=args.skip_reviews,
            max_review_records=args.max_review_records,
            build_negative_samples=not args.no_negative_samples,
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

    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
