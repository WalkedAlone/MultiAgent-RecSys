#!/usr/bin/env python
"""Download the official Amazon Reviews 2023 Electronics files.

Examples:
    python scripts/download_data.py --group core
    python scripts/download_data.py --group metadata
    python scripts/download_data.py --group reviews
    python scripts/download_data.py --group core --verify-only

Downloads are resumable: interrupted range segments remain in
``data/raw/**/.<file>.parts/`` and are reused on the next run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_app_config  # noqa: E402
from src.core.exceptions import RecSysError  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.data.downloader import DownloadResult, DownloadSpec, FileDownloader  # noqa: E402

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"


def build_specs() -> dict[str, list[DownloadSpec]]:
    """Build download specs from the configured data paths."""
    config = get_app_config()
    benchmark = (Path(config.data.paths.raw_benchmark) / "5core").as_posix()
    metadata_dir = Path(config.data.paths.raw_metadata).as_posix()
    reviews_dir = Path(config.data.paths.raw_reviews).as_posix()

    core = [
        DownloadSpec(
            name="5core_train",
            url=f"{BASE}/benchmark/5core/last_out_w_his/Electronics.train.csv.gz",
            relative_path=f"{benchmark}/Electronics.train.csv.gz",
            description="Official 5-Core Leave-Last-Out training split",
        ),
        DownloadSpec(
            name="5core_valid",
            url=f"{BASE}/benchmark/5core/last_out_w_his/Electronics.valid.csv.gz",
            relative_path=f"{benchmark}/Electronics.valid.csv.gz",
            description="Official 5-Core Leave-Last-Out validation split",
        ),
        DownloadSpec(
            name="5core_test",
            url=f"{BASE}/benchmark/5core/last_out_w_his/Electronics.test.csv.gz",
            relative_path=f"{benchmark}/Electronics.test.csv.gz",
            description="Official 5-Core Leave-Last-Out test split",
        ),
    ]
    metadata = [
        DownloadSpec(
            name="meta_electronics",
            url=f"{BASE}/raw/meta_categories/meta_Electronics.jsonl.gz",
            relative_path=f"{metadata_dir}/meta_Electronics.jsonl.gz",
            description="Item metadata for Electronics",
        )
    ]
    reviews = [
        DownloadSpec(
            name="reviews_electronics",
            url=f"{BASE}/raw/review_categories/Electronics.jsonl.gz",
            relative_path=f"{reviews_dir}/Electronics.jsonl.gz",
            description="Raw Electronics reviews (6.5 GB compressed)",
        )
    ]
    return {
        "core": core,
        "metadata": metadata,
        "reviews": reviews,
        "all": core + metadata + reviews,
    }


def _print_result(result: DownloadResult) -> None:
    present = "already present" if result.already_present else "downloaded"
    gzip_status = (
        "n/a"
        if result.gzip_verified is None
        else ("valid" if result.gzip_verified else "INVALID")
    )
    print(
        f"  OK  {result.name:20s} {result.size_bytes / 1024 / 1024:8.1f} MB  "
        f"sha256={result.sha256[:16]}...  gzip={gzip_status}  "
        f"lines={result.line_count}  {present}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download official Amazon Reviews 2023 Electronics files"
    )
    parser.add_argument(
        "--group",
        choices=["core", "metadata", "reviews", "all"],
        default="core",
        help="Which files to download (default: core)",
    )
    parser.add_argument("--connections", type=int, default=None)
    parser.add_argument("--segment-size-mb", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files without downloading",
    )
    parser.add_argument(
        "--no-gzip-verify",
        action="store_true",
        help="Skip full gzip decompression (not recommended)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    config = get_app_config()
    specs = build_specs()[args.group]

    print("Amazon Reviews 2023 / Electronics download plan")
    print(f"project_root={PROJECT_ROOT}")
    print(f"group={args.group} files={len(specs)}")
    for spec in specs:
        print(f"  - {spec.name}: {spec.relative_path}")
        print(f"    {spec.url}")

    failed: list[tuple[str, str]] = []
    try:
        with FileDownloader(
            config,
            connections=args.connections,
            segment_size_bytes=(
                args.segment_size_mb * 1024 * 1024
                if args.segment_size_mb
                else None
            ),
            force=args.force,
            verify_gzip=not args.no_gzip_verify,
        ) as downloader:
            for spec in specs:
                try:
                    result = downloader.download(
                        spec, verify_only=args.verify_only
                    )
                    _print_result(result)
                except RecSysError as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    print(f"  FAIL {spec.name}: {message}")
                    failed.append((spec.name, message))
    except KeyboardInterrupt:
        print("\nInterrupted. Completed range segments are kept for resume.")
        return 130

    print(f"\nmanifest={PROJECT_ROOT / 'data' / 'raw' / 'manifest.json'}")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for name, message in failed:
            print(f"  - {name}: {message}")
        return 1
    print("All requested files are present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
