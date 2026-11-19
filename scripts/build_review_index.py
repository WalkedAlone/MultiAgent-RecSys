#!/usr/bin/env python
"""Build the real review Chroma index from Amazon review text."""

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
from src.rag.embedding import build_embedding_from_config  # noqa: E402
from src.rag.review_processor import (  # noqa: E402
    ReviewIndexBuilder,
    prepare_top_reviews,
)

# Keep the Chroma client alive until interpreter teardown. Dropping the local
# builder immediately after main() returns can skip the local HNSW header flush.
_INDEX_BUILDER = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build review RAG index")
    parser.add_argument(
        "--source",
        default="data/processed/reviews/reviews.parquet",
    )
    parser.add_argument("--max-reviews-per-item", type=int, default=2)
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Include catalog items outside the train item map (default: train-only)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=2000)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    root = config.project_root
    source = config.resolve_path(args.source)
    prepared = root / (
        f"data/processed/reviews/review_top{args.max_reviews_per_item}.parquet"
    )
    started = time.perf_counter()
    if args.rebuild or not prepared.is_file():
        prep = prepare_top_reviews(
            reviews_path=source,
            output_path=prepared,
            max_per_item=args.max_reviews_per_item,
        )
        print(f"prepared={prepared} {prep}")
    else:
        prep = {"output": str(prepared), "reused": True}
    if args.prepare_only:
        save_json({"prepare": prep}, root / "logs/phase5_review_prepare.json")
        return 0

    if args.rebuild:
        import shutil

        shutil.rmtree(
            config.resolve_path(config.agent.rag.review_index_dir),
            ignore_errors=True,
        )
    allowed_item_ids = None
    train_only = not args.all_items
    if train_only:
        item_map = pl.read_parquet(
            root / "data/processed/items/item_id_map.parquet",
            columns=["parent_asin"],
        )
        allowed_item_ids = set(item_map["parent_asin"].to_list())
    embedder = build_embedding_from_config(config)
    if args.shards > 1:
        print(
            "Use scripts/build_sharded_review_index.py --shards N for the "
            "process-isolated sharded build; the single-process sharded path "
            "does not persist HNSW reliably on this Windows host."
        )
        return 2
    global _INDEX_BUILDER
    builder = ReviewIndexBuilder(config=config, embedder=embedder)
    _INDEX_BUILDER = builder
    stats = builder.build(
        source_path=prepared,
        max_per_item=args.max_reviews_per_item,
        limit_items=args.limit_items,
        allowed_item_ids=allowed_item_ids,
        batch_size=args.batch_size,
        upsert_batch_size=args.upsert_batch_size,
        rebuild=args.rebuild,
    )
    # Chroma 0.5.x persists its local HNSW segment during process teardown;
    # do not drop the collection before the process exits.
    stats["train_only"] = train_only
    stats["prepare"] = prep
    stats["runtime_seconds_total"] = round(time.perf_counter() - started, 1)
    save_json(stats, root / "logs/phase5_review_index.json")
    print(f"review_index={stats}")
    print(f"saved={root / 'logs/phase5_review_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
