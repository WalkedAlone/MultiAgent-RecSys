#!/usr/bin/env python
"""Build exactly one bounded review-index shard in a normal subprocess."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_app_config  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.rag.embedding import build_embedding_from_config  # noqa: E402
from src.rag.review_processor import ReviewIndexBuilder  # noqa: E402
from src.rag.vector_store import ChromaCollection  # noqa: E402

_BUILDER = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one review shard")
    parser.add_argument("--shard-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collection", default="product_reviews")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--upsert-batch-size", type=int, default=2000)
    args = parser.parse_args(argv)
    setup_logging(level="INFO")
    config = get_app_config()
    collection = ChromaCollection(
        path=args.output_dir,
        collection_name=args.collection,
    )
    global _BUILDER
    _BUILDER = ReviewIndexBuilder(
        config=config,
        embedder=build_embedding_from_config(config),
        collection=collection,
    )
    stats = _BUILDER.build(
        source_path=args.shard_path,
        max_per_item=100,
        batch_size=args.batch_size,
        upsert_batch_size=args.upsert_batch_size,
    )
    Path(args.output_dir, "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
