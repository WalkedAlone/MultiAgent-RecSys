#!/usr/bin/env python
"""Search real product reviews through the local BGE + Chroma index."""

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
from src.rag.retriever import ReviewRetriever  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search product review evidence")
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    embedder = build_embedding_from_config(config)
    retriever = ReviewRetriever.from_config(config, embedder)
    try:
        evidence = retriever.search(
            item_id=args.item_id,
            query=args.query,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Review search failed: {type(exc).__name__}: {exc}")
        return 1
    if args.json:
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    else:
        print(f"item_id={args.item_id} evidence={len(evidence)}")
        for index, item in enumerate(evidence, start=1):
            print(
                f"  {index}. score={item['score']:.4f} "
                f"rating={item['rating']} verified={item['verified_purchase']} "
                f"time={item['timestamp']} review={item['review_id']}"
            )
            print(f"     {item['text'][:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
