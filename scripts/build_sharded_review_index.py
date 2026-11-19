#!/usr/bin/env python
"""Build bounded Chroma review-index shards in separate processes.

A single very large persistent HNSW collection was not reloadable with the
tested Windows Chroma build.  Each shard is written by its own process (max
~10k vectors), which the 10k integration test proved reloadable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl  # noqa: E402

from src.core.config import get_app_config  # noqa: E402
from src.core.training import save_json  # noqa: E402
from src.rag.review_processor import prepare_top_reviews  # noqa: E402


def shard_index(item_id: str, num_shards: int) -> int:
    import hashlib

    digest = hashlib.sha1(str(item_id).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(num_shards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build sharded review RAG index")
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-reviews-per-item", type=int, default=2)
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--upsert-batch-size", type=int, default=2000)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    config = get_app_config()
    root = config.project_root
    prepared = root / f"data/processed/reviews/review_top{args.max_reviews_per_item}.parquet"
    if not prepared.is_file():
        prepare_top_reviews(
            reviews_path=root / "data/processed/reviews/reviews.parquet",
            output_path=prepared,
            max_per_item=args.max_reviews_per_item,
        )
    shard_root = root / "data/processed/reviews/shards"
    if shard_root.exists():
        shutil.rmtree(shard_root, ignore_errors=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    output_root = config.resolve_path(config.agent.rag.review_index_dir)
    if args.rebuild:
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    frame = pl.read_parquet(prepared)
    item_map = pl.read_parquet(
        root / "data/processed/items/item_id_map.parquet", columns=["parent_asin"]
    )
    frame = frame.filter(pl.col("parent_asin").is_in(item_map["parent_asin"].to_list()))
    if args.limit_items and args.limit_items > 0:
        keep = (
            frame.select("parent_asin")
            .unique()
            .sort("parent_asin")
            .head(args.limit_items)
            .get_column("parent_asin")
            .to_list()
        )
        frame = frame.filter(pl.col("parent_asin").is_in(keep))
    frame = frame.with_columns(
        pl.col("parent_asin").map_elements(
            lambda value: shard_index(str(value), args.shards),
            return_dtype=pl.Int64,
        ).alias("_shard")
    )
    shard_files: list[tuple[int, Path]] = []
    for shard in range(args.shards):
        shard_frame = frame.filter(pl.col("_shard") == shard)
        path = shard_root / f"shard_{shard:02d}.parquet"
        shard_frame.write_parquet(path)
        shard_files.append((shard, path))
    print(f"prepared_shards={len(shard_files)} docs={frame.height}")

    started = time.perf_counter()
    results: list[dict] = []
    script = Path(__file__).resolve().with_name("build_review_shard.py")

    def run_shard(shard: int, path: Path) -> dict:
        output_dir = output_root / f"shard_{shard:02d}"
        log_path = output_root / f"shard_{shard:02d}.log"
        command = [
            sys.executable,
            str(script),
            "--shard-path",
            str(path),
            "--output-dir",
            str(output_dir),
            "--collection",
            config.agent.rag.collection_name,
            "--batch-size",
            str(args.batch_size),
            "--upsert-batch-size",
            str(args.upsert_batch_size),
        ]
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(
                command,
                cwd=str(root),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"shard {shard} failed with exit {completed.returncode}; see {log_path}"
            )
        with open(output_dir / "stats.json", encoding="utf-8") as handle:
            result = json.load(handle)
        result["shard"] = shard
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_shard, shard, path): shard
            for shard, path in shard_files
            if path.stat().st_size > 0
        }
        for future in as_completed(futures):
            shard = futures[future]
            result = future.result()
            results.append(result)
            print(f"shard {shard}: {result['collection_count']} docs")
    results.sort(key=lambda item: item["shard"])
    stats = {
        "mode": "process_sharded",
        "num_shards": args.shards,
        "workers": args.workers,
        "collection": config.agent.rag.collection_name,
        "collection_count": sum(int(item["collection_count"]) for item in results),
        "documents_indexed": sum(int(item["documents_indexed"]) for item in results),
        "max_per_item": args.max_reviews_per_item,
        "train_only": True,
        "shards": results,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    save_json(stats, root / "logs/phase5_review_index.json")
    print(f"review_index={stats}")
    print(f"saved={root / 'logs/phase5_review_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
