#!/usr/bin/env python
"""Rebuild Faiss/NumPy indexes from exported retrieval vectors."""

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

import numpy as np  # noqa: E402

from src.core.config import get_app_config  # noqa: E402
from src.retrieval.faiss_index import VectorIndex  # noqa: E402


def build_two_tower(root: Path) -> None:
    vectors = np.load(root / "indexes/two_tower_item_vectors.npy")
    ids = json.loads(
        (root / "indexes/two_tower_item_ids.json").read_text(encoding="utf-8")
    )
    index = VectorIndex.build(
        vectors[1:], ids, root / "indexes/two_tower.faiss"
    )
    print(f"two_tower: backend={index.backend} size={index.size}")


def build_lightgcn(root: Path) -> None:
    vectors = np.load(root / "indexes/lightgcn_item_embeddings.npy")
    ids = json.loads(
        (root / "indexes/lightgcn_item_ids.json").read_text(encoding="utf-8")
    )
    index = VectorIndex.build(
        vectors, ids, root / "indexes/lightgcn.faiss"
    )
    print(f"lightgcn: backend={index.backend} size={index.size}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild retrieval indexes")
    parser.add_argument(
        "--model", choices=["two_tower", "lightgcn", "all"], default="all"
    )
    args = parser.parse_args(argv)
    root = get_app_config().project_root
    if args.model in {"two_tower", "all"}:
        build_two_tower(root)
    if args.model in {"lightgcn", "all"}:
        build_lightgcn(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
