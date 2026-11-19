#!/usr/bin/env python
"""Run the Phase 3 engine for one user and print an auditable Top-K result."""

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

import polars as pl  # noqa: E402

from src.core.config import get_app_config  # noqa: E402
from src.core.exceptions import RecSysError  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.core.training import save_json  # noqa: E402
from src.services.recommendation_service import RecommendationEngine  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo the Phase 3 engine")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--budget-max", type=float, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--strategy", choices=["quota", "rrf", "quota_rrf"], default=None
    )
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    root = config.project_root
    user_id = args.user_id
    if user_id is None:
        valid = pl.read_parquet(root / "data/splits/valid_eval.parquet")
        user_id = str(valid.sort("user_id")["user_id"][0])
    try:
        engine = RecommendationEngine.from_artifacts(config)
        response = engine.recommend(
            user_id,
            top_k=args.top_k,
            budget_max=args.budget_max,
            category=args.category,
            use_mmr=not args.no_mmr,
            strategy=args.strategy,
        )
    except RecSysError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"FAILED (unexpected): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    save_json(response, root / "logs/phase3_demo.json")
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"user_id={user_id} recommendations={len(response['recommendations'])}")
        for item in response["recommendations"]:
            print(
                f"  {item['rank']:2d}. {item['item_id']} "
                f"score={item['ranking_score']:.4f} "
                f"mmr={item['mmr_score']:.4f} "
                f"price={item.get('price')} "
                f"category={item.get('main_category')} "
                f"sources={item.get('retrieval_sources')}"
            )
        print("\ntrace:")
        print(json.dumps(response["trace"], ensure_ascii=False, indent=2, default=str))
        print(f"\nsaved={root / 'logs/phase3_demo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
