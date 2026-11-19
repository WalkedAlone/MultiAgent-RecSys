#!/usr/bin/env python
"""Inspect and explicitly edit SQLite/Chroma user memory."""

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
from src.memory.memory_manager import MemoryManager  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage explicit user memory")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--add", nargs=2, metavar=("KIND", "VALUE"))
    parser.add_argument("--delete-id", type=int)
    parser.add_argument("--delete-kind-value", nargs=2, metavar=("KIND", "VALUE"))
    parser.add_argument("--feedback", nargs=2, metavar=("ITEM_ID", "TYPE"))
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    manager = MemoryManager.from_artifacts(
        config,
        create_semantic=not args.no_semantic,
    )
    try:
        output = {}
        if args.add:
            output["added"] = manager.add_preference(
                user_id=args.user_id,
                kind=args.add[0],
                value=args.add[1],
            )
        if args.delete_id is not None:
            output["deleted_count"] = manager.delete_preference(
                user_id=args.user_id, preference_id=args.delete_id
            )
        if args.delete_kind_value:
            output["deleted_count"] = manager.delete_preference(
                user_id=args.user_id,
                kind=args.delete_kind_value[0],
                value=args.delete_kind_value[1],
            )
        if args.feedback:
            output["feedback"] = manager.record_feedback(
                user_id=args.user_id,
                item_id=args.feedback[0],
                feedback_type=args.feedback[1],
                query=args.query or None,
            )
        if args.show or not output:
            output["profile"] = manager.get_profile(
                args.user_id, query=args.query or None
            )
            output["preferences"] = manager.list_preferences(args.user_id)
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    finally:
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
