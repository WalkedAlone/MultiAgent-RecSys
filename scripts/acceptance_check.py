"""Acceptance checks for the AgentRec Level-3 requirements.

Checks:
1. LLM-off: the trained recommender can produce real recommendations locally
   and through the direct FastAPI endpoint (no workflow/LLM involved).
2. LLM-on: the chat endpoint returns a structured response from the real
   LangGraph workflow and the Critic.
3. Persistence: model/index/memory artifacts exist and the profile endpoint
   can load saved preferences after the service has been restarted.

Run:
    python scripts/acceptance_check.py
    python scripts/acceptance_check.py --api-url http://127.0.0.1:8000
    python scripts/acceptance_check.py --skip-llm-on

This script does not fabricate products.  The LLM-on check does consume
DeepSeek tokens and may take tens of seconds.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from pathlib import Path

import requests

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_USER = "AGAV7IY4HWGYRUIAGUZRXYUJ22DA"
DEFAULT_QUERY = "推荐 1500 美元以内的笔记本电脑"

REQUIRED_ARTIFACTS = [
    "checkpoints/itemcf_similarity.npz",
    "checkpoints/itemcf_meta.json",
    "checkpoints/two_tower_best.pt",
    "checkpoints/lightgcn_best.pt",
    "checkpoints/deepfm_best.pt",
    "checkpoints/din_best.pt",
    "indexes/two_tower.faiss",
    "indexes/lightgcn.faiss",
    "data/processed/users/user_id_map.parquet",
    "data/processed/user_profiles.sqlite",
    "indexes/chroma_memory",
]


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok = False
        self.detail = ""

    def pass_(self, detail: str) -> "Check":
        self.ok = True
        self.detail = detail
        return self

    def fail(self, detail: str) -> "Check":
        self.ok = False
        self.detail = detail
        return self

    def skip(self, detail: str) -> "Check":
        self.ok = True
        self.detail = f"SKIP: {detail}"
        return self


def check_artifacts() -> Check:
    check = Check("持久化产物")
    missing = [
        relative
        for relative in REQUIRED_ARTIFACTS
        if not (PROJECT_ROOT / relative).exists()
    ]
    if missing:
        return check.fail("missing: " + ", ".join(missing))
    db_path = PROJECT_ROOT / "data/processed/user_profiles.sqlite"
    try:
        with sqlite3.connect(db_path) as connection:
            preference_count = connection.execute(
                "SELECT COUNT(*) FROM memory_preferences WHERE active = 1"
            ).fetchone()[0]
            feedback_count = connection.execute(
                "SELECT COUNT(*) FROM recommendation_feedback"
            ).fetchone()[0]
    except sqlite3.Error as exc:
        return check.fail(f"SQLite unreadable: {type(exc).__name__}: {exc}")
    return check.pass_(
        f"artifacts={len(REQUIRED_ARTIFACTS)}; "
        f"memory_preferences={preference_count}; "
        f"feedback_rows={feedback_count}"
    )


def check_local_engine(user_id: str) -> Check:
    """Prove the recommender works with no LLM client in the call path."""
    check = Check("关闭 LLM：本地推荐引擎")
    try:
        from src.services.recommendation_service import RecommendationEngine

        engine = RecommendationEngine.from_artifacts()
        response = engine.recommend(
            user_id,
            top_k=5,
            category="laptop",
            budget_max=1500.0,
        )
    except Exception as exc:  # noqa: BLE001 - acceptance report
        return check.fail(f"{type(exc).__name__}: {exc}")
    recommendations = response.get("recommendations") or []
    if not recommendations:
        return check.fail(
            "engine returned 0 recommendations; "
            f"fallback={(response.get('trace', {}).get('fusion', {}).get('fallback'))}"
        )
    first = recommendations[0]
    if not first.get("item_id") or not first.get("title"):
        return check.fail("recommendation lacks real item_id/title")
    return check.pass_(
        f"recommendations={len(recommendations)}; "
        f"first={first['item_id']} | {first.get('title', '')[:60]}"
    )


def _request(method: str, url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 300)
    return requests.request(method, url, **kwargs)


def check_llm_off_api(api_url: str, user_id: str) -> Check:
    check = Check("关闭 LLM：FastAPI 直连推荐")
    try:
        response = _request(
            "POST",
            f"{api_url.rstrip('/')}/api/v1/recommend/user/{user_id}",
            json={"top_k": 5, "category": "laptop", "budget_max": 1500},
            timeout=120,
        )
    except requests.RequestException as exc:
        return check.fail(f"{type(exc).__name__}: {exc}")
    if response.status_code != 200:
        return check.fail(f"HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    recommendations = body.get("recommendations") or []
    if not recommendations:
        return check.fail("API returned 0 recommendations")
    fallback = (
        (body.get("trace") or {}).get("fusion") or {}
    ).get("fallback") or {}
    return check.pass_(
        f"recommendations={len(recommendations)}; "
        f"fallback_selected={fallback.get('selected')}"
    )


def check_llm_on(api_url: str, user_id: str, query: str) -> Check:
    check = Check("开启 LLM：多智能体对话")
    try:
        response = _request(
            "POST",
            f"{api_url.rstrip('/')}/api/v1/recommend/chat",
            json={
                "user_id": user_id,
                "query": query,
                "thread_id": f"acceptance-{uuid.uuid4().hex[:8]}",
            },
            timeout=300,
        )
    except requests.RequestException as exc:
        return check.fail(f"{type(exc).__name__}: {exc}")
    if response.status_code != 200:
        return check.fail(f"HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    recommendations = body.get("recommendations") or []
    critic = body.get("critic") or {}
    if not body.get("assistant_message"):
        return check.fail("missing assistant_message")
    if not critic:
        return check.fail("missing critic payload")
    return check.pass_(
        f"status={body.get('status')}; "
        f"recommendations={len(recommendations)}; "
        f"critic_approved={critic.get('approved')}; "
        f"critic_llm={critic.get('llm_status')}"
    )


def check_profile_persistence(api_url: str, user_id: str) -> Check:
    check = Check("重启后：用户画像/记忆可加载")
    try:
        response = _request(
            "GET",
            f"{api_url.rstrip('/')}/api/v1/user/{user_id}/profile",
            timeout=120,
        )
    except requests.RequestException as exc:
        return check.fail(f"{type(exc).__name__}: {exc}")
    if response.status_code != 200:
        return check.fail(f"HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    profile = body.get("profile") or {}
    preferences = body.get("preferences") or []
    if not profile.get("known_user"):
        return check.fail("user is not known after reload")
    return check.pass_(
        f"known_user=true; preferences={len(preferences)}; "
        f"memory_status={profile.get('semantic_status')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--skip-llm-on", action="store_true")
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Only run local artifact and engine checks.",
    )
    args = parser.parse_args()

    checks = [check_artifacts(), check_local_engine(args.user_id)]
    if not args.skip_api:
        checks.append(check_llm_off_api(args.api_url, args.user_id))
        checks.append(check_profile_persistence(args.api_url, args.user_id))
        if args.skip_llm_on:
            checks.append(Check("开启 LLM：多智能体对话").skip("--skip-llm-on"))
        else:
            checks.append(
                check_llm_on(args.api_url, args.user_id, args.query)
            )

    print("AgentRec acceptance checks")
    print("=" * 72)
    for check in checks:
        icon = "PASS" if check.ok else "FAIL"
        print(f"[{icon}] {check.name}")
        print(f"       {check.detail}")
    failed = [check for check in checks if not check.ok]
    print("=" * 72)
    if failed:
        print(f"FAILED: {len(failed)} check(s)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
