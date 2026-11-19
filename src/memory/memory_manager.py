"""Unified structured + semantic user memory manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agents.profile_agent import UserProfileStore
from src.core.logger import get_logger
from src.memory.profile_store import StructuredMemoryStore
from src.memory.vector_memory import UserSemanticMemory
from src.rag.embedding import BGEEmbedding, build_embedding_from_config

logger = get_logger(__name__)


class MemoryManager:
    """Explicit durable memory; temporary query constraints are never written."""

    def __init__(
        self,
        *,
        structured: StructuredMemoryStore,
        semantic: UserSemanticMemory | None,
        semantic_top_k: int = 5,
        semantic_status: str = "ok",
    ) -> None:
        self.structured = structured
        self.semantic = semantic
        self.semantic_top_k = int(semantic_top_k)
        self.semantic_status = semantic_status

    @classmethod
    def from_artifacts(
        cls,
        config: Any,
        *,
        item_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        embedder: Any | None = None,
        create_semantic: bool = True,
        seed_store: UserProfileStore | None = None,
    ) -> "MemoryManager":
        seed = seed_store or UserProfileStore.from_artifacts(
            config.project_root, item_metadata=item_metadata
        )
        structured = StructuredMemoryStore(
            config.resolve_path(config.agent.memory.profile_db_path),
            seed_store=seed,
        )
        semantic = None
        semantic_status = "disabled"
        if create_semantic:
            try:
                embedder = embedder or build_embedding_from_config(config)
                semantic = UserSemanticMemory.from_config(config, embedder)
                semantic_status = "ok"
            except Exception as exc:  # pragma: no cover - Chroma/model edge
                logger.warning(
                    "semantic_memory_unavailable error=%s", type(exc).__name__
                )
                semantic_status = f"unavailable:{type(exc).__name__}"
        return cls(
            structured=structured,
            semantic=semantic,
            semantic_top_k=config.agent.memory.semantic_top_k,
            semantic_status=semantic_status,
        )

    def get_profile(
        self, user_id: str, *, query: str | None = None
    ) -> dict[str, Any]:
        profile = self.structured.get_profile(str(user_id), query=query)
        semantic_memory: list[dict[str, Any]] = []
        if self.semantic is not None and query:
            try:
                semantic_memory = self.semantic.search(
                    user_id=str(user_id),
                    query=str(query),
                    top_k=self.semantic_top_k,
                )
            except Exception as exc:  # pragma: no cover - vector backend edge
                logger.warning(
                    "semantic_memory_search_failed error=%s",
                    type(exc).__name__,
                )
                profile["semantic_status"] = (
                    f"search_failed:{type(exc).__name__}"
                )
        profile["semantic_memory"] = semantic_memory
        profile.setdefault("semantic_status", self.semantic_status)
        return profile

    def add_preference(
        self,
        *,
        user_id: str,
        kind: str,
        value: str,
        source: str = "explicit_user",
        confidence: float = 0.9,
    ) -> dict[str, Any]:
        preference = self.structured.add_preference(
            user_id,
            kind,
            value,
            source=source,
            confidence=confidence,
        )
        semantic_id = None
        if self.semantic is not None:
            try:
                semantic_id = self.semantic.add(
                    user_id=str(user_id),
                    text=f"{kind}: {value}",
                    kind=kind,
                    source=source,
                    confidence=confidence,
                )
            except Exception as exc:  # pragma: no cover - vector backend edge
                logger.warning(
                    "semantic_memory_add_failed error=%s", type(exc).__name__
                )
        return {**preference, "semantic_id": semantic_id}

    def delete_preference(
        self,
        *,
        user_id: str,
        preference_id: int | None = None,
        kind: str | None = None,
        value: str | None = None,
    ) -> int:
        semantic_kind, semantic_value = kind, value
        if semantic_kind is None and preference_id is not None:
            for pref in self.structured.list_preferences(user_id):
                if int(pref["id"]) == int(preference_id):
                    semantic_kind = pref["kind"]
                    semantic_value = pref["value"]
                    break
        count = self.structured.delete_preference(
            user_id=user_id,
            preference_id=preference_id,
            kind=kind,
            value=value,
        )
        if self.semantic is not None and semantic_kind and semantic_value:
            try:
                self.semantic.delete_ids(
                    [
                        UserSemanticMemory.make_id(
                            user_id, semantic_kind, semantic_value
                        )
                    ]
                )
            except Exception as exc:  # pragma: no cover - vector backend edge
                logger.warning(
                    "semantic_memory_delete_failed error=%s",
                    type(exc).__name__,
                )
        return count

    def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        return self.structured.list_preferences(str(user_id))

    def record_feedback(
        self,
        *,
        user_id: str,
        item_id: str,
        feedback_type: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        feedback = self.structured.record_feedback(
            user_id=user_id,
            item_id=item_id,
            feedback_type=feedback_type,
            query=query,
        )
        if self.semantic is not None:
            try:
                self.semantic.add(
                    user_id=str(user_id),
                    text=f"{feedback_type}: item {item_id}",
                    kind=feedback_type,
                    source="feedback",
                    confidence=0.8,
                )
            except Exception as exc:  # pragma: no cover - vector backend edge
                logger.warning(
                    "semantic_feedback_add_failed error=%s",
                    type(exc).__name__,
                )
        return feedback

    def apply_intent_updates(
        self,
        *,
        user_id: str,
        updates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Apply only explicit durable memory updates extracted by IntentAgent."""
        applied: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for update in updates:
            if not isinstance(update, Mapping) or not update.get("explicit", False):
                skipped.append({"reason": "not_explicit", "update": dict(update)})
                continue
            action = str(update.get("action") or "add")
            kind = str(update.get("kind") or "other")
            value = str(update.get("value") or "").strip()
            if not value:
                skipped.append({"reason": "empty_value", "update": dict(update)})
                continue
            if action == "delete":
                deleted = self.delete_preference(
                    user_id=user_id, kind=kind, value=value
                )
                applied.append(
                    {"action": "delete", "kind": kind, "value": value, "count": deleted}
                )
            elif action == "add":
                applied.append(
                    {
                        "action": "add",
                        **self.add_preference(
                            user_id=user_id,
                            kind=kind,
                            value=value,
                            source="explicit_user",
                            confidence=0.9,
                        ),
                    }
                )
            else:
                skipped.append({"reason": "unsupported_action", "update": dict(update)})
        return {
            "applied": applied,
            "skipped": skipped,
            "written": bool(applied),
            "status": "applied" if applied else "no_update",
        }

    def close(self) -> None:
        self.structured.close()
