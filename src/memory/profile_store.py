"""SQLite structured user memory with explicit/inferred provenance.

This module deliberately separates:
* temporary request constraints (never written here), and
* durable preferences/feedback (written only with an explicit source).

The Phase 4 read-only ``UserProfileStore`` can seed inferred train facts, but
those are labelled ``inferred_from_train`` and never silently become hard
constraints.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.core.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserProfileRow(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(120), primary_key=True)
    favorite_category = Column(String(120), nullable=True)
    interaction_count = Column(Integer, default=0)
    source = Column(String(40), default="inferred_from_train")
    confidence = Column(Float, default=0.4)
    updated_at = Column(DateTime(timezone=True), default=_utcnow)


class MemoryPreferenceRow(Base):
    __tablename__ = "memory_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(120), index=True, nullable=False)
    kind = Column(String(30), nullable=False)
    value = Column(String(300), nullable=False)
    source = Column(String(40), nullable=False, default="explicit_user")
    confidence = Column(Float, nullable=False, default=0.9)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow)


class FeedbackRow(Base):
    __tablename__ = "recommendation_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(120), index=True, nullable=False)
    item_id = Column(String(80), index=True, nullable=False)
    feedback_type = Column(String(30), nullable=False)
    query = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class StructuredMemoryStore:
    """SQLAlchemy-backed durable user profile and preference store."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        seed_store: Any | None = None,
    ) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self.engine = create_engine(
            f"sqlite:///{path.as_posix()}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )
        self.seed_store = seed_store

    # ------------------------------------------------------------------
    # Seeding and reads
    # ------------------------------------------------------------------
    def _ensure_seed(self, user_id: str) -> None:
        with self.Session() as session:
            if session.get(UserProfileRow, str(user_id)) is not None:
                return
            if self.seed_store is None:
                return
            seed = self.seed_store.get_profile(str(user_id))
            if not seed.get("known_user"):
                return
            long_term = seed.get("long_term", {})
            session.add(
                UserProfileRow(
                    user_id=str(user_id),
                    favorite_category=long_term.get("favorite_category") or None,
                    interaction_count=int(
                        long_term.get("interaction_count") or 0
                    ),
                    source="inferred_from_train",
                    confidence=0.4,
                    updated_at=_utcnow(),
                )
            )
            session.flush()
            seen: set[tuple[str, str]] = set()
            category = str(long_term.get("favorite_category") or "").strip()
            if category:
                self._insert_preference(
                    session,
                    user_id,
                    "category",
                    category,
                    source="inferred_from_train",
                    confidence=0.4,
                    seen=seen,
                )
            for brand in long_term.get("brand_preferences", []):
                value = str(brand.get("store") or "").strip()
                if value:
                    confidence = min(
                        0.45,
                        0.2
                        + 0.05 * int(brand.get("positive_interactions") or 1),
                    )
                    self._insert_preference(
                        session,
                        user_id,
                        "brand",
                        value,
                        source="inferred_from_train",
                        confidence=round(confidence, 3),
                        seen=seen,
                    )
            price_range = long_term.get("price_range_recent")
            if isinstance(price_range, (list, tuple)) and len(price_range) == 2:
                self._insert_preference(
                    session,
                    user_id,
                    "budget",
                    f"{price_range[0]}-{price_range[1]}",
                    source="inferred_from_train",
                    confidence=0.3,
                    seen=seen,
                )
            session.commit()

    @staticmethod
    def _insert_preference(
        session,
        user_id: str,
        kind: str,
        value: str,
        *,
        source: str,
        confidence: float,
        seen: set[tuple[str, str]],
    ) -> None:
        key = (kind, value)
        if key in seen:
            return
        seen.add(key)
        session.add(
            MemoryPreferenceRow(
                user_id=str(user_id),
                kind=kind,
                value=value,
                source=source,
                confidence=float(confidence),
                active=True,
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )
        )

    def get_profile(
        self, user_id: str, *, query: str | None = None
    ) -> dict[str, Any]:
        user_id = str(user_id)
        self._ensure_seed(user_id)
        with self.Session() as session:
            profile_row = session.get(UserProfileRow, user_id)
            preferences = session.scalars(
                select(MemoryPreferenceRow)
                .where(
                    MemoryPreferenceRow.user_id == user_id,
                    MemoryPreferenceRow.active.is_(True),
                )
                .order_by(MemoryPreferenceRow.updated_at.desc())
            ).all()
            feedback_count = session.scalar(
                select(func.count(FeedbackRow.id)).where(
                    FeedbackRow.user_id == user_id
                )
            )
            recent_feedback_rows = session.scalars(
                select(FeedbackRow)
                .where(FeedbackRow.user_id == user_id)
                .order_by(FeedbackRow.created_at.desc())
                .limit(5)
            ).all()
        if profile_row is None:
            return {
                "user_id": user_id,
                "known_user": False,
                "long_term": {},
                "explicit_preferences": [],
                "inferred_preferences": [],
                "recent_feedback": [],
                "memory_status": "not_found",
            }
        preference_dicts = [self._pref_dict(pref) for pref in preferences]
        explicit = [
            pref
            for pref in preference_dicts
            if pref["source"] in {"explicit_user", "feedback"}
        ]
        inferred = [
            pref for pref in preference_dicts if pref["source"].startswith("inferred")
        ]
        effective = self._effective_constraints(explicit)
        return {
            "user_id": user_id,
            "known_user": True,
            "source": "sqlite_structured_memory",
            "long_term": {
                "favorite_category": profile_row.favorite_category or "",
                "interaction_count": int(profile_row.interaction_count or 0),
                "brand_preferences": [
                    {"store": pref["value"], "source": pref["source"]}
                    for pref in preference_dicts
                    if pref["kind"] == "brand"
                ],
                "explicit_preference_count": len(explicit),
                "inferred_preference_count": len(inferred),
                "feedback_count": int(feedback_count or 0),
                "memory_updated_at": (
                    profile_row.updated_at.isoformat()
                    if profile_row.updated_at
                    else None
                ),
            },
            "explicit_preferences": explicit,
            "inferred_preferences": inferred,
            "effective_constraints": effective,
            "recent_feedback": [
                {
                    "item_id": row.item_id,
                    "feedback_type": row.feedback_type,
                    "query": row.query,
                    "created_at": (
                        row.created_at.isoformat() if row.created_at else None
                    ),
                }
                for row in recent_feedback_rows
            ],
            "memory_status": "sqlite_structured",
        }

    @staticmethod
    def _pref_dict(pref: MemoryPreferenceRow) -> dict[str, Any]:
        return {
            "id": int(pref.id),
            "kind": pref.kind,
            "value": pref.value,
            "source": pref.source,
            "confidence": float(pref.confidence),
            "updated_at": (
                pref.updated_at.isoformat() if pref.updated_at else None
            ),
        }

    @staticmethod
    def _effective_constraints(
        explicit: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Only explicit durable facts may become fallback constraints."""
        category = None
        budget = None
        for pref in explicit:
            value = str(pref["value"]).strip()
            if pref["kind"] == "category" and value:
                category = value
            elif pref["kind"] == "budget" and value:
                numbers = re.findall(r"\d+(?:\.\d+)?", value)
                if numbers:
                    budget = float(numbers[-1])
        return {
            "category": category,
            "budget_max": budget,
            "source": "explicit_memory",
        }

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_preference(
        self,
        user_id: str,
        kind: str,
        value: str,
        *,
        source: str = "explicit_user",
        confidence: float = 0.9,
    ) -> dict[str, Any]:
        user_id = str(user_id)
        value = str(value).strip()
        if not value:
            raise ValueError("preference value must not be empty")
        self._ensure_seed(user_id)
        with self.Session() as session:
            if session.get(UserProfileRow, user_id) is None:
                session.add(
                    UserProfileRow(
                        user_id=user_id,
                        favorite_category=None,
                        interaction_count=0,
                        source=source,
                        confidence=float(confidence),
                        updated_at=_utcnow(),
                    )
                )
                session.flush()
            existing = session.scalars(
                select(MemoryPreferenceRow).where(
                    MemoryPreferenceRow.user_id == user_id,
                    MemoryPreferenceRow.kind == kind,
                    MemoryPreferenceRow.value == value,
                    MemoryPreferenceRow.active.is_(True),
                )
            ).first()
            profile_row = session.get(UserProfileRow, user_id)
            if profile_row is not None:
                profile_row.updated_at = _utcnow()
            if existing is not None:
                existing.source = source
                existing.confidence = float(confidence)
                existing.updated_at = _utcnow()
                session.commit()
                return self._pref_dict(existing)
            row = MemoryPreferenceRow(
                user_id=user_id,
                kind=kind,
                value=value,
                source=source,
                confidence=float(confidence),
                active=True,
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._pref_dict(row)

    def delete_preference(
        self,
        *,
        user_id: str,
        preference_id: int | None = None,
        kind: str | None = None,
        value: str | None = None,
    ) -> int:
        with self.Session() as session:
            statement = select(MemoryPreferenceRow).where(
                MemoryPreferenceRow.user_id == str(user_id),
                MemoryPreferenceRow.active.is_(True),
            )
            if preference_id is not None:
                statement = statement.where(
                    MemoryPreferenceRow.id == int(preference_id)
                )
            if kind is not None:
                statement = statement.where(MemoryPreferenceRow.kind == kind)
            if value is not None:
                statement = statement.where(MemoryPreferenceRow.value == value)
            rows = session.scalars(statement).all()
            for row in rows:
                row.active = False
                row.updated_at = _utcnow()
            session.commit()
            return len(rows)

    def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        self._ensure_seed(str(user_id))
        with self.Session() as session:
            rows = session.scalars(
                select(MemoryPreferenceRow)
                .where(
                    MemoryPreferenceRow.user_id == str(user_id),
                    MemoryPreferenceRow.active.is_(True),
                )
                .order_by(MemoryPreferenceRow.updated_at.desc())
            ).all()
        return [self._pref_dict(row) for row in rows]

    def record_feedback(
        self,
        *,
        user_id: str,
        item_id: str,
        feedback_type: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        allowed = {"like", "dislike", "purchased", "not_interested"}
        if feedback_type not in allowed:
            raise ValueError(f"unsupported feedback_type: {feedback_type}")
        user_id = str(user_id)
        item_id = str(item_id)
        self._ensure_seed(user_id)
        with self.Session() as session:
            row = FeedbackRow(
                user_id=user_id,
                item_id=item_id,
                feedback_type=feedback_type,
                query=(query or "")[:500],
                created_at=_utcnow(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
        preference = self.add_preference(
            user_id,
            feedback_type,
            item_id,
            source="feedback",
            confidence=0.8,
        )
        return {
            "id": int(row.id),
            "user_id": user_id,
            "item_id": item_id,
            "feedback_type": feedback_type,
            "preference": preference,
        }

    def close(self) -> None:
        self.engine.dispose()
