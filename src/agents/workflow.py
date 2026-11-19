"""LangGraph workflow connecting the four Phase 4 agents.

The graph is intentionally small and auditable:

    Intent -> Profile -> Recommendation -> Review RAG (real BGE/Chroma evidence)
           -> Critic -> (retry Recommendation | Finalize -> Memory update)

No node stores hidden chain-of-thought.  The trace contains only status,
latency, tool names, counts and short result summaries.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.critic_agent import CriticAgent
from src.agents.intent_agent import IntentAgent
from src.agents.profile_agent import ProfileAgent, UserProfileStore
from src.agents.product_tools import ProductToolkit
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.state import AgentState
from src.core.config import AppConfig, get_app_config
from src.core.exceptions import (
    AgentWorkflowError,
    ConfigurationError,
    DeepSeekError,
)
from src.core.logger import get_logger
from src.llm.deepseek_client import DeepSeekClient
from src.memory.memory_manager import MemoryManager
from src.rag.embedding import build_embedding_from_config
from src.rag.retriever import ReviewRetriever
from src.services.recommendation_service import RecommendationEngine

logger = get_logger(__name__)

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - depends on optional runtime package
    SqliteSaver = None  # type: ignore[assignment]


def _checkpoint_db_is_usable(path: Path) -> bool:
    """Return False when an existing SQLite checkpoint file is unreadable.

    The API and local ablation scripts can be pointed at the same bind-mounted
    file.  SQLite can still be damaged by separate writers; detecting it at
    startup lets the service back up and recreate the checkpoint DB instead of
    returning HTTP 500 on the first chat request.
    """
    if not path.exists():
        return True
    try:
        with sqlite3.connect(str(path)) as connection:
            row = connection.execute("PRAGMA quick_check").fetchone()
            return bool(row and str(row[0]).lower() == "ok")
    except sqlite3.DatabaseError:
        return False


class AgentWorkflow:
    """Compiled LangGraph workflow with four real agent nodes."""

    def __init__(
        self,
        *,
        config: AppConfig,
        intent_agent: IntentAgent,
        profile_agent: ProfileAgent,
        recommendation_agent: RecommendationAgent,
        critic_agent: CriticAgent,
        toolkit: ProductToolkit | None = None,
        client: Any | None = None,
        checkpointer: Any | None = None,
        memory_manager: MemoryManager | None = None,
        review_retriever: ReviewRetriever | None = None,
        enable_review_rag: bool = True,
        enable_critic: bool = True,
    ) -> None:
        self.config = config
        self.intent_agent = intent_agent
        self.profile_agent = profile_agent
        self.recommendation_agent = recommendation_agent
        self.critic_agent = critic_agent
        self.toolkit = toolkit
        self.client = client
        self.memory_manager = memory_manager
        self.review_retriever = review_retriever
        self.enable_review_rag = bool(enable_review_rag)
        self.enable_critic = bool(enable_critic)
        self._sqlite_connection: sqlite3.Connection | None = None

        if checkpointer is None:
            checkpointer, kind = self._make_checkpointer()
        else:
            kind = "injected"
        self.checkpointer = checkpointer
        self.checkpointer_kind = kind
        self.graph: CompiledStateGraph = self._build_graph()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _make_checkpointer(self) -> tuple[Any, str]:
        if not self.config.agent.workflow.enable_checkpointing:
            return MemorySaver(), "memory"
        if SqliteSaver is None:
            logger.warning(
                "sqlite_checkpoint_unavailable; using in-memory checkpoint"
            )
            return MemorySaver(), "memory"
        path = self.config.resolve_path(
            self.config.agent.workflow.checkpoint_db_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _checkpoint_db_is_usable(path):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = path.with_name(f"{path.name}.corrupt-{stamp}")
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(path) + suffix)
                if candidate.exists():
                    candidate.replace(Path(str(backup) + suffix))
            logger.warning(
                "agent_checkpoint_corrupt backed_up=%s", backup
            )
        try:
            connection = sqlite3.connect(
                str(path), check_same_thread=False
            )
            self._sqlite_connection = connection
            logger.info("agent_checkpoint_sqlite path=%s", path)
            return SqliteSaver(connection), "sqlite"
        except Exception as exc:  # pragma: no cover - filesystem/SQLite edge
            logger.warning(
                "sqlite_checkpoint_failed error=%s; using in-memory checkpoint",
                type(exc).__name__,
            )
            return MemorySaver(), "memory"

    @classmethod
    def from_artifacts(
        cls,
        config: AppConfig | None = None,
        *,
        checkpointer: Any | None = None,
    ) -> "AgentWorkflow":
        config = config or get_app_config()
        try:
            engine = RecommendationEngine.from_artifacts(config)
            client = DeepSeekClient()
        except DeepSeekError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"Could not initialise the agent workflow: {type(exc).__name__}"
            ) from exc

        categories = sorted(
            {
                str(metadata.get("main_category") or "")
                for metadata in engine.item_metadata.values()
                if str(metadata.get("main_category") or "").strip()
            }
        )
        profile_store = UserProfileStore.from_artifacts(
            config.project_root,
            item_metadata=engine.item_metadata,
        )
        embedder = build_embedding_from_config(config)
        memory_manager = MemoryManager.from_artifacts(
            config,
            item_metadata=engine.item_metadata,
            embedder=embedder,
            seed_store=profile_store,
        )
        try:
            review_retriever: ReviewRetriever | None = ReviewRetriever.from_config(
                config, embedder
            )
        except Exception as exc:  # pragma: no cover - Vector backend edge
            logger.warning(
                "review_retriever_unavailable error=%s", type(exc).__name__
            )
            review_retriever = None
        toolkit = ProductToolkit(
            engine=engine,
            profile_store=profile_store,
            item_metadata=engine.item_metadata,
            memory_manager=memory_manager,
            review_retriever=review_retriever,
        )
        return cls(
            config=config,
            intent_agent=IntentAgent(
                client,
                categories=categories,
                max_attempts=config.agent.llm.json_max_attempts,
                temperature=config.agent.llm.temperature,
                max_conversation_messages=(
                    config.agent.memory.short_term_max_turns * 2
                ),
            ),
            profile_agent=ProfileAgent(
                profile_store, memory_manager=memory_manager
            ),
            recommendation_agent=RecommendationAgent(
                client=client,
                config=config,
                toolkit=toolkit,
            ),
            critic_agent=CriticAgent(
                client=client,
                config=config,
                item_metadata=engine.item_metadata,
            ),
            toolkit=toolkit,
            client=client,
            checkpointer=checkpointer,
            memory_manager=memory_manager,
            review_retriever=review_retriever,
        )

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("intent", self._intent_node)
        builder.add_node("profile", self._profile_node)
        builder.add_node("recommendation", self._recommendation_node)
        builder.add_node("review_rag", self._review_rag_node)
        if self.enable_critic:
            builder.add_node("critic", self._critic_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_node("memory_update", self._memory_update_node)

        builder.add_edge(START, "intent")
        builder.add_conditional_edges(
            "intent",
            self._route_after_intent,
            {"clarify": "finalize", "continue": "profile"},
        )
        builder.add_edge("profile", "recommendation")
        builder.add_edge("recommendation", "review_rag")
        if self.enable_critic:
            builder.add_edge("review_rag", "critic")
            builder.add_conditional_edges(
                "critic",
                self._route_after_critic,
                {"retry": "recommendation", "finalize": "finalize"},
            )
        else:
            builder.add_edge("review_rag", "finalize")
        builder.add_edge("finalize", "memory_update")
        builder.add_edge("memory_update", END)
        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _intent_node(self, state: AgentState) -> dict[str, Any]:
        result = self.intent_agent.run(
            state.get("query", ""), state.get("conversation", [])
        )
        intent = result.get("intent", {})
        clarification = (
            intent.get("clarification_question")
            if intent.get("needs_clarification")
            else None
        )
        return {
            "intent": intent,
            "clarification": clarification,
            "trace": list(result.get("trace", [])),
        }

    def _profile_node(self, state: AgentState) -> dict[str, Any]:
        result = self.profile_agent.run(
            state.get("user_id", ""),
            state.get("intent", {}),
            state.get("conversation", []),
            query=state.get("query", ""),
        )
        return {
            "profile": result.get("profile", {}),
            "trace": list(result.get("trace", [])),
        }

    def _recommendation_node(self, state: AgentState) -> dict[str, Any]:
        result = self.recommendation_agent.run(
            user_id=state.get("user_id", ""),
            query=state.get("query", ""),
            intent=state.get("intent", {}),
            profile=state.get("profile", {}),
            extra_exclude_item_ids=state.get("extra_exclude_item_ids", []),
        )
        return {
            "recommendations": result.get("recommendations", []),
            "recommendation_trace": result.get("recommendation_trace", {}),
            "trace": list(result.get("trace", [])),
            "errors": list(result.get("errors", [])),
        }

    def _review_rag_node(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        recommendations = [dict(item) for item in state.get("recommendations", [])]
        if not recommendations:
            return {
                "review_evidence": [],
                "review_status": "no_recommendations",
                "trace": [
                    {
                        "node": "review_rag",
                        "status": "skipped",
                        "latency_ms": 0,
                        "summary": "no recommendations to attach review evidence to",
                    }
                ],
            }
        if not self.enable_review_rag:
            for item in recommendations:
                item["review_evidence"] = []
                item["review_evidence_status"] = "disabled_for_ablation"
            return {
                "recommendations": recommendations,
                "review_evidence": [],
                "review_status": "disabled_for_ablation",
                "trace": [
                    {
                        "node": "review_rag",
                        "status": "skipped",
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "summary": (
                            "review RAG disabled for ablation; no evidence was "
                            "fabricated"
                        ),
                    }
                ],
            }
        if self.review_retriever is None or self.toolkit is None:
            for item in recommendations:
                item["review_evidence"] = []
                item["review_evidence_status"] = "unavailable"
            return {
                "recommendations": recommendations,
                "review_evidence": [],
                "review_status": "unavailable",
                "trace": [
                    {
                        "node": "review_rag",
                        "status": "unavailable",
                        "latency_ms": 0,
                        "summary": (
                            "review index/retriever not configured; no evidence "
                            "was fabricated"
                        ),
                    }
                ],
            }
        try:
            index_count = int(self.review_retriever.count)
        except Exception as exc:  # pragma: no cover - vector backend edge
            logger.warning(
                "review_index_count_failed error=%s", type(exc).__name__
            )
            index_count = 0
        if index_count == 0:
            for item in recommendations:
                item["review_evidence"] = []
                item["review_evidence_status"] = "index_empty"
            return {
                "recommendations": recommendations,
                "review_evidence": [],
                "review_status": "index_empty",
                "trace": [
                    {
                        "node": "review_rag",
                        "status": "unavailable",
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "summary": (
                            "review index is empty/unavailable; build it with "
                            "scripts/build_review_index.py; no evidence invented"
                        ),
                    }
                ],
            }

        flattened: list[dict[str, Any]] = []
        found = 0
        errors = 0
        max_items = min(len(recommendations), 10)
        for index, item in enumerate(recommendations):
            item_id = str(item.get("item_id") or "")
            if index >= max_items:
                item["review_evidence"] = []
                item["review_evidence_status"] = "not_requested"
                continue
            title = str(item.get("title") or "")
            category = str(item.get("main_category") or "")
            # BGE-small-en works best with English review language; use the
            # real English product title plus a neutral review intent. Product
            # scoping is enforced by the Chroma parent_asin filter.
            review_query = (
                f"{title[:120]} {category[:60]} "
                "quality performance reliability pros cons"
            ).strip()
            result = self.toolkit.registry.execute(
                "search_reviews",
                {
                    "item_id": item_id,
                    "query": review_query,
                    "top_k": int(self.config.agent.rag.top_k),
                },
            )
            if not result.ok:
                errors += 1
                item["review_evidence"] = []
                item["review_evidence_status"] = "error"
                continue
            payload = result.result or {}
            evidence = list(payload.get("evidence") or [])
            # A second product-scope guard, independent of the retriever.
            evidence = [
                review
                for review in evidence
                if str(review.get("parent_asin")) == item_id
            ]
            item["review_evidence"] = evidence
            item["review_evidence_status"] = (
                "ok" if evidence else "no_evidence"
            )
            if evidence:
                found += 1
                first = evidence[0]
                sentiment = (
                    f"{first.get('rating')}星"
                    if first.get("rating") is not None
                    else "未评分"
                )
                timestamp = str(first.get("timestamp") or "")[:10]
                excerpt = str(first.get("text") or "").strip()[:180]
                item["evidence_summary"] = (
                    f"真实评论（{sentiment}，{timestamp}）：{excerpt}"
                )
                flattened.extend(
                    [{**review, "item_id": item_id} for review in evidence[:3]]
                )
        if found == len(recommendations[:max_items]):
            status = "ok"
        elif found > 0:
            status = "partial"
        elif errors:
            status = "error"
        else:
            status = "no_evidence"
        return {
            "recommendations": recommendations,
            "review_evidence": flattened,
            "review_status": status,
            "trace": [
                {
                    "node": "review_rag",
                    "status": "ok" if found else status,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "summary": (
                        f"items_with_evidence={found}/{max_items}; "
                        f"evidence_chunks={len(flattened)}; errors={errors}"
                    ),
                }
            ],
        }

    def _critic_node(self, state: AgentState) -> dict[str, Any]:
        result = self.critic_agent.run(
            query=state.get("query", ""),
            intent=state.get("intent", {}),
            profile=state.get("profile", {}),
            recommendations=state.get("recommendations", []),
        )
        critic = result.get("critic", {})
        previous_exclusions = set(state.get("extra_exclude_item_ids", []))
        previous_exclusions.update(critic.get("rejected_item_ids", []))
        retry_count = int(state.get("retry_count", 0)) + (
            0 if critic.get("approved") else 1
        )
        return {
            "critic": critic,
            "extra_exclude_item_ids": sorted(previous_exclusions),
            "retry_count": retry_count,
            "trace": list(result.get("trace", [])),
        }

    def _finalize_node(self, state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", {})
        critic = dict(state.get("critic", {}))
        if not self.enable_critic:
            critic = {
                "enabled": False,
                "skipped": True,
                "approved": None,
                "fatal": False,
                "issues": [],
                "rejected_item_ids": [],
                "summary": "Critic disabled for Phase 7 ablation",
                "llm_status": "skipped",
                "llm_approved": None,
                "review_evidence_status": "not_evaluated",
            }
        recommendations = list(state.get("recommendations", []))
        rejected = set(critic.get("rejected_item_ids", []))
        visible = [
            item
            for item in recommendations
            if str(item.get("item_id")) not in rejected
        ] if not critic.get("approved") else recommendations

        clarification = state.get("clarification")
        if intent.get("needs_clarification"):
            status = "needs_clarification"
        elif not self.enable_critic:
            status = "completed" if visible else "no_recommendations"
        elif critic.get("llm_status") == "failed":
            status = "critic_unavailable"
        elif not visible:
            # ``extra_exclude_item_ids`` accumulates every item the Critic
            # rejected across retries.  If the final retry exhausts the
            # candidate pool, the last Critic call sees an empty list and its
            # own ``rejected`` set is empty; the accumulated exclusions still
            # tell us that earlier candidates were rejected rather than that
            # retrieval never found anything.
            has_prior_rejections = bool(state.get("extra_exclude_item_ids"))
            status = (
                "rejected_after_retries"
                if (recommendations and rejected) or has_prior_rejections
                else "no_recommendations"
            )
        elif critic.get("approved"):
            status = "completed"
        elif critic.get("fatal"):
            status = "completed_with_warnings"
        else:
            status = "completed_with_warnings"

        if status == "needs_clarification":
            assistant_message = clarification or "请补充商品需求后重试。"
        elif visible:
            lines = []
            for item in visible:
                title = item.get("title") or item.get("item_id")
                reason = item.get("recommendation_reason") or ""
                lines.append(f"- {title}（{item.get('item_id')}）：{reason}")
                if item.get("evidence_summary"):
                    lines.append(f"  证据：{item['evidence_summary']}")
            assistant_message = "\n".join(lines)
        else:
            if status == "critic_unavailable":
                assistant_message = (
                    "推荐结果未通过 Critic 审核，且 Critic 服务当前不可用。"
                )
            elif status == "rejected_after_retries":
                assistant_message = (
                    "候选商品均未通过 Critic 的真实性与相关性审核；"
                    "系统没有用占位或虚构商品补齐。"
                )
            else:
                assistant_message = (
                    "没有满足条件的真实商品；系统没有生成占位或虚构商品。"
                )

        conversation = list(state.get("conversation", []))
        conversation.extend(
            [
                {"role": "user", "content": state.get("query", "")},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        final = {
            "user_id": state.get("user_id", ""),
            "query": state.get("query", ""),
            "status": status,
            "intent": intent,
            "profile": state.get("profile", {}),
            "recommendations": visible,
            "recommendation_trace": state.get("recommendation_trace", {}),
            "critic": critic,
            "review_evidence": state.get("review_evidence", []),
            "review_status": state.get("review_status", ""),
            "clarification": clarification,
            "assistant_message": assistant_message,
            "conversation": conversation,
        }
        return {
            "final": final,
            "conversation": conversation,
            "status": status,
            "trace": [
                {
                    "node": "finalize",
                    "status": status,
                    "latency_ms": 0,
                    "summary": (
                        f"visible_recommendations={len(visible)}; "
                        f"critic_approved={bool(critic.get('approved'))}; "
                        f"critic_skipped={not self.enable_critic}"
                    ),
                }
            ],
        }

    def _memory_update_node(self, state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        updates = list(state.get("intent", {}).get("memory_updates", []))
        if self.memory_manager is None:
            memory_update = {
                "status": "unavailable",
                "written": False,
                "reason": "persistent memory manager is not configured",
            }
            trace_status = "unavailable"
        elif not updates:
            memory_update = {
                "status": "no_update",
                "written": False,
                "reason": (
                    "no explicit durable memory update was requested; temporary "
                    "query constraints were not written"
                ),
                "applied": [],
                "skipped": [],
            }
            trace_status = "no_update"
        else:
            try:
                memory_update = self.memory_manager.apply_intent_updates(
                    user_id=state.get("user_id", ""),
                    updates=updates,
                )
                trace_status = str(memory_update.get("status", "no_update"))
            except Exception as exc:  # pragma: no cover - storage edge
                logger.warning(
                    "memory_update_failed error=%s", type(exc).__name__
                )
                memory_update = {
                    "status": "error",
                    "written": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "applied": [],
                    "skipped": [],
                }
                trace_status = "error"
        final = dict(state.get("final", {}))
        final["memory_update"] = memory_update
        return {
            "memory_update": memory_update,
            "final": final,
            "trace": [
                {
                    "node": "memory_update",
                    "status": trace_status,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "summary": (
                        f"status={memory_update.get('status')}; "
                        f"written={memory_update.get('written')}; "
                        f"updates={len(updates)}"
                    ),
                }
            ],
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    @staticmethod
    def _route_after_intent(state: AgentState) -> str:
        return "clarify" if state.get("intent", {}).get("needs_clarification") else "continue"

    def _route_after_critic(self, state: AgentState) -> str:
        critic = state.get("critic", {})
        if critic.get("approved") or critic.get("fatal"):
            return "finalize"
        retry_count = int(state.get("retry_count", 0))
        if retry_count <= int(self.config.agent.workflow.max_critic_retries):
            return "retry"
        return "finalize"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        *,
        user_id: str,
        thread_id: str | None = None,
        conversation: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        query = str(query).strip()
        if not query:
            raise ValueError("query must not be empty")
        thread_id = thread_id or f"agent-{uuid.uuid4().hex[:12]}"
        graph_config = {"configurable": {"thread_id": thread_id}}

        prior: list[dict[str, str]] = []
        if conversation is not None:
            prior = [dict(message) for message in conversation]
        else:
            try:
                snapshot = self.graph.get_state(graph_config)
                if snapshot and snapshot.values:
                    prior = list(snapshot.values.get("conversation", []))
            except Exception:
                prior = []

        initial: AgentState = {
            "user_id": str(user_id),
            "query": query,
            "thread_id": thread_id,
            "conversation": prior,
            "intent": {},
            "profile": {},
            "recommendations": [],
            "recommendation_trace": {},
            "review_evidence": [],
            "review_status": "",
            "critic": {},
            "extra_exclude_item_ids": [],
            "retry_count": 0,
            "clarification": None,
            "status": "running",
            "final": {},
            "memory_update": {},
            "trace": [],
            "errors": [],
        }
        started = time.perf_counter()
        try:
            state = self.graph.invoke(
                initial,
                config={
                    **graph_config,
                    "recursion_limit": int(
                        self.config.agent.workflow.max_total_steps
                    ),
                },
            )
        except GraphRecursionError as exc:
            raise AgentWorkflowError(
                "Agent workflow exceeded max_total_steps; no fake result was returned"
            ) from exc

        final = dict(state.get("final", {}))
        if not final:
            raise AgentWorkflowError(
                "Agent workflow finished without a final response"
            )
        final["thread_id"] = thread_id
        final["agent_trace"] = list(state.get("trace", []))
        final["errors"] = list(state.get("errors", []))
        final["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        final["checkpointing"] = {
            "kind": self.checkpointer_kind,
            "persistent": self.checkpointer_kind == "sqlite",
        }
        logger.info(
            "agent_workflow_completed thread=%s status=%s recommendations=%d latency_ms=%.1f",
            thread_id,
            final.get("status"),
            len(final.get("recommendations", [])),
            final["latency_ms"],
        )
        return final

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:  # pragma: no cover - best effort
                pass
        if self.memory_manager is not None:
            try:
                self.memory_manager.close()
            except Exception:  # pragma: no cover - best effort
                pass
        if self.review_retriever is not None:
            try:
                self.review_retriever.close()
            except Exception:  # pragma: no cover - best effort
                pass
        if self._sqlite_connection is not None:
            try:
                self._sqlite_connection.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._sqlite_connection = None

    def __enter__(self) -> "AgentWorkflow":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
