"""FastAPI application for Phase 6."""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.agents.workflow import AgentWorkflow
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    ModelMetricsResponse,
    PreferenceCreateRequest,
    PreferenceMutationResponse,
    ProductResponse,
    ProfileResponse,
    RecommendationResponse,
    UserRecommendRequest,
)
from src.core.config import PROJECT_ROOT, AppConfig, get_app_config
from src.core.exceptions import (
    AgentWorkflowError,
    DeepSeekError,
    ModelNotReadyError,
    RecSysError,
)
from src.core.logger import get_logger, redact_secrets

logger = get_logger(__name__)
API_VERSION = "1.1.0"


@dataclass
class AppServices:
    config: AppConfig
    workflow: Any
    engine: Any
    memory_manager: Any
    profile_store: Any
    review_retriever: Any | None
    item_metadata: dict[str, dict[str, Any]]
    workflow_lock: threading.Lock = field(default_factory=threading.Lock)
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    request_count: int = 0
    error_count: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    review_count: int | None = None


def build_services(config: AppConfig | None = None) -> AppServices:
    """Load the real Phase 5 workflow and derive API services from it."""
    config = config or get_app_config()
    workflow = AgentWorkflow.from_artifacts(config)
    engine = workflow.toolkit.engine if workflow.toolkit is not None else None
    if engine is None:
        raise ModelNotReadyError(
            "Agent workflow does not expose the recommendation engine"
        )
    return AppServices(
        config=config,
        workflow=workflow,
        engine=engine,
        memory_manager=workflow.memory_manager,
        profile_store=workflow.profile_agent.store,
        review_retriever=workflow.review_retriever,
        item_metadata=dict(engine.item_metadata),
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass


def _latency_stats(values: deque[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return round(float(ordered[index]), 2)

    return {
        "count": float(len(ordered)),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": round(float(max(ordered)), 2),
        "mean": round(float(statistics.mean(ordered)), 2),
    }


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": redact_secrets(message)}}


def create_app(services: AppServices | None = None) -> FastAPI:
    """Create the FastAPI app; tests may inject fake services."""
    _load_dotenv()
    expected_api_key = os.getenv("RECSYS_API_KEY", "").strip() or None
    cors_origins = [
        value.strip()
        for value in os.getenv(
            "RECSYS_CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501",
        ).split(",")
        if value.strip()
    ]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is not None:
            app.state.services = services
            owned = False
        else:
            app.state.services = await run_in_threadpool(build_services)
            owned = True
        logger.info("api_started version=%s", API_VERSION)
        try:
            yield
        finally:
            if owned:
                try:
                    await run_in_threadpool(app.state.services.workflow.close)
                except Exception:
                    logger.warning("api_workflow_close_failed", exc_info=True)
            logger.info("api_stopped")

    app = FastAPI(
        title="AgentRec API",
        version=API_VERSION,
        description=(
            "AgentRec personalized recommendation API with a real LangGraph "
            "workflow, deep-learning recall/ranking, SQLite/Chroma memory and "
            "review evidence."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Auth / metrics middleware
    # ------------------------------------------------------------------
    async def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if expected_api_key is not None and x_api_key != expected_api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing X-API-Key",
            )

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            current: AppServices | None = getattr(
                request.app.state, "services", None
            )
            if current is not None:
                current.request_count += 1
                current.latencies.append(latency_ms)
                if status_code >= 400:
                    current.error_count += 1
            logger.info(
                "api_request method=%s path=%s status=%d latency_ms=%.1f "
                "request_id=%s",
                request.method,
                request.url.path,
                status_code,
                latency_ms,
                request_id,
            )
            if "response" in locals():
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "validation_error", "Request validation failed"
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException):
        if exc.status_code == 401:
            code = "unauthorized"
        elif exc.status_code == 404:
            code = "not_found"
        else:
            code = "http_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code, str(exc.detail)),
        )

    @app.exception_handler(DeepSeekError)
    async def deepseek_handler(request: Request, exc: DeepSeekError):
        logger.warning("api_deepseek_error error=%s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content=_error_payload("deepseek_unavailable", str(exc)),
        )

    @app.exception_handler(ModelNotReadyError)
    async def not_ready_handler(request: Request, exc: ModelNotReadyError):
        return JSONResponse(
            status_code=503,
            content=_error_payload("model_not_ready", str(exc)),
        )

    @app.exception_handler(AgentWorkflowError)
    async def workflow_handler(request: Request, exc: AgentWorkflowError):
        return JSONResponse(
            status_code=500,
            content=_error_payload("agent_workflow_error", str(exc)),
        )

    @app.exception_handler(RecSysError)
    async def recsys_handler(request: Request, exc: RecSysError):
        return JSONResponse(
            status_code=500,
            content=_error_payload("application_error", str(exc)),
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        logger.exception("api_unexpected_error")
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "internal_error", "Internal server error"
            ),
        )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health(request: Request):
        current: AppServices = request.app.state.services
        review_count = current.review_count
        review_status = "not_configured"
        if current.review_retriever is not None:
            if review_count is None:
                try:
                    review_count = int(current.review_retriever.count)
                    current.review_count = review_count
                except Exception as exc:
                    review_status = f"error:{type(exc).__name__}"
                    review_count = None
            if review_count is not None:
                review_status = "ready" if review_count else "empty"
        elif review_count is None:
            review_status = "unavailable"
        return HealthResponse(
            status="ok",
            api_version=API_VERSION,
            components={
                "workflow": "ready",
                "retrievers": {
                    name: "ready" for name in current.engine.retrievers
                },
                "review_rag": {
                    "status": review_status,
                    "count": review_count,
                    "max_reviews_per_train_item": 2,
                },
                "memory": {
                    "status": getattr(
                        current.memory_manager, "semantic_status", "unknown"
                    )
                },
                "checkpointer": getattr(
                    current.workflow, "checkpointer_kind", "unknown"
                ),
                "llm_model": current.workflow.client.model_name
                if getattr(current.workflow, "client", None) is not None
                else "not_configured",
            },
            latency_stats=_latency_stats(current.latencies),
        )

    @app.get(
        "/api/v1/metrics",
        response_model=MetricsResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def request_metrics(request: Request):
        current: AppServices = request.app.state.services
        return MetricsResponse(
            request_count=current.request_count,
            error_count=current.error_count,
            latency_ms=_latency_stats(current.latencies),
            uptime_seconds=round(
                time.perf_counter() - current.started_at, 2
            ),
        )

    @app.get(
        "/api/v1/metrics/model",
        response_model=ModelMetricsResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def model_metrics(request: Request):
        current: AppServices = request.app.state.services
        logs = current.config.project_root / "logs"
        names = {
            "itemcf_valid": "metrics_itemcf_valid.json",
            "itemcf_test": "metrics_itemcf_test.json",
            "two_tower_valid": "metrics_two_tower_valid.json",
            "two_tower_test": "metrics_two_tower_test.json",
            "lightgcn_valid": "metrics_lightgcn_valid.json",
            "lightgcn_test": "metrics_lightgcn_test.json",
            "popularity_valid": "metrics_popularity_valid.json",
            "popularity_test": "metrics_popularity_test.json",
            "deepfm_valid": "metrics_deepfm_valid.json",
            "din_valid": "metrics_din_valid.json",
            "phase3_fusion_valid": "phase3_fusion_valid.json",
            "phase3_fusion_test": "phase3_fusion_test.json",
            "phase3_pipeline_valid": "phase3_pipeline_valid.json",
            "phase3_pipeline_test": "phase3_pipeline_test.json",
            "phase5_review_index": "phase5_review_index.json",
            "phase7_ablation_valid": "phase7_ablation_valid.json",
            "phase7_ablation_test": "phase7_ablation_test.json",
            "phase7_agent_ablation": "phase7_agent_ablation.json",
        }
        metrics: dict[str, Any] = {}
        for key, filename in names.items():
            path = logs / filename
            if path.is_file():
                try:
                    metrics[key] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metrics[key] = {"error": "unreadable"}
        return ModelMetricsResponse(metrics=metrics)

    @app.post(
        "/api/v1/recommend/chat",
        response_model=ChatResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def recommend_chat(request: Request, body: ChatRequest):
        current: AppServices = request.app.state.services

        def run_workflow():
            with current.workflow_lock:
                return current.workflow.run(
                    body.query,
                    user_id=body.user_id,
                    thread_id=body.thread_id,
                )

        final = await run_in_threadpool(run_workflow)
        return ChatResponse(**final)

    @app.post(
        "/api/v1/recommend/user/{user_id}",
        response_model=RecommendationResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def recommend_user(
        request: Request, user_id: str, body: UserRecommendRequest
    ):
        current: AppServices = request.app.state.services
        effective: dict[str, Any] = {}
        if body.use_memory and current.memory_manager is not None:
            try:
                profile = current.memory_manager.get_profile(str(user_id))
                effective = profile.get("effective_constraints", {}) or {}
            except Exception:
                logger.warning("api_memory_profile_failed", exc_info=True)
        budget_min = (
            body.budget_min
            if body.budget_min is not None
            else effective.get("budget_min")
        )
        budget_max = (
            body.budget_max
            if body.budget_max is not None
            else effective.get("budget_max")
        )
        category = body.category or effective.get("category")
        started = time.perf_counter()
        result = await run_in_threadpool(
            current.engine.recommend,
            str(user_id),
            top_k=body.top_k,
            budget_min=budget_min,
            budget_max=budget_max,
            category=category,
            use_mmr=body.use_mmr,
            strategy=body.strategy,
        )
        result["latency_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 1
        )
        result["profile"] = {
            "memory_effective_constraints": effective,
        }
        if not result.get("recommendations"):
            result["message"] = (
                "No real product satisfied the request; no placeholder was generated."
            )
        return RecommendationResponse(**result)

    @app.post(
        "/api/v1/feedback",
        response_model=FeedbackResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def feedback(request: Request, body: FeedbackRequest):
        current: AppServices = request.app.state.services
        if current.memory_manager is None:
            raise ModelNotReadyError("Persistent memory is not configured")
        if str(body.item_id) not in current.item_metadata:
            raise HTTPException(status_code=404, detail="Product not found")
        result = await run_in_threadpool(
            current.memory_manager.record_feedback,
            user_id=body.user_id,
            item_id=body.item_id,
            feedback_type=body.feedback_type,
            query=body.query,
        )
        preferences = current.memory_manager.list_preferences(body.user_id)
        return FeedbackResponse(feedback=result, preferences=preferences)

    @app.get(
        "/api/v1/user/{user_id}/profile",
        response_model=ProfileResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def user_profile(
        request: Request,
        user_id: str,
        query: str | None = Query(default=None, max_length=500),
    ):
        current: AppServices = request.app.state.services
        result = await run_in_threadpool(
            current.workflow.profile_agent.run,
            str(user_id),
            {},
            [],
            query=query,
        )
        profile = result.get("profile", {})
        preferences = (
            current.memory_manager.list_preferences(str(user_id))
            if current.memory_manager is not None
            else []
        )
        return ProfileResponse(
            user_id=str(user_id), profile=profile, preferences=preferences
        )

    @app.post(
        "/api/v1/user/{user_id}/preferences",
        response_model=PreferenceMutationResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def add_user_preference(
        request: Request, user_id: str, body: PreferenceCreateRequest
    ):
        current: AppServices = request.app.state.services
        if current.memory_manager is None:
            raise HTTPException(
                status_code=503,
                detail="persistent memory is not configured",
            )
        preference = await run_in_threadpool(
            current.memory_manager.add_preference,
            user_id=str(user_id),
            kind=body.kind,
            value=body.value,
            source="explicit_user",
            confidence=body.confidence,
        )
        preferences = await run_in_threadpool(
            current.memory_manager.list_preferences, str(user_id)
        )
        return PreferenceMutationResponse(
            preference=preference, preferences=preferences
        )

    @app.delete(
        "/api/v1/user/{user_id}/preferences/{preference_id}",
        response_model=PreferenceMutationResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def delete_user_preference(
        request: Request, user_id: str, preference_id: int
    ):
        current: AppServices = request.app.state.services
        if current.memory_manager is None:
            raise HTTPException(
                status_code=503,
                detail="persistent memory is not configured",
            )
        deleted = await run_in_threadpool(
            current.memory_manager.delete_preference,
            user_id=str(user_id),
            preference_id=int(preference_id),
        )
        preferences = await run_in_threadpool(
            current.memory_manager.list_preferences, str(user_id)
        )
        return PreferenceMutationResponse(
            deleted=int(deleted), preferences=preferences
        )

    @app.get(
        "/api/v1/products/{item_id}",
        response_model=ProductResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def product(request: Request, item_id: str):
        current: AppServices = request.app.state.services
        metadata = current.item_metadata.get(str(item_id))
        if metadata is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return ProductResponse(item_id=str(item_id), **dict(metadata))

    return app


app = create_app()
