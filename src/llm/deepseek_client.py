"""Unified DeepSeek client used by every agent.

Features required by the project:

* OpenAI-compatible SDK pointed at ``DEEPSEEK_BASE_URL``.
* API key from environment only (never logged or committed).
* Configurable model, timeout and retry count.
* Transparent retries for transient failures with exponential backoff.
* Chat, JSON Output (Pydantic-validated) and Function Calling support.
* Per-call token, latency and attempt accounting written to
  ``logs/llm_usage.jsonl`` (no prompts or user data).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar, cast

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.core.config import DeepSeekSettings, PROJECT_ROOT, get_deepseek_settings
from src.core.exceptions import (
    DeepSeekAPIError,
    DeepSeekAuthenticationError,
    DeepSeekConfigurationError,
    DeepSeekConnectionError,
    DeepSeekRateLimitError,
    DeepSeekResponseError,
    DeepSeekServerError,
    DeepSeekTimeoutError,
)
from src.core.logger import get_logger
from src.llm.prompts import build_json_repair_prompt, build_json_system_prompt
from src.llm.schemas import LLMResponse, LLMUsage, ToolCall

T = TypeVar("T", bound=BaseModel)

_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 8.0
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


@dataclass
class ClientStats:
    """In-process counters for observability and agent evaluation."""

    successful_calls: int = 0
    failed_calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: int = 0


class DeepSeekClient:
    """Thread-safe wrapper around the DeepSeek chat-completions API."""

    def __init__(
        self,
        settings: DeepSeekSettings | None = None,
        *,
        client: Any | None = None,
        usage_log_path: str | Path | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else get_deepseek_settings()
        if not self.settings.has_api_key:
            raise DeepSeekConfigurationError(
                "DEEPSEEK_API_KEY is not configured. Copy .env.example to .env, "
                "set DEEPSEEK_API_KEY, and restart. Never commit the key to Git."
            )

        assert self.settings.api_key is not None  # guarded by has_api_key
        api_key = self.settings.api_key.get_secret_value().strip()
        self.model_name = self.settings.model

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            try:
                # Python's urllib/httpx reads the Windows system proxy from the
                # registry.  A stale 127.0.0.1 proxy must not silently break API
                # calls, so environment/system proxies are disabled unless the
                # user explicitly sets DEEPSEEK_PROXY.
                http_client_kwargs: dict[str, Any] = {
                    "trust_env": False,
                    "timeout": self.settings.timeout,
                }
                if self.settings.proxy:
                    http_client_kwargs["proxy"] = self.settings.proxy
                http_client = openai.DefaultHttpxClient(**http_client_kwargs)
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=self.settings.base_url,
                    timeout=self.settings.timeout,
                    # Retries are handled here so attempts are logged consistently.
                    max_retries=0,
                    http_client=http_client,
                )
            except Exception as exc:  # pragma: no cover - SDK construction is stable
                raise DeepSeekConfigurationError(
                    f"Could not initialise the DeepSeek client: {type(exc).__name__}"
                ) from exc

        self._usage_log_path = (
            Path(usage_log_path)
            if usage_log_path is not None
            else PROJECT_ROOT / "logs" / "llm_usage.jsonl"
        )
        self._sleep = sleep_fn or time.sleep
        self._stats_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._stats = ClientStats()
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        call_name: str = "chat",
    ) -> LLMResponse:
        """Run one chat completion and return a normalised response.

        Raises:
            DeepSeekConfigurationError: missing API key.
            DeepSeekAuthenticationError: invalid key / permission.
            DeepSeekTimeoutError: request timed out after retries.
            DeepSeekConnectionError: endpoint unreachable after retries.
            DeepSeekRateLimitError: rate limit persists after retries.
            DeepSeekServerError: DeepSeek 5xx persists after retries.
            DeepSeekResponseError: malformed model output or rejected request.
            DeepSeekAPIError: any other API failure.
        """
        request = self._build_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_body=extra_body,
        )
        response = self._execute(request, call_name=call_name)
        self._logger.info(
            "llm_call_completed call=%s model=%s finish_reason=%s "
            "prompt_tokens=%d completion_tokens=%d latency_ms=%d attempts=%d tool_calls=%d",
            call_name,
            response.model,
            response.finish_reason or "unknown",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.latency_ms,
            response.usage.attempts,
            len(response.tool_calls),
        )
        return response

    def chat_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        schema: type[T],
        *,
        max_attempts: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
        call_name: str | None = None,
    ) -> tuple[T, LLMResponse]:
        """Request JSON Output and validate it against a Pydantic schema.

        If validation fails, the model receives a repair instruction for up to
        ``max_attempts`` calls (network-level retries are separate).

        Returns:
            A ``(validated_model, raw_response)`` tuple.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        schema_json = json.dumps(
            schema.model_json_schema(), ensure_ascii=False, sort_keys=True
        )
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": build_json_system_prompt(schema_json)}
        ]
        conversation.extend(dict(message) for message in messages)

        base_call_name = call_name or f"json:{schema.__name__}"
        last_errors = "unknown validation error"

        for attempt in range(1, max_attempts + 1):
            response = self.chat(
                conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                call_name=f"{base_call_name}:attempt{attempt}",
            )
            raw_content = (response.content or "").strip()
            if not raw_content:
                last_errors = "empty response content"
            else:
                candidate = _strip_code_fences(raw_content)
                try:
                    parsed = schema.model_validate_json(candidate)
                    return cast(T, parsed), response
                except ValidationError as exc:
                    last_errors = _format_validation_errors(exc)

            if attempt < max_attempts:
                if raw_content:
                    conversation.append({"role": "assistant", "content": raw_content})
                conversation.append(
                    {"role": "user", "content": build_json_repair_prompt(last_errors)}
                )

        raise DeepSeekResponseError(
            f"DeepSeek did not return schema-valid JSON for {schema.__name__} "
            f"after {max_attempts} attempt(s): {last_errors}"
        )

    def stats(self) -> dict[str, int]:
        """Return a snapshot of in-process call counters."""
        with self._stats_lock:
            return asdict(self._stats)

    def reset_stats(self) -> None:
        """Reset in-process counters (usage log is append-only)."""
        with self._stats_lock:
            self._stats = ClientStats()

    def close(self) -> None:
        """Close the HTTP client created by this instance (no-op if injected)."""
        if not self._owns_client:
            return
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            self._logger.warning("Could not close DeepSeek HTTP client")

    def __enter__(self) -> "DeepSeekClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------
    def _build_request(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | Mapping[str, Any] | None,
        response_format: Mapping[str, Any] | None,
        extra_body: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages must not be empty")

        normalized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"messages[{index}] must be a mapping")
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"messages[{index}] has unsupported role: {role!r}")
            if role == "tool" and not message.get("tool_call_id"):
                raise ValueError(f"messages[{index}] is a tool message without tool_call_id")
            normalized.append(dict(message))

        resolved_temperature = (
            self.settings.temperature if temperature is None else temperature
        )
        if not 0.0 <= resolved_temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": normalized,
            "temperature": resolved_temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if tools is not None:
            if not tools:
                raise ValueError("tools must not be an empty sequence when provided")
            request["tools"] = [dict(tool) for tool in tools]
            request["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        elif tool_choice is not None:
            raise ValueError("tool_choice requires tools")
        if response_format is not None:
            request["response_format"] = dict(response_format)
        if extra_body is not None:
            request["extra_body"] = dict(extra_body)
        return request

    # ------------------------------------------------------------------
    # Execution, retry and error mapping
    # ------------------------------------------------------------------
    def _execute(self, request: dict[str, Any], call_name: str) -> LLMResponse:
        total_attempts = self.settings.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            started = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**request)
            except openai.OpenAIError as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                if self._is_retryable(exc) and attempt < total_attempts:
                    delay = min(
                        _RETRY_BASE_DELAY_S * (2 ** (attempt - 1)),
                        _RETRY_MAX_DELAY_S,
                    )
                    with self._stats_lock:
                        self._stats.retries += 1
                    self._logger.warning(
                        "DeepSeek transient failure; retrying call=%s attempt=%d/%d "
                        "error=%s delay_s=%.1f",
                        call_name,
                        attempt,
                        total_attempts,
                        type(exc).__name__,
                        delay,
                    )
                    self._sleep(delay)
                    continue

                with self._stats_lock:
                    self._stats.failed_calls += 1
                mapped = self._map_exception(exc, latency_ms)
                self._logger.error(
                    "DeepSeek call failed call=%s attempt=%d error=%s latency_ms=%d",
                    call_name,
                    attempt,
                    type(mapped).__name__,
                    latency_ms,
                )
                raise mapped from exc
            except Exception as exc:
                with self._stats_lock:
                    self._stats.failed_calls += 1
                self._logger.error(
                    "Unexpected DeepSeek client error call=%s error=%s",
                    call_name,
                    type(exc).__name__,
                )
                raise DeepSeekAPIError(
                    f"Unexpected DeepSeek client error: {type(exc).__name__}"
                ) from exc

            latency_ms = int((time.perf_counter() - started) * 1000)
            usage = self._extract_usage(completion, latency_ms, attempt)

            try:
                response = self._parse_completion(completion, usage, call_name)
            except DeepSeekResponseError:
                with self._stats_lock:
                    self._stats.failed_calls += 1
                self._record_usage_event(
                    response=None,
                    call_name=call_name,
                    usage=usage,
                    success=False,
                    error_type="DeepSeekResponseError",
                )
                raise

            with self._stats_lock:
                self._stats.successful_calls += 1
                self._stats.prompt_tokens += usage.prompt_tokens
                self._stats.completion_tokens += usage.completion_tokens
                self._stats.total_latency_ms += usage.latency_ms
            self._record_usage_event(
                response=response,
                call_name=call_name,
                usage=usage,
                success=True,
                error_type=None,
            )
            return response

        raise DeepSeekAPIError("DeepSeek request failed without a mapped error")

    @staticmethod
    def _is_retryable(exc: openai.OpenAIError) -> bool:
        if isinstance(
            exc,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code >= 500
        return False

    def _map_exception(self, exc: openai.OpenAIError, latency_ms: int) -> Exception:
        detail = getattr(exc, "message", None) or type(exc).__name__
        detail = str(detail)[:300]

        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            status_code = getattr(exc, "status_code", None) or "401/403"
            return DeepSeekAuthenticationError(
                f"DeepSeek authentication failed (HTTP {status_code}). "
                "Check DEEPSEEK_API_KEY and account permissions."
            )
        if isinstance(exc, openai.APITimeoutError):
            return DeepSeekTimeoutError(
                f"DeepSeek request timed out after {latency_ms} ms; "
                "increase DEEPSEEK_TIMEOUT if needed."
            )
        if isinstance(exc, openai.APIConnectionError):
            return DeepSeekConnectionError(
                f"Could not reach DeepSeek at {self.settings.base_url}: {detail}"
            )
        if isinstance(exc, openai.RateLimitError):
            return DeepSeekRateLimitError(f"DeepSeek rate limit exceeded: {detail}")
        if isinstance(exc, openai.InternalServerError):
            return DeepSeekServerError(f"DeepSeek server error: {detail}")
        if isinstance(exc, openai.BadRequestError):
            return DeepSeekResponseError(f"DeepSeek rejected the request: {detail}")
        if isinstance(exc, openai.APIStatusError):
            return DeepSeekAPIError(
                f"DeepSeek API returned HTTP {exc.status_code}: {detail}"
            )
        return DeepSeekAPIError(f"DeepSeek API error: {type(exc).__name__}: {detail}")

    # ------------------------------------------------------------------
    # Response parsing and accounting
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_usage(completion: Any, latency_ms: int, attempts: int) -> LLMUsage:
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
        )
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            attempts=attempts,
        )

    def _parse_completion(
        self, completion: Any, usage: LLMUsage, call_name: str
    ) -> LLMResponse:
        choices = getattr(completion, "choices", None)
        if not choices:
            raise DeepSeekResponseError("DeepSeek returned no choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise DeepSeekResponseError("DeepSeek response has no message")

        content = getattr(message, "content", None)
        if content is not None and not isinstance(content, str):
            raise DeepSeekResponseError("DeepSeek returned non-text content")

        tool_calls: list[ToolCall] = []
        for index, raw_call in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(raw_call, "function", None)
            name = getattr(function, "name", None)
            if not isinstance(name, str) or not name.strip():
                raise DeepSeekResponseError("DeepSeek returned a tool call without a name")

            raw_arguments = getattr(function, "arguments", None)
            if isinstance(raw_arguments, dict):
                arguments: Any = raw_arguments
            elif raw_arguments is None or raw_arguments == "":
                arguments = {}
            elif isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise DeepSeekResponseError(
                        f"Tool call '{name}' returned invalid JSON arguments"
                    ) from exc
            else:
                raise DeepSeekResponseError(
                    f"Tool call '{name}' returned unsupported argument type"
                )

            if not isinstance(arguments, dict):
                raise DeepSeekResponseError(
                    f"Tool call '{name}' arguments must be a JSON object"
                )

            tool_calls.append(
                ToolCall(
                    id=str(getattr(raw_call, "id", None) or f"call_{index + 1}"),
                    name=name,
                    arguments=arguments,
                )
            )

        model = getattr(completion, "model", None) or self.model_name
        finish_reason = getattr(choice, "finish_reason", None)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            model=str(model),
            call_name=call_name,
        )

    def _record_usage_event(
        self,
        *,
        response: LLMResponse | None,
        call_name: str,
        usage: LLMUsage,
        success: bool,
        error_type: str | None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "call": call_name,
            "model": response.model if response is not None else self.model_name,
            "success": success,
            "error_type": error_type,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_ms": usage.latency_ms,
            "attempts": usage.attempts,
            "finish_reason": response.finish_reason if response is not None else None,
            "tool_calls": len(response.tool_calls) if response is not None else 0,
        }
        try:
            self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._usage_lock:
                with self._usage_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError as exc:
            self._logger.warning(
                "Could not write LLM usage log: %s", type(exc).__name__
            )


def _strip_code_fences(text: str) -> str:
    """Remove an optional ```json ... ``` wrapper around a JSON object."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    return stripped


def _format_validation_errors(exc: ValidationError) -> str:
    """Summarise validation errors without echoing user-supplied values."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        parts.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "; ".join(parts)[:500]
