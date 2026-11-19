"""Safe tool registry for Function Calling.

The LLM only produces tool names and JSON arguments.  ``ToolRegistry`` checks
that the name is registered, validates the arguments against a Pydantic model,
and invokes a Python function that was registered by application code.  It never
evaluates model-generated code.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ValidationError

from src.core.logger import get_logger
from src.core.exceptions import ToolExecutionError

_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")
_logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: schema + implementation."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]


class ToolExecutionResult(BaseModel):
    """Outcome of one tool execution, safe to send back to the model."""

    tool_name: str
    ok: bool
    result: Any = None
    error: str | None = None
    latency_ms: int = 0

    def as_tool_message(self, tool_call_id: str) -> dict[str, Any]:
        """Build an OpenAI-compatible ``tool`` message."""
        payload = self.result if self.ok else {"error": self.error or "tool execution failed"}
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        }


class ToolRegistry:
    """Registry of explicitly allowed tools and their validated arguments."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[[BaseModel], Any],
    ) -> None:
        """Register a tool.

        Args:
            name: Unique tool name exposed to the model.
            description: Natural-language description used by the model.
            args_model: Pydantic model validating the JSON arguments.
            handler: Python callable accepting the validated model instance.

        Raises:
            ValueError: duplicate or invalid name/description.
            TypeError: invalid schema or handler.
        """
        if not _TOOL_NAME_RE.match(name):
            raise ValueError(
                "Tool name must start with a letter and contain only letters, "
                "digits, underscores or hyphens (max 64 characters)"
            )
        if not description.strip():
            raise ValueError("Tool description must not be empty")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if not isinstance(args_model, type) or not issubclass(args_model, BaseModel):
            raise TypeError("args_model must be a Pydantic BaseModel subclass")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            args_model=args_model,
            handler=handler,
        )

    def openai_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI/DeepSeek function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.args_model.model_json_schema(),
                },
            }
            for spec in self._tools.values()
        ]

    def names(self) -> list[str]:
        """Return registered tool names (for validation and tests)."""
        return list(self._tools)

    def execute(
        self, name: str, arguments: Mapping[str, Any] | str
    ) -> ToolExecutionResult:
        """Validate and execute a tool call.

        Errors are returned as ``ok=False`` results instead of escaping, so an
        agent loop can feed the error back to the model.
        """
        started = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                error=f"Unknown tool: {name}",
                latency_ms=_elapsed_ms(started),
            )

        try:
            if isinstance(arguments, str):
                payload = json.loads(arguments)
            else:
                payload = dict(arguments)
            if not isinstance(payload, dict):
                raise TypeError("arguments must decode to a JSON object")
        except (json.JSONDecodeError, TypeError, ValueError):
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                error="Invalid tool arguments: expected a JSON object",
                latency_ms=_elapsed_ms(started),
            )

        try:
            validated = spec.args_model.model_validate(payload)
        except ValidationError as exc:
            message = _format_validation_errors(exc)
            _logger.warning("tool_arguments_invalid name=%s errors=%s", name, message)
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                error=f"Invalid arguments for {name}: {message}",
                latency_ms=_elapsed_ms(started),
            )

        _logger.debug(
            "tool_invoked name=%s args=%s",
            name,
            json.dumps(payload, ensure_ascii=False)[:1000],
        )

        try:
            raw_result = spec.handler(validated)
            result = _ensure_jsonable(raw_result)
        except ToolExecutionError as exc:
            _logger.warning("tool_failed name=%s error=%s", name, type(exc).__name__)
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                error=str(exc)[:300],
                latency_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            _logger.warning(
                "tool_failed name=%s error=%s", name, type(exc).__name__
            )
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                error=f"{type(exc).__name__}: {str(exc)[:240]}",
                latency_ms=_elapsed_ms(started),
            )

        latency_ms = _elapsed_ms(started)
        _logger.info(
            "tool_completed name=%s ok=true latency_ms=%d", name, latency_ms
        )
        return ToolExecutionResult(
            tool_name=name,
            ok=True,
            result=result,
            latency_ms=latency_ms,
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _format_validation_errors(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        parts.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "; ".join(parts)[:400]


def _ensure_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError("Tool result is not JSON-serializable") from exc
    return value
