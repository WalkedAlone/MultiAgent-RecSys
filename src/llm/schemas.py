"""Pydantic schemas shared by every LLM consumer (agents, services, tests)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A tool call requested by the model.

    The model only *requests* the call.  Application code validates the
    arguments against a registered Pydantic model and executes a known Python
    function; model-generated code is never executed.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    """Token accounting and latency for a single API call."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    attempts: int = Field(default=1, ge=1)


class LLMResponse(BaseModel):
    """Normalised result of one chat completion."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: str | None = None
    model: str = ""
    call_name: str = ""

    def to_assistant_message(self) -> dict[str, Any]:
        """Serialise back to the OpenAI-compatible assistant message format."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }
                for tool_call in self.tool_calls
            ]
        return message
