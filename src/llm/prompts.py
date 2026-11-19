"""Shared prompt fragments for structured output.

Agent-specific prompts live next to their agents; these helpers guarantee that
JSON mode is requested consistently across the project.
"""

from __future__ import annotations

JSON_ONLY_SYSTEM_PROMPT = (
    "You are a precise assistant. Return exactly one valid JSON object and nothing else. "
    "Do not wrap the JSON in markdown fences. Do not add explanations before or after it. "
    "Follow the provided JSON Schema exactly. Use null only when the schema allows it. "
    "Never invent facts or identifiers that were not provided."
)


def build_json_system_prompt(schema_json: str) -> str:
    """Append a JSON Schema to the shared JSON-only system prompt."""
    return f"{JSON_ONLY_SYSTEM_PROMPT}\n\nJSON Schema:\n{schema_json}"


def build_json_repair_prompt(validation_errors: str) -> str:
    """Ask the model to repair a schema-invalid response."""
    return (
        "Your previous answer did not satisfy the JSON Schema. "
        f"Validation errors: {validation_errors}. "
        "Return only a corrected JSON object, without markdown fences or commentary."
    )
