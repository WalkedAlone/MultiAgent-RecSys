"""Application-specific exception hierarchy.

All project code raises subclasses of :class:`RecSysError` so that callers can
distinguish expected application failures from programming errors.  Exception
messages must never contain API keys or raw user records.
"""

from __future__ import annotations


class RecSysError(Exception):
    """Base class for all expected application errors."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigurationError(RecSysError):
    """Raised when configuration files or environment settings are invalid."""


# ---------------------------------------------------------------------------
# LLM / DeepSeek
# ---------------------------------------------------------------------------
class DeepSeekError(RecSysError):
    """Base class for DeepSeek API failures."""


class DeepSeekConfigurationError(DeepSeekError):
    """Raised when the DeepSeek client cannot be constructed safely."""


class DeepSeekAuthenticationError(DeepSeekError):
    """HTTP 401/403: the API key is missing, invalid or lacks permission."""


class DeepSeekTimeoutError(DeepSeekError):
    """The request exceeded the configured timeout."""


class DeepSeekConnectionError(DeepSeekError):
    """The API endpoint could not be reached."""


class DeepSeekRateLimitError(DeepSeekError):
    """HTTP 429: the account hit a rate limit."""


class DeepSeekServerError(DeepSeekError):
    """HTTP 5xx returned by DeepSeek."""


class DeepSeekResponseError(DeepSeekError):
    """The API returned a malformed or schema-invalid response."""


class DeepSeekAPIError(DeepSeekError):
    """Any other DeepSeek API failure."""


# ---------------------------------------------------------------------------
# Tools / agents
# ---------------------------------------------------------------------------
class ToolExecutionError(RecSysError):
    """A registered tool failed to execute."""


class ToolNotFoundError(ToolExecutionError):
    """The model requested a tool that is not registered."""


class ToolArgumentError(ToolExecutionError):
    """Tool arguments failed schema validation."""


class AgentWorkflowError(RecSysError):
    """The multi-agent workflow could not complete."""


# ---------------------------------------------------------------------------
# Models / data
# ---------------------------------------------------------------------------
class ModelNotReadyError(RecSysError):
    """A required checkpoint or index has not been built yet."""


class DataValidationError(RecSysError):
    """Input data does not satisfy the expected schema."""


class DownloadError(RecSysError):
    """A dataset download failed (network, HTTP status or incomplete segment)."""


class IntegrityError(RecSysError):
    """A downloaded file is truncated, corrupt or fails checksum verification."""
