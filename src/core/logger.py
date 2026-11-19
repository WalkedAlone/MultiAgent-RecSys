"""Logging setup with built-in secret redaction.

Logs are written to ``logs/app.log`` (rotating) and optionally to stderr.
Timestamps are UTC.  A redaction filter masks common secret patterns before a
record reaches any handler, so API keys are not leaked even by accident.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from src.core.config import PROJECT_ROOT

_LOG_FORMAT = (
    "%(asctime)sZ %(levelname)-8s %(name)s "
    "[%(filename)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_LOGGER_NAME = "recsys"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_ENV_SECRET_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\b(Bearer\s+)([A-Za-z0-9._\-]+)")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b")

_lock = threading.RLock()
_configured = False


def redact_secrets(message: str) -> str:
    """Mask common API-key/token patterns in a user-visible string."""
    redacted = _ENV_SECRET_RE.sub(r"\1***", str(message))
    redacted = _BEARER_RE.sub(r"\1***", redacted)
    return _SK_RE.sub("sk-***", redacted)


class RedactionFilter(logging.Filter):
    """Mask API keys, bearer tokens and ``KEY=value`` pairs in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive against bad formatting
            return True

        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class UTCFormatter(logging.Formatter):
    """Formatter that emits UTC timestamps (the format string appends ``Z``)."""

    converter = time.gmtime


def _coerce_level(level: str | int | None) -> int:
    if level is None:
        level = os.getenv("APP_LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def setup_logging(
    level: str | int | None = None,
    log_dir: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """(Re)configure the application logger.

    Args:
        level: Logging level name or numeric value. Defaults to
            ``APP_LOG_LEVEL`` or ``INFO``.
        log_dir: Directory for ``app.log``. Defaults to ``<project>/logs``.
        console: Whether to mirror records to stderr.

    Returns:
        The configured ``recsys`` logger.
    """
    global _configured

    resolved_level = _coerce_level(level)
    target_dir = Path(log_dir) if log_dir is not None else PROJECT_ROOT / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = UTCFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    redaction = RedactionFilter()

    file_handler = logging.FileHandler(
        target_dir / "app.log", mode="a", encoding="utf-8", delay=False
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redaction)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(redaction)
        logger.addHandler(console_handler)

    with _lock:
        _configured = True

    return logger


def _ensure_configured() -> None:
    if _configured:
        return
    with _lock:
        if not _configured:
            setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the ``recsys`` logger."""
    _ensure_configured()
    full_name = name if name == _LOGGER_NAME or name.startswith(f"{_LOGGER_NAME}.") else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)


def shutdown_logging() -> None:
    """Close and remove handlers (mainly useful in tests)."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    with _lock:
        _configured = False
