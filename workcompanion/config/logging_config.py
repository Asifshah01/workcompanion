"""Structured logging configuration with secret redaction."""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

_CONFIGURED = False

_SECRET_PATTERNS = [
    re.compile(r"(gsk_[A-Za-z0-9]{16,})"),
    re.compile(r"((?:api[_-]?key|authorization|bearer)[\"']?\s*[:=]\s*[\"']?)([^\s\"',]{6,})", re.I),
]


class SecretRedactingFilter(logging.Filter):
    """Strip API keys and bearer tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        for pattern in _SECRET_PATTERNS:
            message = pattern.sub(lambda m: (m.group(1) + "***") if m.lastindex else "***", message)
        record.msg = message
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """Compact one-line JSON formatter used for machine readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)[-2000:]
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_logs: bool = False, force: bool = False) -> None:
    """Configure root logging once (idempotent unless ``force``)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.addFilter(SecretRedactingFilter())
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s", "%H:%M:%S")
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    # These libraries are chatty at INFO level.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def log_extra(**fields: Any) -> dict[str, Any]:
    """Build the ``extra_fields`` mapping consumed by :class:`JsonFormatter`."""
    return {"extra_fields": fields}