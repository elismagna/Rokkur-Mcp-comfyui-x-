"""Structured logging. Every record carries the current job/project context."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "rokkur_log_context", default=None
)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach fields (job_id, project_id, stage, agent_id…) to every log line in the block."""
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ctx = _context.get() or {}
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **getattr(record, "ctx", {}),
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = " ".join(f"{k}={v}" for k, v in getattr(record, "ctx", {}).items())
        base = f"{self.formatTime(record)} {record.levelname:<7} {record.name}: {record.getMessage()}"
        extra = getattr(record, "data", None)
        if extra:
            base += f" {extra}"
        if ctx:
            base += f" [{ctx}]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    handler.addFilter(_ContextFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
