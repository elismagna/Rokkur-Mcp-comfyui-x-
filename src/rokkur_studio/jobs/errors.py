"""Job failure taxonomy. Handlers raise these; the queue decides retry vs dead-letter."""

from __future__ import annotations

from typing import Any


class JobError(Exception):
    """A failure the queue may retry (network down, ComfyUI restarting…)."""

    retryable = True

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details,
                "retryable": self.retryable}


class PermanentJobError(JobError):
    """Retrying the identical work cannot succeed (bad input, gate violation, budget)."""

    retryable = False


class JobCancelled(PermanentJobError):
    def __init__(self, message: str = "job cancelled") -> None:
        super().__init__("cancelled", message)
