"""ComfyUI HTTP API client. Treats ComfyUI as a rendering service; never touches the GUI.

Endpoints used (ComfyUI ``server.py``): POST /prompt, GET /history/{id}, GET /queue,
POST /queue (delete), POST /interrupt, POST /free, POST /upload/image, GET /view,
GET /object_info, GET /system_stats, and optionally the /ws progress socket.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

_OOM_PATTERN = re.compile(
    r"out of memory|OutOfMemoryError|Allocation on device|CUDA error: out of memory", re.I
)


class ComfyError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code, self.details = code, details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


class ComfyUnavailable(ComfyError):
    """ComfyUI is not reachable (not started, restarting, crashed)."""


class ComfyValidationError(ComfyError):
    """ComfyUI rejected the workflow (missing node, bad input value)."""


class ComfyExecutionError(ComfyError):
    """A node raised during execution."""

    @property
    def is_oom(self) -> bool:
        return bool(self.details.get("oom"))


class ComfyTimeout(ComfyError):
    pass


@dataclass
class OutputFile:
    node_id: str
    filename: str
    subfolder: str
    type: str
    kind: str  # images | gifs | videos | audio …


@dataclass
class RenderResult:
    prompt_id: str
    status: str
    outputs: list[OutputFile]
    execution_seconds: float | None
    messages: list[Any] = field(default_factory=list)


def is_oom_message(text: str) -> bool:
    return bool(_OOM_PATTERN.search(text or ""))


def parse_history_entry(prompt_id: str, entry: dict[str, Any]) -> RenderResult:
    """Turn one ``/history`` entry into outputs or a structured execution error."""
    status = entry.get("status") or {}
    messages = status.get("messages") or []
    for item in messages:
        if isinstance(item, list | tuple) and len(item) == 2 and item[0] == "execution_error":
            err = item[1] or {}
            text = f"{err.get('exception_type', '')}: {err.get('exception_message', '')}"
            raise ComfyExecutionError(
                "oom" if is_oom_message(text) else "node_error",
                f"node {err.get('node_id')} ({err.get('node_type')}) failed: {text.strip()}",
                {
                    "prompt_id": prompt_id,
                    "node_id": err.get("node_id"),
                    "node_type": err.get("node_type"),
                    "exception_type": err.get("exception_type"),
                    "exception_message": err.get("exception_message"),
                    "oom": is_oom_message(text),
                },
            )
    if status.get("status_str") == "error":
        raise ComfyExecutionError("node_error", "ComfyUI reported an error",
                                  {"prompt_id": prompt_id, "messages": messages})
    outputs: list[OutputFile] = []
    for node_id, node_out in (entry.get("outputs") or {}).items():
        for kind, files in (node_out or {}).items():
            if not isinstance(files, list):
                continue
            for f in files:
                if isinstance(f, dict) and "filename" in f:
                    outputs.append(OutputFile(node_id, f["filename"], f.get("subfolder", ""),
                                              f.get("type", "output"), kind))
    start = end = None
    for item in messages:
        if isinstance(item, list | tuple) and len(item) == 2 and isinstance(item[1], dict):
            ts = item[1].get("timestamp")
            if item[0] == "execution_start":
                start = ts
            elif item[0] in ("execution_success", "execution_cached") and ts:
                end = ts
    seconds = (end - start) / 1000 if start and end and end >= start else None
    return RenderResult(prompt_id, status.get("status_str", "success"), outputs, seconds, messages)


class ComfyClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30, client_id: str | None = None,
                 transport: httpx.BaseTransport | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or f"rokkur-{uuid.uuid4().hex[:12]}"
        # headers: e.g. the cloud server's access token, sent with every request
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout_s, transport=transport,
                                 headers=headers)

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            return self.http.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            raise ComfyUnavailable("unavailable", f"ComfyUI unreachable at {self.base_url}: {exc}",
                                   {"url": self.base_url}) from exc

    # -- discovery ------------------------------------------------------------------------
    def system_stats(self) -> dict[str, Any]:
        r = self._request("GET", "/system_stats")
        r.raise_for_status()
        return r.json()

    def object_info(self) -> dict[str, Any]:
        r = self._request("GET", "/object_info")
        r.raise_for_status()
        return r.json()

    def queue(self) -> dict[str, Any]:
        r = self._request("GET", "/queue")
        r.raise_for_status()
        return r.json()

    # -- jobs -----------------------------------------------------------------------------
    def upload_input(self, path: Path, *, subfolder: str = "", overwrite: bool = True) -> str:
        """Upload a file into ComfyUI's input directory; returns the name nodes should use."""
        with Path(path).open("rb") as fh:
            r = self._request(
                "POST", "/upload/image",
                files={"image": (Path(path).name, fh, "application/octet-stream")},
                data={"type": "input", "subfolder": subfolder,
                      "overwrite": "true" if overwrite else "false"},
            )
        if r.status_code != 200:
            raise ComfyError("upload_failed", f"upload rejected ({r.status_code})",
                             {"body": r.text[:2000]})
        info = r.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    def submit(self, workflow: dict[str, Any], *, extra: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {"prompt": workflow, "client_id": self.client_id}
        if extra:
            body["extra_data"] = extra
        r = self._request("POST", "/prompt", json=body)
        if r.status_code == 400:
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            raise ComfyValidationError("invalid_workflow",
                                       (data.get("error") or {}).get("message", "workflow rejected"),
                                       {"error": data.get("error"),
                                        "node_errors": data.get("node_errors")})
        r.raise_for_status()
        data = r.json()
        if data.get("node_errors"):
            raise ComfyValidationError("invalid_workflow", "node errors",
                                       {"node_errors": data["node_errors"]})
        return str(data["prompt_id"])

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        r = self._request("GET", f"/history/{prompt_id}")
        r.raise_for_status()
        return (r.json() or {}).get(prompt_id)

    def queue_position(self, prompt_id: str) -> str | None:
        """``running``, ``pending`` or None if ComfyUI no longer has it queued."""
        q = self.queue()
        if any(item[1] == prompt_id for item in q.get("queue_running", [])):
            return "running"
        if any(item[1] == prompt_id for item in q.get("queue_pending", [])):
            return "pending"
        return None

    def wait(self, prompt_id: str, *, timeout_s: float = 3600, poll_s: float = 2.0,
             on_progress: Callable[[dict[str, Any]], None] | None = None,
             should_cancel: Callable[[], bool] | None = None) -> RenderResult:
        """Poll until the prompt finishes. Raises on node error, OOM, timeout or loss."""
        deadline = time.monotonic() + timeout_s
        missing = 0
        while True:
            entry = self.history(prompt_id)
            if entry is not None:  # ComfyUI writes history only once a prompt has finished
                return parse_history_entry(prompt_id, entry)
            if should_cancel and should_cancel():
                self.cancel(prompt_id)
                raise ComfyError("cancelled", f"prompt {prompt_id} cancelled")
            position = self.queue_position(prompt_id)
            if position is None and entry is None:
                missing += 1
                if missing >= 3:  # ComfyUI restarted and forgot the prompt
                    raise ComfyUnavailable("lost", f"prompt {prompt_id} vanished from ComfyUI",
                                           {"prompt_id": prompt_id})
            else:
                missing = 0
            if on_progress:
                on_progress({"prompt_id": prompt_id, "queue": position})
            if time.monotonic() >= deadline:
                self.cancel(prompt_id)
                raise ComfyTimeout("timeout", f"prompt {prompt_id} exceeded {timeout_s}s")
            time.sleep(poll_s)

    def cancel(self, prompt_id: str) -> None:
        """Remove from the pending queue, or interrupt if it is the running prompt."""
        try:
            if self.queue_position(prompt_id) == "running":
                self._request("POST", "/interrupt", json={"prompt_id": prompt_id})
            else:
                self._request("POST", "/queue", json={"delete": [prompt_id]})
        except ComfyError:
            log.warning("cancel failed", exc_info=True)

    def free(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        r = self._request("POST", "/free", json={"unload_models": unload_models,
                                                  "free_memory": free_memory})
        r.raise_for_status()

    def download(self, output: OutputFile, dest: Path) -> Path:
        r = self._request("GET", "/view", params={"filename": output.filename,
                                                  "subfolder": output.subfolder,
                                                  "type": output.type})
        if r.status_code != 200:
            raise ComfyError("download_failed", f"/view returned {r.status_code}",
                             {"filename": output.filename})
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
