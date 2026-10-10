"""In-memory fakes for external HTTP services (ComfyUI, Ollama)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import httpx


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    import struct
    import zlib

    raw = b"".join(b"\0" + bytes(rgb) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PNG_32 = _png(32, 32, (120, 80, 200))


class FakeComfyUI:
    """Implements the subset of ComfyUI's HTTP API that Studio uses.

    ``behaviour`` per submitted prompt: "ok" echoes the uploaded input video as output,
    "oom" reports a CUDA OOM execution error, "node_error" a generic node failure.
    """

    def __init__(self, behaviours: list[str] | None = None,
                 object_info: dict[str, Any] | None = None) -> None:
        self.behaviours = list(behaviours or [])
        self.uploads: dict[str, bytes] = {}
        self.prompts: dict[str, dict[str, Any]] = {}
        self.history_store: dict[str, Any] = {}
        self.freed = 0
        self.interrupted: list[str] = []
        self.deleted: list[str] = []
        self.object_info_data = object_info or {}
        self.image_sources: dict[str, str | None] = {}
        self.down = False

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _complete(self, prompt_id: str, behaviour: str, workflow: dict[str, Any]) -> None:
        msgs: list[Any] = [["execution_start", {"prompt_id": prompt_id, "timestamp": 1000}]]
        if behaviour in ("oom", "node_error"):
            msg = ("CUDA out of memory. Tried to allocate 2.00 GiB" if behaviour == "oom"
                   else "mat1 and mat2 shapes cannot be multiplied")
            exc_type = "torch.OutOfMemoryError" if behaviour == "oom" else "RuntimeError"
            msgs.append(["execution_error", {"prompt_id": prompt_id, "node_id": "3",
                                             "node_type": "KSampler", "exception_type": exc_type,
                                             "exception_message": msg, "traceback": []}])
            self.history_store[prompt_id] = {"prompt": [], "outputs": {}, "status": {
                "status_str": "error", "completed": False, "messages": msgs}}
            return
        if not any(n["class_type"] == "LoadVideo" for n in workflow.values()):
            self._complete_images(prompt_id, workflow, msgs)
            return
        video_name = next(n["inputs"].get("file") for n in workflow.values()
                          if n["class_type"] == "LoadVideo")
        msgs.append(["execution_success", {"prompt_id": prompt_id, "timestamp": 3500}])
        self.history_store[prompt_id] = {"prompt": [], "outputs": {"16": {"images": [
            {"filename": f"out_{video_name}", "subfolder": "rokkur", "type": "output"}],
            "animated": [True]}}, "status": {"status_str": "success", "completed": True,
                                             "messages": msgs}}

    def _complete_images(self, prompt_id: str, workflow: dict[str, Any], msgs: list[Any]) -> None:
        """A still-image graph: one PNG per picture in the batch, echoing the input picture
        when the graph loads one (edit, inpaint, upscale), else a plain colour."""
        batch = 1
        for n in workflow.values():
            if n["class_type"] in ("EmptyFlux2LatentImage", "EmptySD3LatentImage",
                                   "EmptyLatentImage"):
                batch = int(n["inputs"].get("batch_size", 1))
            elif n["class_type"] == "RepeatLatentBatch":
                batch = int(n["inputs"].get("amount", 1))
        source = next((n["inputs"].get("image") for n in workflow.values()
                       if n["class_type"] == "LoadImage"), None)
        self.image_sources[prompt_id] = source
        msgs.append(["execution_success", {"prompt_id": prompt_id, "timestamp": 2200}])
        self.history_store[prompt_id] = {"prompt": [], "outputs": {"9": {"images": [
            {"filename": f"img_{prompt_id}_{i:05d}.png", "subfolder": "rokkur", "type": "output"}
            for i in range(1, batch + 1)]}}, "status": {
            "status_str": "success", "completed": True, "messages": msgs}}

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        request.read()
        path, method = request.url.path, request.method
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "nt"}, "devices": [
                {"name": "cuda:0 NVIDIA GeForce RTX 3070", "vram_total": 8 * 2**30,
                 "vram_free": 6 * 2**30}]})
        if path == "/object_info":
            return httpx.Response(200, json=self.object_info_data)
        if path == "/upload/image" and method == "POST":
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            name, payload = "", b""
            for part in request.content.split(b"--" + boundary):
                if b'name="image"' in part:
                    name = part.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
                    payload = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
            self.uploads[name] = payload
            return httpx.Response(200, json={"name": name, "subfolder": "", "type": "input"})
        if path == "/prompt" and method == "POST":
            data = json.loads(request.content)
            workflow = data["prompt"]
            missing = [n["class_type"] for n in workflow.values()
                       if self.object_info_data and n["class_type"] not in self.object_info_data]
            if missing:
                return httpx.Response(400, json={"error": {"type": "invalid_prompt",
                                                           "message": f"missing {missing}"},
                                                 "node_errors": {}})
            pid = uuid.uuid4().hex
            self.prompts[pid] = workflow
            self._complete(pid, self.behaviours.pop(0) if self.behaviours else "ok", workflow)
            return httpx.Response(200, json={"prompt_id": pid, "number": len(self.prompts),
                                             "node_errors": {}})
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[1]
            return httpx.Response(200, json={pid: self.history_store[pid]}
                                  if pid in self.history_store else {})
        if path == "/queue" and method == "GET":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        if path == "/queue" and method == "POST":
            self.deleted += json.loads(request.content).get("delete", [])
            return httpx.Response(200, json={})
        if path == "/interrupt":
            self.interrupted.append("x")
            return httpx.Response(200, json={})
        if path == "/free":
            self.freed += 1
            return httpx.Response(200, json={})
        if path == "/view":
            filename = request.url.params["filename"]
            if filename.startswith("img_"):
                pid = filename[4:].rsplit("_", 1)[0]
                source = self.image_sources.get(pid)
                if source and source.split("/")[-1] in self.uploads:
                    return httpx.Response(200, content=self.uploads[source.split("/")[-1]])
                return httpx.Response(200, content=PNG_32)
            source = filename.removeprefix("out_")
            if source not in self.uploads:
                return httpx.Response(404)
            return httpx.Response(200, content=self.uploads[source])
        return httpx.Response(404, json={"error": f"unhandled {method} {path}"})


def ollama_transport(replies: list[str], calls: list[dict[str, Any]] | None = None
                     ) -> httpx.MockTransport:
    replies = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            if calls is not None:
                calls.append(body)
            return httpx.Response(200, json={"message": {"role": "assistant",
                                                         "content": replies.pop(0)},
                                             "done": True})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b-instruct"}]})
        if request.url.path == "/api/generate":
            if calls is not None:
                calls.append(json.loads(request.content))
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    return httpx.MockTransport(handle)


Handler = Callable[[httpx.Request], httpx.Response]
