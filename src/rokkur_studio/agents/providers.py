"""Agent providers: where structured reasoning comes from.

* ``RuleBasedProvider`` – deterministic, no model; the default so the pipeline runs anywhere.
* ``OllamaProvider`` – local LLM via Ollama's documented ``/api/chat`` with a JSON-schema
  ``format``; malformed output is fed back and retried, then raised.
* ``RokkurCollectiveProvider`` – placeholder that refuses to run until the Collective's
  interface is audited (docs/current-state.md). It never pretends to call anything.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentOutputError(RuntimeError):
    def __init__(self, role: str, message: str, raw: str | None = None) -> None:
        super().__init__(f"{role}: {message}")
        self.role, self.raw = role, raw


class AgentUnavailable(RuntimeError):
    pass


class AgentProvider(Protocol):
    name: str

    def generate(self, role: str, instructions: str, payload: dict[str, Any],
                 schema: type[T], images: list[bytes] | None = None) -> T: ...

    def supports_images(self) -> bool: ...


class RuleBasedProvider:
    """Marker provider: roles implement their deterministic logic themselves."""

    name = "rule_based"

    def generate(self, role: str, instructions: str, payload: dict[str, Any],
                 schema: type[T], images: list[bytes] | None = None) -> T:
        raise AgentUnavailable("rule_based provider has no generative model")

    def supports_images(self) -> bool:
        return False


# Ollama compiles ``format`` into a llama.cpp grammar. String length bounds expand into one
# grammar rule per character (a 2000-char maxLength is rejected with HTTP 400), so length and
# item-count caps are left out of the grammar and enforced by pydantic validation instead.
_GRAMMAR_UNSAFE = {"minLength", "maxLength", "maxItems"}


def grammar_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: grammar_schema(v) for k, v in schema.items() if k not in _GRAMMAR_UNSAFE}
    if isinstance(schema, list):
        return [grammar_schema(v) for v in schema]
    return schema


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, *, max_retries: int = 2,
                 timeout_s: float = 300, transport: httpx.BaseTransport | None = None,
                 before_generate: Callable[[], None] | None = None) -> None:
        self.model, self.max_retries = model, max_retries
        # Called before each request, e.g. to make ComfyUI release VRAM so the model is not
        # pushed onto the CPU. Failures are logged and ignored.
        self.before_generate = before_generate
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s,
                                 transport=transport)
        self._vision: bool | None = None

    def generate(self, role: str, instructions: str, payload: dict[str, Any],
                 schema: type[T], images: list[bytes] | None = None) -> T:
        user: dict[str, Any] = {"role": "user", "content": json.dumps(payload, default=str)}
        if images:
            user["images"] = [base64.b64encode(i).decode("ascii") for i in images]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"You are the {role} of Rökkur Studio. {instructions} "
             "Reply with JSON only, matching the provided schema."},
            user,
        ]
        if self.before_generate is not None:
            try:
                self.before_generate()
            except Exception as exc:  # noqa: BLE001 - freeing VRAM is best effort
                log.warning("before_generate hook failed", extra={"data": {"error": str(exc)}})
        last_raw: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                # ``think: false`` keeps reasoning models (qwen3.x) from spending minutes
                # thinking before a 200-token JSON answer; older Ollama ignores the key.
                r = self.http.post("/api/chat", json={
                    "model": self.model, "messages": messages, "stream": False, "think": False,
                    "format": grammar_schema(schema.model_json_schema()),
                    "options": {"temperature": 0.2},
                })
            except httpx.HTTPError as exc:
                raise AgentUnavailable(f"Ollama request failed: {exc}") from exc
            if r.status_code != 200:
                raise AgentUnavailable(
                    f"Ollama returned {r.status_code} for {role}: {r.text[:300]}")
            last_raw = (r.json().get("message") or {}).get("content", "")
            try:
                return schema.model_validate_json(last_raw)
            except ValidationError as exc:
                log.warning("malformed agent output", extra={"data": {
                    "role": role, "attempt": attempt, "errors": exc.errors()[:5]}})
                messages += [{"role": "assistant", "content": last_raw},
                             {"role": "user", "content": "That JSON was invalid: "
                              f"{exc.errors()[:5]}. Return corrected JSON only."}]
        raise AgentOutputError(role, f"invalid output after {self.max_retries + 1} attempts",
                               last_raw)

    def supports_images(self) -> bool:
        """True when Ollama reports the model can read images (``/api/show`` capabilities)."""
        if self._vision is None:
            try:
                r = self.http.post("/api/show", json={"model": self.model}, timeout=10)
                caps = (r.json().get("capabilities") or []) if r.status_code == 200 else []
            except (httpx.HTTPError, ValueError):
                return False  # not cached: Ollama may just be starting
            self._vision = "vision" in caps
        return self._vision

    def installed_models(self) -> list[str]:
        r = self.http.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def check(self) -> str | None:
        """None when Ollama answers and the model is installed, else the reason."""
        try:
            names = self.installed_models()
        except httpx.HTTPError as exc:
            return f"Ollama not reachable: {exc}"
        if self.model not in names and f"{self.model}:latest" not in names:
            return (f"model {self.model!r} is not installed (have: {', '.join(names) or 'none'});"
                    f" run: ollama pull {self.model}")
        return None

    def loaded_models(self) -> list[str]:
        r = self.http.get("/api/ps")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def unload_all(self) -> list[str]:
        """Free VRAM before a heavy GPU job (``keep_alive: 0`` unloads a model)."""
        unloaded = []
        for name in self.loaded_models():
            self.http.post("/api/generate", json={"model": name, "keep_alive": 0})
            unloaded.append(name)
        return unloaded


class RokkurCollectiveProvider:
    name = "rokkur_collective"

    def supports_images(self) -> bool:
        return False

    def generate(self, role: str, instructions: str, payload: dict[str, Any],
                 schema: type[T], images: list[bytes] | None = None) -> T:
        raise AgentUnavailable(
            "Rökkur Collective integration is not implemented: its interface has not been "
            "audited yet (see docs/current-state.md). Use agents.provider=ollama or rule_based.")
