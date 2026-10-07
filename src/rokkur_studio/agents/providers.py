"""Agent providers: where structured reasoning comes from.

* ``RuleBasedProvider`` – deterministic, no model; the default so the pipeline runs anywhere.
* ``OllamaProvider`` – local LLM via Ollama's documented ``/api/chat`` with a JSON-schema
  ``format``; malformed output is fed back and retried, then raised.
* ``RokkurCollectiveProvider`` – placeholder that refuses to run until the Collective's
  interface is audited (docs/current-state.md). It never pretends to call anything.
"""

from __future__ import annotations

import json
import logging
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
                 schema: type[T]) -> T: ...


class RuleBasedProvider:
    """Marker provider: roles implement their deterministic logic themselves."""

    name = "rule_based"

    def generate(self, role: str, instructions: str, payload: dict[str, Any],
                 schema: type[T]) -> T:
        raise AgentUnavailable("rule_based provider has no generative model")


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, *, max_retries: int = 2,
                 timeout_s: float = 300, transport: httpx.BaseTransport | None = None) -> None:
        self.model, self.max_retries = model, max_retries
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s,
                                 transport=transport)

    def generate(self, role: str, instructions: str, payload: dict[str, Any],
                 schema: type[T]) -> T:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": f"You are the {role} of Rökkur Studio. {instructions} "
             "Reply with JSON only, matching the provided schema."},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ]
        last_raw: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.http.post("/api/chat", json={
                    "model": self.model, "messages": messages, "stream": False,
                    "format": schema.model_json_schema(), "options": {"temperature": 0.2},
                })
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise AgentUnavailable(f"Ollama request failed: {exc}") from exc
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

    def generate(self, role: str, instructions: str, payload: dict[str, Any],
                 schema: type[T]) -> T:
        raise AgentUnavailable(
            "Rökkur Collective integration is not implemented: its interface has not been "
            "audited yet (see docs/current-state.md). Use agents.provider=ollama or rule_based.")
