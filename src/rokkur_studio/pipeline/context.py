"""Wiring: one object that carries the services a stage handler needs (dependency injection)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from rokkur_studio.agents.providers import (
    AgentProvider,
    OllamaProvider,
    RokkurCollectiveProvider,
    RuleBasedProvider,
)
from rokkur_studio.comfyui.client import ComfyClient, ComfyError
from rokkur_studio.comfyui.compiler import TemplateRegistry
from rokkur_studio.config import Settings
from rokkur_studio.db.session import Database
from rokkur_studio.gpu.lease import GpuLeaseManager
from rokkur_studio.media.ffmpeg import FFmpeg
from rokkur_studio.storage.base import AssetStore
from rokkur_studio.storage.local import LocalAssetStore


@dataclass
class StudioContext:
    settings: Settings
    db: Database
    store: AssetStore
    ffmpeg: FFmpeg
    gpu: GpuLeaseManager
    registry: TemplateRegistry
    provider: AgentProvider
    comfy_factory: Callable[[], ComfyClient]
    extras: dict[str, object] = field(default_factory=dict)
    dp_provider: AgentProvider | None = None  # None: the DP pass uses ``provider``


def profile_availability(ctx: StudioContext) -> dict[str, str | None]:
    """Check configured profiles against the live ComfyUI when it is reachable.

    Keep the static file and VRAM checks useful while ComfyUI is offline. When it responds,
    also catch missing custom nodes before a project can enter the render pipeline.
    """
    object_info = None
    if ctx.settings.render.renderer == "comfyui":
        client = None
        try:
            client = ctx.comfy_factory()
            object_info = client.object_info()
        except (ComfyError, httpx.HTTPError):
            pass
        finally:
            if client is not None:
                client.close()
    return {name: ctx.settings.profile_problem(name, object_info=object_info)
            for name in ctx.settings.profiles}


def free_idle_comfyui(settings: Settings) -> Callable[[], None]:
    """A hook that unloads ComfyUI's models, but only while ComfyUI has nothing running."""

    def hook() -> None:
        client = ComfyClient(settings.comfyui.url)
        try:
            q = client.queue()
            if q.get("queue_running") or q.get("queue_pending"):
                return  # never pull models out from under a render
            client.free()
        finally:
            client.close()

    return hook


def make_provider(settings: Settings) -> AgentProvider:
    if settings.agents.provider == "ollama":
        return OllamaProvider(
            settings.ollama.url, settings.ollama.model,
            max_retries=settings.agents.max_output_retries,
            before_generate=free_idle_comfyui(settings)
            if settings.gpu.free_comfyui_before_agents else None)
    if settings.agents.provider == "rokkur_collective":
        return RokkurCollectiveProvider()
    return RuleBasedProvider()


def make_dp_provider(settings: Settings) -> AgentProvider | None:
    """A separate Ollama model for the Director of Photography, when one is configured."""
    model = settings.director.vision_model.strip()
    if settings.agents.provider != "ollama" or not model or model == settings.ollama.model:
        return None
    return OllamaProvider(
        settings.ollama.url, model, max_retries=settings.agents.max_output_retries,
        before_generate=free_idle_comfyui(settings)
        if settings.gpu.free_comfyui_before_agents else None)


def build_context(settings: Settings, db: Database | None = None) -> StudioContext:
    db = db or Database(settings.database.url)
    comfy_factory = lambda: ComfyClient(settings.comfyui.url)  # noqa: E731

    def unload_ollama(holder: str) -> None:
        OllamaProvider(settings.ollama.url, settings.ollama.model, timeout_s=10).unload_all()

    def free_comfy(holder: str) -> None:
        client = comfy_factory()
        try:
            client.free()
        finally:
            client.close()

    gpu = GpuLeaseManager(
        db, settings.gpu,
        before_heavy=[unload_ollama] if settings.gpu.unload_ollama_before_heavy else [],
        after_heavy=[free_comfy] if (settings.gpu.free_comfyui_after_heavy
                                     and settings.render.renderer == "comfyui") else [],
    )
    return StudioContext(
        settings=settings,
        db=db,
        store=LocalAssetStore(settings.studio.data_dir),
        ffmpeg=FFmpeg(),
        gpu=gpu,
        registry=TemplateRegistry(settings.workflows_dir),
        provider=make_provider(settings),
        comfy_factory=comfy_factory,
        dp_provider=make_dp_provider(settings),
    )
