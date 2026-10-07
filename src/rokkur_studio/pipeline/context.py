"""Wiring: one object that carries the services a stage handler needs (dependency injection)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rokkur_studio.agents.providers import (
    AgentProvider,
    OllamaProvider,
    RokkurCollectiveProvider,
    RuleBasedProvider,
)
from rokkur_studio.comfyui.client import ComfyClient
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


def make_provider(settings: Settings) -> AgentProvider:
    if settings.agents.provider == "ollama":
        return OllamaProvider(settings.ollama.url, settings.ollama.model,
                              max_retries=settings.agents.max_output_retries)
    if settings.agents.provider == "rokkur_collective":
        return RokkurCollectiveProvider()
    return RuleBasedProvider()


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
    )
