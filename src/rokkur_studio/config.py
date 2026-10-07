"""Configuration: YAML files overridden by ``STUDIO_<SECTION>__<KEY>`` environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ResourceClass = Literal["GPU_LIGHT", "GPU_MEDIUM", "GPU_HEAVY"]


class StudioSection(BaseModel):
    autonomy_level: int = Field(2, ge=0, le=4)
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    log_json: bool = True


class DatabaseSection(BaseModel):
    url: str = "postgresql+psycopg://rokkur:rokkur@localhost:5432/rokkur"


class GpuSection(BaseModel):
    vram_gb: float = 8
    max_heavy_jobs: int = 1
    lease_seconds: int = 3600
    unload_ollama_before_heavy: bool = True
    free_comfyui_after_heavy: bool = True
    free_comfyui_before_agents: bool = True  # an 8 GB card cannot hold Wan and a 9B LLM
    class_vram_gb: dict[str, float] = Field(
        default_factory=lambda: {"GPU_LIGHT": 2.0, "GPU_MEDIUM": 5.0, "GPU_HEAVY": 8.0}
    )


class ComfySection(BaseModel):
    url: str = "http://127.0.0.1:8188"
    poll_interval_s: float = 2.0
    timeout_s: float = 3600


class OllamaSection(BaseModel):
    url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:9b"


class YoutubeSection(BaseModel):
    enabled: bool = False
    auto_publish: bool = False
    default_privacy: Literal["private", "unlisted", "public"] = "private"
    allow_public: bool = False       # a public upload needs this AND an explicit privacy=public
    secrets_dir: Path = Path("secrets")
    client_secret_file: str = "youtube_client_secret.json"
    token_file: str = "youtube_token.json"
    auth_port: int = 8401            # loopback redirect for the sign-in flow

    @property
    def client_secret_path(self) -> Path:
        return self.secrets_dir / self.client_secret_file

    @property
    def token_path(self) -> Path:
        return self.secrets_dir / self.token_file


class AgentsSection(BaseModel):
    provider: Literal["rule_based", "ollama", "rokkur_collective"] = "rule_based"
    max_per_project: int = 8
    max_output_retries: int = 2


class RenderSection(BaseModel):
    renderer: Literal["ffmpeg_preview", "comfyui"] = "ffmpeg_preview"
    default_profile: str = "RTX3070_QUALITY"
    max_retries: int = 3
    max_renders_per_project: int = 40


class QualitySection(BaseModel):
    pass_threshold: float = 6.5


class RightsSection(BaseModel):
    block_unknown: bool = True


class CommentsSection(BaseModel):
    auto_reply_enabled: bool = False


class CostsSection(BaseModel):
    max_gpu_minutes_per_project: float = 120
    max_cloud_gpu_minutes: float = 0
    max_cost_per_project_usd: float = 0


class JobsSection(BaseModel):
    default_max_attempts: int = 3
    lease_seconds: int = 900
    backoff_base_s: float = 5
    backoff_max_s: float = 600
    poll_interval_s: float = 1.0


class RenderProfile(BaseModel):
    """One render profile from ``config/render_profiles.yaml``."""

    name: str = ""
    description: str = ""
    workflow: str
    max_width: int
    max_height: int
    fps: int
    max_frames: int
    steps: int = 20
    denoise: float = 0.6
    controls: dict[str, bool] = Field(default_factory=dict)
    offload: bool = False
    quantization: str | None = None
    resource_class: ResourceClass = "GPU_HEAVY"
    location: Literal["local", "remote"] = "local"
    degrade: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    studio: StudioSection = StudioSection()
    database: DatabaseSection = DatabaseSection()
    gpu: GpuSection = GpuSection()
    comfyui: ComfySection = ComfySection()
    ollama: OllamaSection = OllamaSection()
    youtube: YoutubeSection = YoutubeSection()
    agents: AgentsSection = AgentsSection()
    render: RenderSection = RenderSection()
    quality: QualitySection = QualitySection()
    rights: RightsSection = RightsSection()
    comments: CommentsSection = CommentsSection()
    costs: CostsSection = CostsSection()
    jobs: JobsSection = JobsSection()
    profiles: dict[str, RenderProfile] = Field(default_factory=dict)
    config_dir: Path = Path("config")
    workflows_dir: Path = Path("workflows")

    def profile(self, name: str) -> RenderProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise KeyError(f"unknown render profile {name!r}; known: {sorted(self.profiles)}") from exc


ENV_PREFIX = "STUDIO_"


def _deep_set(target: dict[str, Any], keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """Turn ``STUDIO_GPU__VRAM_GB=24`` into ``{"gpu": {"vram_gb": 24}}``.

    Values are parsed as YAML scalars so ``true``/``8``/``[a, b]`` keep their types.
    ``DATABASE_URL`` is accepted as a shortcut for ``STUDIO_DATABASE__URL``.
    """
    out: dict[str, Any] = {}
    if "DATABASE_URL" in environ:
        _deep_set(out, ["database", "url"], environ["DATABASE_URL"])
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX) or "__" not in name:
            continue
        parts = name[len(ENV_PREFIX):].split("__")
        keys = [k if i > 0 and parts[i - 1].lower() == "profiles" else k.lower()
                for i, k in enumerate(parts)]  # profile names keep their case
        _deep_set(out, keys, yaml.safe_load(raw) if raw != "" else "")
    return out


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    config_dir: Path | None = None, environ: dict[str, str] | None = None
) -> Settings:
    environ = dict(os.environ if environ is None else environ)
    config_dir = Path(config_dir or environ.get("STUDIO_CONFIG_DIR", "config"))
    raw: dict[str, Any] = {}
    main = config_dir / "studio.yaml"
    if main.exists():
        raw = yaml.safe_load(main.read_text(encoding="utf-8")) or {}
    profiles_file = config_dir / "render_profiles.yaml"
    if profiles_file.exists():
        profiles = (yaml.safe_load(profiles_file.read_text(encoding="utf-8")) or {}).get(
            "profiles", {}
        )
        raw["profiles"] = {name: {**p, "name": name} for name, p in profiles.items()}
    raw = _merge(raw, env_overrides(environ))
    raw.setdefault("config_dir", str(config_dir))
    if "STUDIO_WORKFLOWS_DIR" in environ:
        raw["workflows_dir"] = environ["STUDIO_WORKFLOWS_DIR"]
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
