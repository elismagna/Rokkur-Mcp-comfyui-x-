"""Configuration: YAML files overridden by ``STUDIO_<SECTION>__<KEY>`` environment variables."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, field_validator

ResourceClass = Literal["GPU_LIGHT", "GPU_MEDIUM", "GPU_HEAVY"]
PLAYLIST_ID = re.compile(r"[A-Za-z0-9_-]{10,64}")


class StudioSection(BaseModel):
    autonomy_level: int = Field(2, ge=0, le=4)
    data_dir: Path = Path("data")
    media_dir: Path = Path("media")  # source videos you own; /media inside Docker
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
    auto_publish: bool = False       # not used: nothing uploads without a person's click
    default_privacy: Literal["private", "unlisted", "public"] = "private"
    allow_public: bool = False       # a public upload or a scheduled release (which goes public)
                                     # needs this AND an explicit request
    secrets_dir: Path = Path("secrets")
    client_secret_file: str = "youtube_client_secret.json"
    token_file: str = "youtube_token.json"
    auth_port: int = 8401            # loopback redirect for the sign-in flow
    timezone: str = "UTC"            # release times and dates in the dashboard, e.g. Europe/Oslo
    release_times: list[str] = Field(default_factory=list)  # daily release slots, e.g. ["18:00"]
    min_lead_minutes: int = Field(30, ge=5, le=1440)  # a scheduled release is at least this far off
    default_playlist_id: str = ""    # playlist a finished video is added to unless you pick another

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown time zone {v!r}; use a name like Europe/Oslo or UTC") from exc
        return v

    @field_validator("release_times", mode="before")
    @classmethod
    def _clock_times(cls, v: Any) -> list[str]:
        if isinstance(v, str | int):  # "18:00, 21:00" from an env var
            v = str(v).split(",") if isinstance(v, str) else [v]
        out = set()
        for t in v or []:
            if isinstance(t, int):  # unquoted 18:00 in YAML is the base-60 integer 1080
                t = f"{t // 60}:{t % 60:02d}"
            m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", str(t).strip())
            if not m:
                raise ValueError(f"release time {t!r} is not HH:MM (24-hour)")
            out.add(f"{int(m.group(1)):02d}:{m.group(2)}")
        return sorted(out)

    @field_validator("default_playlist_id")
    @classmethod
    def _playlist_id(cls, v: str) -> str:
        v = v.strip()
        if v and not PLAYLIST_ID.fullmatch(v):
            raise ValueError(f"{v!r} does not look like a YouTube playlist id (PL…)")
        return v

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


class DirectorSection(BaseModel):
    """The cinematography pass and prompt compiler (docs/director.md)."""

    enabled: bool = True             # false: every shot renders with the brief's single prompt
    framing_weight: float = Field(1.3, ge=0.5, le=2.0)   # (close-up shot:1.3)
    angle_weight: float = Field(1.25, ge=0.5, le=2.0)    # (low-angle shot:1.25)
    vision: Literal["auto", "off"] = "auto"  # auto: show the DP a frame of each shot if it can see
    vision_model: str = ""           # optional separate Ollama model for the DP pass
    max_framing_calls: int = Field(24, ge=0)  # shots beyond this get rule-based framing
    schedule_fps: int = Field(24, ge=1, le=120)
    schedule_interval: int = Field(24, ge=1)
    schedule_inline_negative: bool = True    # "--neg" inside each keyframe (FizzNodes syntax)


class RenderSection(BaseModel):
    renderer: Literal["ffmpeg_preview", "comfyui"] = "ffmpeg_preview"
    default_profile: str = "RTX3070_QUALITY"
    max_retries: int = 3
    max_renders_per_project: int = 40


class QualitySection(BaseModel):
    pass_threshold: float = 6.5


class SubjectSection(BaseModel):
    """Keeping the real main subject over a restyled render (pipeline/subject.py)."""

    model: Literal["u2net", "isnet-general-use"] = "u2net"
    model_dir: Path | None = None     # default <data_dir>/models
    download: bool = True             # fetch the model (checksum-verified) on first use
    min_coverage: float = Field(0.01, ge=0, le=1)   # less of the frame: no clear subject
    max_coverage: float = Field(0.75, ge=0, le=1)   # more: the subject is the whole frame
    grow: float = Field(0.012, ge=0, le=0.1)        # mask growth, share of the short side
    feather: float = Field(0.012, ge=0, le=0.1)     # soft edge, share of the short side
    harmonize: float = Field(0.5, ge=0, le=1)       # move the subject's colours toward the new room
    threads: int = Field(0, ge=0)     # CPU threads for the mask model; 0 = automatic


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
    # Workflow used instead when the real subject is kept and its masks exist: VACE then keeps
    # the subject's pixels as context and regenerates only the room (docs/subject.md).
    keep_workflow: str | None = None
    max_width: int   # the box turns with the source: a landscape clip gets max_height wide
    max_height: int
    max_pixels: int | None = None  # area cap, e.g. 399360 = 480x832, what Wan 1.3B was trained on
    negative_base: str = ""        # prepended to every shot's negative prompt (Wan's own default)
    fps: int
    max_frames: int
    frame_multiple: int = Field(1, ge=1)
    min_vram_gb: float = Field(0, ge=0)
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
    director: DirectorSection = DirectorSection()
    render: RenderSection = RenderSection()
    quality: QualitySection = QualitySection()
    subject: SubjectSection = SubjectSection()
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

    def profile_problem(self, name: str, *,
                        object_info: dict[str, Any] | None = None) -> str | None:
        from rokkur_studio.comfyui.compiler import (
            TemplateError,
            TemplateRegistry,
            validate_against_object_info,
        )

        profile = self.profile(name)
        if self.render.renderer != "comfyui":
            return None
        if profile.location == "remote":
            return "Remote rendering is not configured; choose a local profile."
        if profile.min_vram_gb > self.gpu.vram_gb:
            return (f"Needs {profile.min_vram_gb:g} GB VRAM; this studio is configured "
                    f"for {self.gpu.vram_gb:g} GB.")
        try:
            template = TemplateRegistry(self.workflows_dir).get(profile.workflow)
        except (TemplateError, OSError, ValueError) as exc:
            return str(exc)
        except Exception as exc:  # e.g. a YAML typo in params.yaml: report it, don't 500
            return f"workflow {profile.workflow} could not be loaded: {exc}"
        if object_info is not None:
            problems = validate_against_object_info(template, object_info)
            if problems:
                return f"workflow {profile.workflow} is unavailable: {problems[0]}"
        return None


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
