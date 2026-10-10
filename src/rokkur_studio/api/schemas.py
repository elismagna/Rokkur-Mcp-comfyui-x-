"""Pydantic request/response models for the Studio API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from rokkur_studio.domain.rights import RightsCategory

# KSampler names offered for Wan; all are in ComfyUI core's KSampler lists.
Sampler = Literal["uni_pc", "uni_pc_bh2", "euler", "dpmpp_2m", "res_multistep"]
Scheduler = Literal["simple", "beta", "normal", "sgm_uniform"]
# Post-render stabilizer; auto lets the studio decide.
Stabilize = Literal["auto", "off", "light", "strong"]
SAMPLERS: tuple[str, ...] = get_args(Sampler)
SCHEDULERS: tuple[str, ...] = get_args(Scheduler)
STABILIZE_MODES: tuple[str, ...] = get_args(Stabilize)
# Edge detail on the New video form: Canny (low, high) thresholds; None keeps the workflow's
# 0.2/0.5. 0.4/0.8 is Comfy-Org's VACE template default and draws fewer fur and texture edges.
EDGE_PRESETS: dict[str, tuple[float, float] | None] = {
    "default": None, "calm": (0.4, 0.8), "tight": (0.1, 0.3)}


class SourceIn(BaseModel):
    platform: Literal["local", "upload", "youtube", "url"] = "local"
    local_path: str | None = Field(None, description="File path readable by the worker")
    url: str | None = None
    video_id: str | None = None
    title: str | None = None
    creator: str | None = None


class RightsIn(BaseModel):
    category: RightsCategory = RightsCategory.UNKNOWN
    license: str | None = None
    owner: str | None = None
    permission_evidence: str | None = None
    attribution_required: bool = False
    attribution_text: str | None = None
    allowed_transformations: list[str] = Field(default_factory=list)
    commercial_use: bool | None = None


class CreativeIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    theme: str = Field(min_length=1, max_length=2000)
    style: str | None = None
    prompt: str | None = None
    character_description: str | None = None
    character_key: str | None = None      # a character from the director's asset tracker
    use_global_look: bool = True          # apply the tracker's prefix, modifiers, negatives
    character_reference_path: str | None = None
    # When no character reference is uploaded (manifest/schema.py IdentitySpec): "auto" gives
    # Wan a cutout of the real subject when it is kept, and no reference otherwise.
    reference_mode: Literal["auto", "cutout", "source", "none"] = "auto"
    # auto: the studio decides from the prompt (pipeline/subject.py); keep: the real subject is
    # laid back over the render; restyle: the render's own subject is used.
    subject: Literal["auto", "keep", "restyle"] = "auto"
    control_strength: float = Field(1.0, ge=0, le=2)
    canny_low: float | None = Field(None, gt=0, lt=1)   # edge thresholds; the workflow's
    canny_high: float | None = Field(None, gt=0, lt=1)  # defaults (0.2/0.5) when unset
    seed: int | None = Field(None, ge=0, le=4294967295)
    steps: int | None = Field(None, ge=8, le=40)
    cfg: float = Field(6.0, ge=1, le=12)
    # Sampling options; None keeps the workflow's own value (shift 8, uni_pc, simple). A
    # workflow that fixes one (the Self-Forcing drafts' sampler) reports it as ignored.
    shift: float | None = Field(None, ge=1, le=20)
    sampler: Sampler | None = None
    scheduler: Scheduler | None = None
    # Below 1 renders smaller than the profile's size box, e.g. 0.67 for 480P on CLOUD_14B.
    resolution_scale: float | None = Field(None, ge=0.5, le=1.0)
    # Temporal stability: the post-render stabilizer (auto: the studio decides), and how much
    # the source clip is smoothed over time before it becomes the Canny/depth guide.
    stabilize: Stabilize = "auto"
    smooth_control: float = Field(0.0, ge=0, le=1)
    # Repairs: auto_tune lets QC change settings when a shot fails (off: only the seed changes);
    # min_stability > 0 re-renders shots whose stability score is below it.
    auto_tune: bool = True
    min_stability: float = Field(0.0, ge=0, le=10)
    picture_review: bool = True   # AI picture review after rendering, on the local vision model
    negative_prompt: str | None = Field(None, max_length=2000)
    keep_source_audio: bool = True
    audio_bed_path: str | None = None
    audio_bed_gain: float = Field(0.25, ge=0, le=2)
    audio_bed_rights_confirmed: bool = False
    audio_bed_rights_evidence: str | None = Field(None, max_length=1000)
    style_strength: float = Field(0.7, ge=0, le=1)
    identity_strength: float = Field(0.8, ge=0, le=1)
    title: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    made_for_kids: bool = False
    # Where the shots render: this PC's ComfyUI or the cloud server (docs/cloud.md). Unset:
    # the configured default (cloud.default when cloud is set up, else local).
    render_on: Literal["local", "cloud"] | None = None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    target_format: Literal["youtube_short", "youtube_video"] = "youtube_short"
    render_profile: str | None = None
    channel_id: str | None = None
    source: SourceIn
    rights: RightsIn = RightsIn()
    creative: CreativeIn
    autostart: bool = False


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    failed_from_state: str | None
    channel_id: str | None
    target_format: str
    render_profile: str
    repair_rounds: int
    created_at: datetime
    updated_at: datetime


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: str | None
    job_id: str | None
    type: str
    from_state: str | None
    to_state: str | None
    actor: str
    data: dict[str, Any]
    created_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str | None
    kind: str
    stage: str | None
    agent_id: str | None
    status: str
    retry_count: int
    max_attempts: int
    error: dict[str, Any] | None
    result: dict[str, Any] | None
    locked_by: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_s: float | None


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    rel_path: str
    size_bytes: int | None
    mime: str | None
    sha256: str | None
    meta: dict[str, Any]
    created_at: datetime


class RenderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    shot_id: str
    attempt: int
    profile: str
    renderer: str
    workflow: str | None
    params: dict[str, Any] = Field(default_factory=dict)
    status: str
    remote_id: str | None
    error: dict[str, Any] | None
    duration_s: float | None
    started_at: datetime
    finished_at: datetime | None


class RightsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    category: str
    status: str
    reason: str | None
    license: str | None
    owner: str | None
    permission_evidence: str | None
    attribution_required: bool
    attribution_text: str | None
    allowed_transformations: list[Any]
    commercial_use: bool | None
    decided_by: str
    decided_at: datetime


class RightsDecisionIn(RightsIn):
    approve: bool
    decided_by: str = "human"
    note: str | None = None


class PublicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dry_run: bool
    status: str
    request: dict[str, Any]
    youtube_video_id: str | None
    error: dict[str, Any] | None = None
    created_at: datetime


class ProjectDetail(BaseModel):
    project: ProjectOut
    source: dict[str, Any] | None
    rights: RightsOut | None
    documents: dict[str, dict[str, Any]]
    renders: list[RenderOut]
    assets: list[AssetOut]
    jobs: list[JobOut]
    publications: list[PublicationOut]
    next_job: str | None


class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    target: str
    render_id: str | None
    asset_id: str | None
    rater: str
    value: int
    tags: list[str]
    note: str | None
    snapshot: dict[str, Any]
    updated_at: datetime


class RedoIn(BaseModel):
    shots: list[str] = Field(min_length=1)


class PublishIn(BaseModel):
    dry_run: bool = True
    privacy: Literal["private", "unlisted", "public"] | None = None
    publish_at: datetime | None = None
    playlist_id: str | None = None


class ChannelIn(BaseModel):
    name: str
    youtube_channel_id: str | None = None
    autonomy_level: int = Field(2, ge=0, le=4)
    default_privacy: Literal["private", "unlisted", "public"] = "private"
    guidelines: dict[str, Any] = Field(default_factory=dict)


class ChannelOut(ChannelIn):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class ApprovalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str | None
    kind: str
    status: str
    summary: str
    payload: dict[str, Any]
    requested_by: str
    requested_at: datetime
    decided_by: str | None
    decided_at: datetime | None
    note: str | None


class ApprovalDecision(BaseModel):
    approve: bool
    decided_by: str = "human"
    note: str | None = None
