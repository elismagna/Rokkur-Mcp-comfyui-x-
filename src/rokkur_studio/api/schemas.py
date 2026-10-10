"""Pydantic request/response models for the Studio API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rokkur_studio.domain.rights import RightsCategory


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
    # Stable mode (docs/stability.md): one seed and one reference for every shot, no per-shot
    # framing changes from the director pass, full steps, source guide at 1.0 and a stricter
    # quality pass mark, so drift and artifacts are caught instead of accepted.
    stable: bool = False
    # Where the shots render: this PC's ComfyUI or the cloud server (docs/cloud.md). Unset:
    # the configured default (cloud.default when cloud is set up, else local).
    render_on: Literal["local", "cloud"] | None = None


class AdjustIn(BaseModel):
    """Safe changes for the shots that have not rendered yet (commands.adjust_remaining_shots)."""

    prompt_extra: str | None = Field(None, max_length=500)  # appended to those shots' prompts
    seed: int | None = Field(None, ge=0, le=4294967295)
    steps: int | None = Field(None, ge=4, le=60)
    cfg: float | None = Field(None, ge=1, le=12)
    control_strength: float | None = Field(None, ge=0, le=2)
    canny_low: float | None = Field(None, gt=0, lt=1)
    canny_high: float | None = Field(None, gt=0, lt=1)
    reference_image_id: str | None = None   # a picture from the library as the appearance reference
    clear_reference: bool = False


class SoundtrackIn(BaseModel):
    """A video's soundtrack, from the sound library (commands.set_soundtrack)."""

    clip_id: str | None = None            # None: remove the added track
    gain: float | None = Field(None, ge=0, le=2)
    keep_source_audio: bool | None = None


class ExtendIn(BaseModel):
    """Continue a finished video (pipeline/extend.py)."""

    seconds: float = Field(3.0, gt=0, le=8)
    prompt: str = Field("", max_length=2000)   # empty: the brief's scene prompt
    negative_prompt: str | None = Field(None, max_length=2000)
    seed: int | None = Field(None, ge=0, le=4294967295)
    steps: int | None = Field(None, ge=4, le=60)
    reference_image_path: str | None = None
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


class UpgradeIn(BaseModel):
    shots: list[str] = Field(default_factory=list)  # empty: every shot


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
