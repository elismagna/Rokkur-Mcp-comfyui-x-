"""Structured agent outputs. Agents return these, never free prose."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rokkur_studio.director.vocabulary import CameraAngle, CameraMovement, Lighting, ShotSize
from rokkur_studio.domain.rights import RightsCategory

TargetFormat = Literal["youtube_short", "youtube_video"]


class ShotStory(BaseModel):
    """Story pass (Creative Director): what one shot shows, written as visible states."""

    shot_id: str
    start: float
    end: float
    intent: str = ""
    subject: str = ""      # concrete physical states: pose, expression, wardrobe details
    background: str = ""   # setting details as nouns
    motion_type: str = "unknown"
    camera: str = "unknown"


class ShotFraming(BaseModel):
    """Cinematography pass (Director of Photography) for one shot: allowed terms only."""

    shot_size: ShotSize
    camera_angle: CameraAngle
    camera_movement: CameraMovement
    lighting: Lighting
    observed_subject: str = ""
    observed_background: str = ""


class ObservedShotFraming(ShotFraming):
    """Vision calls must supply observations rather than silently omitting optional fields."""

    observed_subject: str = Field(min_length=1, max_length=1000)
    observed_background: str = Field(max_length=1000)


class ShotPlan(ShotStory):
    """A shot as stored in the brief: story + framing + the compiled diffusion prompt."""

    shot_size: ShotSize | None = None
    camera_angle: CameraAngle | None = None
    camera_movement: CameraMovement | None = None
    lighting: Lighting | None = None
    framing_by: str | None = None
    prompt: str = ""


class DirectorNotes(BaseModel):
    """How the director passes produced this brief (shown on the dashboard)."""

    story_by: str = "rule_based"
    framing_by: str = "rule_based"
    vision: bool = False
    character_key: str | None = None
    global_look: bool = True
    weights: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class _BriefFields(BaseModel):
    style: str
    theme: str
    character: str | None = None
    visual_identity: str = ""
    prompt: str
    negative_prompt: str = "blurry, deformed, extra limbs, watermark, text"
    style_strength: float = Field(0.7, ge=0, le=1)
    identity_strength: float = Field(0.8, ge=0, le=1)
    motion_preservation: Literal["strict", "loose"] = "strict"
    identity_requirements: str = ""
    background_requirements: str = ""
    target_duration: float = Field(gt=0)
    target_aspect_ratio: Literal["9:16", "16:9", "1:1"] = "9:16"
    rationale: str = ""


class StoryBrief(_BriefFields):
    """What the Creative Director model is asked to return (no framing fields)."""

    shot_plan: list[ShotStory] = Field(min_length=1)


class CreativeBrief(_BriefFields):
    """The stored brief: story, framing and per-shot prompts."""

    shot_plan: list[ShotPlan] = Field(min_length=1)
    director: DirectorNotes | None = None


class MetadataDraft(BaseModel):
    """Channel Manager output: YouTube title/description/tags for one video."""

    title: str = Field(min_length=3, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=15)
    rationale: str = ""


class TrendScore(BaseModel):
    """Trend Analyst output for one candidate (0..1 each)."""

    candidate_id: str
    novelty: float = Field(ge=0, le=1)
    visual_movement: float = Field(ge=0, le=1)
    transformation_potential: float = Field(ge=0, le=1)
    channel_relevance: float = Field(ge=0, le=1)
    competition: float = Field(ge=0, le=1)
    timeliness: float = Field(ge=0, le=1)
    production_difficulty: float = Field(ge=0, le=1)
    reuse_value: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    notes: str = ""


class RightsAssessment(BaseModel):
    """Rights agent output. Advisory only: the deterministic gate in domain.rights decides."""

    category: RightsCategory
    license: str | None = None
    owner: str | None = None
    attribution_required: bool = False
    commercial_use: bool | None = None
    evidence_summary: str = ""
    confidence: float = Field(ge=0, le=1)


QcRecommendation = Literal[
    "PASS", "RERENDER_SHOT", "REDUCE_STYLE_STRENGTH", "INCREASE_IDENTITY", "CHANGE_SEED",
    "ADD_POSE_CONTROL", "ADD_DEPTH_CONTROL", "REPAIR_FRAMES", "ADJUST_CONTROL_STRENGTH",
]


class RepairAction(BaseModel):
    shot_id: str
    recommendations: list[QcRecommendation]
    changes: dict[str, float | int | str | bool]
    reason: str
    unsupported: list[str] = Field(default_factory=list)


class RepairPlan(BaseModel):
    round: int
    actions: list[RepairAction]
