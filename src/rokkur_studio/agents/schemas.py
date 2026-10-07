"""Structured agent outputs. Agents return these, never free prose."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rokkur_studio.domain.rights import RightsCategory

TargetFormat = Literal["youtube_short", "youtube_video"]


class ShotPlan(BaseModel):
    shot_id: str
    start: float
    end: float
    intent: str = ""
    motion_type: str = "unknown"
    camera: str = "unknown"


class CreativeBrief(BaseModel):
    """Creative Director output."""

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
    shot_plan: list[ShotPlan] = Field(min_length=1)
    rationale: str = ""


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
    "ADD_POSE_CONTROL", "ADD_DEPTH_CONTROL", "REPAIR_FRAMES",
]


class RepairAction(BaseModel):
    shot_id: str
    recommendations: list[QcRecommendation]
    changes: dict[str, float | int | str | bool]
    reason: str


class RepairPlan(BaseModel):
    round: int
    actions: list[RepairAction]
