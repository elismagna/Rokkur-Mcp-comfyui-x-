"""Versioned video reconstruction manifest: the only input to workflow compilation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

MANIFEST_VERSION = 1


class VideoSpec(BaseModel):
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration: float = Field(gt=0)


class StyleSpec(BaseModel):
    theme: str
    prompt: str
    negative_prompt: str = "blurry, deformed, extra limbs, watermark, text"
    strength: float = Field(0.7, ge=0, le=1)


class IdentitySpec(BaseModel):
    reference_image: str | None = None  # store-relative path
    reference_mode: Literal["source", "none"] = "source"
    strength: float = Field(0.8, ge=0, le=1)


class MotionSpec(BaseModel):
    type: str = "unknown"
    intensity: float = Field(0.0, ge=0, le=1)


class CameraSpec(BaseModel):
    type: str = "unknown"


class ControlsSpec(BaseModel):
    pose: bool = False
    depth: bool = False
    segmentation: bool = False


class ShotSpec(BaseModel):
    shot_id: str
    start: float = Field(ge=0)
    end: float
    motion: MotionSpec = MotionSpec()
    camera: CameraSpec = CameraSpec()
    controls: ControlsSpec = ControlsSpec()
    seed: int = 0
    prompt: str = ""  # compiled per-shot prompt; empty means the manifest's style prompt
    overrides: dict[str, float | int | str | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered(self) -> ShotSpec:
        if self.end <= self.start:
            raise ValueError(f"{self.shot_id}: end must be after start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class ReconstructionManifest(BaseModel):
    version: Literal[1] = 1
    project_id: str
    source_asset: str  # store-relative path of the ingested source
    video: VideoSpec
    style: StyleSpec
    identity: IdentitySpec = IdentitySpec()
    shots: list[ShotSpec] = Field(min_length=1)
    render_profile: str
    target_format: str = "youtube_short"

    @model_validator(mode="after")
    def _shots_ordered(self) -> ReconstructionManifest:
        ids = [s.shot_id for s in self.shots]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate shot ids")
        for a, b in zip(self.shots, self.shots[1:], strict=False):
            if b.start < a.end - 1e-3:
                raise ValueError(f"shots overlap: {a.shot_id} and {b.shot_id}")
        return self

    def shot(self, shot_id: str) -> ShotSpec:
        for s in self.shots:
            if s.shot_id == shot_id:
                return s
        raise KeyError(shot_id)
