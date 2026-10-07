"""/director endpoints: the asset tracker, the allowed vocabulary and a prompt preview."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from rokkur_studio.api.deps import get_ctx
from rokkur_studio.director.assets import (
    AssetTracker,
    TrackerError,
    load_tracker,
    save_tracker,
    tracker_path,
)
from rokkur_studio.director.prompts import Weights, preview
from rokkur_studio.director.vocabulary import (
    VOCABULARY,
    CameraAngle,
    CameraMovement,
    Lighting,
    ShotSize,
)
from rokkur_studio.pipeline.context import StudioContext

router = APIRouter(prefix="/director", tags=["director"])

Ctx = Annotated[StudioContext, Depends(get_ctx)]


def current_tracker(ctx: StudioContext) -> AssetTracker:
    try:
        return load_tracker(tracker_path(ctx.settings.studio.data_dir))
    except TrackerError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/assets")
def get_assets(ctx: Ctx) -> dict[str, Any]:
    """The asset tracker (global look, global negative prompt, characters)."""
    return current_tracker(ctx).to_json()


@router.put("/assets")
def put_assets(body: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    """Replace the asset tracker. Keys: PROMPT_PREFIX, GLOBAL_STYLE_MODIFIERS,
    GLOBAL_NEGATIVE_PROMPT, CHARACTERS (name -> description)."""
    try:
        tracker = AssetTracker.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
    save_tracker(tracker_path(ctx.settings.studio.data_dir), tracker)
    return tracker.to_json()


@router.get("/vocabulary")
def vocabulary() -> dict[str, list[str]]:
    """The only cinematography terms the Director of Photography may choose."""
    return {k: list(v) for k, v in VOCABULARY.items()}


class PreviewIn(BaseModel):
    theme: str = ""
    subject: str = ""
    background: str = ""
    shot_size: ShotSize | None = None
    camera_angle: CameraAngle | None = None
    camera_movement: CameraMovement | None = None
    lighting: Lighting | None = None
    character_key: str | None = None
    use_global_look: bool = True


@router.post("/preview")
def preview_prompt(body: PreviewIn, ctx: Ctx) -> dict[str, Any]:
    """Compile one prompt with the studio's rules, without any model or render."""
    d = ctx.settings.director
    return preview(current_tracker(ctx), theme=body.theme, subject=body.subject,
                   background=body.background, shot_size=body.shot_size,
                   camera_angle=body.camera_angle, camera_movement=body.camera_movement,
                   lighting=body.lighting, character_key=body.character_key,
                   global_look=body.use_global_look,
                   weights=Weights(framing=d.framing_weight, angle=d.angle_weight))
