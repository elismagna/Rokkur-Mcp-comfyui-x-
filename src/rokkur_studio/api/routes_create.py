"""/create endpoints: image and video takes from a prompt (docs/create.md)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.db.models import Job, Take
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import create

router = APIRouter(prefix="/create", tags=["create"])

Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]


class TakeIn(BaseModel):
    mode: Literal["image", "video"] = "image"
    prompt: str = Field(min_length=1, max_length=2000)
    aspect: str = ""
    seconds: int = 5
    seed: int | None = Field(default=None, ge=0)
    count: int = Field(default=1, ge=1, le=create.MAX_BATCH)
    negative_prompt: str = Field(default="", max_length=2000)
    camera: dict[str, str] = Field(default_factory=dict,
                                   description=f"keys: {', '.join(create.CINEMA)}")
    start_take: str | None = Field(default=None, description="a finished picture to animate")
    render_on: Literal["local", "cloud"] = "local"


@router.get("/options")
def options() -> dict[str, Any]:
    """Shapes, lengths and Cinema choices the Create page offers."""
    return {"image_sizes": create.IMAGE_SIZES, "video_sizes": create.VIDEO_SIZES,
            "seconds": create.SECONDS, "cinema": {k: list(v) for k, v in create.CINEMA.items()}}


@router.get("/takes")
def list_takes(session: Db, limit: int = 60) -> list[dict[str, Any]]:
    return create.listing(session, limit=min(max(limit, 1), 500))


@router.post("/takes", status_code=201)
def make_takes(body: TakeIn, ctx: Ctx, session: Db) -> list[dict[str, Any]]:
    start = None
    if body.start_take:
        picked = session.get(Take, body.start_take)
        if picked is None or picked.mode != "image" or not picked.output:
            raise HTTPException(422, "start_take must be a finished picture")
        start = create.path_for(ctx.settings, picked.output)
    mode = "animate" if body.mode == "video" and start is not None else body.mode
    try:
        takes = create.submit(session, ctx.settings, create.TakeRequest(
            mode=mode, prompt=body.prompt, aspect=body.aspect, seconds=body.seconds,
            seed=body.seed, count=body.count, negative_prompt=body.negative_prompt,
            camera=body.camera, start_image=start, render_on=body.render_on,
            parent_id=body.start_take))
    except create.CreateError as exc:
        raise HTTPException(422, str(exc)) from exc
    return [create.view(t, None) for t in takes]


@router.get("/takes/{take_id}")
def get_take(take_id: str, session: Db) -> dict[str, Any]:
    take = session.get(Take, take_id)
    if take is None:
        raise HTTPException(404, "no such take")
    return create.view(take, session.get(Job, take.job_id) if take.job_id else None)
