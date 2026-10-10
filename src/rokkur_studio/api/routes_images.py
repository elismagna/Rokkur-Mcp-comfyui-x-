"""/images endpoints: the still-picture library (docs/images.md)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.db.models import Image
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import images as svc
from rokkur_studio.services.images import IMAGE_EXTS, ImageRequest, ImageStore, image_view

router = APIRouter(prefix="/images", tags=["images"])
Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]


def _store(ctx: StudioContext) -> ImageStore:
    return ImageStore(ctx.settings.studio.data_dir)


def _image(session: Session, image_id: str) -> Image:
    try:
        return svc.get_image(session, image_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("", summary="The picture library, newest first")
def list_pictures(session: Db, limit: int = Query(60, le=500), offset: int = 0,
                  kind: str | None = None, project_id: str | None = None,
                  status: str | None = None) -> list[dict[str, Any]]:
    return [image_view(i) for i in svc.list_images(session, limit=limit, offset=offset, kind=kind,
                                                   project_id=project_id, status=status)]


@router.get("/profiles", summary="Image profiles and whether each operation can run")
def profiles(ctx: Ctx, target: str = "local") -> dict[str, Any]:
    out = {}
    for name, profile in ctx.settings.image_profiles.items():
        out[name] = {"description": profile.description, "operations": {
            op: ctx.settings.image_profile_problem(name, op, target=target)
            for op in profile.workflows}}
    return out


@router.post("", status_code=201, summary="Ask for pictures; one job renders them")
def create_pictures(body: ImageRequest, ctx: Ctx, session: Db) -> list[dict[str, Any]]:
    try:
        rows = svc.request_images(session, ctx.settings, _store(ctx), ctx.ffmpeg, body)
    except (ValueError, LookupError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return [image_view(r) for r in rows]


@router.post("/upload", status_code=201, summary="Put your own picture into the library")
def upload_picture(ctx: Ctx, session: Db, file: Annotated[UploadFile, File()],
                   rights_confirmed: Annotated[bool, Form()] = False,
                   rights_evidence: Annotated[str, Form()] = "",
                   title: Annotated[str, Form()] = "") -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise HTTPException(422, "Use a PNG, JPG or WebP picture.")
    if not rights_confirmed or not rights_evidence.strip():
        raise HTTPException(422, "Confirm that you may use and transform the picture and say "
                                 "where it comes from.")
    uploads = Path(ctx.settings.studio.data_dir) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{uuid.uuid4().hex[:12]}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        row = svc.import_image(session, _store(ctx), ctx.ffmpeg, dest, kind="upload",
                               title=title or Path(file.filename or "").stem,
                               request={"rights_confirmed": True,
                                        "rights_evidence": rights_evidence.strip()})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        dest.unlink(missing_ok=True)
    return image_view(row)


@router.get("/{image_id}")
def get_picture(image_id: str, session: Db) -> dict[str, Any]:
    return image_view(_image(session, image_id))


@router.get("/{image_id}/file")
def picture_file(image_id: str, ctx: Ctx, session: Db) -> FileResponse:
    image = _image(session, image_id)
    if not image.rel_path:
        raise HTTPException(404, "this picture has no file yet")
    return FileResponse(_store(ctx).path_for(image.rel_path), media_type="image/png")


@router.get("/{image_id}/source")
def picture_source(image_id: str, ctx: Ctx, session: Db) -> FileResponse:
    image = _image(session, image_id)
    if not image.source_rel_path:
        raise HTTPException(404, "this picture was not made from another one")
    return FileResponse(_store(ctx).path_for(image.source_rel_path), media_type="image/png")


class VerdictIn(BaseModel):
    value: int | None = None


@router.post("/{image_id}/verdict")
def set_picture_verdict(image_id: str, body: VerdictIn, session: Db) -> dict[str, Any]:
    try:
        return image_view(svc.set_verdict(session, _image(session, image_id), body.value))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/{image_id}", status_code=204)
def delete_picture(image_id: str, ctx: Ctx, session: Db) -> None:
    svc.delete_image(session, _store(ctx), _image(session, image_id))
