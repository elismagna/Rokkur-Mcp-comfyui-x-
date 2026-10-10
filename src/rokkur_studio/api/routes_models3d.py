"""/models3d endpoints: the 3D studio (docs/three.md)."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.db.models import Model3D
from rokkur_studio.mesh import ops
from rokkur_studio.mesh.io import MESH_EXTS, MeshError
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import models3d as svc
from rokkur_studio.services.models3d import (
    METHODS,
    EditRequest,
    ModelStore,
    PointsRequest,
    ReconRequest,
    ReliefRequest,
    model_view,
)

router = APIRouter(prefix="/models3d", tags=["3d"])
Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]
_MIME = {"stl": "model/stl", "obj": "model/obj", "ply": "application/octet-stream",
         "glb": "model/gltf-binary"}


def _store(ctx: StudioContext) -> ModelStore:
    return ModelStore(ctx.settings.studio.data_dir)


def _model(session: Session, model_id: str) -> Model3D:
    try:
        return svc.get_model(session, model_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("", summary="The 3D models, newest first")
def list_models(session: Db, limit: int = Query(60, le=500), offset: int = 0,
                kind: str | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
    return [model_view(m) for m in svc.list_models(session, limit=limit, offset=offset,
                                                   kind=kind, project_id=project_id)]


@router.get("/methods", summary="Ways to make a model and whether each can run now")
def methods(ctx: Ctx, target: str = "local") -> dict[str, Any]:
    status = svc.availability(ctx.settings, target)
    return {key: {**spec, "problem": status.get(key)} for key, spec in METHODS.items()}


@router.post("/upload", status_code=201, summary="Open an STL, OBJ, PLY or GLB file")
def upload_model(ctx: Ctx, session: Db, file: Annotated[UploadFile, File()],
                 title: Annotated[str, Form()] = "",
                 rights_evidence: Annotated[str, Form()] = "") -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in MESH_EXTS:
        raise HTTPException(422, "Open an STL, OBJ, PLY or GLB file.")
    uploads = Path(ctx.settings.studio.data_dir) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{uuid.uuid4().hex[:12]}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        row = svc.import_model(session, _store(ctx), dest, title=title or Path(file.filename or "").stem,
                               rights_evidence=rights_evidence)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        dest.unlink(missing_ok=True)
    return model_view(row)


@router.post("/edit", status_code=201, summary="Scale, fit, rotate, mirror, centre, flip, repair or combine")
def edit_model(body: EditRequest, ctx: Ctx, session: Db) -> dict[str, Any]:
    try:
        return model_view(svc.edit_model(session, _store(ctx), body))
    except (ValueError, LookupError) as exc:
        raise HTTPException(422, str(exc)) from exc


class BoxIn(BaseModel):
    size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    title: str = Field("", max_length=200)


@router.post("/box", status_code=201, summary="A box of the given size")
def make_box(body: BoxIn, ctx: Ctx, session: Db) -> dict[str, Any]:
    try:
        return model_view(svc.make_box(session, _store(ctx), body.size, title=body.title))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/relief", status_code=201, summary="A relief or lithophane from a picture")
def make_relief(body: ReliefRequest, ctx: Ctx, session: Db) -> dict[str, Any]:
    try:
        return model_view(svc.make_relief(session, ctx.settings, _store(ctx), ctx.ffmpeg, body))
    except (ValueError, LookupError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/points", status_code=201, summary="A 2.5D solid from a LiDAR scan or point cloud")
def points_to_model(body: PointsRequest, ctx: Ctx, session: Db) -> dict[str, Any]:
    try:
        return model_view(svc.points_to_model(session, _store(ctx), body))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/reconstruct", status_code=202,
             summary="From one picture (Hunyuan3D), photos or a video (Meshroom), or points (Poisson)")
def reconstruct(body: ReconRequest, ctx: Ctx, session: Db) -> dict[str, Any]:
    try:
        return model_view(svc.request_reconstruction(session, ctx.settings, _store(ctx),
                                                     ctx.ffmpeg, body))
    except (ValueError, LookupError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{model_id}")
def get_model(model_id: str, session: Db) -> dict[str, Any]:
    return model_view(_model(session, model_id))


@router.get("/{model_id}/file", summary="The model as STL, or ?format=obj|ply|glb, ?ascii=1 for ASCII STL")
def model_file(model_id: str, ctx: Ctx, session: Db, format: str = "stl",
               ascii: bool = False) -> FileResponse:
    row = _model(session, model_id)
    if not row.rel_path:
        raise HTTPException(404, "this model has no file yet")
    try:
        path = svc.export_model(_store(ctx), row, format, ascii_stl=ascii)
    except (ValueError, MeshError) as exc:
        raise HTTPException(422, str(exc)) from exc
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", row.title or row.id).strip("_") or row.id
    return FileResponse(path, media_type=_MIME.get(path.suffix.lstrip("."), "application/octet-stream"),
                        filename=f"{name}{path.suffix}")


@router.get("/{model_id}/view.json", summary="Vertices and faces for the page's 3D view")
def model_view_data(model_id: str, ctx: Ctx, session: Db) -> dict[str, Any]:
    row = _model(session, model_id)
    try:
        return svc.view_data(_store(ctx), row, ctx.settings.three.view_max_faces)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{model_id}/preview.svg", summary="A shaded drawing of the model")
def model_preview(model_id: str, ctx: Ctx, session: Db) -> Response:
    row = _model(session, model_id)
    store = _store(ctx)
    if not row.rel_path:
        raise HTTPException(404, "this model has no file yet")
    cached = store.dir(row.id) / "preview.svg"
    if not cached.is_file():
        cached.write_text(ops.preview_svg(svc.load(store, row), width=320, height=240,
                                          max_faces=6000), encoding="utf-8")
    return Response(cached.read_text(encoding="utf-8"), media_type="image/svg+xml")


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: str, ctx: Ctx, session: Db) -> None:
    svc.delete_model(session, _store(ctx), _model(session, model_id))
