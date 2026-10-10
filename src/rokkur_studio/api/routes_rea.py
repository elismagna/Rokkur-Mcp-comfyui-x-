"""/rea endpoints: runs of the REA reverse-engineering CLI (docs/rea.md)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.db.models import ReaRun
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import rea as svc
from rokkur_studio.services.rea import PRESETS, USES, ReaRequest, ReaStore, run_view

router = APIRouter(prefix="/rea", tags=["rea"])
Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]


def _store(ctx: StudioContext) -> ReaStore:
    return ReaStore(ctx.settings.studio.data_dir)


def _run(session: Session, run_id: str) -> ReaRun:
    try:
        return svc.get_run(session, run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("", summary="REA runs, newest first")
def list_runs(session: Db, limit: int = Query(60, le=500), offset: int = 0,
              preset: str | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
    return [run_view(r) for r in svc.list_runs(session, limit=limit, offset=offset, preset=preset,
                                               project_id=project_id)]


@router.get("/status", summary="Whether the worker can run REA, and the presets it offers")
def rea_status(ctx: Ctx) -> dict[str, Any]:
    avail = svc.availability(ctx.settings)
    return {"available": avail.ok, "reason": avail.reason, "version": avail.version,
            "command": avail.command, "presets": PRESETS, "uses": USES}


@router.post("", status_code=202, summary="Queue one REA run; the worker executes it")
def create_run(body: ReaRequest, ctx: Ctx, session: Db) -> dict[str, Any]:
    try:
        return run_view(svc.request_run(session, ctx.settings, _store(ctx), body))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{run_id}")
def get_run(run_id: str, session: Db) -> dict[str, Any]:
    return run_view(_run(session, run_id))


@router.get("/{run_id}/output", summary="The run's full output (JSON, or text when REA gave none)")
def run_output(run_id: str, ctx: Ctx, session: Db) -> FileResponse:
    run = _run(session, run_id)
    if not run.rel_path:
        raise HTTPException(404, "this run has no output yet")
    path = _store(ctx).path_for(run.rel_path)
    return FileResponse(path, media_type="application/json" if path.suffix == ".json"
                        else "text/plain", filename=f"rea-{run.id}{path.suffix}")


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: str, ctx: Ctx, session: Db) -> None:
    svc.delete_run(session, _store(ctx), _run(session, run_id))
