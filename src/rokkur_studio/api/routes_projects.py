"""/projects endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.api.schemas import (
    AssetOut,
    EventOut,
    JobOut,
    ProjectCreate,
    ProjectDetail,
    ProjectOut,
    PublicationOut,
    PublishIn,
    RatingOut,
    RedoIn,
    RenderOut,
    RightsDecisionIn,
    RightsOut,
)
from rokkur_studio.db.models import Asset, Event, Job, Project, Publication, Rating, Render
from rokkur_studio.domain.states import ProjectStatus
from rokkur_studio.pipeline.context import (
    StudioContext,
    profile_availability,
    supported_controls,
)
from rokkur_studio.pipeline.driver import advance, next_job_kind
from rokkur_studio.services import commands, publishing, ratings
from rokkur_studio.services.assets import import_file, project_assets
from rokkur_studio.services.projects import get_project, latest_document, latest_rights
from rokkur_studio.youtube.client import YouTubeError
from rokkur_studio.youtube.oauth import OAuthError

router = APIRouter(prefix="/projects", tags=["projects"])

Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]

DOC_KINDS = ("analysis", "creative_brief", "prompt_schedule", "manifest", "qc_report",
             "repair_plan", "metadata")


def _project(session: Session, project_id: str, for_update: bool = False) -> Project:
    try:
        return get_project(session, project_id, for_update=for_update)
    except LookupError as exc:
        raise HTTPException(404, f"project {project_id} not found") from exc


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, ctx: Ctx, session: Db) -> Project:
    try:
        profile = body.render_profile or ctx.settings.render.default_profile
        ctx.settings.profile(profile)
        if problem := profile_availability(ctx)[profile]:
            raise ValueError(problem)
        return commands.create_project(session, body, ctx.settings)
    except (KeyError, LookupError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("", response_model=list[ProjectOut])
def list_projects(session: Db, status: ProjectStatus | None = None,
                  limit: int = Query(100, le=500)) -> list[Project]:
    stmt = select(Project).order_by(Project.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Project.status == status.value)
    return list(session.scalars(stmt))


def project_detail(session: Session, project: Project) -> ProjectDetail:
    docs = {}
    for kind in DOC_KINDS:
        doc = latest_document(session, project.id, kind)
        if doc is not None:
            docs[kind] = {"version": doc.version, "created_by": doc.created_by,
                          "created_at": doc.created_at.isoformat(), "data": doc.data}
    src = project.source
    rights = latest_rights(session, project.id)
    return ProjectDetail(
        project=ProjectOut.model_validate(project),
        source={c: getattr(src, c) for c in ("platform", "video_id", "url", "title", "creator",
                                             "local_path", "asset_id")} if src else None,
        rights=RightsOut.model_validate(rights) if rights else None,
        documents=docs,
        renders=[RenderOut.model_validate(r) for r in session.scalars(
            select(Render).where(Render.project_id == project.id)
            .order_by(Render.started_at))],
        assets=[AssetOut.model_validate(a) for a in project_assets(session, project.id)],
        jobs=[JobOut.model_validate(j) for j in session.scalars(
            select(Job).where(Job.project_id == project.id).order_by(Job.created_at))],
        publications=[PublicationOut.model_validate(p) for p in session.scalars(
            select(Publication).where(Publication.project_id == project.id))],
        next_job=next_job_kind(project),
    )


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project_detail(project_id: str, session: Db) -> ProjectDetail:
    return project_detail(session, _project(session, project_id))


@router.post("/{project_id}/start", response_model=ProjectOut)
def start_project(project_id: str, ctx: Ctx, session: Db) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.start(session, project, ctx.settings)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.post("/{project_id}/advance", response_model=JobOut | None,
             summary="Manually enqueue the next stage (autonomy level 0/1)")
def advance_project(project_id: str, ctx: Ctx, session: Db) -> Job | None:
    project = _project(session, project_id)
    if next_job_kind(project) is None:
        raise HTTPException(409, f"no stage job runs in state {project.status}")
    return advance(session, project, ctx.settings, manual=True)


@router.post("/{project_id}/cancel", response_model=ProjectOut)
def cancel_project(project_id: str, session: Db) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.cancel(session, project)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.post("/{project_id}/resume", response_model=ProjectOut)
def resume_project(project_id: str, ctx: Ctx, session: Db) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.resume_project(session, project, ctx.settings)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.get("/{project_id}/ratings", response_model=list[RatingOut], tags=["ratings"])
def list_ratings(project_id: str, session: Db) -> list[Rating]:
    return ratings.project_ratings(session, _project(session, project_id).id)


@router.put("/{project_id}/ratings", response_model=RatingOut | None, tags=["ratings"],
            summary="Rate a shot attempt or the video (value 0 removes your rating)")
def put_rating(project_id: str, body: ratings.RatingIn, session: Db) -> Rating | None:
    try:
        return ratings.rate(session, _project(session, project_id), body, actor="api")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/{project_id}/redo", response_model=ProjectOut, tags=["ratings"],
             summary="Render the given shots again; every other shot keeps its render")
def redo(project_id: str, body: RedoIn, ctx: Ctx, session: Db) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.redo_shots(session, project, ctx.settings, shot_ids=body.shots, actor="api",
                            supported=supported_controls(ctx, project.render_profile))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.post("/{project_id}/repair-more", response_model=ProjectOut,
             summary="Allow more repair rounds for a project stopped at the repair limit")
def repair_more(project_id: str, ctx: Ctx, session: Db, rounds: int | None = None) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.repair_more(session, project, ctx.settings, actor="api", rounds=rounds)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.post("/{project_id}/allow-more-renders", response_model=ProjectOut,
             summary="Raise the render budget of a project stopped at it, and resume it")
def allow_more_renders(project_id: str, ctx: Ctx, session: Db,
                       renders: int | None = None) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.allow_more_renders(session, project, ctx.settings, actor="api", renders=renders)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.post("/{project_id}/recheck-quality", response_model=ProjectOut,
             summary="Run the quality check again on the current renders, rendering nothing")
def recheck_quality(project_id: str, ctx: Ctx, session: Db) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.recheck_quality(session, project, ctx.settings, actor="api")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.post("/{project_id}/keep-renders", response_model=ProjectOut,
             summary="Keep the renders QC rejected and finish the video from them")
def keep_renders(project_id: str, ctx: Ctx, session: Db, note: str | None = None) -> Project:
    project = _project(session, project_id, for_update=True)
    try:
        commands.keep_renders(session, project, ctx.settings, actor="api", note=note)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return project


@router.post("/{project_id}/rights", response_model=ProjectOut,
             summary="Record a human rights decision for a project awaiting one")
def decide_rights(project_id: str, body: RightsDecisionIn, ctx: Ctx, session: Db) -> Project:
    project = _project(session, project_id, for_update=True)
    fields = body.model_dump(exclude={"approve", "decided_by", "note"})
    fields["category"] = body.category.value
    try:
        return commands.decide_rights(session, project, ctx.settings, approve=body.approve,
                                      decided_by=body.decided_by, fields=fields, note=body.note)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{project_id}/source", response_model=AssetOut,
             summary="Upload the permitted source video")
def upload_source(project_id: str, ctx: Ctx, session: Db,
                  file: UploadFile = File(...)) -> Asset:  # noqa: B008
    project = _project(session, project_id, for_update=True)
    if project.status not in (ProjectStatus.DISCOVERED, ProjectStatus.SCORED,
                              ProjectStatus.RIGHTS_PENDING, ProjectStatus.RIGHTS_OK):
        raise HTTPException(409, "source can only be replaced before ingestion")
    suffix = Path(file.filename or "source.mp4").suffix.lower() or ".mp4"
    tmp = ctx.store.project_dir(project.id, "work") / f"upload{suffix}"
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    asset = import_file(session, ctx.store, project.id, "source", "source", tmp,
                        name=f"source{suffix}")
    tmp.unlink(missing_ok=True)
    assert project.source is not None
    project.source.asset_id = asset.id
    project.source.platform = "upload"
    return asset


@router.get("/{project_id}/events", response_model=list[EventOut])
def project_events(project_id: str, session: Db) -> list[Event]:
    _project(session, project_id)
    return list(session.scalars(select(Event).where(Event.project_id == project_id)
                                .order_by(Event.id)))


@router.get("/{project_id}/assets", response_model=list[AssetOut])
def list_assets(project_id: str, session: Db, kind: str | None = None) -> list[Asset]:
    _project(session, project_id)
    return project_assets(session, project_id, kind)


@router.get("/{project_id}/prompt-schedule", response_class=PlainTextResponse)
def prompt_schedule(project_id: str, session: Db) -> PlainTextResponse:
    """The Batch Prompt Schedule text (FizzNodes format), ready to paste into the node."""
    _project(session, project_id)
    doc = latest_document(session, project_id, "prompt_schedule")
    if doc is None:
        raise HTTPException(404, "no prompt schedule yet: the brief stage writes it")
    return PlainTextResponse(doc.data["text"] + "\n")


@router.get("/{project_id}/assets/{asset_id}/file")
def asset_file(project_id: str, asset_id: str, ctx: Ctx, session: Db) -> FileResponse:
    asset = session.get(Asset, asset_id)
    if asset is None or asset.project_id != project_id:
        raise HTTPException(404, "asset not found")
    return FileResponse(ctx.store.path_for(asset.rel_path), media_type=asset.mime)


@router.post("/{project_id}/render", response_model=JobOut | None,
             summary="Queue rendering for a project whose workflow is ready")
def render_project(project_id: str, ctx: Ctx, session: Db) -> Job | None:
    project = _project(session, project_id)
    if project.status not in (ProjectStatus.WORKFLOW_READY, ProjectStatus.RENDER_QUEUED):
        raise HTTPException(409, f"cannot render in state {project.status}")
    return advance(session, project, ctx.settings, manual=True)


@router.post("/{project_id}/publish", response_model=PublicationOut,
             summary="Dry-run (default) or really upload a READY_TO_PUBLISH project to YouTube")
def publish(project_id: str, body: PublishIn, ctx: Ctx, session: Db) -> Publication | Response:
    """``publish_at`` schedules a public release (private until then; needs allow_public)."""
    project = _project(session, project_id, for_update=True)
    try:
        if body.dry_run:
            return publishing.dry_run(session, project, privacy=body.privacy,
                                      publish_at=body.publish_at, playlist_id=body.playlist_id,
                                      settings=ctx.settings)
        client = ctx.extras.get("youtube_client") or publishing.make_client(ctx.settings)
        return publishing.upload(session, project, settings=ctx.settings, store=ctx.store,
                                 client=client, privacy=body.privacy,  # type: ignore[arg-type]
                                 publish_at=body.publish_at, playlist_id=body.playlist_id)
    except publishing.PublishGateError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OAuthError as exc:
        raise HTTPException(409, str(exc)) from exc
    except YouTubeError as exc:
        # Returned, not raised: the request transaction then commits the failed publication
        # and its audit event instead of rolling them back.
        return JSONResponse({"detail": str(exc)}, status_code=502)
