"""Server-rendered control dashboard (Jinja2, no frontend build step).

Every action here goes through the same services as the API and CLI, so the gates (rights,
QC, YouTube private-by-default) apply identically. Messages are passed back to the page as
``?msg=`` / ``?err=`` query parameters after a redirect.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.api.routes_projects import project_detail
from rokkur_studio.api.routes_system import gpu_leases as gpu_info
from rokkur_studio.api.routes_system import system as system_info
from rokkur_studio.api.routes_system import workers as workers_info
from rokkur_studio.api.routes_youtube import release_overview
from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
from rokkur_studio.db.models import (
    ApprovalRequest,
    Asset,
    CostEntry,
    Event,
    Job,
    Project,
    Publication,
)
from rokkur_studio.director.assets import (
    AssetTracker,
    TrackerError,
    load_tracker,
    normalise_key,
    save_tracker,
    tracker_path,
)
from rokkur_studio.director.prompts import Weights, preview
from rokkur_studio.director.vocabulary import VOCABULARY
from rokkur_studio.domain.rights import RightsCategory
from rokkur_studio.domain.states import ProjectStatus as S
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import commands, publishing
from rokkur_studio.services.projects import (
    at_budget_limit,
    at_repair_limit,
    budget_usage,
    failure_reason,
    get_project,
    latest_document,
    save_document,
)
from rokkur_studio.youtube.client import YouTubeError, video_url
from rokkur_studio.youtube.oauth import OAuthError

router = APIRouter(prefix="/ui", include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["pretty"] = lambda v: json.dumps(v, indent=2, default=str)
templates.env.globals["video_url"] = video_url


def _release_label(value: datetime | str | None, settings: Any) -> str:
    when = value if isinstance(value, datetime) else publishing.parse_iso(value)
    return publishing.local_label(settings, when) if when else ""


templates.env.filters["release_label"] = _release_label

Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]

WORKING = {"RIGHTS_PENDING", "DOWNLOADED_OR_INGESTED", "ANALYZING", "ANALYZED",
           "CREATIVE_PLANNING", "CREATIVE_READY", "WORKFLOW_COMPILING", "WORKFLOW_READY",
           "RENDER_QUEUED", "RENDERING", "QUALITY_CHECK", "QUALITY_FAILED", "REPAIRING",
           "QUALITY_PASSED", "EDITING", "PUBLISHING"}
ATTENTION = {"RIGHTS_PENDING", "FAILED"}
GROUPS: dict[str, set[str] | None] = {
    "all": None,
    "working": WORKING - {"RIGHTS_PENDING"},
    "attention": ATTENTION,
    "ready": {"READY_TO_PUBLISH"},
    "published": {"PUBLISHED", "MONITORING"},
    "closed": {"ARCHIVED", "CANCELLED", "RIGHTS_REJECTED"},
}

# The pipeline as a person thinks about it; each step lists the states that belong to it.
STAGES: list[tuple[str, set[str]]] = [
    ("Rights", {S.RIGHTS_PENDING, S.RIGHTS_OK}),
    ("Ingest", {S.DOWNLOADED_OR_INGESTED}),
    ("Analyse", {S.ANALYZING, S.ANALYZED}),
    ("Brief", {S.CREATIVE_PLANNING, S.CREATIVE_READY}),
    ("Workflow", {S.WORKFLOW_COMPILING, S.WORKFLOW_READY}),
    ("Render", {S.RENDER_QUEUED, S.RENDERING}),
    ("Quality", {S.QUALITY_CHECK, S.QUALITY_FAILED, S.REPAIRING, S.QUALITY_PASSED}),
    ("Edit", {S.EDITING}),
    ("Ready", {S.READY_TO_PUBLISH, S.PUBLISHING}),
    ("Published", {S.PUBLISHED, S.MONITORING}),
]
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def stage_progress(status: str, failed_from: str | None = None) -> list[dict[str, str]]:
    """``[{"label", "state"}]`` with state done / current / failed / todo."""
    effective = failed_from if status == S.FAILED and failed_from else status
    index = next((i for i, (_, states) in enumerate(STAGES) if effective in states), -1)
    if status in (S.PUBLISHED, S.MONITORING, S.ARCHIVED) and index == -1:
        index = len(STAGES) - 1
    out = []
    for i, (label, _) in enumerate(STAGES):
        if i < index:
            state = "done"
        elif i == index:
            state = "failed" if status in (S.FAILED, S.RIGHTS_REJECTED) else (
                "done" if status in (S.PUBLISHED, S.MONITORING) else "current")
        else:
            state = "todo"
        out.append({"label": label, "state": state})
    return out


def percent(status: str, failed_from: str | None = None) -> int:
    steps = stage_progress(status, failed_from)
    done = sum(1 for s in steps if s["state"] == "done")
    return round(100 * done / len(steps))


templates.env.globals["stage_progress"] = stage_progress
templates.env.globals["percent"] = percent


def _page(request: Request, name: str, ctx: StudioContext, **data: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {
        "profile_status": {n: ctx.settings.profile_problem(n) for n in ctx.settings.profiles},
        "working": WORKING, "attention": ATTENTION, "settings": ctx.settings,
        "msg": request.query_params.get("msg"), "err": request.query_params.get("err"),
        "path": request.url.path, **data})


def _back(url: str, *, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    sep = "&" if "?" in url else "?"
    if msg:
        url += f"{sep}msg={quote(msg)}"
    elif err:
        url += f"{sep}err={quote(err)}"
    return RedirectResponse(url, status_code=303)


def _thumbs(session: Session, project_ids: list[str]) -> dict[str, str]:
    if not project_ids:
        return {}
    rows = session.execute(select(Asset.project_id, Asset.id)
                           .where(Asset.project_id.in_(project_ids), Asset.kind == "thumbnail")
                           .order_by(Asset.created_at)).all()
    return {pid: f"/projects/{pid}/assets/{aid}/file" for pid, aid in rows}


def media_files(ctx: StudioContext) -> list[dict[str, Any]]:
    root = ctx.settings.studio.media_dir
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
             and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"path": str(p), "name": str(p.relative_to(root)),
             "mb": round(p.stat().st_size / 1e6, 1)} for p in files[:300]]


def _probe(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).is_success
    except httpx.HTTPError:
        return False


def services(ctx: StudioContext) -> list[dict[str, Any]]:
    s = ctx.settings
    yt = s.youtube
    signed_in = yt.token_path.is_file()
    return [
        {"name": "ComfyUI", "ok": _probe(f"{s.comfyui.url}/system_stats"),
         "detail": s.comfyui.url, "needed": s.render.renderer == "comfyui"},
        {"name": "Ollama", "ok": _probe(f"{s.ollama.url}/api/version"),
         "detail": f"{s.ollama.model}", "needed": s.agents.provider == "ollama"},
        {"name": "FFmpeg", "ok": ctx.ffmpeg.available(), "detail": "encode + QC", "needed": True},
        {"name": "YouTube", "ok": signed_in and yt.enabled,
         "detail": ("signed in, uploads on" if signed_in and yt.enabled else
                    "signed in, uploads off" if signed_in else "not signed in"),
         "needed": False},
    ]


def quota_today(session: Session) -> float:
    start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return float(session.scalar(select(func.coalesce(func.sum(CostEntry.amount), 0.0)).where(
        CostEntry.kind == "youtube_quota", CostEntry.created_at >= start)) or 0.0)


# -- overview ------------------------------------------------------------------------------
@router.get("", response_class=HTMLResponse)
def overview(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    counts = dict(session.execute(select(Project.status, func.count(Project.id))
                                  .group_by(Project.status)).all())
    tiles = {name: sum(n for st, n in counts.items() if states is None or st in states)
             for name, states in GROUPS.items()}
    active = list(session.scalars(select(Project).where(Project.status.in_(WORKING - ATTENTION))
                                  .order_by(Project.updated_at.desc()).limit(8)))
    attention = list(session.scalars(select(Project).where(Project.status.in_(ATTENTION))
                                     .order_by(Project.updated_at.desc()).limit(8)))
    finished = list(session.scalars(
        select(Project).where(Project.status.in_(["READY_TO_PUBLISH", "PUBLISHED", "MONITORING"]))
        .order_by(Project.updated_at.desc()).limit(8)))
    pending_approvals = session.scalar(select(func.count(ApprovalRequest.id))
                                       .where(ApprovalRequest.status == "pending")) or 0
    return _page(request, "overview.html", ctx, tiles=tiles, active=active, attention=attention,
                 finished=finished, thumbs=_thumbs(session, [p.id for p in finished]),
                 services=services(ctx), gpu=gpu_info(ctx, session),
                 workers=workers_info(session), pending_approvals=pending_approvals)


# -- projects ------------------------------------------------------------------------------
@router.get("/projects", response_class=HTMLResponse)
def projects_page(request: Request, ctx: Ctx, session: Db, group: str = "all",
                  q: str = "") -> HTMLResponse:
    stmt = select(Project).order_by(Project.created_at.desc()).limit(300)
    states = GROUPS.get(group)
    if states is not None:
        stmt = stmt.where(Project.status.in_(states))
    if q:
        stmt = stmt.where(Project.name.ilike(f"%{q}%"))
    projects = list(session.scalars(stmt))
    return _page(request, "projects.html", ctx, projects=projects, group=group, q=q,
                 groups=list(GROUPS), thumbs=_thumbs(session, [p.id for p in projects]))


def _tracker(ctx: StudioContext) -> tuple[AssetTracker | None, str | None]:
    try:
        return load_tracker(tracker_path(ctx.settings.studio.data_dir)), None
    except TrackerError as exc:
        return None, str(exc)


@router.get("/new", response_class=HTMLResponse)
def new_page(request: Request, ctx: Ctx) -> HTMLResponse:
    tracker, _ = _tracker(ctx)
    return _page(request, "new.html", ctx, media=media_files(ctx),
                 characters=tracker.characters if tracker else {},
                 profiles=ctx.settings.profiles,
                 default_profile=ctx.settings.render.default_profile,
                 categories=[c.value for c in RightsCategory
                             if c.value not in ("UNKNOWN", "REJECTED", "REFERENCE_ONLY")])


@router.get("/media/preview")
def media_preview(ctx: Ctx, path: str) -> FileResponse:
    if path not in {m["path"] for m in media_files(ctx)}:
        raise HTTPException(404, "Choose a file from the media folder")
    return FileResponse(path)


@router.post("/projects")
def create_from_form(ctx: Ctx, session: Db, theme: Annotated[str, Form()],
                     rights_category: Annotated[str, Form()],
                     name: Annotated[str, Form()] = "",
                     local_path: Annotated[str, Form()] = "",
                     media_file: Annotated[str, Form()] = "",
                     source_file: Annotated[UploadFile | None, File()] = None,
                     reference_file: Annotated[UploadFile | None, File()] = None,
                     reference_mode: Annotated[str, Form()] = "source",
                     control_strength: Annotated[float, Form()] = 1.0,
                     seed: Annotated[str, Form()] = "",
                     steps: Annotated[str, Form()] = "",
                     cfg: Annotated[float, Form()] = 6.0,
                     negative_prompt: Annotated[str, Form()] = "",
                     prompt: Annotated[str, Form()] = "",
                     permission_evidence: Annotated[str, Form()] = "",
                     character_reference_path: Annotated[str, Form()] = "",
                     character_key: Annotated[str, Form()] = "",
                     character_description: Annotated[str, Form()] = "",
                     use_global_look: Annotated[bool, Form()] = False,
                     render_profile: Annotated[str, Form()] = "",
                     target_format: Annotated[str, Form()] = "youtube_short",
                     autostart: Annotated[bool, Form()] = False) -> RedirectResponse:
    path = ""
    try:
        selected = render_profile or ctx.settings.render.default_profile
        if problem := ctx.settings.profile_problem(selected):
            return _back("/ui/new", err=problem)
    except KeyError as exc:
        return _back("/ui/new", err=str(exc))
    written: list[Path] = []

    def fail(message: str) -> RedirectResponse:
        for upload in written:  # a retry uploads again; do not keep a copy per attempt
            upload.unlink(missing_ok=True)
        return _back("/ui/new", err=message)

    def save_upload(upload: UploadFile, name: str) -> str:
        uploads = Path(ctx.settings.studio.data_dir) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / f"{uuid.uuid4().hex[:12]}_{name}"
        written.append(dest)
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        return str(dest)

    has_reference = reference_file is not None and bool(reference_file.filename)
    ref_suffix = Path(reference_file.filename or "").suffix.lower() if reference_file else ""
    if has_reference and ref_suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        return fail("Use a PNG, JPG or WebP reference image.")
    if source_file is not None and source_file.filename:
        suffix = Path(source_file.filename).suffix.lower()
        if suffix not in VIDEO_EXTS:
            return fail(f"{source_file.filename} is not a video file")
        path = save_upload(source_file, Path(source_file.filename).name)
        name = name or Path(source_file.filename).stem
    elif media_file:
        allowed = {m["path"] for m in media_files(ctx)}
        if media_file not in allowed:
            return fail("that file is not in the media folder")
        path = media_file
    elif local_path:
        path = local_path
    if not path:
        return fail("pick a video from the media folder or upload one")
    if not Path(path).is_file():
        return fail("The worker cannot read that source. Use Upload or the media folder.")
    if has_reference and reference_file is not None:
        character_reference_path = save_upload(reference_file, f"reference{ref_suffix}")
    try:
        body = ProjectCreate(
            name=name or Path(path).stem, target_format=target_format,  # type: ignore[arg-type]
            render_profile=render_profile or None,
            source=SourceIn(platform="local", local_path=path),
            rights=RightsIn(category=RightsCategory(rights_category),
                            permission_evidence=permission_evidence or None),
            creative=CreativeIn(theme=theme.strip(), prompt=prompt or None,
                                reference_mode=reference_mode,  # type: ignore[arg-type]
                                control_strength=control_strength, cfg=cfg,
                                seed=int(seed) if seed.strip() else None,
                                steps=int(steps) if steps.strip() else None,
                                negative_prompt=negative_prompt or None,
                                character_key=character_key or None,
                                character_description=character_description or None,
                                use_global_look=use_global_look,
                                character_reference_path=character_reference_path or None),
            autostart=autostart)
        project = commands.create_project(session, body, ctx.settings, actor="dashboard")
    except (ValueError, LookupError) as exc:
        return fail(str(exc))
    return _back(f"/ui/projects/{project.id}", msg="Project created" + (
        "; the worker picks it up now" if autostart else ""))


def _budget_stop(session: Session, ctx: StudioContext, project: Project) -> dict[str, Any] | None:
    """What a project stopped at its render budget has used, for the page to explain."""
    if not at_budget_limit(session, project, ctx.settings):
        return None
    return {**budget_usage(session, project.id, ctx.settings),
            "grant": commands.render_grant(ctx.settings)}


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(project_id: str, request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    try:
        project = get_project(session, project_id)
    except LookupError as exc:
        raise HTTPException(404) from exc
    detail = project_detail(session, project)
    events = list(session.scalars(select(Event).where(Event.project_id == project_id)
                                  .order_by(Event.id.desc()).limit(150)))
    yt = ctx.settings.youtube
    publish_block = None
    if not yt.enabled:
        publish_block = "YouTube uploads are off. Set STUDIO_YOUTUBE__ENABLED=true in .env."
    elif not yt.token_path.is_file():
        publish_block = "Not signed in to YouTube. Run: .\\scripts\\studio.ps1 youtube-auth"
    keyframes = {a.meta.get("shot_id"): f"/projects/{project_id}/assets/{a.id}/file"
                 for a in detail.assets if a.kind == "keyframe"}
    next_slot = (publishing.next_release_slot(session, ctx.settings, for_project=project_id)
                 if project.status == S.READY_TO_PUBLISH else None)
    proposals = publishing.pending_proposals(session, project_id)
    failed = project.status == S.FAILED
    previews: dict[str, Any] = {}
    for asset in detail.assets:
        if asset.kind == "render" and asset.meta.get("shot_id"):
            key = asset.meta["shot_id"]
            if key not in previews or asset.meta.get("attempt", 0) > previews[key].meta.get("attempt", 0):
                previews[key] = asset
    return _page(request, "project.html", ctx, d=detail, p=project, events=events,
                 shot_previews=previews,
                 keyframes=keyframes,
                 failure=failure_reason(session, project) if failed else None,
                 repair_limit=at_repair_limit(project),
                 budget=_budget_stop(session, ctx, project),
                 steps=stage_progress(project.status, project.failed_from_state),
                 publish_block=publish_block,
                 schedule_block=None if yt.allow_public else (
                     "Scheduling makes the video public at the release time, and public "
                     "uploads are off (youtube.allow_public in config/studio.yaml)."),
                 next_slot=next_slot,
                 next_slot_local=(next_slot.astimezone(publishing.zone(ctx.settings))
                                  .strftime("%Y-%m-%dT%H:%M") if next_slot else ""),
                 playlists=(publishing.load_playlists(ctx.settings) or {}).get("items", []),
                 proposal=proposals[0] if proposals else None,
                 privacies=["private", "unlisted"] + (["public"] if yt.allow_public else []))


@router.post("/projects/{project_id}/metadata")
def save_metadata(project_id: str, ctx: Ctx, session: Db, title: Annotated[str, Form()],
                  description: Annotated[str, Form()],
                  tags: Annotated[str, Form()] = "") -> RedirectResponse:
    url = f"/ui/projects/{project_id}"
    project = get_project(session, project_id, for_update=True)
    current = latest_document(session, project.id, "metadata")
    if current is None:
        return _back(url, err="no metadata yet; it is written when the video is finished")
    if project.status not in (S.READY_TO_PUBLISH, S.EDITING):
        return _back(url, err=f"metadata can only be edited before publishing ({project.status})")
    data = {**current.data, "title": title.strip(),
            "description": description.replace("\r\n", "\n").strip(),
            "tags": [t.strip() for t in tags.split(",") if t.strip()]}
    errors = publishing.validate_metadata(data)
    if errors:
        return _back(url, err="; ".join(errors))
    save_document(session, project.id, "metadata", data, created_by="dashboard")
    return _back(url, msg="Title, description and tags saved")


@router.post("/projects/{project_id}/publish")
def publish_from_ui(project_id: str, ctx: Ctx, session: Db,
                    privacy: Annotated[str, Form()] = "private",
                    mode: Annotated[str, Form()] = "dry",
                    release: Annotated[str, Form()] = "now",
                    publish_at: Annotated[str, Form()] = "",
                    publish_local: Annotated[str, Form()] = "",
                    playlist_id: Annotated[str, Form()] = "") -> RedirectResponse:
    url = f"/ui/projects/{project_id}"
    project = get_project(session, project_id, for_update=True)
    try:
        when: datetime | None = None
        if release == "schedule":
            # The page's script sends UTC; without it the field is read in the studio's zone.
            if publish_at:
                when = publishing.parse_iso(publish_at)
            elif publish_local:
                when = publishing.parse_when(publish_local, ctx.settings)
            else:
                return _back(url, err="Pick the release time")
            privacy = "private"
        plan = (f"private, public {publishing.local_label(ctx.settings, when)}" if when
                else privacy)
        if mode == "dry":
            publishing.dry_run(session, project, privacy=privacy, publish_at=when,
                               playlist_id=playlist_id or None, actor="dashboard",
                               settings=ctx.settings)
            return _back(url, msg=f"Dry run OK ({plan}): everything checks out, "
                                  "nothing was uploaded")
        privacy = publishing.resolve_privacy(ctx.settings, project, privacy)
        with publishing.youtube_client(ctx.settings, ctx.extras.get("youtube_client")) as yt:
            pub = publishing.upload(session, project, settings=ctx.settings, store=ctx.store,
                                    client=yt, privacy=privacy, publish_at=when,
                                    playlist_id=playlist_id or None, actor="dashboard")
    except (publishing.PublishGateError, OAuthError, ValueError) as exc:
        return _back(url, err=str(exc))
    except YouTubeError as exc:
        # Returning normally commits the request transaction, which keeps the failed
        # publication row and the audit event that upload() recorded.
        return _back(url, err=f"YouTube refused the upload: {exc}")
    done = f"Uploaded ({plan}): {video_url(pub.youtube_video_id or '')}"
    if pub.error:
        done += f". Warning: {pub.error['message']}"
    return _back(url, msg=done)


@router.post("/projects/{project_id}/{action}")
def project_action(project_id: str, action: str, ctx: Ctx, session: Db) -> RedirectResponse:
    url = f"/ui/projects/{project_id}"
    project = get_project(session, project_id, for_update=True)
    try:
        if action == "start":
            commands.start(session, project, ctx.settings, actor="dashboard")
        elif action == "cancel":
            commands.cancel(session, project, actor="dashboard")
        elif action == "resume":
            commands.resume_project(session, project, ctx.settings, actor="dashboard")
        elif action == "repair-more":
            commands.repair_more(session, project, ctx.settings, actor="dashboard")
        elif action == "allow-more-renders":
            commands.allow_more_renders(session, project, ctx.settings, actor="dashboard")
        elif action == "recheck-quality":
            commands.recheck_quality(session, project, ctx.settings, actor="dashboard")
        elif action == "keep-renders":
            commands.keep_renders(session, project, ctx.settings, actor="dashboard")
        elif action in ("approve-rights", "reject-rights"):
            commands.decide_rights(session, project, ctx.settings,
                                   approve=action == "approve-rights", decided_by="dashboard",
                                   fields={}, note="decided in dashboard")
        else:
            raise HTTPException(404)
    except ValueError as exc:
        return _back(url, err=str(exc))
    return _back(url, msg={"start": "Started", "cancel": "Cancelled", "resume": "Resumed",
                           "repair-more": "Repairing again",
                           "allow-more-renders": "Allowed more renders; rendering continues",
                           "recheck-quality": "Checking quality again",
                           "keep-renders": "Kept the renders; the video is being edited",
                           "approve-rights": "Rights approved",
                           "reject-rights": "Rights rejected"}[action])


# -- queue / approvals ---------------------------------------------------------------------
@router.get("/queue", response_class=HTMLResponse)
def queue_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    jobs = list(session.scalars(select(Job).order_by(Job.created_at.desc()).limit(200)))
    return _page(request, "queue.html", ctx, jobs=jobs, workers=workers_info(session),
                 gpu=gpu_info(ctx, session))


@router.get("/approvals", response_class=HTMLResponse)
def approvals_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    items = list(session.scalars(select(ApprovalRequest)
                                 .order_by(ApprovalRequest.requested_at.desc()).limit(200)))
    names = dict(session.execute(select(Project.id, Project.name).where(
        Project.id.in_(list({a.project_id for a in items if a.project_id})))).all())
    return _page(request, "approvals.html", ctx, items=items, names=names,
                 thumbs=_thumbs(session, [a.project_id for a in items
                                          if a.project_id and a.kind == "publish"]))


@router.post("/approvals/{approval_id}/{decision}")
def decide_approval(approval_id: str, decision: str, ctx: Ctx, session: Db) -> RedirectResponse:
    req = session.get(ApprovalRequest, approval_id, with_for_update=True)
    if req is None or decision not in ("approve", "reject"):
        raise HTTPException(404)
    if req.kind == "publish" and decision == "approve":
        try:
            with publishing.youtube_client(ctx.settings, ctx.extras.get("youtube_client")) as yt:
                pub = publishing.approve_proposal(session, req, settings=ctx.settings,
                                                  store=ctx.store, client=yt,
                                                  decided_by="dashboard")
        except (publishing.PublishGateError, OAuthError) as exc:
            return _back("/ui/approvals", err=str(exc))
        except YouTubeError as exc:  # returning commits the failed attempt's audit trail
            return _back("/ui/approvals", err=f"YouTube refused the upload: {exc}")
        done = f"Uploaded: {video_url(pub.youtube_video_id or '')}"
        if pub.error:
            done += f". Warning: {pub.error['message']}"
        return _back("/ui/approvals", msg=done)
    try:
        commands.decide_approval(session, req, ctx.settings, approve=decision == "approve",
                                 decided_by="dashboard", note="decided in dashboard")
    except ValueError as exc:
        return _back("/ui/approvals", err=str(exc))
    if req.kind == "repair_budget" and decision == "approve":
        return _back("/ui/approvals", msg="Repairing again")
    return _back("/ui/approvals", msg=f"{'Approved' if decision == 'approve' else 'Rejected'}")


# -- YouTube -------------------------------------------------------------------------------
@router.get("/youtube", response_class=HTMLResponse)
def youtube_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    yt = ctx.settings.youtube
    pubs = session.execute(select(Publication, Project.name)
                           .join(Project, Project.id == Publication.project_id)
                           .order_by(Publication.created_at.desc()).limit(100)).all()
    ready = list(session.scalars(select(Project).where(Project.status == "READY_TO_PUBLISH")
                                 .order_by(Project.updated_at.desc())))
    return _page(request, "youtube.html", ctx, yt=yt,
                 client_file=yt.client_secret_path.is_file(), signed_in=yt.token_path.is_file(),
                 quota=quota_today(session), pubs=pubs, ready=ready,
                 playlists=publishing.load_playlists(ctx.settings),
                 default_playlist=publishing.playlist_title(ctx.settings,
                                                            yt.default_playlist_id),
                 releases=release_overview(ctx, session),
                 proposals=len(publishing.pending_proposals(session)))


@router.post("/youtube/playlists")
def youtube_playlists(ctx: Ctx) -> RedirectResponse:
    try:
        with publishing.youtube_client(ctx.settings, ctx.extras.get("youtube_client"),
                                       uploads=False) as yt:
            data = publishing.refresh_playlists(ctx.settings, yt)
    except (OAuthError, YouTubeError, httpx.HTTPError) as exc:
        return _back("/ui/youtube", err=f"Could not load playlists: {exc}")
    n = len(data["items"])
    return _back("/ui/youtube", msg=f"Loaded {n} playlist{'' if n == 1 else 's'}")


@router.post("/youtube/check")
def youtube_check(ctx: Ctx) -> RedirectResponse:
    from rokkur_studio.youtube.client import YouTubeClient
    from rokkur_studio.youtube.oauth import OAuthClient, TokenStore

    yt = ctx.settings.youtube
    try:
        client: YouTubeClient = ctx.extras.get("youtube_client") or YouTubeClient(  # type: ignore[assignment]
            OAuthClient.load(yt.client_secret_path), TokenStore(yt.token_path))
        channel = client.my_channel()
    except (OAuthError, YouTubeError, httpx.HTTPError) as exc:
        return _back("/ui/youtube", err=f"Sign-in check failed: {exc}")
    return _back("/ui/youtube", msg=f"Signed in as '{channel['title']}' "
                                    f"({channel.get('custom_url') or channel['id']})")


# -- agents --------------------------------------------------------------------------------
@router.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    s = ctx.settings
    installed: list[str] | None = None
    loaded: list[str] | None = None
    try:
        installed = [m["name"] for m in httpx.get(f"{s.ollama.url}/api/tags", timeout=3)
                     .json().get("models", [])]
        loaded = [m["name"] for m in httpx.get(f"{s.ollama.url}/api/ps", timeout=3)
                  .json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        pass
    briefs = session.execute(
        select(Event.project_id, Event.actor, Event.created_at, Project.name)
        .join(Project, Project.id == Event.project_id)
        .where(Event.type == "CREATIVE_BRIEF_CREATED")
        .order_by(Event.id.desc()).limit(15)).all()
    return _page(request, "agents.html", ctx, installed=installed, loaded=loaded, briefs=briefs,
                 check_output=request.query_params.get("out"))


@router.post("/agents/check")
def agents_check(ctx: Ctx, theme: Annotated[str, Form()] = "") -> RedirectResponse:
    from rokkur_studio.cli import run_agent_check

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run_agent_check(ctx.settings,
                               theme=theme or "1970s stop-motion claymation, warm film grain")
    out = buf.getvalue()[-3000:]
    return RedirectResponse(f"/ui/agents?out={quote(out)}&"
                            + ("msg=All+agents+answered" if code == 0
                               else "err=Agent+check+failed%3B+see+the+output"),
                            status_code=303)


# -- director ------------------------------------------------------------------------------
@router.get("/director", response_class=HTMLResponse)
def director_page(request: Request, ctx: Ctx, subject: str = "", theme: str = "",
                  background: str = "", shot_size: str = "", camera_angle: str = "",
                  camera_movement: str = "", lighting: str = "", character_key: str = "",
                  global_look: str = "", run: str = "") -> HTMLResponse:
    tracker, problem = _tracker(ctx)
    look_on = global_look == "on" or not run  # an unticked box is absent from the form
    form = {"subject": subject, "theme": theme, "background": background,
            "shot_size": shot_size, "camera_angle": camera_angle,
            "camera_movement": camera_movement, "lighting": lighting,
            "character_key": character_key, "global_look": look_on}
    result = None
    if run and tracker is not None:
        d = ctx.settings.director
        try:
            result = preview(tracker, theme=theme, subject=subject, background=background,
                             shot_size=shot_size or None, camera_angle=camera_angle or None,
                             camera_movement=camera_movement or None,
                             lighting=lighting or None, character_key=character_key or None,
                             global_look=look_on,
                             weights=Weights(framing=d.framing_weight, angle=d.angle_weight))
        except ValueError as exc:
            result = {"error": str(exc)}
    return _page(request, "director.html", ctx, tracker=tracker, problem=problem,
                 vocabulary=VOCABULARY, form=form, result=result)


def _save(ctx: StudioContext, change: Any) -> RedirectResponse:
    tracker, problem = _tracker(ctx)
    if tracker is None:
        return _back("/ui/director", err=problem)
    try:
        updated = AssetTracker.model_validate(change(tracker.to_json()))
    except ValidationError as exc:
        return _back("/ui/director", err="; ".join(
            str(e["msg"]).removeprefix("Value error, ") for e in exc.errors())[:300])
    save_tracker(tracker_path(ctx.settings.studio.data_dir), updated)
    return _back("/ui/director", msg="Saved")


@router.post("/director/look")
def director_look(ctx: Ctx, prompt_prefix: Annotated[str, Form()] = "",
                  style_modifiers: Annotated[str, Form()] = "",
                  negative_prompt: Annotated[str, Form()] = "") -> RedirectResponse:
    return _save(ctx, lambda t: {**t, "PROMPT_PREFIX": prompt_prefix.strip(),
                                 "GLOBAL_STYLE_MODIFIERS": style_modifiers.strip(),
                                 "GLOBAL_NEGATIVE_PROMPT": negative_prompt.strip()})


@router.post("/director/characters")
def director_character(ctx: Ctx, key: Annotated[str, Form()],
                       description: Annotated[str, Form()]) -> RedirectResponse:
    if not normalise_key(key):
        return _back("/ui/director", err="give the character a name, e.g. NEO")
    return _save(ctx, lambda t: {**t, "CHARACTERS": {**t["CHARACTERS"],
                                                     normalise_key(key): description}})


@router.post("/director/characters/{key}/delete")
def director_character_delete(key: str, ctx: Ctx) -> RedirectResponse:
    return _save(ctx, lambda t: {**t, "CHARACTERS": {
        k: v for k, v in t["CHARACTERS"].items() if k != normalise_key(key)}})


# -- system --------------------------------------------------------------------------------
@router.get("/system", response_class=HTMLResponse)
def system_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    return _page(request, "system.html", ctx, info=system_info(ctx, session),
                 services=services(ctx), gpu=gpu_info(ctx, session),
                 workers=workers_info(session), profiles=ctx.settings.profiles)
