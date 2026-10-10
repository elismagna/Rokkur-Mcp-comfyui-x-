"""Server-rendered control dashboard (Jinja2, no frontend build step).

Every action here goes through the same services as the API and CLI, so the gates (rights,
QC, YouTube private-by-default) apply identically. Messages are passed back to the page as
``?msg=`` / ``?err=`` query parameters after a redirect.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import httpx
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rokkur_studio.agents.providers import AgentOutputError, AgentUnavailable
from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.api.routes_projects import project_detail
from rokkur_studio.api.routes_system import gpu_leases as gpu_info
from rokkur_studio.api.routes_system import system as system_info
from rokkur_studio.api.routes_system import workers as workers_info
from rokkur_studio.api.routes_youtube import release_overview
from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
from rokkur_studio.config import Settings, project_target
from rokkur_studio.db.models import (
    ApprovalRequest,
    Asset,
    CostEntry,
    Event,
    Image,
    Job,
    Project,
    Publication,
    Render,
)
from rokkur_studio.director.assets import (
    AssetTracker,
    TrackerError,
    load_tracker,
    normalise_key,
    save_tracker,
    tracker_path,
)
from rokkur_studio.director.prompt_editor import (
    PromptDraft,
    PromptEditor,
    PromptEditRequest,
)
from rokkur_studio.director.prompts import Weights, preview
from rokkur_studio.director.vocabulary import VOCABULARY
from rokkur_studio.domain.rights import RightsCategory
from rokkur_studio.domain.states import ProjectStatus as S
from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.pipeline.context import StudioContext, profile_availability, supported_controls
from rokkur_studio.pipeline.extend import ExtendRequest, latest_extension, request_extension
from rokkur_studio.pipeline.stages import _subject_masker
from rokkur_studio.pipeline.subject import (
    MaskerUnavailable,
    OnnxSubjectMasker,
    decide_subject,
    resize,
)
from rokkur_studio.services import characters as character_svc
from rokkur_studio.services import commands, publishing, ratings, taste
from rokkur_studio.services import images as image_svc
from rokkur_studio.services.assets import import_file
from rokkur_studio.services.images import SIZE_PRESETS, ImageRequest, ImageStore
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
STATIC_DIR = Path(__file__).parent / "static"
# Changes whenever the stylesheet or script does, so a browser never keeps an old copy.
templates.env.globals["static_version"] = hashlib.sha256(b"".join(
    p.read_bytes() for p in sorted(STATIC_DIR.glob("*")) if p.is_file())).hexdigest()[:10]
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
AUDIO_EXTS = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg",
              ".opus", ".wav", ".weba", ".webm", ".wma"}


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
    base, hash_sign, fragment = url.partition("#")
    sep = "&" if "?" in base else "?"
    if msg:
        base += f"{sep}msg={quote(msg)}"
    elif err:
        base += f"{sep}err={quote(err)}"
    return RedirectResponse(base + hash_sign + fragment, status_code=303)


def _thumbs(session: Session, project_ids: list[str]) -> dict[str, str]:
    """A picture per project: its thumbnail, else the first shot's middle frame."""
    if not project_ids:
        return {}
    rows = session.execute(select(Asset.project_id, Asset.id, Asset.kind)
                           .where(Asset.project_id.in_(project_ids),
                                  Asset.kind.in_(("thumbnail", "keyframe")))
                           .order_by(Asset.created_at)).all()
    out: dict[str, str] = {}
    for pid, aid, kind in rows:
        if kind == "keyframe" and pid in out:
            continue  # the first keyframe stands in until a thumbnail exists
        out[pid] = f"/projects/{pid}/assets/{aid}/file"
    return out


STEP_LABELS = {"rights_check": "Checking rights", "ingest": "Copying the source",
               "analyze": "Analysing the footage", "creative_plan": "Writing the brief",
               "compile_workflow": "Building workflows", "render": "Rendering",
               "qc": "Checking quality", "repair": "Planning repairs",
               "edit": "Editing the final video"}


def running_now(session: Session) -> list[dict[str, Any]]:
    """What the workers are doing right now, in words, newest first."""
    rows = session.execute(select(Job, Project.name).join(Project, Project.id == Job.project_id)
                           .where(Job.status == "RUNNING").order_by(Job.started_at.desc())).all()
    out = []
    for job, name in rows:
        step = STEP_LABELS.get(job.kind, job.kind.replace("_", " ").capitalize())
        if job.kind == "render":
            active = session.scalars(select(Render).where(Render.project_id == job.project_id,
                                                          Render.status == "running")
                                     .order_by(Render.started_at.desc()).limit(1)).one_or_none()
            if active is not None:
                step = f"Rendering shot {active.shot_id.replace('shot_', '')}" + (
                    f", attempt {active.attempt}" if active.attempt > 1 else "")
        out.append({"project_id": job.project_id, "name": name, "step": step,
                    "since": job.started_at.isoformat() if job.started_at else None})
    return out


@router.get("/status")
def live_status(session: Db) -> dict[str, Any]:
    """The rail's live line: cheap database reads only, polled by every page."""
    return {"running": running_now(session),
            "queued": session.scalar(select(func.count(Job.id)).where(Job.status == "QUEUED")) or 0,
            "approvals": session.scalar(select(func.count(ApprovalRequest.id))
                                        .where(ApprovalRequest.status == "pending")) or 0}


def media_files(ctx: StudioContext) -> list[dict[str, Any]]:
    root = ctx.settings.studio.media_dir
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
             and not any(part.startswith(".") for part in p.relative_to(root).parts)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"path": str(p), "name": str(p.relative_to(root)),
             "mb": round(p.stat().st_size / 1e6, 1)} for p in files[:300]]


def _probe(url: str, headers: dict[str, str] | None = None) -> bool:
    try:
        return httpx.get(url, timeout=2, headers=headers).is_success
    except httpx.HTTPError:
        return False


def _cloud_detail(s: Settings) -> str:
    """The cloud server's host and price, never its token or full URL (it may hold a secret)."""
    host = httpx.URL(s.cloud.url).host or "configured"
    price = s.cloud.price_per_hour_usd
    return f"{host} · {s.cloud.vram_gb:g} GB" + (f" · ${price:g}/h" if price else "")


def services(ctx: StudioContext) -> list[dict[str, Any]]:
    s = ctx.settings
    yt = s.youtube
    signed_in = yt.token_path.is_file()
    masker = OnnxSubjectMasker(s.subject, s.studio.data_dir)
    mask_problem = masker.problem()
    return [
        {"name": "ComfyUI", "ok": _probe(f"{s.comfyui.url}/system_stats"),
         "detail": s.comfyui.url, "needed": s.render.renderer == "comfyui"},
        *([{"name": "Cloud ComfyUI", "ok": _probe(f"{s.cloud.url.rstrip('/')}/system_stats",
                                                   s.cloud.headers()),
            "detail": _cloud_detail(s), "needed": False}] if s.cloud.ready else []),
        {"name": "Ollama", "ok": _probe(f"{s.ollama.url}/api/version"),
         "detail": f"{s.ollama.model}", "needed": s.agents.provider == "ollama"},
        {"name": "FFmpeg", "ok": ctx.ffmpeg.available(), "detail": "encode + QC", "needed": True},
        {"name": "Subject masks", "ok": mask_problem is None,
         "detail": mask_problem or (f"{s.subject.model} ready" if masker.path.is_file()
                                    else f"{s.subject.model}, downloads on first use"),
         "needed": False},
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
        .order_by(Project.updated_at.desc()).limit(12)))
    pending_approvals = session.scalar(select(func.count(ApprovalRequest.id))
                                       .where(ApprovalRequest.status == "pending")) or 0
    running = {r["project_id"]: r["step"] for r in running_now(session)}
    profile = taste.build_profile(session)
    return _page(request, "overview.html", ctx, tiles=tiles, active=active, attention=attention,
                 finished=finished, running=running,
                 thumbs=_thumbs(session, [p.id for p in finished + active + attention]),
                 verdicts=ratings.video_verdicts(session, [p.id for p in finished]),
                 unrated=ratings.unrated_count(session), taste=profile,
                 services=services(ctx), gpu=gpu_info(ctx, session),
                 workers=workers_info(session), pending_approvals=pending_approvals)


# -- projects ------------------------------------------------------------------------------
@router.get("/projects", response_class=HTMLResponse)
def projects_page(request: Request, ctx: Ctx, session: Db, group: str = "all",
                  q: str = "", view: str = "grid") -> HTMLResponse:
    stmt = select(Project).order_by(Project.created_at.desc()).limit(300)
    states = GROUPS.get(group)
    if states is not None:
        stmt = stmt.where(Project.status.in_(states))
    if q:
        stmt = stmt.where(Project.name.ilike(f"%{q}%"))
    projects = list(session.scalars(stmt))
    ids = [p.id for p in projects]
    return _page(request, "projects.html", ctx, projects=projects, group=group, q=q,
                 groups=list(GROUPS), thumbs=_thumbs(session, ids),
                 verdicts=ratings.video_verdicts(session, ids),
                 view="list" if view == "list" else "grid")


def _tracker(ctx: StudioContext) -> tuple[AssetTracker | None, str | None]:
    try:
        return load_tracker(tracker_path(ctx.settings.studio.data_dir)), None
    except TrackerError as exc:
        return None, str(exc)


@router.get("/new", response_class=HTMLResponse)
def new_page(request: Request, ctx: Ctx, session: Db, reference_image: str = "",
             source_video: str = "") -> HTMLResponse:
    tracker, _ = _tracker(ctx)
    profile = taste.build_profile(session)
    cloud = ctx.settings.cloud
    reference = None
    if reference_image:  # a picture from the library, handed over as the appearance reference
        picture = session.get(Image, reference_image)
        if picture is not None and picture.rel_path:
            reference = {"id": picture.id, "title": picture.title or picture.prompt or picture.kind,
                         "path": str(_images(ctx).path_for(picture.rel_path))}
    return _page(request, "new.html", ctx, media=media_files(ctx), reference=reference,
                 source_video=source_video,
                 suggestions=taste.suggestions(profile), taste=profile,
                 profile_status=profile_availability(ctx),
                 cloud=cloud if cloud.ready else None,
                 cloud_status=(profile_availability(ctx, "cloud", live=False)
                               if cloud.ready else {}),
                 characters=tracker.characters if tracker else {},
                 profiles=ctx.settings.profiles,
                 default_profile=ctx.settings.render.default_profile,
                 categories=[c.value for c in RightsCategory
                             if c.value not in ("UNKNOWN", "REJECTED", "REFERENCE_ONLY")])


@router.post("/new/characters")
def find_characters_in_clip(ctx: Ctx, session: Db, media_file: Annotated[str, Form()] = "",
                            local_path: Annotated[str, Form()] = "",
                            source_file: Annotated[UploadFile | None, File()] = None
                            ) -> JSONResponse:
    """Cutouts of the subjects seen in a clip, put into the picture library as characters."""
    path = ""
    uploaded: Path | None = None
    if source_file is not None and source_file.filename:
        if Path(source_file.filename).suffix.lower() not in VIDEO_EXTS:
            raise HTTPException(422, f"{source_file.filename} is not a video file")
        uploads = Path(ctx.settings.studio.data_dir) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        uploaded = uploads / f"{uuid.uuid4().hex[:12]}_{Path(source_file.filename).name}"
        with uploaded.open("wb") as fh:
            shutil.copyfileobj(source_file.file, fh)
        path = str(uploaded)
    elif media_file:
        if media_file not in {m["path"] for m in media_files(ctx)}:
            raise HTTPException(422, "that file is not in the media folder")
        path = media_file
    elif local_path:
        path = local_path
    if not path or not Path(path).is_file():
        raise HTTPException(422, "Pick or upload a video first.")
    work = Path(ctx.settings.studio.data_dir) / "uploads" / f"characters_{uuid.uuid4().hex[:8]}"
    try:
        found = character_svc.find_characters(ctx.ffmpeg, _subject_masker(ctx), Path(path), work,
                                              settings=ctx.settings.subject)
    except MaskerUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except FFmpegError as exc:
        raise HTTPException(422, f"The studio could not read that video: {exc.summary}") from exc
    clip = Path(path).stem
    store = _images(ctx)
    out = []
    for i, c in enumerate(found, 1):
        row = image_svc.import_image(session, store, ctx.ffmpeg, Path(c["path"]), kind="character",
                                     title=f"Subject {i} from {clip}",
                                     request={"clip": clip, "time": c["time"], "share": c["share"],
                                              "seen": c["seen"], "uploaded_source": str(uploaded or "")})
        crop = store.dir(row.id) / "crop.png"
        shutil.copy2(c["crop"], crop)
        out.append({"id": row.id, "title": row.title, "url": f"/images/{row.id}/file",
                    "crop_url": f"/ui/images/{row.id}/crop",
                    "path": str(store.path_for(row.rel_path or "")),
                    "time": c["time"], "share": c["share"], "seen": c["seen"],
                    "description": character_svc.describe(f"Subject {i}", clip, c["share"]),
                    "uploaded": str(uploaded) if uploaded else ""})
    shutil.rmtree(work, ignore_errors=True)
    return JSONResponse({"characters": out, "clip": clip,
                         "note": ("The subject model sees one salient subject per frame; the "
                                  "pictures show the clearest view of each distinct subject.")})


@router.get("/images/{image_id}/crop")
def image_crop(image_id: str, ctx: Ctx, session: Db) -> FileResponse:
    image = _image_or_404(session, image_id)
    crop = _images(ctx).dir(image.id) / "crop.png"
    if not crop.is_file():
        if not image.rel_path:
            raise HTTPException(404)
        return FileResponse(_images(ctx).path_for(image.rel_path), media_type="image/png")
    return FileResponse(crop, media_type="image/png")


@router.post("/projects/{project_id}/storyboard")
def storyboard_from_ui(project_id: str, ctx: Ctx, session: Db) -> RedirectResponse:
    """One still per shot from the brief's prompts, through the picture pipeline, so the look
    can be judged before a long video render (or compared with one after)."""
    try:
        project = get_project(session, project_id)
    except LookupError as exc:
        raise HTTPException(404) from exc
    brief = latest_document(session, project_id, "creative_brief")
    if brief is None:
        return _back(f"/ui/projects/{project_id}", err="No creative brief yet: the stills come "
                     "from its shot prompts.")
    size = "short" if project.target_format == "youtube_short" else "widescreen"
    made = 0
    try:
        for shot in brief.data.get("shot_plan", []):
            prompt = (shot.get("prompt") or brief.data.get("style_prompt")
                      or project.creative_input.get("theme", ""))
            if not prompt:
                continue
            image_svc.request_images(session, ctx.settings, _images(ctx), ctx.ffmpeg, ImageRequest(
                operation="generate", prompt=prompt, size=size, count=1, kind="storyboard",
                project_id=project_id, shot_id=shot.get("shot_id"),
                title=f"{project.name} · {str(shot.get('shot_id', '')).replace('shot_', 'shot ')}",
                render_on=project_target(project.creative_input)), actor="dashboard")
            made += 1
    except (ValueError, LookupError) as exc:
        return _back(f"/ui/projects/{project_id}", err=str(exc))
    return _back(f"/ui/projects/{project_id}#storyboard",
                 msg=f"{made} storyboard still{'s' if made != 1 else ''} queued")


def _storyboard(session: Session, project_id: str) -> list[Image]:
    latest: dict[str, Image] = {}
    for row in session.scalars(select(Image).where(Image.project_id == project_id,
                                                   Image.kind == "storyboard")
                               .order_by(Image.created_at)):
        latest[row.shot_id or row.id] = row
    return list(latest.values())


@router.get("/subject-decision")
def subject_decision(theme: str = "", prompt: str = "", subject: str = "auto",
                     character_key: str = "", character_description: str = "",
                     reference: bool = False) -> dict[str, str]:
    """What the studio would do with the main subject, for the hint on the New video form."""
    return decide_subject({"theme": theme, "prompt": prompt, "subject": subject,
                           "character_key": character_key,
                           "character_description": character_description,
                           "character_reference_path": "form" if reference else ""}).to_dict()


@router.post("/prompt-enhance")
def prompt_enhance(body: PromptEditRequest, ctx: Ctx) -> PromptDraft:
    """Return a local-model suggestion; the caller must explicitly apply it to the form."""
    if ctx.settings.agents.provider != "ollama":
        raise HTTPException(
            503, "Local Ollama is not enabled. Your prompt draft is unchanged.")
    lease = ctx.gpu.try_acquire(f"prompt-editor-{uuid.uuid4().hex[:10]}", "GPU_HEAVY")
    if lease is None:
        raise HTTPException(
            409, "A render is using the GPU. Your draft is unchanged; try again after it finishes.")
    try:
        return PromptEditor(ctx.provider).polish(body)
    except AgentUnavailable as exc:
        raise HTTPException(
            503, "The local prompt model is unavailable. Your draft is unchanged; check Ollama "
                 "and try again.") from exc
    except AgentOutputError as exc:
        raise HTTPException(
            502, "The model could not return a usable draft. Your prompts are unchanged; "
                 "try again or edit them manually.") from exc
    finally:
        ctx.gpu.release(lease.id)


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
                     audio_bed_file: Annotated[UploadFile | None, File()] = None,
                     reference_mode: Annotated[str, Form()] = "auto",
                     mute_source_audio: Annotated[bool, Form()] = False,
                     audio_bed_gain: Annotated[float, Form()] = 0.25,
                     audio_bed_rights_confirmed: Annotated[bool, Form()] = False,
                     audio_bed_rights_evidence: Annotated[str, Form()] = "",
                     subject: Annotated[str, Form()] = "auto",
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
                     stable: Annotated[bool, Form()] = False,
                     render_profile: Annotated[str, Form()] = "",
                     render_on: Annotated[str, Form()] = "",
                     target_format: Annotated[str, Form()] = "youtube_short",
                     autostart: Annotated[bool, Form()] = False) -> RedirectResponse:
    path = ""
    try:
        selected = render_profile or ctx.settings.render.default_profile
        ctx.settings.profile(selected)
        target = ctx.settings.new_project_target(render_on or None)
        if problem := profile_availability(ctx, target)[selected]:
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
    audio_bed_path: str | None = None
    if audio_bed_file is not None and audio_bed_file.filename:
        audio_suffix = Path(audio_bed_file.filename).suffix.lower()
        if audio_suffix not in AUDIO_EXTS:
            return fail("Use a supported audio file such as MP3, WAV, M4A, FLAC or OGG.")
        audio_bed_path = save_upload(audio_bed_file, f"soundtrack{audio_suffix}")
        try:
            if not ctx.ffmpeg.has_audio(Path(audio_bed_path)):
                return fail("The selected soundtrack file does not contain an audio stream.")
        except FFmpegError:
            return fail("The studio could not read that soundtrack file. Choose another audio file.")
    try:
        body = ProjectCreate(
            name=name or Path(path).stem, target_format=target_format,  # type: ignore[arg-type]
            render_profile=render_profile or None,
            source=SourceIn(platform="local", local_path=path),
            rights=RightsIn(category=RightsCategory(rights_category),
                            permission_evidence=permission_evidence or None),
            creative=CreativeIn(theme=theme.strip(), prompt=prompt or None,
                                reference_mode=reference_mode,  # type: ignore[arg-type]
                                subject=subject,  # type: ignore[arg-type]
                                control_strength=control_strength, cfg=cfg,
                                seed=int(seed) if seed.strip() else None,
                                steps=int(steps) if steps.strip() else None,
                                negative_prompt=negative_prompt or None,
                                keep_source_audio=not mute_source_audio,
                                audio_bed_path=audio_bed_path,
                                audio_bed_gain=audio_bed_gain,
                                audio_bed_rights_confirmed=audio_bed_rights_confirmed,
                                audio_bed_rights_evidence=audio_bed_rights_evidence or None,
                                character_key=character_key or None,
                                character_description=character_description or None,
                                use_global_look=use_global_look, stable=stable,
                                render_on=target,
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


def _shot_reviews(session: Session, project: Project, detail: Any,
                  keyframes: dict[Any, str]) -> list[dict[str, Any]]:
    """Everything the review room shows per shot: the plan, every attempt, the latest render
    with its QC result and your rating of it."""
    docs = detail.documents
    planned = {s["shot_id"]: s for s in (docs.get("creative_brief") or {}).get("data", {})
               .get("shot_plan", [])}
    manifest_shots = (docs.get("manifest") or {}).get("data", {}).get("shots", [])
    timing = {s["shot_id"]: (s["start"], s["end"]) for s in manifest_shots}
    qc = {r.get("render_id"): r for r in (docs.get("qc_report") or {}).get("data", {})
          .get("shots", [])}
    rows = list(session.scalars(select(Render).where(Render.project_id == project.id)
                                .order_by(Render.attempt)))
    by_asset = {r.output_asset_id: r for r in rows if r.output_asset_id}
    verdicts = {r.render_id: r for r in ratings.project_ratings(session, project.id)
                if r.render_id}
    order = list(timing) or list(planned)
    for r in rows:
        if r.shot_id not in order:
            order.append(r.shot_id)
    out = []
    for sid in order:
        attempts = [a for a in detail.assets if a.kind in ("render", "render_raw")
                    and a.meta.get("shot_id") == sid]
        finals = [a for a in attempts if a.kind == "render"]
        latest = max(finals, key=lambda a: a.meta.get("attempt", 0), default=None)
        render = by_asset.get(latest.id) if latest else None
        running = next((r for r in rows if r.shot_id == sid and r.status == "running"), None)
        start, end = timing.get(sid, (planned.get(sid, {}).get("start", 0),
                                      planned.get(sid, {}).get("end", 0)))
        rating = verdicts.get(render.id) if render else None
        earlier = None
        if render and rating is None:  # a disliked attempt that was redone stays visible
            earlier = next((verdicts[r.id] for r in reversed(rows) if r.shot_id == sid
                            and r.id in verdicts), None)
        details = (render.params or {}).get("_details", {}) if render else {}
        out.append({
            "id": sid, "label": sid.replace("shot_", "Shot "), "number": sid.replace("shot_", ""),
            "start": start, "end": end, "plan": planned.get(sid, {}),
            "keyframe": keyframes.get(sid), "asset": latest, "render": render,
            "attempts": [{"asset": a, "render_id": by_asset[a.id].id if a.id in by_asset else None}
                         for a in sorted(attempts, key=lambda a: (a.meta.get("attempt", 0),
                                                                   a.kind))],
            "qc": qc.get(render.id) if render else None, "rating": rating, "earlier": earlier,
            "subject": details.get("subject"), "running": running,
        })
    return out


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
    manifest = detail.documents.get("manifest")
    if manifest is not None:  # manifests from before subject handling have none: they restyled
        subject = manifest["data"].get("subject")
    else:
        subject = {**decide_subject(project.creative_input).to_dict(), "planned": True}
    shots = _shot_reviews(session, project, detail, keyframes)
    video_asset = ratings.latest_video_asset(session, project_id)
    verdicts = ratings.project_ratings(session, project_id)
    video_rating = next((r for r in reversed(verdicts) if r.target == ratings.VIDEO
                         and video_asset is not None and r.asset_id == video_asset.id), None)
    earlier_video_rating = None if video_rating else next(
        (r for r in reversed(verdicts) if r.target == ratings.VIDEO), None)
    source = next((a for a in reversed(detail.assets) if a.kind == "source"), None)
    repair_limit = at_repair_limit(project)
    usage = budget_usage(session, project_id, ctx.settings)
    return _page(request, "project.html", ctx, d=detail, p=project, events=events,
                 shots=shots, subject=subject, keyframes=keyframes, tags=ratings.TAGS,
                 storyboard=_storyboard(session, project_id),
                 adjustable=project.status in commands.ADJUSTABLE,
                 extendable=(project.status in (S.READY_TO_PUBLISH, S.PUBLISHED, S.MONITORING)
                             and ctx.settings.render.renderer == "comfyui"),
                 extension=latest_extension(session, project),
                 extend_jobs=list(session.scalars(select(Job).where(
                     Job.project_id == project_id, Job.kind == "extend",
                     Job.status.in_(("QUEUED", "RUNNING", "RETRY_WAIT"))))),
                 rendered_shot_ids=sorted(commands.rendered_shots(session, project_id)),
                 library_pictures=[i for i in image_svc.list_images(session, limit=40)
                                   if i.status == "done" and i.rel_path],
                 can_storyboard=detail.documents.get("creative_brief") is not None
                 and not image_availability(ctx).get(ctx.settings.images.default_profile),
                 video_asset=video_asset, video_rating=video_rating,
                 earlier_video_rating=earlier_video_rating,
                 source_url=(f"/projects/{project_id}/assets/{source.id}/file" if source else None),
                 can_redo=project.status == S.READY_TO_PUBLISH or repair_limit,
                 gpu_minutes=usage["gpu"], cloud_usage=usage,
                 render_on=project_target(project.creative_input),
                 failure=failure_reason(session, project) if failed else None,
                 repair_limit=repair_limit,
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
    data = {**current.data, "text_by": "you", "title": title.strip(),
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


@router.post("/projects/{project_id}/rate")
async def rate_from_ui(project_id: str, request: Request, session: Db) -> Any:
    """Rate a shot attempt or the video. Answers JSON to the page's script, else redirects."""
    url = f"/ui/projects/{project_id}"
    wants_json = "application/json" in request.headers.get("accept", "")
    form = await request.form()
    try:
        body = ratings.RatingIn(target=str(form.get("target", "")),
                                value=int(str(form.get("value", "0"))),  # type: ignore[arg-type]
                                tags=[str(t) for t in form.getlist("tags")],
                                note=str(form.get("note") or "") or None,
                                render_id=str(form.get("render_id") or "") or None)
        project = get_project(session, project_id)
        rating = ratings.rate(session, project, body, actor="dashboard")
    except (ValidationError, ValueError, LookupError) as exc:
        message = (exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc))
        if wants_json:
            return JSONResponse({"error": message}, status_code=422)
        return _back(url, err=message)
    label = ratings.VALUES.get(rating.value, "") if rating else "Rating removed"
    if wants_json:
        return {"value": rating.value if rating else 0, "label": label,
                "tags": rating.tags if rating else [], "note": rating.note if rating else None}
    return _back(url, msg=f"{label} saved" if rating else label)


@router.post("/projects/{project_id}/redo")
def redo_from_ui(project_id: str, ctx: Ctx, session: Db,
                 shots: Annotated[list[str] | None, Form()] = None) -> RedirectResponse:
    url = f"/ui/projects/{project_id}"
    project = get_project(session, project_id, for_update=True)
    try:
        commands.redo_shots(session, project, ctx.settings, shot_ids=shots or [],
                            actor="dashboard", supported=supported_controls(ctx, project.render_profile))
    except ValueError as exc:
        return _back(url, err=str(exc))
    names = ", ".join(s.replace("shot_", "") for s in shots or [])
    return _back(url, msg=f"Redoing shot{'s' if len(shots or []) > 1 else ''} {names}; "
                          "the other shots stay as they are")


def _reference_from_library(ctx: StudioContext, session: Session, project: Project,
                            image_id: str) -> str:
    """Copy a library picture into the project's references; returns its store path."""
    picture = _image_or_404(session, image_id)
    if not picture.rel_path:
        raise ValueError("that picture has no file yet")
    asset = import_file(session, ctx.store, project.id, "reference", "references",
                        _images(ctx).path_for(picture.rel_path),
                        name=f"reference_{picture.id}.png", meta={"image_id": picture.id})
    return asset.rel_path


@router.post("/projects/{project_id}/extend")
def extend_from_ui(project_id: str, ctx: Ctx, session: Db,
                   seconds: Annotated[float, Form()] = 3.0,
                   prompt: Annotated[str, Form()] = "",
                   reference_image_id: Annotated[str, Form()] = "") -> RedirectResponse:
    """Continue the finished video past its last frame (docs/video-tools.md)."""
    reference = None
    if reference_image_id:
        picture = _image_or_404(session, reference_image_id)
        reference = str(_images(ctx).path_for(picture.rel_path)) if picture.rel_path else None
    try:
        job = request_extension(session, ctx.settings, ExtendRequest(
            project_id=project_id, seconds=seconds, prompt=prompt,
            reference_image_path=reference), actor="dashboard")
    except (ValueError, LookupError) as exc:
        return _back(f"/ui/projects/{project_id}#extend", err=str(exc))
    return _back(f"/ui/projects/{project_id}#extend",
                 msg=f"Extension of {seconds:g} s queued (job {job.id[-8:]}); the longer cut "
                     "appears as the latest final video when it is done")


@router.post("/new/extend")
def extend_clip_from_ui(ctx: Ctx, session: Db, media_file: Annotated[str, Form()],
                        prompt: Annotated[str, Form()],
                        seconds: Annotated[float, Form()] = 3.0,
                        rights_confirmed: Annotated[bool, Form()] = False,
                        rights_evidence: Annotated[str, Form()] = "") -> RedirectResponse:
    """Extend a clip in the media folder before making a video from it."""
    if media_file not in {m["path"] for m in media_files(ctx)}:
        return _back("/ui/new", err="that file is not in the media folder")
    try:
        job = request_extension(session, ctx.settings, ExtendRequest(
            source_path=media_file, seconds=seconds, prompt=prompt,
            rights_confirmed=rights_confirmed, rights_evidence=rights_evidence), actor="dashboard")
    except (ValueError, LookupError) as exc:
        return _back("/ui/new", err=str(exc))
    return _back("/ui/new", msg=f"Extending {Path(media_file).name} by {seconds:g} s (job "
                               f"{job.id[-8:]}); the longer clip appears in the media folder "
                               "list when it is done")


@router.post("/projects/{project_id}/adjust")
def adjust_from_ui(project_id: str, ctx: Ctx, session: Db,
                   prompt_extra: Annotated[str, Form()] = "",
                   seed: Annotated[str, Form()] = "", steps: Annotated[str, Form()] = "",
                   cfg: Annotated[str, Form()] = "",
                   control_strength: Annotated[str, Form()] = "",
                   canny_low: Annotated[str, Form()] = "", canny_high: Annotated[str, Form()] = "",
                   reference_image_id: Annotated[str, Form()] = "",
                   clear_reference: Annotated[bool, Form()] = False) -> RedirectResponse:
    try:
        project = get_project(session, project_id, for_update=True)
    except LookupError as exc:
        raise HTTPException(404) from exc
    back = f"/ui/projects/{project_id}#adjust"
    try:
        changes = {"prompt_extra": prompt_extra.strip() or None,
                   "seed": int(seed) if seed.strip() else None,
                   "steps": int(steps) if steps.strip() else None,
                   "cfg": float(cfg) if cfg.strip() else None,
                   "control_strength": float(control_strength) if control_strength.strip() else None,
                   "canny_low": float(canny_low) if canny_low.strip() else None,
                   "canny_high": float(canny_high) if canny_high.strip() else None}
        reference = (_reference_from_library(ctx, session, project, reference_image_id)
                     if reference_image_id and not clear_reference else None)
        result = commands.adjust_remaining_shots(session, project, changes=changes,
                                                 actor="dashboard", reference_image=reference,
                                                 clear_reference=clear_reference)
    except (ValueError, LookupError) as exc:
        return _back(back, err=str(exc))
    n = len(result["shots"])
    return _back(back, msg=f"Applied to the {n} shot{'s' if n != 1 else ''} still to render; "
                           f"{len(result['rendered_untouched'])} finished shot(s) untouched")


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


# -- taste ---------------------------------------------------------------------------------
# -- pictures ---------------------------------------------------------------------------------

def _images(ctx: StudioContext) -> ImageStore:
    return ImageStore(ctx.settings.studio.data_dir)


def _image_or_404(session: Session, image_id: str) -> Image:
    try:
        return image_svc.get_image(session, image_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


def image_availability(ctx: StudioContext, target: str = "local") -> dict[str, str | None]:
    """Why each image profile cannot run right now, by its first operation; None when it can."""
    out = {}
    for name, profile in ctx.settings.image_profiles.items():
        first = "generate" if "generate" in profile.workflows else next(iter(profile.workflows), "generate")
        out[name] = ctx.settings.image_profile_problem(name, first, target=target)
    return out


PAGE_SIZE = 60


@router.get("/images", response_class=HTMLResponse)
def images_page(request: Request, ctx: Ctx, session: Db, image: str = "", kind: str = "",
                page: int = 1) -> HTMLResponse:
    page = max(1, page)
    gallery = image_svc.list_images(session, limit=PAGE_SIZE + 1, offset=(page - 1) * PAGE_SIZE,
                                    kind=kind or None)
    more = len(gallery) > PAGE_SIZE
    gallery = gallery[:PAGE_SIZE]
    selected = session.get(Image, image) if image else None
    library = [i for i in image_svc.list_images(session, limit=200) if i.status == "done" and i.rel_path]
    status = image_availability(ctx)
    cloud = ctx.settings.cloud
    projects = list(session.scalars(select(Project).where(Project.status.in_(
        [S.READY_TO_PUBLISH.value, S.PUBLISHED.value, S.MONITORING.value, S.QUALITY_PASSED.value]))
        .order_by(Project.updated_at.desc()).limit(50)))
    default = ctx.settings.images.default_profile
    unavailable = status.get(default) or next((r for r in status.values() if r), None)
    form = {"operation": request.query_params.get("op", "generate"),
            "prompt": request.query_params.get("prompt", ""),
            "source_id": request.query_params.get("source", "")}
    return _page(request, "images.html", ctx, gallery=gallery, more=more, page=page, kind=kind,
                 selected=selected, library=library, image_profiles=ctx.settings.image_profiles,
                 image_status=status, available=not status.get(default),
                 unavailable_reason=unavailable, sizes=SIZE_PRESETS,
                 cloud=cloud if cloud.ready else None, projects=projects, form=form,
                 busy=any(i.status in ("queued", "running") for i in gallery),
                 parent=(session.get(Image, selected.parent_id)
                         if selected and selected.parent_id else None),
                 children=(list(session.scalars(select(Image).where(Image.parent_id == selected.id)
                                                .order_by(Image.created_at)))
                           if selected else []))


@router.post("/images")
def create_images_from_form(ctx: Ctx, session: Db, operation: Annotated[str, Form()] = "generate",
                            prompt: Annotated[str, Form()] = "",
                            profile: Annotated[str, Form()] = "",
                            size: Annotated[str, Form()] = "square",
                            width: Annotated[str, Form()] = "", height: Annotated[str, Form()] = "",
                            count: Annotated[int, Form()] = 1,
                            steps: Annotated[str, Form()] = "", cfg: Annotated[str, Form()] = "",
                            seed: Annotated[str, Form()] = "",
                            source_id: Annotated[str, Form()] = "",
                            source_file: Annotated[UploadFile | None, File()] = None,
                            rights_confirmed: Annotated[bool, Form()] = False,
                            rights_evidence: Annotated[str, Form()] = "",
                            mask_data: Annotated[str, Form()] = "",
                            mask_grow: Annotated[int, Form()] = 8,
                            pad_left: Annotated[int, Form()] = 0, pad_top: Annotated[int, Form()] = 0,
                            pad_right: Annotated[int, Form()] = 0,
                            pad_bottom: Annotated[int, Form()] = 0,
                            feather: Annotated[int, Form()] = 24,
                            scale: Annotated[int, Form()] = 2,
                            render_on: Annotated[str, Form()] = "",
                            title: Annotated[str, Form()] = "") -> RedirectResponse:
    uploads = Path(ctx.settings.studio.data_dir) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def fail(message: str) -> RedirectResponse:
        for f in written:
            f.unlink(missing_ok=True)
        return _back("/ui/images", err=message)

    source_path = None
    if source_file is not None and source_file.filename:
        suffix = Path(source_file.filename).suffix.lower()
        dest = uploads / f"{uuid.uuid4().hex[:12]}{suffix}"
        with dest.open("wb") as fh:
            shutil.copyfileobj(source_file.file, fh)
        written.append(dest)
        source_path, source_id = str(dest), ""
    mask_path = None
    if mask_data.startswith("data:image/png;base64,"):
        mask = uploads / f"{uuid.uuid4().hex[:12]}_mask.png"
        try:
            mask.write_bytes(base64.b64decode(mask_data.split(",", 1)[1]))
        except ValueError:
            return fail("The painted mask could not be read; paint it again.")
        written.append(mask)
        mask_path = str(mask)
    custom = size == "custom" and width.strip() and height.strip()
    try:
        body = ImageRequest(
            operation=operation,  # type: ignore[arg-type]
            prompt=prompt, profile=profile or None,
            size=None if custom else (size or None),
            width=int(width) if custom else None, height=int(height) if custom else None,
            count=count, steps=int(steps) if steps.strip() else None,
            cfg=float(cfg) if cfg.strip() else None, seed=int(seed) if seed.strip() else None,
            source_id=source_id or None, source_path=source_path, mask_path=mask_path,
            mask_grow=mask_grow, pad={"left": pad_left, "top": pad_top, "right": pad_right,
                                      "bottom": pad_bottom},
            feather=feather, scale=scale,  # type: ignore[arg-type]
            render_on=render_on or None,  # type: ignore[arg-type]
            title=title, rights_confirmed=rights_confirmed, rights_evidence=rights_evidence)
        rows = image_svc.request_images(session, ctx.settings, _images(ctx), ctx.ffmpeg, body,
                                        actor="dashboard")
    except (ValueError, LookupError, ValidationError) as exc:
        return fail(str(exc).splitlines()[0] if isinstance(exc, ValidationError) else str(exc))
    for f in written:  # the request copied what it needs
        f.unlink(missing_ok=True)
    n = len(rows)
    return _back(f"/ui/images?image={rows[0].id}#selected",
                 msg=f"{n} picture{'s' if n > 1 else ''} queued; the worker renders "
                     f"{'them' if n > 1 else 'it'} now")


@router.post("/images/{image_id}/verdict")
def image_verdict(image_id: str, request: Request, session: Db,
                  value: Annotated[int, Form()] = 0) -> Response:
    image = _image_or_404(session, image_id)
    new_value = None if value == 0 or image.verdict == value else value
    try:
        image_svc.set_verdict(session, image, new_value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"verdict": image.verdict})
    return _back(f"/ui/images?image={image_id}#selected")


@router.post("/images/{image_id}/delete")
def image_delete(image_id: str, ctx: Ctx, session: Db) -> RedirectResponse:
    image_svc.delete_image(session, _images(ctx), _image_or_404(session, image_id))
    return _back("/ui/images", msg="Picture deleted")


@router.post("/images/{image_id}/thumbnail")
def image_as_thumbnail(image_id: str, ctx: Ctx, session: Db,
                       project_id: Annotated[str, Form()]) -> RedirectResponse:
    image = _image_or_404(session, image_id)
    if not image.rel_path:
        return _back(f"/ui/images?image={image_id}", err="This picture has no file yet.")
    try:
        project = get_project(session, project_id)
    except LookupError:
        return _back(f"/ui/images?image={image_id}", err="Pick a video.")
    import_file(session, ctx.store, project.id, "thumbnail", "thumbnails",
                _images(ctx).path_for(image.rel_path), name=f"thumb_{image.id}.png",
                meta={"image_id": image.id, "source": "pictures"})
    return _back(f"/ui/projects/{project.id}#publish",
                 msg="Thumbnail set from the picture; the next upload uses it")


@router.get("/images/{image_id}/automask")
def image_automask(image_id: str, ctx: Ctx, session: Db, invert: bool = False) -> FileResponse:
    """The main subject of a picture (white) on black, drawn by the subject model."""
    image = _image_or_404(session, image_id)
    if not image.rel_path:
        raise HTTPException(404, "this picture has no file yet")
    store = _images(ctx)
    path = store.path_for(image.rel_path)
    masker = _subject_masker(ctx)
    try:
        width, height = ctx.ffmpeg.image_size(path)
        frames = ctx.ffmpeg.read_rgb_frames(path, masker.size, masker.size)
        probs = masker.predict(frames[:1])[0]
    except MaskerUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except FFmpegError as exc:
        raise HTTPException(422, f"could not read the picture: {exc.summary}") from exc
    chosen = resize(probs, height, width) > 0.5
    if invert:
        chosen = ~chosen
    frame = np.repeat((chosen.astype(np.uint8) * 255)[..., None], 3, axis=2)
    out = store.dir(image.id) / ("automask_inverted.png" if invert else "automask.png")
    ctx.ffmpeg.write_image(frame, out)
    return FileResponse(out, media_type="image/png")


@router.get("/taste", response_class=HTMLResponse)
def taste_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    profile = taste.build_profile(session)
    return _page(request, "taste.html", ctx, profile=profile,
                 suggestions=taste.suggestions(profile), report=taste.report(profile),
                 queue=ratings.unrated_renders(session, limit=12),
                 unrated=ratings.unrated_count(session), tags=ratings.TAGS,
                 kinds=taste.KINDS, min_projects=taste.MIN_PROJECTS)


# -- queue / approvals ---------------------------------------------------------------------
@router.get("/queue", response_class=HTMLResponse)
def queue_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    jobs = list(session.scalars(select(Job).order_by(Job.created_at.desc()).limit(200)))
    names = dict(session.execute(select(Project.id, Project.name).where(
        Project.id.in_(list({j.project_id for j in jobs if j.project_id})))).all())
    return _page(request, "queue.html", ctx, jobs=jobs, names=names, workers=workers_info(session),
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
