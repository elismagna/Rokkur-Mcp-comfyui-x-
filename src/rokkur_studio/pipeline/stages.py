"""Stage handlers. Each runs one pipeline stage for one project and is safe to re-run:
a handler resumed after a crash continues from what is already recorded in the database."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rokkur_studio.agents.roles import ChannelManager, RepairPlanner
from rokkur_studio.agents.schemas import CreativeBrief
from rokkur_studio.comfyui.client import ComfyError
from rokkur_studio.comfyui.compiler import (
    TemplateError,
    compile_workflow,
    validate_against_object_info,
)
from rokkur_studio.config import RenderProfile, project_target
from rokkur_studio.db.models import (
    ApprovalRequest,
    Asset,
    CostEntry,
    Document,
    Event,
    Job,
    Project,
    Render,
    RightsDecision,
    utcnow,
)
from rokkur_studio.director.assets import TrackerError, load_tracker, tracker_path
from rokkur_studio.director.passes import direct
from rokkur_studio.director.prompts import build_schedule
from rokkur_studio.domain.rights import RightsCategory, RightsStatus, evaluate
from rokkur_studio.domain.states import ProjectStatus
from rokkur_studio.gpu.lease import GpuUnavailable
from rokkur_studio.jobs.errors import JobCancelled, JobError, PermanentJobError
from rokkur_studio.jobs.queue import is_cancelled
from rokkur_studio.manifest.builder import build_manifest, shot_params, shot_seed
from rokkur_studio.manifest.schema import ReconstructionManifest, ShotSpec, SubjectSpec
from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.pipeline import qc as qc_mod
from rokkur_studio.pipeline.analysis import analyze_video
from rokkur_studio.pipeline.audio import audio_job
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.pipeline.driver import autonomy_level
from rokkur_studio.pipeline.extend import extend_job
from rokkur_studio.pipeline.images import image_job
from rokkur_studio.pipeline.models3d import mesh_job
from rokkur_studio.pipeline.rea import rea_job
from rokkur_studio.pipeline.renderers import (
    ComfyUIRenderer,
    FFmpegPreviewRenderer,
    Renderer,
    RenderOOM,
    RenderOutcome,
    RenderRejected,
    RenderUnavailable,
)
from rokkur_studio.pipeline.subject import (
    MaskerUnavailable,
    OnnxSubjectMasker,
    SubjectMasker,
    cutout_reference,
    decide_subject,
    keep_subject,
    shot_masks,
    subject_share,
    unusable,
    vace_mask_video,
)
from rokkur_studio.services import publishing, ratings
from rokkur_studio.services.assets import import_file, register_asset
from rokkur_studio.services.events import EventType, record_event
from rokkur_studio.services.projects import (
    budget_usage,
    get_project,
    latest_document,
    latest_rights,
    repair_budget,
    save_document,
    transition,
)

log = logging.getLogger(__name__)
S = ProjectStatus

Handler = Callable[[StudioContext, Job], dict[str, Any]]


def _enter(ctx: StudioContext, job: Job, trigger: set[ProjectStatus],
           working: ProjectStatus | None) -> Project:
    """Move the project into the stage's working state (idempotent on resume)."""
    with ctx.db.transaction() as s:
        project = get_project(s, job.project_id or "", for_update=True)
        status = S(project.status)
        if working is not None and status == working:
            return project
        if status not in trigger:
            raise PermanentJobError("wrong_state", f"{job.kind} cannot run in state {status}",
                                    {"state": status.value})
        if working is not None:
            transition(s, project, working, actor=job.kind, job_id=job.id)
        return project


def _finish(ctx: StudioContext, job: Job, target: ProjectStatus,
            data: dict[str, Any] | None = None) -> None:
    with ctx.db.transaction() as s:
        project = get_project(s, job.project_id or "", for_update=True)
        transition(s, project, target, actor=job.kind, job_id=job.id, data=data)


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def _manifest(ctx: StudioContext, project_id: str) -> tuple[ReconstructionManifest, int]:
    with ctx.db.session() as s:
        doc = latest_document(s, project_id, "manifest")
        if doc is None:
            raise PermanentJobError("no_manifest", "project has no manifest")
        return ReconstructionManifest.model_validate(doc.data), doc.version


# -- rights -------------------------------------------------------------------------------
def rights_check(ctx: StudioContext, job: Job) -> dict[str, Any]:
    with ctx.db.transaction() as s:
        project = get_project(s, job.project_id or "", for_update=True)
        if S(project.status) is not S.RIGHTS_PENDING:
            raise PermanentJobError("wrong_state", f"rights_check in {project.status}")
        current = latest_rights(s, project.id)
        category = RightsCategory(current.category) if current else RightsCategory.UNKNOWN
        status, reason = evaluate(
            category, has_evidence=bool(current and current.permission_evidence),
            block_unknown=ctx.settings.rights.block_unknown,
            commercial_use=current.commercial_use if current else None)
        decision = RightsDecision(
            project_id=project.id, category=category.value, status=status.value, reason=reason,
            license=current.license if current else None,
            owner=current.owner if current else None,
            permission_evidence=current.permission_evidence if current else None,
            attribution_required=current.attribution_required if current else False,
            attribution_text=current.attribution_text if current else None,
            allowed_transformations=current.allowed_transformations if current else [],
            commercial_use=current.commercial_use if current else None,
            decided_by="rights_gate")
        s.add(decision)
        s.flush()
        if status is RightsStatus.APPROVED:
            transition(s, project, S.RIGHTS_OK, actor="rights_gate", reason=reason, job_id=job.id)
        elif status is RightsStatus.REJECTED:
            transition(s, project, S.RIGHTS_REJECTED, actor="rights_gate", reason=reason,
                       job_id=job.id)
        else:
            s.add(ApprovalRequest(project_id=project.id, kind="rights_ambiguity",
                                  summary=reason, requested_by="rights_gate",
                                  payload={"category": category.value}))
            record_event(s, EventType.RIGHTS_NEEDS_HUMAN, project_id=project.id,
                         actor="rights_gate", job_id=job.id, data={"reason": reason})
            record_event(s, EventType.APPROVAL_REQUESTED, project_id=project.id,
                         actor="rights_gate", job_id=job.id, data={"kind": "rights_ambiguity"})
        return {"rights": status.value, "reason": reason,
                "awaiting_human": status is RightsStatus.NEEDS_HUMAN}


# -- ingest -------------------------------------------------------------------------------
def ingest(ctx: StudioContext, job: Job) -> dict[str, Any]:
    pid = job.project_id or ""
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        if S(project.status) is not S.RIGHTS_OK:
            raise PermanentJobError("wrong_state", f"ingest in {project.status}")
        source = project.source
        if source is None:
            raise PermanentJobError("no_source", "project has no source")
        if source.asset_id is None:
            if not source.local_path:
                raise PermanentJobError(
                    "no_source_file",
                    "Studio never downloads platform videos. Supply the permitted file "
                    "(upload it, or set local_path to a file the worker can read).")
            src = Path(source.local_path)
            if not src.is_file():
                raise PermanentJobError("source_missing", f"source file not found: {src}")
            asset = import_file(s, ctx.store, pid, "source", "source", src,
                                name=f"source{src.suffix.lower() or '.mp4'}")
            source.asset_id = asset.id
        found = s.get(Asset, source.asset_id)
        assert found is not None
        asset = found
        try:
            info = ctx.ffmpeg.probe(ctx.store.path_for(asset.rel_path))
        except FFmpegError as exc:
            raise PermanentJobError("unreadable_source", "source is not a readable video",
                                    exc.to_dict()) from exc
        asset.meta = {**asset.meta, "probe": info.to_dict()}
        ref = project.creative_input.get("character_reference_path")
        if ref and not project.creative_input.get("character_reference_asset"):
            ref_path = Path(ref)
            if not ref_path.is_file():
                raise PermanentJobError("reference_missing", f"reference image not found: {ref}")
            ref_asset = import_file(s, ctx.store, pid, "reference", "references", ref_path,
                                    name=f"character{ref_path.suffix.lower()}")
            project.creative_input = {**project.creative_input,
                                      "character_reference_asset": ref_asset.rel_path}
        transition(s, project, S.DOWNLOADED_OR_INGESTED, actor="ingest", job_id=job.id,
                   data={"asset": asset.rel_path})
        return {"source_asset": asset.rel_path, "probe": info.to_dict()}


# -- analysis -----------------------------------------------------------------------------
def analyze(ctx: StudioContext, job: Job) -> dict[str, Any]:
    project = _enter(ctx, job, {S.DOWNLOADED_OR_INGESTED}, S.ANALYZING)
    pid = project.id
    with ctx.db.session() as s:
        p = get_project(s, pid)
        source = p.source
        asset = s.get(Asset, source.asset_id) if source and source.asset_id else None
        profile = ctx.settings.profile(p.render_profile)
    if asset is None:
        raise PermanentJobError("no_source_asset", "source was not ingested")
    try:
        analysis = analyze_video(ctx.ffmpeg, ctx.store.path_for(asset.rel_path),
                                 max_shot_s=profile.max_frames / profile.fps)
    except FFmpegError as exc:
        raise JobError("ffmpeg_failed", "analysis failed", exc.to_dict()) from exc
    analysis["source_asset"] = asset.rel_path
    _write_json(ctx.store.project_dir(pid, "analysis") / "analysis.json", analysis)
    with ctx.db.transaction() as s:
        save_document(s, pid, "analysis", analysis, created_by="video_analyst")
        p = get_project(s, pid, for_update=True)
        transition(s, p, S.ANALYZED, actor="video_analyst", job_id=job.id,
                   data={"shots": len(analysis["shots"])})
    return {"shots": len(analysis["shots"]), "duration": analysis["duration"]}


# -- creative -----------------------------------------------------------------------------
def _keyframes(ctx: StudioContext, pid: str, analysis: dict[str, Any]) -> dict[str, bytes]:
    """The middle frame of every shot (for the DP's eyes and the dashboard), made once."""
    source = ctx.store.path_for(analysis["source_asset"])
    out_dir = ctx.store.project_dir(pid, "analysis/keyframes")
    with ctx.db.session() as s:
        known = {a.meta.get("shot_id") for a in s.scalars(select(Asset).where(
            Asset.project_id == pid, Asset.kind == "keyframe"))}
    frames: dict[str, bytes] = {}
    new: list[tuple[str, Path]] = []
    for shot in analysis["shots"]:
        path = out_dir / f"{shot['shot_id']}.jpg"
        if not path.is_file():
            try:
                ctx.ffmpeg.thumbnail(source, path, at=(shot["start"] + shot["end"]) / 2,
                                     width=512)
            except FFmpegError as exc:
                log.warning("keyframe extraction failed", extra={"data": {
                    "shot": shot["shot_id"], "error": exc.summary}})
                continue
        if shot["shot_id"] not in known:
            new.append((shot["shot_id"], path))
        frames[shot["shot_id"]] = path.read_bytes()
    if new:
        with ctx.db.transaction() as s:
            for shot_id, path in new:
                register_asset(s, ctx.store, pid, "keyframe", path, meta={"shot_id": shot_id})
    return frames


def creative_plan(ctx: StudioContext, job: Job) -> dict[str, Any]:
    project = _enter(ctx, job, {S.ANALYZED}, S.CREATIVE_PLANNING)
    with ctx.db.session() as s:
        analysis = latest_document(s, project.id, "analysis")
        if analysis is None:
            raise PermanentJobError("no_analysis", "analysis missing")
        data = analysis.data
    path = tracker_path(ctx.settings.studio.data_dir)
    try:
        tracker = load_tracker(path)
    except TrackerError as exc:
        raise PermanentJobError("asset_tracker_invalid",
                                f"{exc}. Fix it on the Director page or delete the file.") from exc
    keyframes = _keyframes(ctx, project.id, data)
    director_settings = ctx.settings.director
    if project.creative_input.get("stable"):
        # Stable mode: no per-shot framing or lighting changes; every shot renders the brief's
        # one prompt, so the look cannot drift from shot to shot (docs/stability.md).
        director_settings = director_settings.model_copy(update={"enabled": False})
    brief, by = direct(ctx.provider, project.creative_input, data, project.target_format,
                       tracker=tracker, settings=director_settings,
                       dp_provider=ctx.dp_provider, keyframes=keyframes)
    d = ctx.settings.director
    schedule = None
    if d.enabled and all(shot.prompt for shot in brief.shot_plan):
        schedule = build_schedule(brief.shot_plan, brief.negative_prompt,
                                  duration=data["duration"], fps=d.schedule_fps,
                                  interval=d.schedule_interval,
                                  inline_negative=d.schedule_inline_negative)
    notes = brief.director
    with ctx.db.transaction() as s:
        doc = save_document(s, project.id, "creative_brief", brief.model_dump(), created_by=by)
        info: dict[str, Any] = {"version": doc.version, "story_by": by,
                                "framing_by": notes.framing_by if notes else None,
                                "vision": notes.vision if notes else False}
        if schedule is not None:
            sdoc = save_document(s, project.id, "prompt_schedule", schedule.to_dict(),
                                 created_by="prompt_compiler")
            (ctx.store.project_dir(project.id, "prompts")
             / f"prompt_schedule_v{sdoc.version}.txt").write_text(schedule.text + "\n",
                                                                  encoding="utf-8")
            info["schedule_keyframes"] = len(schedule.keyframes)
        record_event(s, EventType.CREATIVE_BRIEF_CREATED, project_id=project.id, actor=by,
                     job_id=job.id, data=info)
        p = get_project(s, project.id, for_update=True)
        transition(s, p, S.CREATIVE_READY, actor="creative_director", job_id=job.id)
    return {"brief_version": doc.version, "provider": by, **{
        k: v for k, v in info.items() if k != "version"}}


# -- manifest + workflow compilation ------------------------------------------------------
def compile_stage(ctx: StudioContext, job: Job) -> dict[str, Any]:
    project = _enter(ctx, job, {S.CREATIVE_READY, S.WORKFLOW_READY}, S.WORKFLOW_COMPILING)
    pid = project.id
    profile = ctx.settings.profile(project.render_profile)
    with ctx.db.session() as s:
        brief_doc = latest_document(s, pid, "creative_brief")
        analysis = latest_document(s, pid, "analysis")
        if brief_doc is None or analysis is None:
            raise PermanentJobError("missing_inputs", "brief or analysis missing")
        brief = CreativeBrief.model_validate(brief_doc.data)
        a = analysis.data
    manifest = build_manifest(
        project_id=pid, source_asset=a["source_asset"], analysis=a, brief=brief,
        profile=profile, target_format=project.target_format,
        reference_image=project.creative_input.get("character_reference_asset"))
    manifest.identity.reference_mode = project.creative_input.get("reference_mode", "source")
    manifest.subject = SubjectSpec(**decide_subject(project.creative_input).to_dict())
    for shot in manifest.shots:
        shot.overrides.update({k: project.creative_input[k]
                               for k in ("control_strength", "steps", "cfg", "seed",
                                         "canny_low", "canny_high")
                               if project.creative_input.get(k) is not None})
    if project.creative_input.get("stable"):
        stable = stable_overrides(project, profile)
        for shot in manifest.shots:
            shot.overrides.update({k: v for k, v in stable.items() if k not in shot.overrides
                                   or k == "seed" and project.creative_input.get("seed") is None})
        if manifest.identity.reference_mode == "auto" and not manifest.identity.reference_image:
            manifest.identity.reference_mode = "cutout"  # the same subject reference for every shot
    compiled: dict[str, Any] = {}
    if ctx.settings.render.renderer == "comfyui":
        try:
            template = ctx.registry.get(profile.workflow)
            for shot in manifest.shots:
                c = compile_workflow(template, {**shot_params(manifest, shot, profile),
                                                "INPUT_VIDEO": f"{pid}_{shot.shot_id}.mp4"})
                compiled[shot.shot_id] = {"applied": c.applied, "ignored": sorted(c.ignored)}
        except TemplateError as exc:
            raise PermanentJobError("template_error", str(exc)) from exc
    with ctx.db.transaction() as s:
        doc = save_document(s, pid, "manifest", manifest.model_dump(), created_by="workflow_planner",
                            schema_version=manifest.version)
        _write_json(ctx.store.project_dir(pid, "manifests") / f"manifest_v{doc.version}.json",
                    manifest.model_dump())
        record_event(s, EventType.MANIFEST_CREATED, project_id=pid, actor="workflow_planner",
                     job_id=job.id, data={"version": doc.version, "shots": len(manifest.shots),
                                          "subject": manifest.subject.mode,
                                          "subject_reason": manifest.subject.reason})
        if compiled:
            record_event(s, EventType.WORKFLOW_COMPILED, project_id=pid,
                         actor="workflow_compiler", job_id=job.id,
                         data={"workflow": profile.workflow, "shots": compiled})
        p = get_project(s, pid, for_update=True)
        transition(s, p, S.WORKFLOW_READY, actor="workflow_planner", job_id=job.id)
    return {"manifest_version": doc.version, "shots": len(manifest.shots),
            "renderer": ctx.settings.render.renderer}


def stable_overrides(project: Project, profile: RenderProfile) -> dict[str, Any]:
    """What Stable mode fixes for every shot: one seed, the profile's full steps (at least
    20), the source guide at 1.0. Values you set yourself on New video win."""
    return {"seed": int(project.creative_input.get("seed")
                        or shot_seed(project.id, "stable")),
            "steps": max(int(profile.steps), 20), "control_strength": 1.0}


# -- rendering ----------------------------------------------------------------------------
def _renderer(ctx: StudioContext, job: Job, target: str = "local") -> Renderer:
    if ctx.settings.render.renderer == "comfyui":
        def cancelled() -> bool:
            with ctx.db.session() as s:
                return is_cancelled(s, job.id)
        try:
            client = ctx.comfy_for(target)
        except ComfyError as exc:
            raise PermanentJobError("cloud_not_configured", str(exc)) from exc
        return ComfyUIRenderer(client, ctx.registry, ctx.ffmpeg,
                               timeout_s=ctx.settings.comfyui.timeout_s,
                               poll_s=ctx.settings.comfyui.poll_interval_s,
                               should_cancel=cancelled)
    custom = ctx.extras.get("renderer")
    if custom is not None:
        return custom  # type: ignore[return-value]
    return FFmpegPreviewRenderer(ctx.ffmpeg)


def _shot_clip(ctx: StudioContext, manifest: ReconstructionManifest, shot: ShotSpec,
               fps: float | None = None) -> Path:
    fps = fps or manifest.video.fps
    clip = ctx.store.project_dir(manifest.project_id, "work/clips") / \
        f"{shot.shot_id}_{fps:g}fps.mp4"
    if not clip.exists():
        ctx.ffmpeg.cut(ctx.store.path_for(manifest.source_asset), clip, start=shot.start,
                       end=shot.end, fps=fps, width=manifest.video.width,
                       height=manifest.video.height)
    return clip


def _latest_renders(ctx: StudioContext, project_id: str) -> dict[str, Render]:
    with ctx.db.session() as s:
        rows = s.scalars(select(Render).where(Render.project_id == project_id,
                                              Render.status == "succeeded")
                         .order_by(Render.attempt)).all()
    return {r.shot_id: r for r in rows}


def _check_budget(ctx: StudioContext, project_id: str) -> None:
    with ctx.db.session() as s:
        used = budget_usage(s, project_id, ctx.settings)
    reason = None
    if used["renders"] >= used["max_renders"]:
        reason = f"render budget exhausted ({used['renders']}/{used['max_renders']})"
    elif used["gpu"] >= used["max_gpu"]:
        reason = f"GPU budget exhausted ({used['gpu']:.1f}/{used['max_gpu']:g} min)"
    elif used["cloud_exhausted"]:
        reason = f"cloud GPU budget exhausted ({used['cloud']:.1f}/{used['max_cloud']:g} min)"
    elif used["exhausted"]:
        reason = f"cost limit reached (about ${used['usd']:.2f} of ${used['max_usd']:g})"
    if reason:
        with ctx.db.transaction() as s:
            record_event(s, EventType.BUDGET_EXCEEDED, project_id=project_id, actor="cost_guard",
                         data={"reason": reason})
        raise PermanentJobError("budget_exceeded", reason)


def _apply_degrade(ctx: StudioContext, manifest: ReconstructionManifest, shot: ShotSpec,
                   steps: list[str], target: str = "local") -> str | None:
    """Apply the next unused OOM recovery step to the shot. Returns the step or None."""
    raw = str(shot.overrides.get("_oom_steps", ""))
    applied = raw.split(",") if raw else []
    for step in steps:
        if step in applied:
            continue
        o = shot.overrides
        if step == "clear_cache":
            if ctx.settings.render.renderer == "comfyui":
                client = ctx.comfy_for(target)
                try:
                    client.free()
                finally:
                    client.close()
        elif step == "reduce_frames":
            o["fps"] = max(6.0, round(float(o.get("fps", manifest.video.fps)) * 0.75, 2))
        elif step == "reduce_resolution":
            o["resolution_scale"] = round(float(o.get("resolution_scale", 1.0)) * 0.75, 3)
        elif step == "enable_offload":
            o["offload"] = True
        elif step.startswith("switch_profile:"):
            o["profile"] = step.split(":", 1)[1]
        else:  # split_shot and anything unknown: not automated yet → escalate
            return None
        applied.append(step)
        o["_oom_steps"] = ",".join(applied)
        return step
    return None


def render(ctx: StudioContext, job: Job) -> dict[str, Any]:
    # RENDERING is accepted so a render interrupted by a crash or retry resumes in place.
    project = _enter(ctx, job, {S.WORKFLOW_READY, S.RENDER_QUEUED, S.RENDERING}, None)
    pid = project.id
    with ctx.db.transaction() as s:
        p = get_project(s, pid, for_update=True)
        if S(p.status) is S.WORKFLOW_READY:
            transition(s, p, S.RENDER_QUEUED, actor="render", job_id=job.id)
        if S(p.status) is S.RENDER_QUEUED:
            transition(s, p, S.RENDERING, actor="render", job_id=job.id)
    manifest, manifest_version = _manifest(ctx, pid)
    base_profile = ctx.settings.profile(manifest.render_profile)
    target = project_target(project.creative_input)
    renderer = _renderer(ctx, job, target)
    cloud = renderer.name == "comfyui" and target == "cloud"
    done = _latest_renders(ctx, pid)
    faults = project.creative_input.get("test_faults") or {}
    keep = manifest.subject.mode == "keep"
    masker = (_subject_masker(ctx)
              if keep or manifest.identity.reference_mode == "cutout" else None)
    rendered: list[str] = []

    # One batch for every shot: ComfyUI keeps the Wan model and text encoder loaded between
    # shots instead of reloading them from disk for each one (docs/speed.md).
    # A cloud render uses the cloud server's GPU, not this PC's: no batch and no lease.
    batch = (ctx.gpu.heavy_batch(job.id) if renderer.name != "ffmpeg_preview" and not cloud
             else contextlib.nullcontext())
    with batch:
        for position in range(len(manifest.shots)):
            shot = manifest.shots[position]
            if shot.shot_id in done:
                continue
            while True:
                with ctx.db.session() as s:
                    if is_cancelled(s, job.id):
                        raise JobCancelled()
                _check_budget(ctx, pid)
                newer, newer_version = _manifest(ctx, pid)
                if newer_version != manifest_version:  # adjusted while rendering
                    manifest, manifest_version = newer, newer_version
                    shot = manifest.shots[position]
                    log.info("manifest reloaded before a shot", extra={"data": {
                        "shot": shot.shot_id, "version": manifest_version}})
                profile = ctx.settings.profile(str(shot.overrides.get("profile",
                                                                       base_profile.name)))
                params = shot_params(manifest, shot, profile)
                if params.get("REFERENCE_IMAGE"):
                    params["REFERENCE_IMAGE"] = str(ctx.store.path_for(params["REFERENCE_IMAGE"]))
                with ctx.db.session() as s:
                    attempt = (s.scalar(select(func.count(Render.id)).where(
                        Render.project_id == pid, Render.shot_id == shot.shot_id)) or 0) + 1
                fault = faults.get(shot.shot_id)
                if fault and attempt in fault.get("attempts", [1]):
                    params["_FAULT"] = fault.get("kind", "black")
                clip = _shot_clip(ctx, manifest, shot, fps=params["FPS"])
                workflow = profile.workflow
                guidance: dict[str, Any] = {}
                if masker is not None and renderer.name == "comfyui":
                    workflow, guidance = _subject_guidance(ctx, manifest, shot, clip, params,
                                                           profile, masker, target)
                out_dir = ctx.store.project_dir(pid, f"renders/{shot.shot_id}")
                out = out_dir / f"attempt_{attempt:02d}.mp4"
                with ctx.db.transaction() as s:
                    row = Render(project_id=pid, job_id=job.id, shot_id=shot.shot_id,
                                 attempt=attempt, profile=profile.name, renderer=renderer.name,
                                 workflow=workflow if renderer.name == "comfyui" else None,
                                 status="running", params={k: v for k, v in params.items()})
                    s.add(row)
                    s.flush()
                    render_id = row.id
                    record_event(s, EventType.RENDER_SUBMITTED, project_id=pid, actor="render",
                                 job_id=job.id, data={"shot_id": shot.shot_id, "attempt": attempt,
                                                      "renderer": renderer.name})
                try:
                    if renderer.name == "ffmpeg_preview" or cloud:
                        outcome = renderer.render_shot(clip=clip, params=params,
                                                       workflow=workflow, out=out)
                    else:
                        with ctx.gpu.lease(job.id, profile.resource_class):
                            outcome = renderer.render_shot(clip=clip, params=params,
                                                           workflow=workflow, out=out)
                except RenderOOM as exc:
                    step = _apply_degrade(ctx, manifest, shot, profile.degrade, target)
                    _render_failed(ctx, job, render_id, "oom", str(exc), {"next_step": step})
                    with ctx.db.transaction() as s:
                        record_event(s, EventType.GPU_OOM, project_id=pid, actor="render",
                                     job_id=job.id, data={"shot_id": shot.shot_id,
                                                          "recovery_step": step})
                        save_document(s, pid, "manifest", manifest.model_dump(),
                                      created_by="oom_recovery")
                    if step is None:
                        raise PermanentJobError(
                            "oom_unrecoverable",
                            f"{shot.shot_id}: CUDA OOM after the full recovery ladder; needs a "
                            "smaller shot, a lighter profile or cloud GPU (approval required)") from exc
                    continue  # retry with the degraded settings (never identical)
                except RenderUnavailable as exc:
                    _render_failed(ctx, job, render_id, "unavailable", str(exc))
                    raise JobError("renderer_unavailable", str(exc)) from exc
                except (RenderRejected, TemplateError) as exc:
                    _render_failed(ctx, job, render_id, "rejected", str(exc))
                    raise PermanentJobError("render_rejected", str(exc)) from exc
                except GpuUnavailable as exc:
                    _render_failed(ctx, job, render_id, "gpu_busy", str(exc))
                    raise JobError("gpu_busy", str(exc)) from exc
                except JobCancelled:
                    _render_failed(ctx, job, render_id, "cancelled", "Render cancelled by the user")
                    raise
                except FFmpegError as exc:
                    _render_failed(ctx, job, render_id, "ffmpeg", str(exc), exc.to_dict())
                    raise JobError("ffmpeg_failed", str(exc), exc.to_dict()) from exc
                details, final = outcome.details, outcome.path
                if guidance:
                    details = {**details, "guidance": guidance}
                if keep and masker is not None:
                    subject, kept = _keep_subject(ctx, manifest, shot, clip, outcome, params, masker)
                    details = {**details, "subject": subject}
                    final = kept or final
                meta = {"shot_id": shot.shot_id, "attempt": attempt}
                with ctx.db.transaction() as s:
                    asset = register_asset(s, ctx.store, pid, "render", final,
                                           {**meta, "subject": "kept"} if final != outcome.path
                                           else meta)
                    if final != outcome.path:  # the render before the subject went back in
                        register_asset(s, ctx.store, pid, "render_raw", outcome.path, meta)
                    done_row = s.get(Render, render_id)
                    assert done_row is not None
                    done_row.status, done_row.output_asset_id = "succeeded", asset.id
                    done_row.remote_id, done_row.duration_s = outcome.remote_id, outcome.seconds
                    done_row.finished_at = utcnow()
                    done_row.params = {**done_row.params, "_details": details}
                    if cloud:  # an estimate: a rented server bills while it is on
                        minutes = outcome.seconds / 60
                        s.add(CostEntry(project_id=pid, job_id=job.id, kind="cloud_gpu_minutes",
                                        amount=minutes, unit="min",
                                        # unrounded: a short shot must not cost $0
                                        usd=minutes / 60 * ctx.settings.cloud.price_per_hour_usd))
                    elif renderer.name != "ffmpeg_preview":
                        s.add(CostEntry(project_id=pid, job_id=job.id, kind="gpu_minutes",
                                        amount=outcome.seconds / 60, unit="min"))
                    data: dict[str, Any] = {"shot_id": shot.shot_id, "attempt": attempt,
                                            "seconds": round(outcome.seconds, 2)}
                    if cloud:
                        data["render_on"] = "cloud"
                    if "subject" in details:
                        data["subject"] = ("kept" if details["subject"].get("kept")
                                           else details["subject"].get("reason"))
                    record_event(s, EventType.RENDER_COMPLETED, project_id=pid, actor="render",
                                 job_id=job.id, data=data)
                rendered.append(shot.shot_id)
                break

    assembled = _assemble(ctx, manifest)
    _finish(ctx, job, S.QUALITY_CHECK, {"assembled": assembled, "rendered": rendered,
                                        "manifest_version": manifest_version})
    return {"rendered": rendered, "assembled": assembled}


def _subject_masker(ctx: StudioContext) -> SubjectMasker:
    custom = ctx.extras.get("subject_masker")
    if custom is not None:
        return custom  # type: ignore[return-value]
    return OnnxSubjectMasker(ctx.settings.subject, ctx.settings.studio.data_dir)


def _mask_cache(ctx: StudioContext, manifest: ReconstructionManifest, shot: ShotSpec,
                params: dict[str, Any]) -> Path:
    """Base name of a shot's mask files: one per shot, frame rate and render shape."""
    return ctx.store.project_dir(manifest.project_id, "work/masks") / \
        f"{shot.shot_id}_{float(params['FPS']):g}fps_{params['WIDTH']}x{params['HEIGHT']}"


def _subject_guidance(ctx: StudioContext, manifest: ReconstructionManifest, shot: ShotSpec,
                      clip: Path, params: dict[str, Any], profile: RenderProfile,
                      masker: SubjectMasker, target: str = "local") -> tuple[str, dict[str, Any]]:
    """Use the subject's masks before rendering. Returns the workflow to run and what was done.

    When the subject is kept and the profile has a keep workflow, Wan gets the subject's masks
    (``MASK_VIDEO``): VACE then keeps the subject's own pixels as context and redraws only the
    room around it. The reference image becomes a cutout of the subject on white when
    ``reference_mode`` is ``cutout``, or ``auto`` with the subject kept. Anything that fails
    leaves the plain workflow and no reference; the render itself still runs.
    """
    workflow = profile.workflow
    keep = manifest.subject.mode == "keep"
    mode = manifest.identity.reference_mode
    want_mask = keep and bool(profile.keep_workflow)
    want_cutout = (not params.get("REFERENCE_IMAGE")
                   and (mode == "cutout" or (mode == "auto" and keep)))
    if not (want_mask or want_cutout):
        return workflow, {}
    fps, width, height = float(params["FPS"]), int(params["WIDTH"]), int(params["HEIGHT"])
    cache = _mask_cache(ctx, manifest, shot, params)
    info: dict[str, Any] = {}
    try:
        probs = shot_masks(ctx.ffmpeg, masker, clip=clip, fps=fps,
                           frames=int(params["FRAME_COUNT"]), aspect=(width, height),
                           cache=cache)
        if reason := unusable(subject_share(probs), ctx.settings.subject):
            return workflow, {"masks": reason}
        if want_mask and profile.keep_workflow:
            problem = (_live_problem(ctx, profile.keep_workflow, target)
                       if _accepts(ctx, profile.keep_workflow, "MASK_VIDEO") else None)
            if _accepts(ctx, profile.keep_workflow, "MASK_VIDEO") and not problem:
                params["MASK_VIDEO"] = str(vace_mask_video(
                    ctx.ffmpeg, probs, cache.with_name(f"{cache.name}_vace.mp4"),
                    width=width, height=height, fps=fps))
                workflow = profile.keep_workflow
                info["keep_workflow"] = workflow
            else:
                info["keep_workflow"] = (f"{profile.keep_workflow} is not available"
                                         + (f": {problem}" if problem else ""))
        if want_cutout and _accepts(ctx, workflow, "REFERENCE_IMAGE"):
            params["REFERENCE_IMAGE"] = str(cutout_reference(
                ctx.ffmpeg, probs, clip=clip, fps=fps, width=width, height=height,
                out=cache.with_name(f"{cache.name}_cutout.png")))
            params["_REFERENCE_KIND"] = "subject cutout"
            info["reference"] = "subject cutout"
    except MaskerUnavailable as exc:
        return profile.workflow, {"masks": f"unavailable: {exc}"}
    except FFmpegError as exc:
        params.pop("MASK_VIDEO", None)
        return profile.workflow, {"masks": f"could not be prepared: {exc.summary}"}
    except (OSError, ValueError) as exc:  # disk full, a damaged mask cache
        params.pop("MASK_VIDEO", None)
        return profile.workflow, {"masks": f"could not be prepared: {exc}"}
    return workflow, info


def _live_problem(ctx: StudioContext, workflow: str, target: str = "local") -> str | None:
    """Why this ComfyUI cannot run ``workflow`` (e.g. a missing node), checked once per worker.
    None when it can, or when ComfyUI cannot be asked: the render itself will tell then."""
    checked = cast(dict[str, str | None],
                   ctx.extras.setdefault(f"live_workflow_problems_{target}", {}))
    if workflow not in checked:
        client = None
        try:
            client = ctx.comfy_for(target)
            info = client.object_info()
            if not info:
                return None
            problems = validate_against_object_info(ctx.registry.get(workflow), info)
            checked[workflow] = problems[0] if problems else None
        except (ComfyError, httpx.HTTPError, TemplateError):
            return None
        finally:
            if client is not None:
                client.close()
    return checked[workflow]


def _accepts(ctx: StudioContext, workflow: str, param: str) -> bool:
    try:
        return param in ctx.registry.get(workflow).spec.parameters
    except TemplateError:
        return False


def _keep_subject(ctx: StudioContext, manifest: ReconstructionManifest, shot: ShotSpec,
                  clip: Path, outcome: RenderOutcome, params: dict[str, Any],
                  masker: SubjectMasker) -> tuple[dict[str, Any], Path | None]:
    """Lay the real subject over a finished render. The composite's path, or None when the raw
    render stands (no clear subject, or masks could not be made: the reason is recorded)."""
    out = outcome.path.with_name(f"{outcome.path.stem}_subject.mp4")
    try:
        result = keep_subject(ctx.ffmpeg, masker, ctx.settings.subject, clip=clip,
                              render=outcome.path, fps=float(params["FPS"]), out=out,
                              cache=_mask_cache(ctx, manifest, shot, params))
    except MaskerUnavailable as exc:
        log.warning("subject not kept", extra={"data": {"shot": shot.shot_id, "error": str(exc)}})
        return {"kept": False, "reason": f"subject masks unavailable: {exc}"}, None
    except FFmpegError as exc:
        return {"kept": False, "reason": f"compositing failed: {exc.summary}"}, None
    except (OSError, ValueError) as exc:  # disk full, a damaged mask cache: the render stands
        log.warning("subject not kept", extra={"data": {"shot": shot.shot_id, "error": str(exc)}})
        return {"kept": False, "reason": f"compositing failed: {exc}"}, None
    if result.get("mask_video"):
        result["mask_video"] = ctx.store.rel(Path(result["mask_video"]))
    return result, out if result.get("kept") else None


def _render_failed(ctx: StudioContext, job: Job, render_id: str, code: str, message: str,
                   details: dict[str, Any] | None = None) -> None:
    with ctx.db.transaction() as s:
        row = s.get(Render, render_id)
        if row is not None:
            row.status, row.finished_at = "failed", utcnow()
            row.error = {"code": code, "message": message, **(details or {})}
        record_event(s, EventType.RENDER_FAILED, project_id=job.project_id, actor="render",
                     job_id=job.id, data={"render_id": render_id, "code": code,
                                          "message": message})


def _assemble(ctx: StudioContext, manifest: ReconstructionManifest) -> str:
    """Splice the latest good render of every shot, in manifest order."""
    pid = manifest.project_id
    latest = _latest_renders(ctx, pid)
    clips: list[Path] = []
    with ctx.db.session() as s:
        for shot in manifest.shots:
            r = latest.get(shot.shot_id)
            if r is None or r.output_asset_id is None:
                raise PermanentJobError("missing_shot", f"no successful render for {shot.shot_id}")
            asset = s.get(Asset, r.output_asset_id)
            assert asset is not None
            clip = ctx.store.path_for(asset.rel_path)
            # Normalise every shot to the manifest grid so OOM-degraded shots still splice.
            norm = clip.with_name(clip.stem + "_norm_v3.mp4")
            if not norm.exists():
                # Align cuts to the full timeline, avoiding cumulative one-frame rounding
                # errors when shots used different frame rates after OOM recovery.
                frames = max(1, round(shot.end * manifest.video.fps)
                             - round(shot.start * manifest.video.fps))
                ctx.ffmpeg.filter_video(clip, norm, f"fps={manifest.video.fps},scale="
                    f"{manifest.video.width}:{manifest.video.height},"
                    f"tpad=stop=-1:stop_mode=clone,trim=end_frame={frames},setpts=PTS-STARTPTS",
                    fps=manifest.video.fps)
            clips.append(norm)
    renders_dir = ctx.store.project_dir(pid, "renders")
    n = len(list(renders_dir.glob("assembled_v*.mp4"))) + 1
    out = ctx.ffmpeg.concat(clips, renders_dir / f"assembled_v{n:02d}.mp4")
    with ctx.db.transaction() as s:
        register_asset(s, ctx.store, pid, "assembled", out, {"version": n})
    return ctx.store.rel(out)


# -- quality control ----------------------------------------------------------------------
def quality_check(ctx: StudioContext, job: Job) -> dict[str, Any]:
    project = _enter(ctx, job, {S.QUALITY_CHECK}, S.QUALITY_CHECK)
    pid = project.id
    manifest, _ = _manifest(ctx, pid)
    latest = _latest_renders(ctx, pid)
    threshold = ctx.settings.quality.pass_threshold
    if project.creative_input.get("stable"):
        threshold = min(9.0, threshold + 1.0)  # artifacts fail and get repaired, not accepted
    fps = manifest.video.fps
    shots = []
    offset = 0
    with ctx.db.session() as s:
        accepted = ratings.accepted_renders(s, pid)
        for shot in manifest.shots:
            r = latest[shot.shot_id]
            asset = s.get(Asset, r.output_asset_id)
            assert asset is not None
            src = ctx.ffmpeg.read_gray_frames(_shot_clip(ctx, manifest, shot), 64, 64, fps=fps)
            out = ctx.ffmpeg.read_gray_frames(ctx.store.path_for(asset.rel_path), 64, 64, fps=fps)
            result = qc_mod.score_shot(src, out, threshold=threshold, shot_id=shot.shot_id,
                                       frame_offset=offset)
            result["render_id"] = r.id
            result["attempt"] = r.attempt
            if result["decision"] != "PASS" and r.id in accepted:
                # You liked this render, or kept it while redoing other shots: QC still
                # records what it measured, but never sends it back for repair.
                result["decision"] = "PASS"
                result["accepted_by"] = "you"
            shots.append(result)
            offset += len(src)
    report = qc_mod.summarize(shots, threshold)
    with ctx.db.transaction() as s:
        doc = save_document(s, pid, "qc_report", report, created_by="qc")
        _write_json(ctx.store.project_dir(pid, "qc") / f"qc_v{doc.version}.json", report)
        p = get_project(s, pid, for_update=True)
        target = S.QUALITY_PASSED if report["decision"] == "PASS" else S.QUALITY_FAILED
        transition(s, p, target, actor="qc", job_id=job.id,
                   data={"overall": report["overall"], "failed_shots": report["failed_shots"]})
    return {"decision": report["decision"], "overall": report["overall"],
            "failed_shots": report["failed_shots"]}


# -- repair -------------------------------------------------------------------------------
def _stop_at_repair_limit(ctx: StudioContext, job: Job, report: dict[str, Any],
                          reason: str) -> dict[str, Any]:
    with ctx.db.transaction() as s:
        p = get_project(s, job.project_id or "", for_update=True)
        record_event(s, EventType.REPAIR_BUDGET_EXHAUSTED, project_id=p.id, actor="repair",
                     job_id=job.id, data={"rounds": p.repair_rounds,
                                          "failed_shots": report["failed_shots"]})
        s.add(ApprovalRequest(project_id=p.id, kind="repair_budget", requested_by="repair",
                              summary=f"QC still failing after {p.repair_rounds} repair "
                              "rounds", payload={"failed_shots": report["failed_shots"]}))
        transition(s, p, S.REPAIRING, actor="repair", job_id=job.id)
        transition(s, p, S.FAILED, actor="repair", job_id=job.id, reason=reason)
    return {"repaired": False, "reason": reason}


STALL_MIN_GAIN = 0.2  # QC points a failing shot must gain over the window


def _repairs_stalled(session: Session, project_id: str, window: int) -> bool:
    """True when the last ``window`` QC reports (``render.stall_reports``) since a human last
    granted rounds show no failing shot gaining STALL_MIN_GAIN points: rerolling again is
    wasted GPU time."""
    granted = session.scalar(
        select(func.max(Event.created_at)).where(
            Event.project_id == project_id, Event.type == EventType.REPAIR_BUDGET_EXTENDED))
    query = select(Document.data).where(Document.project_id == project_id,
                                        Document.kind == "qc_report")
    if granted is not None:
        query = query.where(Document.created_at > granted)
    reports = session.scalars(query.order_by(Document.version.desc())
                              .limit(window)).all()
    if len(reports) < window:
        return False
    oldest = {r["shot_id"]: float(r.get("overall", 0)) for r in reports[-1].get("shots", [])}
    failing = [r for r in reports[0].get("shots", []) if r.get("decision") != "PASS"]
    return bool(failing) and all(
        r["shot_id"] in oldest and float(r.get("overall", 0)) < oldest[r["shot_id"]] + STALL_MIN_GAIN
        for r in failing)


def repair(ctx: StudioContext, job: Job) -> dict[str, Any]:
    project = _enter(ctx, job, {S.QUALITY_FAILED}, None)
    pid = project.id
    manifest, _ = _manifest(ctx, pid)
    with ctx.db.session() as s:
        qc_doc = latest_document(s, pid, "qc_report")
        assert qc_doc is not None
        report = qc_doc.data
        budget = repair_budget(s, pid, ctx.settings.render.max_retries)
    if project.repair_rounds >= budget:
        return _stop_at_repair_limit(ctx, job, report, "repair budget exhausted")
    with ctx.db.session() as s:
        stalled = _repairs_stalled(s, pid, ctx.settings.render.stall_reports)
    if stalled:
        return _stop_at_repair_limit(ctx, job, report,
                                     "repeated re-renders did not improve QC")

    round_ = project.repair_rounds + 1
    current = {s.shot_id: {"seed": s.overrides.get("seed", s.seed),
                           "control_strength": s.overrides.get("control_strength", 1.0),
                           "style_strength": s.overrides.get("style_strength",
                                                             manifest.style.strength),
                           "identity_strength": s.overrides.get("identity_strength",
                                                                manifest.identity.strength)}
               for s in manifest.shots}
    supported = None
    if ctx.settings.render.renderer == "comfyui":
        supported = set(ctx.registry.get(ctx.settings.profile(manifest.render_profile).workflow)
                        .spec.parameters)
    plan = RepairPlanner().plan(report, round_, current, supported=supported)
    if not plan.actions:
        # Stop where a human can keep the renders or check again, as at the repair limit.
        return _stop_at_repair_limit(ctx, job, report,
                                     "this workflow has no automatic repair controls")
    for action in plan.actions:
        manifest.shot(action.shot_id).overrides.update(action.changes)
    failing = [a.shot_id for a in plan.actions]
    with ctx.db.transaction() as s:
        p = get_project(s, pid, for_update=True)
        transition(s, p, S.REPAIRING, actor="repair_planner", job_id=job.id,
                   data={"round": round_, "shots": failing})
        save_document(s, pid, "repair_plan", plan.model_dump(), created_by="repair_planner")
        mdoc = save_document(s, pid, "manifest", manifest.model_dump(),
                             created_by="repair_planner")
        _write_json(ctx.store.project_dir(pid, "manifests") / f"manifest_v{mdoc.version}.json",
                    manifest.model_dump())
        for r in s.scalars(select(Render).where(Render.project_id == pid,
                                                Render.shot_id.in_(failing),
                                                Render.status == "succeeded")):
            r.status = "superseded"
        p.repair_rounds = round_
        transition(s, p, S.RENDER_QUEUED, actor="repair_planner", job_id=job.id)
    return {"repaired": True, "round": round_, "shots": failing}


# -- edit / encode ------------------------------------------------------------------------
_FORMATS = {"youtube_short": (1080, 1920), "youtube_video": (1920, 1080)}


def edit(ctx: StudioContext, job: Job) -> dict[str, Any]:
    project = _enter(ctx, job, {S.QUALITY_PASSED}, S.EDITING)
    pid = project.id
    manifest, _ = _manifest(ctx, pid)
    assembled = _assemble(ctx, manifest)
    final_dir = ctx.store.project_dir(pid, "final")
    width, height = _FORMATS.get(project.target_format, (1080, 1920))
    video = ctx.store.path_for(assembled)
    source = ctx.store.path_for(manifest.source_asset)
    try:
        creative = project.creative_input or {}
        bed_path_value = creative.get("audio_bed_path")
        bed_path: Path | None = None
        if bed_path_value:
            data_dir = Path(ctx.settings.studio.data_dir).resolve()
            allowed = (data_dir / "uploads", data_dir / "audio")  # an upload, or the sound library
            bed_path = Path(str(bed_path_value)).resolve()
            if not any(bed_path.is_relative_to(root) for root in allowed) or not bed_path.is_file():
                raise JobError("audio_bed_missing", "The music or effects file for the soundtrack "
                               "is missing.")
            if not ctx.ffmpeg.has_audio(bed_path):
                raise JobError("audio_bed_invalid", "The uploaded file has no audio stream.")
        source_audio = (ctx.ffmpeg.probe(source).has_audio
                        if creative.get("keep_source_audio", True) else False)
        if source_audio or bed_path is not None:
            with_audio = final_dir / "with_audio.mp4"
            if bed_path is not None:
                ctx.ffmpeg.mix_audio(
                    video, with_audio, source_audio=source if source_audio else None,
                    audio_bed=bed_path, bed_volume=float(creative.get("audio_bed_gain", 0.25)))
            else:
                ctx.ffmpeg.attach_audio(video, source, with_audio)
            video = with_audio
        final = ctx.ffmpeg.encode_final(video, final_dir / "final.mp4", width=width,
                                        height=height, fps=manifest.video.fps)
        info = ctx.ffmpeg.probe(final)
        thumbs = ctx.store.project_dir(pid, "thumbnails")
        thumb = ctx.ffmpeg.thumbnail(final, thumbs / "thumbnail.jpg", at=info.duration * 0.4)
        preview = ctx.ffmpeg.preview_gif(final, final_dir / "preview.gif",
                                         seconds=min(4.0, info.duration))
    except FFmpegError as exc:
        raise JobError("ffmpeg_failed", str(exc), exc.to_dict()) from exc
    with ctx.db.session() as s:
        brief = latest_document(s, pid, "creative_brief")
        brief_data = brief.data if brief else {}
    # Model call (can take a minute on a 9B model) stays outside the DB transaction.
    draft, drafted_by = ChannelManager(ctx.provider).run(project.creative_input, brief_data,
                                                         project.target_format, info.duration)
    with ctx.db.transaction() as s:
        p = get_project(s, pid, for_update=True)
        final_asset = register_asset(s, ctx.store, pid, "final", final, {"probe": info.to_dict()})
        thumb_asset = register_asset(s, ctx.store, pid, "thumbnail", thumb)
        register_asset(s, ctx.store, pid, "preview", preview)
        rights = latest_rights(s, pid)
        metadata = publishing.draft_metadata(p, brief_data, rights, duration=info.duration)
        if draft is not None:
            metadata = publishing.apply_draft(metadata, draft.model_dump(),
                                              target_format=p.target_format, rights=rights)
        previous = latest_document(s, pid, "metadata")
        if previous is not None and (previous.created_by == "dashboard"
                                     or previous.data.get("text_by") == "you"):
            # A video re-edited after redone shots keeps the text you already wrote, on every
            # later redo too.
            metadata = {**metadata, "text_by": "you",
                        **{k: previous.data[k] for k in ("title", "description", "tags")
                           if k in previous.data}}
        save_document(s, pid, "metadata", metadata, created_by=f"channel_manager:{drafted_by}")
        record_event(s, EventType.FINAL_ENCODED, project_id=pid, actor="editor", job_id=job.id,
                     data={"final": final_asset.rel_path, "thumbnail": thumb_asset.rel_path,
                           "duration": info.duration, "warnings": metadata["warnings"]})
        transition(s, p, S.READY_TO_PUBLISH, actor="editor", job_id=job.id)
        if autonomy_level(p, ctx.settings) >= 3 and ctx.settings.youtube.enabled:
            _propose_upload(s, p, ctx, job)
    return {"final": ctx.store.rel(final), "duration": info.duration}


def _propose_upload(s: Session, p: Project, ctx: StudioContext, job: Job) -> None:
    """Autonomy 3+: queue the finished video for a person's approval; nothing uploads here."""
    try:
        with s.begin_nested():
            req = publishing.propose(s, p, ctx.settings, actor="channel_manager")
        log.info("publish proposal %s", req.id)
    except publishing.PublishGateError as exc:
        record_event(s, EventType.PUBLISH_PROPOSAL_SKIPPED, project_id=p.id,
                     actor="channel_manager", job_id=job.id, data={"reason": str(exc)})


HANDLERS: dict[str, Handler] = {
    "image": image_job, "audio": audio_job, "rea": rea_job, "mesh": mesh_job,
    "extend": extend_job,
    "rights_check": rights_check,
    "ingest": ingest,
    "analyze": analyze,
    "creative_plan": creative_plan,
    "compile_workflow": compile_stage,
    "render": render,
    "qc": quality_check,
    "repair": repair,
    "edit": edit,
}
