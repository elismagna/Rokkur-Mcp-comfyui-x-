"""Extend a video: Wan VACE continues a clip past its last frames (docs/video-tools.md).

Works on a clip in the media folder before a video is made (the extended clip lands next to
the original and New video can pick it) and on a finished video after its render (the
extended cut becomes the project's latest final). The ``extend`` job builds VACE's control
video (the clip's last frames, then white frames) and mask video (black over the kept frames,
white over the new ones) with FFmpeg, renders through the ordinary ComfyUI renderer, trims the
overlap and joins the continuation to the original.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from rokkur_studio.comfyui.compiler import TemplateError
from rokkur_studio.config import Settings
from rokkur_studio.db.models import Asset, CostEntry, Job, Project
from rokkur_studio.gpu.lease import GpuUnavailable
from rokkur_studio.jobs.errors import JobCancelled, JobError, PermanentJobError
from rokkur_studio.jobs.queue import enqueue
from rokkur_studio.manifest.builder import fit_within
from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.pipeline.renderers import RenderOOM, RenderRejected, RenderUnavailable
from rokkur_studio.services.assets import register_asset
from rokkur_studio.services.events import EventType, record_event
from rokkur_studio.services.projects import get_project, latest_document

log = logging.getLogger(__name__)

EXTEND_WORKFLOW = "v2v_3070_extend"
OVERLAP_FRAMES = 9          # kept frames the continuation starts from (4n+1 friendly)
MAX_SECONDS = 8.0
VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


class ExtendRequest(BaseModel):
    """What to extend and how. One of ``project_id`` (its latest final) or ``source_path``."""

    project_id: str | None = None
    source_path: str | None = None
    seconds: float = Field(3.0, gt=0, le=MAX_SECONDS)
    prompt: str = Field("", max_length=2000)
    negative_prompt: str | None = Field(None, max_length=2000)
    profile: str | None = None
    seed: int | None = Field(None, ge=0, le=4294967295)
    steps: int | None = Field(None, ge=4, le=60)
    reference_image_path: str | None = None   # a picture on the worker, e.g. from the library
    render_on: str | None = None
    rights_confirmed: bool = False            # for a clip in the media folder
    rights_evidence: str = Field("", max_length=1000)


def plan_frames(seconds: float, fps: float, max_frames: int, multiple: int = 4) -> tuple[int, int]:
    """``(frame_count, new_frames)``: the VACE length (overlap + new, 4n+1) and the new frames."""
    limit = (max_frames - 1) // multiple * multiple + 1
    wanted = OVERLAP_FRAMES + max(1, round(seconds * fps))
    total = min(limit, (wanted - 1 + multiple - 1) // multiple * multiple + 1)
    return total, total - OVERLAP_FRAMES


def request_extension(session: Session, settings: Settings, request: ExtendRequest, *,
                      actor: str = "api") -> Job:
    """Validate and queue one ``extend`` job."""
    if settings.render.renderer != "comfyui":
        raise ValueError("Extending a video needs the ComfyUI renderer (render.renderer: comfyui).")
    target = settings.new_project_target(request.render_on)
    profile_name = request.profile or settings.render.default_profile
    if problem := settings.profile_problem(profile_name, target=target):
        raise ValueError(problem)
    payload: dict[str, Any] = {"seconds": request.seconds, "prompt": request.prompt.strip(),
                               "negative_prompt": request.negative_prompt, "profile": profile_name,
                               "seed": request.seed, "steps": request.steps,
                               "reference": request.reference_image_path, "render_on": target,
                               "actor": actor}
    project_id = None
    if request.project_id:
        project = get_project(session, request.project_id)
        finals = [a for a in session.query(Asset).filter_by(project_id=project.id, kind="final")
                  .order_by(Asset.created_at)]
        if not finals:
            raise ValueError("This video has no finished render to extend yet.")
        payload["asset_id"] = finals[-1].id
        project_id = project.id
        if not payload["prompt"]:
            brief = latest_document(session, project.id, "creative_brief")
            payload["prompt"] = (brief.data.get("prompt") if brief else None) or \
                project.creative_input.get("theme", "")
    elif request.source_path:
        path = Path(request.source_path)
        if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
            raise ValueError("Pick a video file the worker can read.")
        if not request.rights_confirmed or not request.rights_evidence.strip():
            raise ValueError("Confirm that you may use and transform this clip and say where it "
                             "comes from.")
        payload["source_path"] = str(path)
        payload["rights_evidence"] = request.rights_evidence.strip()
    else:
        raise ValueError("Choose a finished video or a clip to extend.")
    if not payload["prompt"]:
        raise ValueError("Describe what the continuation should show.")
    job = enqueue(session, "extend", project_id=project_id, payload=payload, priority=95,
                  max_attempts=settings.jobs.default_max_attempts)
    assert job is not None
    if project_id:
        record_event(session, EventType.EXTENSION_REQUESTED, project_id=project_id, actor=actor,
                     job_id=job.id, data={"seconds": request.seconds, "prompt": payload["prompt"]})
    return job


def _control_videos(ctx: StudioContext, source: Path, work: Path, *, width: int, height: int,
                    fps: float, frame_count: int) -> tuple[Path, Path, Path]:
    """VACE's control video, its mask video and the clip's last frame as a picture."""
    info = ctx.ffmpeg.probe(source)
    tail_start = max(0.0, info.duration - (OVERLAP_FRAMES + 2) / fps)
    tail = ctx.ffmpeg.cut(source, work / "tail.mp4", start=tail_start, end=info.duration, fps=fps)
    frames = ctx.ffmpeg.read_rgb_frames(tail, width, height, fps=fps, aspect=(width, height))
    if len(frames) == 0:
        raise RenderRejected("the clip has no frames to continue from")
    known = frames[-OVERLAP_FRAMES:]
    if len(known) < OVERLAP_FRAMES:  # a very short clip: hold its last frame
        known = np.concatenate([np.repeat(known[-1:], OVERLAP_FRAMES - len(known), axis=0), known])
    white = np.full((frame_count - OVERLAP_FRAMES, height, width, 3), 255, np.uint8)
    control = ctx.ffmpeg.write_frames(np.concatenate([known, white]), work / "control.mp4", fps)
    mask = np.zeros((frame_count, height, width), np.uint8)
    mask[OVERLAP_FRAMES:] = 255
    mask_video = ctx.ffmpeg.write_frames(mask, work / "mask.mp4", fps)
    last = ctx.ffmpeg.write_image(known[-1], work / "last_frame.png")
    return control, mask_video, last


def extend_job(ctx: StudioContext, job: Job) -> dict[str, Any]:
    from rokkur_studio.pipeline.stages import _renderer

    payload = job.payload
    target = str(payload.get("render_on", "local"))
    profile = ctx.settings.profile(str(payload["profile"]))
    if payload.get("asset_id"):
        with ctx.db.session() as s:
            asset = s.get(Asset, payload["asset_id"])
            if asset is None:
                raise PermanentJobError("asset_missing", "the final video is gone")
            source = ctx.store.path_for(asset.rel_path)
    else:
        source = Path(str(payload["source_path"]))
        if not source.is_file():
            raise PermanentJobError("source_missing", f"the clip is gone: {source}")
    info = ctx.ffmpeg.probe(source)
    fps = float(profile.fps)
    width, height = fit_within(info.width, info.height, profile.max_width, profile.max_height,
                               max_pixels=profile.max_pixels)
    frame_count, new_frames = plan_frames(float(payload["seconds"]), fps, profile.max_frames,
                                          profile.frame_multiple)
    work = (ctx.store.project_dir(job.project_id, f"work/extend_{job.id[-8:]}") if job.project_id
            else Path(ctx.settings.studio.data_dir) / "videos" / job.id)
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        control, mask_video, last_frame = _control_videos(
            ctx, source, work, width=width, height=height, fps=fps, frame_count=frame_count)
    except FFmpegError as exc:
        raise PermanentJobError("ffmpeg", f"could not read the clip: {exc.summary}") from exc
    params: dict[str, Any] = {
        "STYLE_PROMPT": str(payload["prompt"]),
        "NEGATIVE_PROMPT": payload.get("negative_prompt") or profile.negative_base or "",
        "SEED": int(payload.get("seed") or 0) or int(job.id[-8:], 16) % 2**31,
        "WIDTH": width, "HEIGHT": height, "FPS": fps, "FRAME_COUNT": frame_count,
        "STEPS": int(payload.get("steps") or profile.steps), "CFG": 6.0,
        "CONTROL_STRENGTH": 1.0, "MASK_VIDEO": str(mask_video), "_OUTPUT_FRAMES": frame_count,
        "_REFERENCE_MODE": "source",
    }
    reference = payload.get("reference")
    if reference and Path(str(reference)).is_file():
        params["REFERENCE_IMAGE"] = str(reference)
        params["_REFERENCE_KIND"] = "picture"
    renderer = _renderer(ctx, job, target)
    if renderer.name != "comfyui":
        raise PermanentJobError("no_renderer", "extending needs the ComfyUI renderer")
    rendered = work / "continuation_raw.mp4"
    try:
        if target == "cloud":
            outcome = renderer.render_shot(clip=control, params=params, workflow=EXTEND_WORKFLOW,
                                           out=rendered)
        else:
            with ctx.gpu.lease(job.id, profile.resource_class):
                outcome = renderer.render_shot(clip=control, params=params,
                                               workflow=EXTEND_WORKFLOW, out=rendered)
    except RenderOOM as exc:
        raise PermanentJobError("oom", f"CUDA out of memory: {exc}; ask for fewer seconds or "
                                       "use the cloud server") from exc
    except RenderUnavailable as exc:
        raise JobError("renderer_unavailable", str(exc)) from exc
    except (RenderRejected, TemplateError) as exc:
        raise PermanentJobError("render_rejected", str(exc)) from exc
    except GpuUnavailable as exc:
        raise JobError("gpu_busy", str(exc)) from exc
    except JobCancelled:
        raise
    # drop the kept frames, match the original's size and rate, and join
    continuation = ctx.ffmpeg.filter_video(
        rendered, work / "continuation.mp4",
        f"trim=start_frame={OVERLAP_FRAMES},setpts=PTS-STARTPTS,scale={info.width}:{info.height},"
        f"fps={info.fps or fps}", fps=info.fps or fps)
    original = ctx.ffmpeg.filter_video(source, work / "original.mp4",
                                       f"scale={info.width}:{info.height},fps={info.fps or fps}",
                                       fps=info.fps or fps)
    joined = ctx.ffmpeg.concat([original, continuation], work / "extended_silent.mp4")
    if info.has_audio:
        joined = ctx.ffmpeg.attach_audio(joined, source, work / "extended.mp4", pad=True)
    seconds = time.monotonic() - started
    result: dict[str, Any] = {"frames_added": new_frames, "took_s": round(seconds, 1),
                              "workflow": EXTEND_WORKFLOW, "width": width, "height": height,
                              "prompt_id": outcome.remote_id}
    if job.project_id:
        final_dir = ctx.store.project_dir(job.project_id, "final")
        version = len(list(final_dir.glob("*.mp4"))) + 1
        dest = final_dir / f"final_extended_v{version}.mp4"
        dest.write_bytes(Path(joined).read_bytes())
        with ctx.db.transaction() as s:
            asset = register_asset(s, ctx.store, job.project_id, "final", dest,
                                   meta={"extended": True, "seconds": payload["seconds"],
                                         "from_asset": payload.get("asset_id"), **result})
            minutes = (outcome.seconds or 0) / 60
            s.add(CostEntry(project_id=job.project_id, job_id=job.id,
                            kind="cloud_gpu_minutes" if target == "cloud" else "gpu_minutes",
                            amount=minutes, unit="min",
                            usd=(minutes / 60 * ctx.settings.cloud.price_per_hour_usd
                                 if target == "cloud" else 0.0)))
            record_event(s, EventType.VIDEO_EXTENDED, project_id=job.project_id, actor="extend",
                         job_id=job.id, data={**result, "asset_id": asset.id})
            result["asset_id"] = asset.id
    else:
        dest = source.with_name(f"{source.stem}_extended_{job.id[-6:]}.mp4")
        dest.write_bytes(Path(joined).read_bytes())
        result["path"] = str(dest)
        with ctx.db.transaction() as s:
            s.add(CostEntry(project_id=None, job_id=job.id,
                            kind="cloud_gpu_minutes" if target == "cloud" else "gpu_minutes",
                            amount=(outcome.seconds or 0) / 60, unit="min"))
    return result


def latest_extension(session: Session, project: Project) -> Asset | None:
    rows = [a for a in session.query(Asset).filter_by(project_id=project.id, kind="final")
            if a.meta.get("extended")]
    return rows[-1] if rows else None
