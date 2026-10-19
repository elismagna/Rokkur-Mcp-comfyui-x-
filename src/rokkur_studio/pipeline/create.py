"""The ``create_take`` job: render one Create-page take (services/create.py, docs/create.md).

On ComfyUI the take's workflow gets its prompt, size, seed and (to animate) start picture,
runs under the GPU lease on this PC or without it on the cloud server, and the result is
saved under ``<data_dir>/create/<take id>/``. With the ``ffmpeg_preview`` renderer (no GPU)
a labelled test pattern stands in, so the page and the queue can be tried anywhere.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rokkur_studio.comfyui.client import (
    ComfyError,
    ComfyExecutionError,
    ComfyUnavailable,
    ComfyValidationError,
)
from rokkur_studio.comfyui.compiler import TemplateError, compile_workflow
from rokkur_studio.db.models import CostEntry, Job, Take, utcnow
from rokkur_studio.gpu.lease import GpuUnavailable
from rokkur_studio.jobs.errors import JobCancelled, JobError, PermanentJobError
from rokkur_studio.jobs.queue import is_cancelled
from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import create

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".gif")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def run_take(ctx: StudioContext, job: Job) -> dict[str, Any]:
    take_id = str((job.payload or {}).get("take_id") or "")
    with ctx.db.transaction() as s:
        take = s.get(Take, take_id)
        if take is None:  # deleted while it waited
            return {"skipped": "deleted"}
        take.status, take.job_id, take.error = "running", job.id, None
        mode, workflow, target = take.mode, take.workflow, take.render_on
        params = dict((take.settings or {}).get("params") or {})
        params.update(PROMPT=take.final_prompt, NEGATIVE_PROMPT=take.negative_prompt or None)
        start = create.path_for(ctx.settings, take.start_image) if take.start_image else None
    folder = create.create_dir(ctx.settings) / take_id
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        if ctx.settings.render.renderer == "comfyui":
            output, remote_id = _comfy(ctx, job, workflow, target, params, start, folder)
        else:
            output, remote_id = _stand_in(ctx, mode, params, folder), None
        thumb = output if mode == "image" else ctx.ffmpeg.thumbnail(
            output, folder / "thumb.jpg", at=min(1.0, _seconds(params) / 2), width=480)
    except PermanentJobError as exc:
        _failed(ctx, take_id, str(exc), final=True)
        raise
    except JobCancelled:
        _failed(ctx, take_id, "Cancelled", final=True, status="cancelled")
        raise
    except JobError as exc:
        _failed(ctx, take_id, str(exc), final=False)
        raise
    except FFmpegError as exc:
        _failed(ctx, take_id, f"FFmpeg could not read the result: {exc}", final=True)
        raise PermanentJobError("create_ffmpeg", str(exc)) from exc
    seconds = time.monotonic() - started
    root = create.create_dir(ctx.settings)
    with ctx.db.transaction() as s:
        row = s.get(Take, take_id)
        if row is None:  # deleted while it rendered: drop the files it left
            for p in folder.glob("*"):
                p.unlink(missing_ok=True)
            return {"skipped": "deleted"}
        row.status, row.error = "succeeded", None
        row.output = output.relative_to(root).as_posix()
        row.thumbnail = thumb.relative_to(root).as_posix()
        row.remote_id, row.duration_s, row.finished_at = remote_id, seconds, utcnow()
        if ctx.settings.render.renderer == "comfyui":
            cloud = target == "cloud"
            minutes = seconds / 60
            s.add(CostEntry(job_id=job.id, kind="cloud_gpu_minutes" if cloud else "gpu_minutes",
                            amount=minutes, unit="min",
                            usd=minutes / 60 * ctx.settings.cloud.price_per_hour_usd
                            if cloud else 0.0))
    return {"take_id": take_id, "output": output.name, "seconds": round(seconds, 2)}


def _seconds(params: dict[str, Any]) -> float:
    return float(params.get("FRAME_COUNT", 1)) / float(params.get("FPS", 16) or 16)


def _failed(ctx: StudioContext, take_id: str, message: str, *, final: bool,
            status: str = "failed") -> None:
    with ctx.db.transaction() as s:
        row = s.get(Take, take_id)
        if row is not None:
            row.status = status if final else "retrying"
            row.error = message
            if final:
                row.finished_at = utcnow()


def _comfy(ctx: StudioContext, job: Job, workflow: str, target: str, params: dict[str, Any],
           start: Path | None, folder: Path) -> tuple[Path, str]:
    from rokkur_studio.pipeline.stages import _live_problem  # stages imports this package

    try:
        template = ctx.registry.get(workflow)
    except TemplateError as exc:
        raise PermanentJobError("create_workflow", str(exc)) from exc
    if problem := _live_problem(ctx, workflow, target):
        raise PermanentJobError("create_workflow", f"{workflow} cannot run here: {problem}. "
                                "Run comfy-check to see which file or node is missing.")
    try:
        client = ctx.comfy_for(target)
    except ComfyError as exc:
        raise PermanentJobError("cloud_not_configured", str(exc)) from exc

    def cancelled() -> bool:
        with ctx.db.session() as s:
            return is_cancelled(s, job.id)

    accepted = template.spec.parameters
    values = {k: v for k, v in params.items() if k in accepted and v is not None}
    try:
        if start is not None:
            values["START_IMAGE"] = client.upload_input(
                start, subfolder=f"rokkur/{client.client_id}/{folder.name}")
        compiled = compile_workflow(template, values)
        if target == "cloud":
            prompt_id, result = _run(ctx, client, compiled.workflow, cancelled)
        else:
            with ctx.gpu.lease(job.id, template.spec.resource_class):
                prompt_id, result = _run(ctx, client, compiled.workflow, cancelled)
        videos = [o for o in result.outputs if Path(o.filename).suffix.lower() in VIDEO_EXTS]
        images = [o for o in result.outputs if Path(o.filename).suffix.lower() in IMAGE_EXTS]
        pick = videos[-1] if videos else images[-1] if images else None
        if pick is None:
            raise PermanentJobError("create_no_output",
                                    f"{workflow} finished but saved no picture or video")
        out = client.download(pick, folder / f"output{Path(pick.filename).suffix.lower()}")
        return out, prompt_id
    except TemplateError as exc:
        raise PermanentJobError("create_params", str(exc)) from exc
    except ComfyExecutionError as exc:
        if exc.is_oom:
            raise PermanentJobError(
                "create_oom", "The GPU ran out of memory. Try a smaller size or a shorter "
                "clip, or render on the cloud server.") from exc
        raise PermanentJobError("create_failed", str(exc)) from exc
    except ComfyValidationError as exc:
        raise PermanentJobError("create_rejected", f"{exc}: {exc.details}") from exc
    except ComfyUnavailable as exc:
        raise JobError("renderer_unavailable", str(exc)) from exc
    except ComfyError as exc:
        if exc.code == "cancelled":
            raise JobCancelled() from exc
        raise JobError("renderer_unavailable", str(exc)) from exc
    except GpuUnavailable as exc:
        raise JobError("gpu_busy", str(exc)) from exc
    finally:
        client.close()


def _run(ctx: StudioContext, client: Any, workflow: dict[str, Any], cancelled: Any
         ) -> tuple[str, Any]:
    prompt_id = client.submit(workflow)
    result = client.wait(prompt_id, timeout_s=ctx.settings.comfyui.timeout_s,
                         poll_s=ctx.settings.comfyui.poll_interval_s, should_cancel=cancelled)
    return prompt_id, result


def _stand_in(ctx: StudioContext, mode: str, params: dict[str, Any], folder: Path) -> Path:
    """A test pattern at the take's size: the preview renderer makes no real pictures."""
    width, height = int(params["WIDTH"]), int(params["HEIGHT"])
    if mode == "image":
        return ctx.ffmpeg.test_pattern(folder / "output.png", width=width, height=height)
    return ctx.ffmpeg.test_pattern(folder / "output.mp4", width=width, height=height,
                                   frames=int(params["FRAME_COUNT"]), fps=int(params["FPS"]))
