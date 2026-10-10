"""The ``audio`` job: make one request's clips through ComfyUI (docs/audio.md).

Same shape as the ``image`` job: a GPU lease of the profile's class for local jobs, the cloud
server for cloud jobs, and a CUDA-OOM ladder (``clear_cache``, ``single_clip``, ``shorter``)
that never resubmits identical work. ComfyUI's SaveAudio writes FLAC; every clip is kept as
``clip.flac`` with its length, rate and channels measured by ffprobe.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from rokkur_studio.comfyui.client import (
    ComfyClient,
    ComfyError,
    ComfyExecutionError,
    ComfyUnavailable,
    ComfyValidationError,
)
from rokkur_studio.comfyui.compiler import TemplateError, compile_workflow
from rokkur_studio.db.models import AudioClip, CostEntry, Job, utcnow
from rokkur_studio.gpu.lease import GpuUnavailable
from rokkur_studio.jobs.errors import JobCancelled, JobError, PermanentJobError
from rokkur_studio.jobs.queue import is_cancelled
from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services.audio import AudioStore

log = logging.getLogger(__name__)
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a")
MIN_SECONDS = 5.0


def audio_store(ctx: StudioContext) -> AudioStore:
    return AudioStore(ctx.settings.studio.data_dir)


def _rows(ctx: StudioContext, ids: list[str]) -> list[AudioClip]:
    with ctx.db.session() as s:
        rows = [s.get(AudioClip, i) for i in ids]
        found = [r for r in rows if r is not None]
        for r in found:
            s.expunge(r)
    return found


def _set(ctx: StudioContext, ids: list[str], **fields: Any) -> None:
    with ctx.db.transaction() as s:
        for i in ids:
            row = s.get(AudioClip, i)
            if row is not None:
                for key, value in fields.items():
                    setattr(row, key, value)


def _fail(ctx: StudioContext, ids: list[str], code: str, message: str) -> None:
    _set(ctx, ids, status="failed", error={"code": code, "message": message},
         finished_at=utcnow())


def _degrade(step: str, params: dict[str, Any], client: ComfyClient) -> bool:
    """Apply one OOM recovery step to ``params``; False when it does not apply here."""
    if step == "clear_cache":
        with contextlib.suppress(ComfyError):
            client.free()
        return True
    if step == "single_clip":
        if int(params.get("BATCH", 1)) <= 1:
            return False
        params["BATCH"] = 1
        return True
    if step == "shorter":
        seconds = float(params.get("SECONDS", 0))
        if seconds <= MIN_SECONDS:
            return False
        params["SECONDS"] = max(MIN_SECONDS, round(seconds / 2, 1))
        return True
    return False


def audio_job(ctx: StudioContext, job: Job) -> dict[str, Any]:
    payload = job.payload
    ids: list[str] = list(payload["clip_ids"])
    target = str(payload.get("render_on", "local"))
    workflow = str(payload["workflow"])
    rows = _rows(ctx, ids)
    ids = [r.id for r in rows]
    if not rows:
        return {"clips": [], "skipped": "deleted"}
    store = audio_store(ctx)
    if ctx.settings.render.renderer != "comfyui":
        _fail(ctx, ids, "no_renderer", "Sound generation needs the ComfyUI renderer")
        raise PermanentJobError("no_renderer", "render.renderer is not comfyui")
    try:
        client = ctx.comfy_for(target)
    except ComfyError as exc:
        _fail(ctx, ids, "cloud_not_configured", str(exc))
        raise PermanentJobError("cloud_not_configured", str(exc)) from exc
    try:
        template = ctx.registry.get(workflow)
    except TemplateError as exc:
        _fail(ctx, ids, "template", str(exc))
        raise PermanentJobError("template", str(exc)) from exc
    _set(ctx, ids, status="running", error=None)
    params: dict[str, Any] = dict(payload["params"])
    params["BATCH"] = len(ids)
    params["OUTPUT_PREFIX"] = f"rokkur/audio/{job.id}"
    work = store.dir(ids[0]) / "comfy"
    work.mkdir(parents=True, exist_ok=True)
    ladder = list(payload.get("degrade") or [])
    applied: list[str] = []
    started = time.monotonic()

    def cancelled() -> bool:
        with ctx.db.session() as s:
            return is_cancelled(s, job.id)

    def submit(values: dict[str, Any]) -> tuple[list[Path], str, float | None]:
        compiled = compile_workflow(template, values)
        prompt_id = client.submit(compiled.workflow)
        result = client.wait(prompt_id, timeout_s=ctx.settings.comfyui.timeout_s,
                             poll_s=ctx.settings.comfyui.poll_interval_s, should_cancel=cancelled)
        outputs = sorted((o for o in result.outputs
                          if Path(o.filename).suffix.lower() in AUDIO_SUFFIXES),
                         key=lambda o: o.filename)
        if not outputs:
            raise PermanentJobError("no_output", f"prompt {prompt_id} produced no sound")
        files = [client.download(o, work / f"out_{i:02d}{Path(o.filename).suffix.lower()}")
                 for i, o in enumerate(outputs, 1)]
        return files, prompt_id, result.execution_seconds

    def run_all() -> list[tuple[Path, str, float | None]]:
        made: list[tuple[Path, str, float | None]] = []
        while len(made) < len(ids):
            remaining = len(ids) - len(made)
            values = dict(params)
            values["BATCH"] = min(int(params["BATCH"]), remaining)
            values["SEED"] = int(params["SEED"]) + len(made)
            try:
                files, prompt_id, seconds = submit(values)
            except ComfyExecutionError as exc:
                if not exc.is_oom:
                    raise PermanentJobError("render_rejected", str(exc)) from exc
                step = next((s for s in ladder if s not in applied), None)
                while step is not None and not _degrade(step, params, client):
                    applied.append(step)
                    step = next((s for s in ladder if s not in applied), None)
                if step is None:
                    raise PermanentJobError(
                        "oom_unrecoverable", "CUDA out of memory after the full recovery "
                        "ladder; try a shorter clip, one clip at a time, or the cloud "
                        "server") from exc
                applied.append(step)
                log.warning("audio oom, degrading", extra={"data": {"step": step}})
                continue
            for f in files[:remaining]:
                made.append((f, prompt_id, seconds))
        return made

    try:
        if target == "cloud" or payload.get("resource_class") == "GPU_LIGHT":
            made = run_all()
        else:
            with ctx.gpu.lease(job.id, str(payload.get("resource_class", "GPU_HEAVY")),
                               should_abort=cancelled):
                made = run_all()
    except JobCancelled:
        _fail(ctx, ids, "cancelled", "Cancelled by the user")
        raise
    except PermanentJobError as exc:
        _fail(ctx, ids, exc.code, str(exc))
        raise
    except ComfyValidationError as exc:
        _fail(ctx, ids, "invalid_workflow", f"{exc}: {exc.details}")
        raise PermanentJobError("invalid_workflow", str(exc)) from exc
    except TemplateError as exc:
        _fail(ctx, ids, "template", str(exc))
        raise PermanentJobError("template", str(exc)) from exc
    except ComfyUnavailable as exc:
        _set(ctx, ids, status="queued", error={"code": "unavailable", "message": str(exc)})
        raise JobError("renderer_unavailable", str(exc)) from exc
    except ComfyError as exc:
        if exc.code == "cancelled":
            _fail(ctx, ids, "cancelled", "Cancelled by the user")
            raise JobCancelled() from exc
        _set(ctx, ids, status="queued", error={"code": exc.code, "message": str(exc)})
        raise JobError(exc.code, str(exc)) from exc
    except GpuUnavailable as exc:
        _set(ctx, ids, status="queued", error={"code": "gpu_busy", "message": str(exc)})
        raise JobError("gpu_busy", str(exc)) from exc
    finally:
        client.close()
    elapsed = time.monotonic() - started
    with ctx.db.transaction() as s:
        for row_id, (file, prompt_id, seconds) in zip(ids, made, strict=False):
            row = s.get(AudioClip, row_id)
            if row is None:
                continue
            dest = store.dir(row_id) / "clip.flac"
            try:
                ctx.ffmpeg.transcode_audio(file, dest)
                info = ctx.ffmpeg.audio_info(dest)
            except FFmpegError as exc:
                row.status, row.finished_at = "failed", utcnow()
                row.error = {"code": "unreadable", "message": exc.summary}
                continue
            row.rel_path = store.rel(dest)
            row.duration_s = round(info.duration, 3)
            row.sample_rate, row.channels = info.sample_rate, info.channels
            row.status, row.remote_id, row.finished_at = "done", prompt_id, utcnow()
            row.took_s = round(seconds if seconds is not None else elapsed / len(ids), 2)
            row.params = {**row.params, "SECONDS": params["SECONDS"], "_oom_steps": applied} \
                if applied else {**row.params}
            row.error = None
            minutes = (row.took_s or 0) / 60
            if target == "cloud":
                s.add(CostEntry(project_id=row.project_id, job_id=job.id,
                                kind="cloud_gpu_minutes", amount=minutes, unit="min",
                                usd=minutes / 60 * ctx.settings.cloud.price_per_hour_usd))
            else:
                s.add(CostEntry(project_id=row.project_id, job_id=job.id, kind="gpu_minutes",
                                amount=minutes, unit="min"))
    return {"clips": ids, "seconds": round(elapsed, 2), "oom_steps": applied}
