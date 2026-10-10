"""The ``mesh`` job: reconstruct a 3D model (docs/three.md).

``image``: one picture through ComfyUI's Hunyuan3D 2.0 graph, with a GPU lease on this PC or on
the cloud server, like pictures and sound. ``photos`` and ``video``: Meshroom's
``meshroom_batch`` on a photo folder (for a video, frames taken with FFmpeg first).
``poisson``: Open3D's screened Poisson reconstruction of a point cloud. The result is read,
measured and kept as ``model.stl``; what the tool printed is kept in ``tool.log``.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from rokkur_studio.comfyui.client import (
    ComfyError,
    ComfyExecutionError,
    ComfyUnavailable,
    ComfyValidationError,
)
from rokkur_studio.comfyui.compiler import TemplateError, compile_workflow
from rokkur_studio.db.models import CostEntry, Job, Model3D, utcnow
from rokkur_studio.gpu.lease import GpuUnavailable
from rokkur_studio.jobs.errors import JobCancelled, JobError, PermanentJobError
from rokkur_studio.jobs.queue import is_cancelled
from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.mesh.io import MESH_EXTS, MeshError, read_mesh, read_points
from rokkur_studio.mesh.io import Mesh as MeshData
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services.models3d import ModelStore, _finish


def _set(ctx: StudioContext, model_id: str, **fields: Any) -> None:
    with ctx.db.transaction() as s:
        row = s.get(Model3D, model_id)
        if row is not None:
            for key, value in fields.items():
                setattr(row, key, value)


def _fail(ctx: StudioContext, model_id: str, code: str, message: str) -> None:
    _set(ctx, model_id, status="failed", error={"code": code, "message": message},
         finished_at=utcnow())


def _largest_mesh(folder: Path) -> Path | None:
    found = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in MESH_EXTS]
    return max(found, key=lambda p: p.stat().st_size, default=None)


def _image(ctx: StudioContext, job: Job, row_id: str, store: ModelStore, work: Path
           ) -> tuple[MeshData, str | None, float | None]:
    payload = job.payload
    target = str(payload.get("render_on", "local"))
    if ctx.settings.render.renderer != "comfyui":
        raise PermanentJobError("no_renderer", "Hunyuan3D needs the ComfyUI renderer")
    try:
        client = ctx.comfy_for(target)
    except ComfyError as exc:
        raise PermanentJobError("cloud_not_configured", str(exc)) from exc
    try:
        template = ctx.registry.get(str(payload["workflow"]))
    except TemplateError as exc:
        raise PermanentJobError("template", str(exc)) from exc

    def cancelled() -> bool:
        with ctx.db.session() as s:
            return is_cancelled(s, job.id)

    def run() -> tuple[MeshData, str, float | None]:
        source = store.path_for(str(payload["source"]))
        values = dict(payload.get("params") or {})
        values["SOURCE_IMAGE"] = client.upload_input(source, subfolder=f"rokkur/{client.client_id}/{row_id}")
        values["OUTPUT_PREFIX"] = f"rokkur/3d/{job.id}"
        prompt_id = client.submit(compile_workflow(template, values).workflow)
        result = client.wait(prompt_id, timeout_s=ctx.settings.comfyui.timeout_s,
                             poll_s=ctx.settings.comfyui.poll_interval_s, should_cancel=cancelled)
        outputs = [o for o in result.outputs if Path(o.filename).suffix.lower() in MESH_EXTS]
        if not outputs:
            raise PermanentJobError("no_output", f"prompt {prompt_id} produced no 3D file")
        out = client.download(outputs[0], work / f"comfy{Path(outputs[0].filename).suffix.lower()}")
        return read_mesh(out), prompt_id, result.execution_seconds

    try:
        if target == "cloud":
            return run()
        with ctx.gpu.lease(job.id, str(payload.get("resource_class", "GPU_HEAVY")),
                           should_abort=cancelled):
            return run()
    except ComfyExecutionError as exc:
        code = "oom" if exc.is_oom else "render_rejected"
        hint = (" Hunyuan3D needs about 6 GB of free VRAM; lower the octree resolution or use "
                "the cloud server.") if exc.is_oom else ""
        raise PermanentJobError(code, str(exc) + hint) from exc
    except ComfyValidationError as exc:
        raise PermanentJobError("invalid_workflow", f"{exc}: {exc.details}") from exc
    except ComfyUnavailable as exc:
        raise JobError("renderer_unavailable", str(exc)) from exc
    except ComfyError as exc:
        if exc.code == "cancelled":
            raise JobCancelled() from exc
        raise JobError(exc.code, str(exc)) from exc
    except GpuUnavailable as exc:
        raise JobError("gpu_busy", str(exc)) from exc
    finally:
        client.close()


def _photogrammetry(ctx: StudioContext, photos: Path, work: Path) -> MeshData:
    cmd = [*shlex.split(ctx.settings.three.photogrammetry_command), "--input", str(photos),
           "--output", str(work / "meshroom")]
    log = work / "tool.log"
    try:
        with log.open("w", encoding="utf-8") as fh:
            fh.write(shlex.join(cmd) + "\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False,
                                  timeout=ctx.settings.three.photogrammetry_timeout_s)
    except FileNotFoundError as exc:
        raise PermanentJobError("not_installed", f"{cmd[0]} is not installed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PermanentJobError("timeout", "Meshroom ran longer than "
                                f"{ctx.settings.three.photogrammetry_timeout_s:g} s") from exc
    tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
    if proc.returncode != 0:
        raise PermanentJobError("photogrammetry_failed",
                                f"Meshroom exited {proc.returncode}: {' / '.join(tail)}")
    found = _largest_mesh(work / "meshroom")
    if found is None:
        raise PermanentJobError("no_output", "Meshroom finished without a mesh; the photos may "
                                "not overlap enough (see tool.log)")
    return read_mesh(found)


def _video_frames(ctx: StudioContext, video: Path, work: Path, fps: float, max_frames: int
                  ) -> tuple[Path, int]:
    frames = work / "frames"
    files = ctx.ffmpeg.extract_frames(video, frames, fps=fps, pattern="frame_%05d.jpg")
    if len(files) > max_frames:  # keep an even spread across the whole video
        keep = set(np.linspace(0, len(files) - 1, max_frames).astype(int).tolist())
        for i, f in enumerate(files):
            if i not in keep:
                f.unlink()
    count = min(len(files), max_frames)
    if count < 3:
        raise PermanentJobError("too_few_frames", f"only {count} frames from the video; it may "
                                "be too short, or raise three.video_fps")
    return frames, count


def _poisson(source: Path, depth: int) -> MeshData:
    import open3d as o3d  # optional; availability() checks it before queueing

    points = read_points(source)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals()
    cloud.orient_normals_consistent_tangent_plane(15)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    d = np.asarray(densities)
    mesh.remove_vertices_by_mask(d < np.quantile(d, 0.02))  # trim the thin far-away shell
    return MeshData(np.asarray(mesh.vertices), np.asarray(mesh.triangles), source.stem)


def mesh_job(ctx: StudioContext, job: Job) -> dict[str, Any]:
    payload = job.payload
    row_id = str(payload["model_id"])
    method = str(payload["method"])
    store = ModelStore(ctx.settings.studio.data_dir)
    with ctx.db.session() as s:
        if s.get(Model3D, row_id) is None:
            return {"model": row_id, "skipped": "deleted"}
    _set(ctx, row_id, status="running", error=None)
    work = store.dir(row_id) / "work"
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prompt_id: str | None = None
    seconds: float | None = None
    try:
        if method == "image":
            mesh, prompt_id, seconds = _image(ctx, job, row_id, store, work)
        elif method == "photos":
            mesh = _photogrammetry(ctx, store.path_for(str(payload["source"])), work)
        elif method == "video":
            frames, count = _video_frames(ctx, Path(str(payload["video"])), work,
                                          float(payload.get("fps", 2.0)),
                                          int(payload.get("max_frames", 150)))
            _set(ctx, row_id, params={"fps": payload.get("fps"), "frames": count})
            mesh = _photogrammetry(ctx, frames, work)
        elif method == "poisson":
            mesh = _poisson(store.path_for(str(payload["source"])), int(payload.get("depth", 9)))
        else:
            raise PermanentJobError("unknown_method", method)
    except JobCancelled:
        _fail(ctx, row_id, "cancelled", "Cancelled by the user")
        raise
    except PermanentJobError as exc:
        _fail(ctx, row_id, exc.code, str(exc))
        raise
    except JobError as exc:
        _set(ctx, row_id, status="queued", error={"code": exc.code, "message": str(exc)})
        raise
    except (MeshError, FFmpegError, ValueError, OSError) as exc:
        message = exc.summary if isinstance(exc, FFmpegError) else str(exc)
        _fail(ctx, row_id, "unreadable", message)
        raise PermanentJobError("unreadable", message) from exc
    elapsed = time.monotonic() - started
    with ctx.db.transaction() as s:
        row = s.get(Model3D, row_id)
        if row is None:
            return {"model": row_id, "skipped": "deleted"}
        try:
            _finish(row, store, mesh)
        except MeshError as exc:
            row.status, row.error, row.finished_at = "failed", {"code": "empty",
                                                                "message": str(exc)}, utcnow()
            return {"model": row_id, "error": str(exc)}
        row.remote_id = prompt_id
        row.took_s = round(seconds if seconds is not None else elapsed, 2)
        if method == "image":
            minutes = (row.took_s or 0) / 60
            if row.render_on == "cloud":
                s.add(CostEntry(project_id=row.project_id, job_id=job.id, kind="cloud_gpu_minutes",
                                amount=minutes, unit="min",
                                usd=minutes / 60 * ctx.settings.cloud.price_per_hour_usd))
            else:
                s.add(CostEntry(project_id=row.project_id, job_id=job.id, kind="gpu_minutes",
                                amount=minutes, unit="min"))
    return {"model": row_id, "method": method, "seconds": round(elapsed, 2)}
