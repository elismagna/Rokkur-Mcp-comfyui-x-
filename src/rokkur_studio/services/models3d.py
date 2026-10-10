"""The 3D studio: import, measure, edit, convert and make printable models (docs/three.md).

Everything that runs on the CPU happens at once (import, edits, combining, conversion, a
relief from a picture, a 2.5D solid from LiDAR points). Reconstruction that needs a model or
an external tool is queued as a ``mesh`` job (``pipeline/models3d.py``): one picture through
ComfyUI's Hunyuan3D 2.0 (this PC or the cloud server), a photo set or a video through
Meshroom, a point cloud through Open3D's Poisson reconstruction. A method whose tool is not
installed says so and is not offered; nothing is faked.
"""

from __future__ import annotations

import importlib.util
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.config import Settings
from rokkur_studio.db.models import Image, Job, Model3D, utcnow
from rokkur_studio.jobs.queue import enqueue
from rokkur_studio.media.ffmpeg import FFmpeg, FFmpegError
from rokkur_studio.mesh import ops
from rokkur_studio.mesh.io import (
    MESH_EXTS,
    POINT_EXTS,
    WRITE_EXTS,
    Mesh,
    MeshError,
    read_mesh,
    read_points,
    write_mesh,
)

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
MAX_SEED = 2**53 - 1
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
EditOp = Literal["scale", "fit", "rotate", "mirror", "center", "flip", "repair", "combine"]
ReconMethod = Literal["image", "photos", "video", "poisson"]

# Ways to make a model, in the order the page shows them, with what each needs.
METHODS: dict[str, dict[str, str]] = {
    "upload": {"label": "Open a model", "needs": "an STL, OBJ, PLY or GLB file",
               "how": "Read and measured on the CPU; kept as STL."},
    "relief": {"label": "Picture to relief or lithophane", "needs": "a picture",
               "how": "Brightness becomes height on a solid base (CPU, at once). Inverted for a "
                      "backlit lithophane."},
    "points": {"label": "LiDAR or point cloud to solid (2.5D)",
               "needs": "a PLY, OBJ, XYZ, PTS or CSV point list",
               "how": "The highest point per grid cell becomes the surface of a closed solid "
                      "(CPU, at once). Right for terrain, walls and reliefs; not for the back "
                      "of an object."},
    "poisson": {"label": "Point cloud to surface (Poisson)", "needs": "a point cloud; Open3D",
                "how": "Open3D estimates normals and reconstructs a closed surface around "
                       "the points (CPU job)."},
    "image": {"label": "One picture to a 3D shape (AI)", "needs": "a picture; Hunyuan3D 2.0 in ComfyUI",
              "how": "Hunyuan3D 2.0 imagines the whole shape from one view (GPU job, on this "
                     "PC or the cloud server). Untextured."},
    "photos": {"label": "Photo set to model (photogrammetry)",
               "needs": "10+ overlapping photos; Meshroom",
               "how": "Meshroom (AliceVision) matches the photos and rebuilds the real "
                      "surface (CPU/GPU job, can take an hour)."},
    "video": {"label": "Video to model (photogrammetry)", "needs": "a walk-around video; Meshroom",
              "how": "Frames are taken from the video with FFmpeg and given to Meshroom."},
}


class ModelStore:
    """Files of the 3D studio, under ``<data_dir>/3d``."""

    def __init__(self, data_dir: Path) -> None:
        self.root = (Path(data_dir) / "3d").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, model_id: str) -> Path:
        if not _SAFE.match(model_id):
            raise ValueError(f"unsafe model id {model_id!r}")
        path = self.root / model_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path_for(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if self.root not in path.parents:
            raise ValueError(f"path escapes the 3D folder: {rel_path!r}")
        return path

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()


def availability(settings: Settings, target: str = "local") -> dict[str, str | None]:
    """Why each way of making a model cannot run now, or None when it can."""
    out: dict[str, str | None] = {"upload": None, "relief": None, "points": None}
    out["poisson"] = (None if importlib.util.find_spec("open3d") is not None else
                      "Open3D is not installed where the worker runs: `pip install open3d` "
                      "(MIT) in the studio's environment.")
    if settings.render.renderer != "comfyui":
        out["image"] = "Needs the ComfyUI renderer (render.renderer: comfyui)."
    elif target == "cloud" and not settings.cloud.ready:
        out["image"] = "Cloud rendering is not set up (docs/cloud.md)."
    else:
        from rokkur_studio.comfyui.compiler import TemplateError, TemplateRegistry

        try:
            TemplateRegistry(settings.workflows_dir).get(settings.three.image_workflow)
            out["image"] = None
        except (TemplateError, OSError, ValueError) as exc:
            out["image"] = str(exc)
    cmd = settings.three.photogrammetry_command.strip()
    problem = None
    if not cmd:
        problem = "Photogrammetry is turned off (three.photogrammetry_command is empty)."
    elif shutil.which(cmd.split()[0]) is None:
        problem = (f"`{cmd.split()[0]}` is not on the worker's PATH. Install Meshroom "
                   "(alicevision.org, MPL-2.0) and put its folder on PATH, or set "
                   "three.photogrammetry_command to the full path of meshroom_batch.")
    out["photos"] = out["video"] = problem
    return out


def _finish(row: Model3D, store: ModelStore, mesh: Mesh) -> None:
    if len(mesh.faces) == 0:
        raise MeshError("the result has no triangles")
    out = write_mesh(mesh, store.dir(row.id) / "model.stl")
    row.rel_path = store.rel(out)
    row.stats = ops.measure(mesh)
    row.status, row.error, row.finished_at = "done", None, utcnow()


def _new(session: Session, **fields: Any) -> Model3D:
    row = Model3D(status="running", render_on="local", **fields)
    session.add(row)
    session.flush()
    return row


def _fail_clean(session: Session, store: ModelStore, row: Model3D) -> None:
    session.delete(row)
    session.flush()
    shutil.rmtree(store.root / row.id, ignore_errors=True)


def load(store: ModelStore, row: Model3D) -> Mesh:
    if not row.rel_path:
        raise LookupError(f"model {row.id} has no file yet")
    return read_mesh(store.path_for(row.rel_path))


def _check_rights(confirmed: bool, evidence: str, what: str) -> None:
    if not confirmed or not evidence.strip():
        raise ValueError(f"Confirm that you may use {what} and say where it comes from.")


def import_model(session: Session, store: ModelStore, path: Path, *, title: str = "",
                 rights_evidence: str = "", project_id: str | None = None,
                 kind: str = "upload", method: str = "file") -> Model3D:
    """Put a model file into the studio. A point cloud without triangles is refused here,
    with the methods that turn points into a solid."""
    path = Path(path)
    if path.suffix.lower() not in MESH_EXTS:
        raise ValueError(f"{path.name}: open an STL, OBJ, PLY or GLB file")
    try:
        mesh = read_mesh(path)
    except (MeshError, ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"The studio could not read {path.name}: {exc}") from exc
    if len(mesh.faces) == 0:
        raise ValueError(f"{path.name} is a point cloud (no triangles). Use 'LiDAR or point "
                         "cloud to solid' or Poisson to make a surface from it.")
    row = _new(session, kind=kind, method=method, title=title or path.stem,
               project_id=project_id,
               request={"file": path.name, "rights_evidence": rights_evidence.strip()})
    source = store.dir(row.id) / f"source{path.suffix.lower()}"
    shutil.copy2(path, source)
    row.source_rel_path = store.rel(source)
    _finish(row, store, mesh)
    session.flush()
    return row


class EditRequest(BaseModel):
    operation: EditOp
    source_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)   # combine
    factor: float = Field(1.0, gt=0, le=1000)              # scale
    size: float = Field(100.0, gt=0, le=100000)            # fit: the longest side (or axis)
    axis: Literal["x", "y", "z", "max"] = "max"            # fit / rotate / mirror
    degrees: float = Field(90.0, ge=-360, le=360)          # rotate
    on_floor: bool = True                                  # center
    tolerance: float = Field(1e-4, gt=0, le=10)            # repair
    title: str = Field("", max_length=200)


def edit_model(session: Session, store: ModelStore, edit: EditRequest, *,
               actor: str = "api") -> Model3D:
    """Apply one edit and keep the result as a new model; the original stays."""
    op = edit.operation
    ids = list(edit.source_ids) if op == "combine" else ([edit.source_id] if edit.source_id else [])
    if not ids:
        raise ValueError("Pick the model to change.")
    parents = [get_model(session, i) for i in ids]
    meshes = [load(store, p) for p in parents]
    mesh = meshes[0]
    axis = "z" if edit.axis == "max" and op in ("rotate", "mirror") else edit.axis
    try:
        if op == "scale":
            out = ops.scale(mesh, edit.factor)
        elif op == "fit":
            out = ops.fit(mesh, edit.size, edit.axis)
        elif op == "rotate":
            out = ops.rotate(mesh, axis, edit.degrees)
        elif op == "mirror":
            out = ops.mirror(mesh, axis)
        elif op == "center":
            out = ops.center(mesh, on_floor=edit.on_floor)
        elif op == "flip":
            out = ops.flip_normals(mesh)
        elif op == "repair":
            out = ops.drop_degenerate(ops.weld(mesh, edit.tolerance))
            if ops.measure(out).get("inverted"):
                out = ops.flip_normals(out)
        elif op == "combine":
            if len(meshes) < 2:
                raise MeshError("Combining needs two or more models.")
            out = ops.combine(meshes)
        else:
            raise ValueError(f"unknown edit {op!r}")
    except MeshError as exc:
        raise ValueError(str(exc)) from exc
    title = edit.title.strip() or f"{parents[0].title or parents[0].kind} · {op}"
    row = _new(session, kind="combine" if op == "combine" else "edit", method=op,
               title=title[:200], parent_id=parents[0].id, project_id=parents[0].project_id,
               params=edit.model_dump(mode="json", exclude={"title"}),
               request={"actor": actor, "inputs": ids})
    _finish(row, store, out)
    session.flush()
    return row


def make_box(session: Session, store: ModelStore, size: tuple[float, float, float], *,
             title: str = "") -> Model3D:
    if min(size) <= 0:
        raise ValueError("Every side must be longer than 0.")
    row = _new(session, kind="primitive", method="box", title=title or "Box",
               params={"size": list(size)})
    _finish(row, store, ops.box(size))
    session.flush()
    return row


class ReliefRequest(BaseModel):
    image_id: str | None = None          # a picture from the library
    image_path: str | None = None        # or a file the worker can read
    width_mm: float = Field(100.0, gt=1, le=2000)
    depth_mm: float = Field(3.0, gt=0, le=200)
    base_mm: float = Field(0.8, ge=0, le=100)
    resolution: int = Field(200, ge=16, le=1000)   # samples across the width
    invert: bool = False                  # a lithophane: dark parts thick
    title: str = Field("", max_length=200)
    rights_confirmed: bool = False
    rights_evidence: str = Field("", max_length=1000)


def _picture(session: Session, data_dir: Path, image_id: str | None, image_path: str | None,
             confirmed: bool, evidence: str) -> tuple[Path, str]:
    """The picture's file and where its right to be used comes from."""
    if image_id:
        from rokkur_studio.services.images import ImageStore

        picture = session.get(Image, image_id)
        if picture is None or not picture.rel_path:
            raise LookupError(f"picture {image_id} is not in the library")
        return ImageStore(data_dir).path_for(picture.rel_path), f"picture {picture.id} from the library"
    if image_path:
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            raise ValueError("Use a PNG, JPG or WebP picture the worker can read.")
        _check_rights(confirmed, evidence, "this picture")
        return path, evidence.strip()
    raise ValueError("Pick a picture from the library or upload one.")


def make_relief(session: Session, settings: Settings, store: ModelStore, ffmpeg: FFmpeg,
                request: ReliefRequest) -> Model3D:
    path, rights = _picture(session, settings.studio.data_dir, request.image_id,
                            request.image_path, request.rights_confirmed, request.rights_evidence)
    try:
        width, height = ffmpeg.image_size(path)
        cols = request.resolution
        rows = max(2, round(cols * height / max(1, width)))
        gray = ffmpeg.read_gray_frames(path, cols, rows)[0]
    except (FFmpegError, IndexError) as exc:
        raise ValueError(f"The studio could not read that picture: {exc}") from exc
    mesh = ops.relief_from_gray(gray, width_mm=request.width_mm, depth_mm=request.depth_mm,
                                base_mm=request.base_mm, invert=request.invert)
    row = _new(session, kind="relief", method="lithophane" if request.invert else "relief",
               title=request.title.strip() or ("Lithophane" if request.invert else "Relief"),
               params=request.model_dump(mode="json", exclude={"rights_confirmed", "title"}),
               request={"rights_evidence": rights})
    shutil.copy2(path, store.dir(row.id) / f"source{path.suffix.lower()}")
    row.source_rel_path = store.rel(store.dir(row.id) / f"source{path.suffix.lower()}")
    _finish(row, store, mesh)
    session.flush()
    return row


class PointsRequest(BaseModel):
    path: str
    cell: float = Field(0.0, ge=0, le=10000)   # grid step; 0 = chosen from the point spacing
    base: float = Field(1.0, ge=0, le=1000)
    fill: int = Field(1, ge=0, le=20)
    title: str = Field("", max_length=200)
    rights_confirmed: bool = False
    rights_evidence: str = Field("", max_length=1000)


def auto_cell(points: np.ndarray) -> float:
    """A grid step that gives each cell a few points on average."""
    span = points.max(axis=0) - points.min(axis=0)
    area = max(float(span[0] * span[1]), 1e-12)
    return float(max(np.sqrt(area / max(1, len(points)) * 4), 1e-6))


def points_to_model(session: Session, store: ModelStore, request: PointsRequest) -> Model3D:
    path = Path(request.path)
    if path.suffix.lower() not in MESH_EXTS | POINT_EXTS or not path.is_file():
        raise ValueError("Use a PLY, OBJ, XYZ, PTS or CSV point file the worker can read.")
    _check_rights(request.rights_confirmed, request.rights_evidence, "this scan")
    try:
        points = read_points(path)
        cell = request.cell or auto_cell(points)
        mesh = ops.relief_from_points(points, cell=cell, base=request.base, fill=request.fill)
    except (MeshError, ValueError) as exc:
        raise ValueError(f"The studio could not make a solid from {path.name}: {exc}") from exc
    row = _new(session, kind="points", method="heightfield", title=request.title.strip() or path.stem,
               params={"cell": round(cell, 6), "base": request.base, "fill": request.fill,
                       "points": int(len(points))},
               request={"file": path.name, "rights_evidence": request.rights_evidence.strip()})
    source = store.dir(row.id) / f"source{path.suffix.lower()}"
    shutil.copy2(path, source)
    row.source_rel_path = store.rel(source)
    _finish(row, store, mesh)
    session.flush()
    return row


class ReconRequest(BaseModel):
    """A reconstruction the worker runs: from one picture, photos, a video or points."""

    method: ReconMethod
    image_id: str | None = None
    image_path: str | None = None
    photo_paths: list[str] = Field(default_factory=list)
    video_path: str | None = None
    points_path: str | None = None
    seed: int | None = Field(None, ge=0, le=MAX_SEED)
    steps: int | None = Field(None, ge=1, le=100)
    cfg: float | None = Field(None, ge=1, le=15)
    octree: int = Field(256, ge=64, le=512)
    depth: int = Field(9, ge=5, le=12)                 # Poisson octree depth
    render_on: Literal["local", "cloud"] | None = None
    project_id: str | None = None
    title: str = Field("", max_length=200)
    rights_confirmed: bool = False
    rights_evidence: str = Field("", max_length=1000)


def request_reconstruction(session: Session, settings: Settings, store: ModelStore,
                           ffmpeg: FFmpeg, request: ReconRequest, *, actor: str = "api"
                           ) -> Model3D:
    """Validate, copy the inputs next to the new model and queue one ``mesh`` job."""
    method = request.method
    target = settings.new_project_target(request.render_on) if method == "image" else "local"
    if problem := availability(settings, target).get(method):
        raise ValueError(problem)
    payload: dict[str, Any] = {"method": method, "render_on": target, "actor": actor}
    row = Model3D(kind=method, method=method, status="queued", render_on=target,
                  title=request.title.strip(), project_id=request.project_id,
                  request=request.model_dump(mode="json", exclude={"rights_confirmed"}))
    session.add(row)
    session.flush()
    folder = store.dir(row.id)
    try:
        if method == "image":
            path, rights = _picture(session, settings.studio.data_dir, request.image_id,
                                    request.image_path, request.rights_confirmed,
                                    request.rights_evidence)
            prepared = folder / "source.png"
            ffmpeg.fit_image(path, prepared, max_pixels=1024 * 1024, multiple=8)
            row.source_rel_path = store.rel(prepared)
            seed = request.seed if request.seed is not None else random.randint(0, MAX_SEED)
            params: dict[str, Any] = {"SEED": seed, "OCTREE": request.octree}
            if request.steps:
                params["STEPS"] = request.steps
            if request.cfg:
                params["CFG"] = request.cfg
            row.params = params
            payload.update(workflow=settings.three.image_workflow, params=params,
                           source=row.source_rel_path,
                           resource_class=settings.three.image_resource_class)
            row.request = {**row.request, "rights_evidence": rights}
            row.title = row.title or "From a picture"
        elif method == "photos":
            photos = [Path(p) for p in request.photo_paths]
            if len(photos) < 3:
                raise ValueError("Photogrammetry needs at least 3 photos (10 or more, overlapping, "
                                 "for a good result).")
            if any(p.suffix.lower() not in IMAGE_EXTS or not p.is_file() for p in photos):
                raise ValueError("Every photo must be a PNG, JPG or WebP the worker can read.")
            _check_rights(request.rights_confirmed, request.rights_evidence, "these photos")
            target_dir = folder / "photos"
            target_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(photos, 1):
                shutil.copy2(p, target_dir / f"photo_{i:04d}{p.suffix.lower()}")
            row.source_rel_path = store.rel(target_dir)
            row.params = {"photos": len(photos)}
            payload["source"] = row.source_rel_path
            row.title = row.title or f"From {len(photos)} photos"
        elif method == "video":
            video = Path(request.video_path or "")
            if video.suffix.lower() not in VIDEO_EXTS or not video.is_file():
                raise ValueError("Pick a video the worker can read.")
            _check_rights(request.rights_confirmed, request.rights_evidence, "this video")
            payload.update(video=str(video), fps=settings.three.video_fps,
                           max_frames=settings.three.max_video_frames)
            row.params = {"fps": settings.three.video_fps}
            row.title = row.title or f"From {video.name}"
        elif method == "poisson":
            src = Path(request.points_path or "")
            if src.suffix.lower() not in MESH_EXTS | POINT_EXTS or not src.is_file():
                raise ValueError("Use a PLY, OBJ, XYZ, PTS or CSV point file the worker can read.")
            _check_rights(request.rights_confirmed, request.rights_evidence, "this scan")
            copy = folder / f"source{src.suffix.lower()}"
            shutil.copy2(src, copy)
            row.source_rel_path = store.rel(copy)
            row.params = {"depth": request.depth}
            payload.update(source=row.source_rel_path, depth=request.depth)
            row.title = row.title or f"Surface of {src.stem}"
    except (ValueError, LookupError, FFmpegError) as exc:
        _fail_clean(session, store, row)
        raise ValueError(exc.summary if isinstance(exc, FFmpegError) else str(exc)) from exc
    payload["model_id"] = row.id
    job = enqueue(session, "mesh", payload=payload, priority=95,
                  max_attempts=settings.jobs.default_max_attempts if method == "image" else 1)
    assert job is not None
    row.job_id = job.id
    session.flush()
    return row


def export_model(store: ModelStore, row: Model3D, fmt: str, *, ascii_stl: bool = False) -> Path:
    """The model in another format, written once next to it."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in WRITE_EXTS:
        raise ValueError(f"Export as {', '.join(WRITE_EXTS)}.")
    out = store.dir(row.id) / (f"export_ascii.{fmt}" if ascii_stl else f"export.{fmt}")
    if not out.is_file():
        write_mesh(load(store, row), out, ascii_stl=ascii_stl)
    return out


def view_data(store: ModelStore, row: Model3D, max_faces: int) -> dict[str, Any]:
    """Flat vertex and face lists for the page's 3D view, thinned evenly above ``max_faces``."""
    mesh = load(store, row)
    faces = mesh.faces
    thinned = len(faces) > max_faces
    if thinned:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces).astype(np.int64)]
        mesh = ops.drop_unused(Mesh(mesh.vertices, faces))
    return {"vertices": np.round(mesh.vertices, 4).ravel().tolist(),
            "faces": mesh.faces.ravel().tolist(), "thinned": thinned,
            "total_faces": int(row.stats.get("triangles", len(faces)))}


def list_models(session: Session, *, limit: int = 60, offset: int = 0, kind: str | None = None,
                project_id: str | None = None) -> list[Model3D]:
    stmt = (select(Model3D).order_by(Model3D.created_at.desc(), Model3D.id.desc())
            .offset(offset).limit(limit))
    if kind:
        stmt = stmt.where(Model3D.kind == kind)
    if project_id:
        stmt = stmt.where(Model3D.project_id == project_id)
    return list(session.scalars(stmt))


def get_model(session: Session, model_id: str) -> Model3D:
    row = session.get(Model3D, model_id)
    if row is None:
        raise LookupError(f"model {model_id} not found")
    return row


def delete_model(session: Session, store: ModelStore, row: Model3D) -> None:
    job = session.get(Job, row.job_id) if row.job_id else None
    if row.status in ("queued", "running") and job is not None and job.status in ("QUEUED", "RETRY_WAIT"):
        job.status = "CANCELLED"
        job.finished_at = utcnow()
    for child in session.scalars(select(Model3D).where(Model3D.parent_id == row.id)):
        child.parent_id = None
    session.delete(row)
    session.flush()
    shutil.rmtree(store.root / row.id, ignore_errors=True)


def model_view(row: Model3D) -> dict[str, Any]:
    return {"id": row.id, "kind": row.kind, "method": row.method, "status": row.status,
            "title": row.title, "params": row.params, "stats": row.stats,
            "parent_id": row.parent_id, "project_id": row.project_id, "render_on": row.render_on,
            "error": row.error, "took_s": row.took_s,
            "file_url": f"/models3d/{row.id}/file" if row.rel_path else None,
            "preview_url": f"/models3d/{row.id}/preview.svg" if row.rel_path else None,
            "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else None}
