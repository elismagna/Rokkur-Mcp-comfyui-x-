"""Still pictures: generate, edit, vary, inpaint, outpaint and upscale through ComfyUI.

The dashboard, API and CLI all call ``request_images``; the worker's ``image`` job
(``pipeline/images.py``) does the rendering. Every picture is an ``Image`` row with its files
under ``<data_dir>/images/<id>/``. See docs/images.md.
"""

from __future__ import annotations

import math
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.config import IMAGE_OPERATIONS, ImageOperation, Settings
from rokkur_studio.db.models import Image, Job, utcnow
from rokkur_studio.jobs.queue import enqueue
from rokkur_studio.media.ffmpeg import FFmpeg, FFmpegError

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
MULTIPLE = 16
MAX_SEED = 2**53 - 1  # what the dashboard's JavaScript can hold exactly
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")

# What the klein edit model is told when you ask for variations of a picture.
VARIATION_PROMPT = ("Create a new variation of this picture: keep the same subject, style, "
                    "lighting and framing, but change the details and the composition slightly.")

SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "square": (1024, 1024),
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "short": (768, 1344),      # 9:16, a YouTube Short's thumbnail
    "widescreen": (1344, 768),  # 16:9
}


class ImageStore:
    """Files of the image library, under ``<data_dir>/images``."""

    def __init__(self, data_dir: Path) -> None:
        self.root = (Path(data_dir) / "images").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, image_id: str) -> Path:
        if not _SAFE.match(image_id):
            raise ValueError(f"unsafe image id {image_id!r}")
        path = self.root / image_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path_for(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if self.root not in path.parents:
            raise ValueError(f"path escapes the image library: {rel_path!r}")
        return path

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()


class ImageRequest(BaseModel):
    """What a person asks for. One request makes ``count`` pictures in one job."""

    operation: ImageOperation = "generate"
    prompt: str = Field("", max_length=4000)
    profile: str | None = None
    size: str | None = None            # a SIZE_PRESETS key, or "WxH"
    width: int | None = Field(None, ge=64, le=4096)
    height: int | None = Field(None, ge=64, le=4096)
    count: int = Field(1, ge=1, le=8)
    steps: int | None = Field(None, ge=1, le=100)
    cfg: float | None = Field(None, ge=0, le=30)
    seed: int | None = Field(None, ge=0, le=MAX_SEED)
    source_id: str | None = None       # an existing picture to edit, vary, repaint, extend or upscale
    source_path: str | None = None     # or a file on the worker (an upload); needs the rights lines
    mask_path: str | None = None       # inpaint: white = repaint, same framing as the source
    mask_grow: int = Field(8, ge=0, le=128)
    pad: dict[str, int] = Field(default_factory=dict)  # outpaint: left/top/right/bottom pixels
    feather: int = Field(24, ge=0, le=256)
    scale: Literal[2, 4] = 2           # upscale factor
    render_on: Literal["local", "cloud"] | None = None
    project_id: str | None = None
    shot_id: str | None = None
    kind: str | None = None            # recorded kind when it differs from the operation (storyboard)
    title: str = Field("", max_length=200)
    rights_confirmed: bool = False     # for a picture you upload: you may use and transform it
    rights_evidence: str = Field("", max_length=1000)


def snap_size(width: int, height: int, max_pixels: int, multiple: int = MULTIPLE
              ) -> tuple[int, int]:
    """Round a size to multiples of 16 and shrink it, keeping the aspect, under ``max_pixels``."""
    scale = min(1.0, math.sqrt(max_pixels / max(1, width * height)))
    w = max(multiple, int(width * scale) // multiple * multiple)
    h = max(multiple, int(height * scale) // multiple * multiple)
    return w, h


def parse_size(request: ImageRequest, settings: Settings) -> tuple[int, int]:
    if request.width and request.height:
        return request.width, request.height
    if request.size:
        if request.size in SIZE_PRESETS:
            return SIZE_PRESETS[request.size]
        m = re.fullmatch(r"\s*(\d{2,4})\s*[xX×]\s*(\d{2,4})\s*", request.size)
        if not m:
            raise ValueError(f"size {request.size!r}: use one of {', '.join(SIZE_PRESETS)} "
                             "or WIDTHxHEIGHT")
        return int(m.group(1)), int(m.group(2))
    return settings.images.default_width, settings.images.default_height


def pad_values(pad: dict[str, int], width: int, height: int, multiple: int = MULTIPLE
               ) -> dict[str, int]:
    """Outpaint padding per side, rounded up so the padded picture stays a multiple of 16."""
    out = {}
    for side in ("left", "top", "right", "bottom"):
        value = int(pad.get(side, 0) or 0)
        if value < 0:
            raise ValueError(f"pad {side} cannot be negative")
        out[side] = value
    total_w = width + out["left"] + out["right"]
    total_h = height + out["top"] + out["bottom"]
    out["right"] += (-total_w) % multiple
    out["bottom"] += (-total_h) % multiple
    if sum(out.values()) == 0:
        raise ValueError("choose at least one side to extend")
    return out


def _profile_for(request: ImageRequest, settings: Settings) -> str:
    if request.profile:
        return request.profile
    if request.operation == "upscale":
        return settings.images.upscale_profile
    return settings.images.default_profile


def _source(session: Session, store: ImageStore, request: ImageRequest) -> tuple[Path, str | None]:
    """The input picture's file and, when it came from the library, its id."""
    if request.source_id:
        parent = session.get(Image, request.source_id)
        if parent is None or not parent.rel_path:
            raise LookupError(f"picture {request.source_id} is not in the library")
        path = store.path_for(parent.rel_path)
        if not path.is_file():
            raise LookupError(f"the file of picture {request.source_id} is gone")
        return path, parent.id
    if request.source_path:
        path = Path(request.source_path)
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError("Use a PNG, JPG or WebP picture.")
        if not path.is_file():
            raise ValueError("The worker cannot read that picture. Upload it or pick one from "
                             "the library.")
        if not request.rights_confirmed or not request.rights_evidence.strip():
            raise ValueError("Confirm that you may use and transform the uploaded picture and "
                             "say where it comes from (its rights or license).")
        return path, None
    raise ValueError(f"{request.operation} needs a picture: pick one from the library or "
                     "upload one")


def request_images(session: Session, settings: Settings, store: ImageStore, ffmpeg: FFmpeg,
                   request: ImageRequest, *, actor: str = "api") -> list[Image]:
    """Validate, prepare the input files, create the rows and queue one job for them."""
    op = request.operation
    if op not in IMAGE_OPERATIONS:
        raise ValueError(f"unknown operation {op!r}")
    if op != "upscale" and op != "variation" and not request.prompt.strip():
        raise ValueError("Write a prompt: what the picture should show or how it should change.")
    count = min(request.count, settings.images.max_batch)
    profile_name = _profile_for(request, settings)
    target = settings.new_project_target(request.render_on)
    if problem := settings.image_profile_problem(profile_name, op, target=target):
        raise ValueError(problem)
    profile = settings.image_profile(profile_name)
    workflow = profile.workflow_for(op)
    max_pixels = min(profile.max_pixels, settings.images.max_pixels)
    seed = request.seed if request.seed is not None else random.randint(0, MAX_SEED)
    params: dict[str, Any] = {"SEED": seed, "STEPS": request.steps or profile.steps,
                              "CFG": request.cfg if request.cfg is not None else profile.cfg}
    prompt = request.prompt.strip()
    rows = [Image(kind=request.kind or op, status="queued", profile=profile_name,
                  workflow=workflow, prompt=prompt, request=request.model_dump(mode="json"),
                  project_id=request.project_id, shot_id=request.shot_id, render_on=target,
                  title=request.title.strip(), seed=seed) for _ in range(count)]
    for row in rows:
        session.add(row)
    session.flush()
    first = store.dir(rows[0].id)
    source_rel: str | None = None
    parent_id: str | None = None
    if op == "generate":
        width, height = snap_size(*parse_size(request, settings), max_pixels)
        params.update(WIDTH=width, HEIGHT=height)
        for row in rows:
            row.width, row.height = width, height
    else:
        source, parent_id = _source(session, store, request)
        prepared = first / "source.png"
        try:
            if op == "upscale":
                ffmpeg.fit_image(source, prepared, max_pixels=max_pixels, multiple=1)
                params["SCALE_BY"] = request.scale / 4
            else:
                ffmpeg.fit_image(source, prepared, max_pixels=max_pixels)
            if op == "inpaint":
                if not request.mask_path or not Path(request.mask_path).is_file():
                    raise ValueError("Paint the area to change first.")
                masked = first / "source_masked.png"
                ffmpeg.alpha_from_mask(prepared, Path(request.mask_path), masked)
                prepared = masked
                params["MASK_GROW"] = request.mask_grow
            elif op == "outpaint":
                width, height = ffmpeg.image_size(prepared)
                pads = pad_values(request.pad, width, height)
                params.update(PAD_LEFT=pads["left"], PAD_TOP=pads["top"],
                              PAD_RIGHT=pads["right"], PAD_BOTTOM=pads["bottom"],
                              FEATHER=request.feather, MASK_GROW=request.mask_grow)
            elif op == "variation":
                prompt = prompt or VARIATION_PROMPT
            elif op == "edit":
                params["MEGAPIXELS"] = round(max_pixels / 1_000_000, 3)
        except FFmpegError as exc:
            raise ValueError(f"The studio could not read that picture: {exc.summary}") from exc
        source_rel = store.rel(prepared)
        for row in rows:
            row.parent_id = parent_id
            row.source_rel_path = source_rel
            row.prompt = prompt
    params["PROMPT"] = prompt
    for row in rows:
        row.params = dict(params)
    payload = {"image_ids": [r.id for r in rows], "operation": op, "profile": profile_name,
               "workflow": workflow, "params": params, "render_on": target,
               "source": source_rel, "degrade": list(profile.degrade),
               "resource_class": profile.resource_class, "actor": actor}
    job = enqueue(session, "image", payload=payload, priority=90,
                  max_attempts=settings.jobs.default_max_attempts)
    assert job is not None
    for row in rows:
        row.job_id = job.id
    session.flush()
    return rows


def list_images(session: Session, *, limit: int = 60, offset: int = 0, kind: str | None = None,
                project_id: str | None = None, status: str | None = None) -> list[Image]:
    stmt = select(Image).order_by(Image.created_at.desc(), Image.id.desc()).offset(offset).limit(limit)
    if kind:
        stmt = stmt.where(Image.kind == kind)
    if project_id:
        stmt = stmt.where(Image.project_id == project_id)
    if status:
        stmt = stmt.where(Image.status == status)
    return list(session.scalars(stmt))


def get_image(session: Session, image_id: str) -> Image:
    image = session.get(Image, image_id)
    if image is None:
        raise LookupError(f"picture {image_id} not found")
    return image


def import_image(session: Session, store: ImageStore, ffmpeg: FFmpeg, path: Path, *,
                 kind: str = "upload", title: str = "", prompt: str = "",
                 project_id: str | None = None, shot_id: str | None = None,
                 request: dict[str, Any] | None = None) -> Image:
    """Put a finished picture (an upload, a character cutout) into the library."""
    row = Image(kind=kind, status="done", title=title, prompt=prompt, project_id=project_id,
                shot_id=shot_id, request=request or {}, finished_at=utcnow())
    session.add(row)
    session.flush()
    dest = store.dir(row.id) / f"image{Path(path).suffix.lower() or '.png'}"
    if Path(path).resolve() != dest.resolve():
        shutil.copy2(path, dest)
    row.rel_path = store.rel(dest)
    try:
        row.width, row.height = ffmpeg.image_size(dest)
    except FFmpegError as exc:
        raise ValueError(f"not a picture the studio can read: {exc.summary}") from exc
    session.flush()
    return row


def set_verdict(session: Session, image: Image, value: int | None) -> Image:
    if value is not None and value not in (-2, -1, 1, 2):
        raise ValueError("a verdict is -2, -1, 1 or 2")
    image.verdict = value
    session.flush()
    return image


def delete_image(session: Session, store: ImageStore, image: Image) -> None:
    """Remove a picture and its files. Pictures made from it keep their own files."""
    job = session.get(Job, image.job_id) if image.job_id else None
    if image.status in ("queued", "running") and job is not None and job.status in ("QUEUED", "RETRY_WAIT"):
        job.status = "CANCELLED"
        job.finished_at = utcnow()
    for child in session.scalars(select(Image).where(Image.parent_id == image.id)):
        child.parent_id = None
    session.delete(image)
    session.flush()
    folder = store.root / image.id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def image_view(image: Image) -> dict[str, Any]:
    """What pages and the API show for a picture."""
    return {
        "id": image.id, "kind": image.kind, "status": image.status, "title": image.title,
        "prompt": image.prompt, "profile": image.profile, "workflow": image.workflow,
        "width": image.width, "height": image.height, "seed": image.seed,
        "parent_id": image.parent_id, "project_id": image.project_id, "shot_id": image.shot_id,
        "render_on": image.render_on, "verdict": image.verdict, "error": image.error,
        "duration_s": image.duration_s, "params": image.params, "request": image.request,
        "file_url": f"/images/{image.id}/file" if image.rel_path else None,
        "source_url": f"/images/{image.id}/source" if image.source_rel_path else None,
        "created_at": image.created_at.isoformat() if isinstance(image.created_at, datetime) else None,
    }
