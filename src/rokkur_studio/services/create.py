"""The Create page: image and video takes from a prompt, in the style of Open Higgsfield's
studios (docs/create.md).

A take is one generation: a still (``image``), a clip from text (``video``) or a clip that
starts from a picture (``animate``). Cinema choices (camera body, lens, focal length,
aperture, shot size, angle, lighting and, for clips, camera movement) are written into the
prompt as plain phrases, the same way the director's cinematography vocabulary is
(director/vocabulary.py), because Wan and Z-Image only read text. Each take is one
``create_take`` job, so it waits for the GPU lease like every render does and never runs
next to a video's shots on the 8 GB card.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from rokkur_studio.config import Settings
from rokkur_studio.db.models import Job, Take, utcnow
from rokkur_studio.director.vocabulary import (
    CAMERA_ANGLES,
    LIGHTING_STYLES,
    PHRASES,
    SHOT_SIZES,
)
from rokkur_studio.jobs.queue import JobStatus, enqueue

KIND = "create_take"
MODES = ("image", "video", "animate")
MAX_BATCH = 4
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Sizes per aspect ratio. Stills follow Z-Image's ~1 megapixel training size; clips stay at
# Wan 2.1 1.3B's 480P (about 832x480 = 399,360 pixels), every side a multiple of 16.
IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
    "4:3": (1152, 864), "3:4": (864, 1152), "21:9": (1536, 640),
}
VIDEO_SIZES: dict[str, tuple[int, int]] = {
    "16:9": (832, 480), "9:16": (480, 832), "1:1": (624, 624),
    "4:3": (704, 528), "3:4": (528, 704),
}
SECONDS = (2, 3, 5)

# Cinema controls, after Open Higgsfield's Cinema Studio (camera, lens, focal length,
# aperture) plus the camera moves Higgsfield is known for. Each choice maps to the phrase
# written into the prompt; an empty choice writes nothing.
CAMERAS: dict[str, str] = {
    "8K digital cinema": "shot on an 8K digital cinema camera, crisp detail",
    "Full-frame digital": "shot on a full-frame digital cinema camera, wide dynamic range",
    "Super 35 digital": "shot on a Super 35 digital cinema camera",
    "Large-format digital": "shot on a large-format digital camera",
    "70mm film": "shot on 70mm large-format film, fine grain, rich colour",
    "35mm film": "shot on 35mm film, natural grain",
    "16mm film": "shot on 16mm film, visible grain, soft halation",
    "VHS camcorder": "VHS camcorder footage, scan lines, soft colour bleed",
}
LENSES: dict[str, str] = {
    "Modern prime": "sharp modern prime lens",
    "Vintage prime": "vintage prime lens, gentle softness and warm flare",
    "Classic anamorphic": "anamorphic lens, oval bokeh, horizontal lens flares",
    "Swirl bokeh": "swirly bokeh portrait lens",
    "Extreme macro": "extreme macro lens, tiny details",
    "Tilt-shift": "tilt-shift lens, miniature-style focus",
    "Halation diffusion": "diffusion filter, glowing halation around highlights",
    "Fisheye": "fisheye lens, curved wide view",
}
FOCAL_LENGTHS: dict[str, str] = {
    "8mm": "8mm ultra-wide lens", "14mm": "14mm wide-angle lens", "24mm": "24mm wide lens",
    "35mm": "35mm lens, natural perspective", "50mm": "50mm lens",
    "85mm": "85mm portrait lens", "135mm": "135mm telephoto lens, compressed background",
}
APERTURES: dict[str, str] = {
    "f/1.4": "f/1.4, shallow depth of field, creamy background blur",
    "f/2.8": "f/2.8, soft background",
    "f/4": "f/4, balanced depth of field",
    "f/11": "f/11, deep focus, everything sharp",
}
MOVES: dict[str, str] = {
    "Static": "static camera, locked-off shot",
    "Slow push-in": "slow push-in toward the subject",
    "Pull back": "camera slowly pulls back to reveal the scene",
    "Pan left": "camera pans left",
    "Pan right": "camera pans right",
    "Tilt up": "camera tilts up",
    "Crane up": "crane shot rising upward",
    "Orbit": "camera orbits around the subject",
    "Tracking": "tracking shot following the subject",
    "Handheld": "handheld camera, subtle shake",
    "Dolly zoom": "dolly zoom, background stretches while the subject stays the same size",
    "Crash zoom": "fast crash zoom into the subject",
    "FPV drone": "FPV drone shot flying fast through the scene",
}
SHOTS: dict[str, str] = {s: PHRASES[s] for s in SHOT_SIZES}
ANGLES: dict[str, str] = {a: PHRASES[a] for a in CAMERA_ANGLES}
LIGHTS: dict[str, str] = {lg: PHRASES[lg] for lg in LIGHTING_STYLES}

CINEMA: dict[str, dict[str, str]] = {
    "shot": SHOTS, "angle": ANGLES, "move": MOVES, "camera": CAMERAS, "lens": LENSES,
    "focal": FOCAL_LENGTHS, "aperture": APERTURES, "light": LIGHTS,
}
CINEMA_LABELS = {"shot": "Shot size", "angle": "Angle", "move": "Camera move",
                 "camera": "Camera", "lens": "Lens", "focal": "Focal length",
                 "aperture": "Aperture", "light": "Lighting"}


class CreateError(ValueError):
    """A request the Create page cannot run; the message is shown to the person."""


@dataclass(frozen=True)
class TakeRequest:
    mode: str
    prompt: str
    aspect: str = ""
    seconds: int = 5
    seed: int | None = None
    count: int = 1
    negative_prompt: str = ""
    camera: dict[str, str] | None = None
    start_image: Path | None = None  # a picture the worker can read (animate only)
    render_on: str = "local"
    parent_id: str | None = None


def clean_camera(camera: dict[str, str] | None, mode: str) -> dict[str, str]:
    """Only known choices; a camera move only for clips (a still cannot move)."""
    out = {}
    for key, value in (camera or {}).items():
        if key in CINEMA and value in CINEMA[key] and not (key == "move" and mode == "image"):
            out[key] = value
    return out


def compose_prompt(prompt: str, camera: dict[str, str]) -> str:
    """The prompt the model reads: framing first (shot size, angle, move), then the person's
    words, then the look (camera body, lens, focal length, aperture, lighting). This follows
    the director's order, which puts the framing noun before the subject."""
    words = " ".join(prompt.split())
    front = [CINEMA[k][camera[k]] for k in ("shot", "angle", "move") if k in camera]
    back = [CINEMA[k][camera[k]] for k in ("camera", "lens", "focal", "aperture", "light")
            if k in camera]
    parts = [", ".join(front)] if front else []
    parts.append(words.rstrip("."))
    if back:
        parts.append(", ".join(back))
    return ". ".join(p for p in parts if p)


def frames_for(seconds: int, fps: int, max_frames: int) -> int:
    """Wan wants 4n+1 frames; ``seconds`` of clip at ``fps``, capped at ``max_frames``."""
    frames = min(int(seconds) * fps + 1, max_frames)
    return max(5, (frames - 1) // 4 * 4 + 1)


def negative_for(settings: Settings) -> str:
    if settings.create.negative_prompt.strip():
        return settings.create.negative_prompt.strip()
    profile = settings.profiles.get(settings.render.default_profile)
    return profile.negative_base if profile and profile.negative_base else (
        "blurry, deformed, watermark, text")


def workflow_params(settings: Settings, req: TakeRequest, seed: int) -> dict[str, Any]:
    """The workflow parameters a take renders with (semantic names, docs/comfyui.md)."""
    cfg = settings.create
    if req.mode == "image":
        width, height = IMAGE_SIZES.get(req.aspect) or IMAGE_SIZES["1:1"]
        return {"WIDTH": width, "HEIGHT": height, "SEED": seed, "STEPS": cfg.image_steps}
    width, height = VIDEO_SIZES.get(req.aspect) or VIDEO_SIZES["16:9"]
    frames = frames_for(req.seconds, cfg.video_fps, cfg.max_frames)
    params: dict[str, Any] = {"WIDTH": width, "HEIGHT": height, "FRAME_COUNT": frames,
                              "SEED": seed, "STEPS": cfg.video_steps, "FPS": cfg.video_fps}
    if req.mode == "animate":
        params["UNKNOWN_FRAMES"] = frames - 1
    return params


def create_dir(settings: Settings) -> Path:
    root = (Path(settings.studio.data_dir) / "create").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def path_for(settings: Settings, rel: str) -> Path:
    root = create_dir(settings)
    path = (root / rel).resolve()
    if root not in path.parents:
        raise CreateError(f"path escapes the Create folder: {rel!r}")
    return path


def submit(session: Session, settings: Settings, req: TakeRequest) -> list[Take]:
    """Queue ``req.count`` takes (seeds counting up from ``req.seed``, or random)."""
    if not settings.create.enabled:
        raise CreateError("The Create page is turned off (create.enabled).")
    if req.mode not in MODES:
        raise CreateError(f"unknown kind of take {req.mode!r}")
    prompt = " ".join(req.prompt.split())
    if not prompt:
        raise CreateError("Write what you want to see first.")
    if len(prompt) > 2000:
        raise CreateError("Keep the prompt under 2,000 characters.")
    if req.mode == "animate" and (req.start_image is None or not req.start_image.is_file()):
        raise CreateError("Pick a start picture to animate.")
    if req.render_on == "cloud" and not settings.cloud.ready:
        raise CreateError("The cloud server is not set up (docs/cloud.md).")
    if req.seconds not in SECONDS:
        raise CreateError(f"Clips can be {', '.join(map(str, SECONDS))} seconds long.")
    count = max(1, min(int(req.count), MAX_BATCH))
    camera = clean_camera(req.camera, req.mode)
    final = compose_prompt(prompt, camera)
    negative = " ".join(req.negative_prompt.split()) or (
        "" if req.mode == "image" else negative_for(settings))
    first_seed = req.seed if req.seed is not None and req.seed >= 0 else random.randrange(2**31)
    takes = []
    for i in range(count):
        seed = (first_seed + i) % 2**32
        take = Take(mode=req.mode, status="queued", prompt=prompt, final_prompt=final,
                    negative_prompt=negative, camera=camera,
                    settings={"aspect": req.aspect, "seconds": req.seconds,
                              "params": workflow_params(settings, req, seed)},
                    workflow=settings.create.workflow_for(req.mode),
                    render_on="cloud" if req.render_on == "cloud" else "local",
                    parent_id=req.parent_id)
        session.add(take)
        session.flush()
        if req.start_image is not None and req.mode == "animate":
            folder = create_dir(settings) / take.id
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f"start{req.start_image.suffix.lower() or '.png'}"
            shutil.copyfile(req.start_image, dest)
            take.start_image = dest.relative_to(create_dir(settings)).as_posix()
        job = enqueue(session, KIND, payload={"take_id": take.id},
                      max_attempts=settings.jobs.default_max_attempts)
        take.job_id = job.id if job else None
        takes.append(take)
    session.flush()
    return takes


def again(session: Session, settings: Settings, take: Take) -> Take:
    """The same take with the next seed."""
    params = (take.settings or {}).get("params") or {}
    start = path_for(settings, take.start_image) if take.start_image else None
    req = TakeRequest(mode=take.mode, prompt=take.prompt,
                      aspect=str((take.settings or {}).get("aspect") or ""),
                      seconds=int((take.settings or {}).get("seconds") or 5),
                      seed=(int(params.get("SEED", 0)) + 1) % 2**32,
                      negative_prompt=take.negative_prompt, camera=take.camera,
                      start_image=start, render_on=take.render_on, parent_id=take.id)
    return submit(session, settings, req)[0]


def state(take: Take, job: Job | None) -> tuple[str, str | None]:
    """What a person should see: the take's status, unless its job gave up (or was lost)."""
    if take.status in ("queued", "running", "retrying") and job is not None:
        if job.status == JobStatus.FAILED:
            return "failed", take.error or (job.error or {}).get("message")
        if job.status == JobStatus.CANCELLED:
            return "cancelled", None
    if take.status == "retrying":
        return "queued", take.error
    return take.status, take.error


def listing(session: Session, *, limit: int = 60) -> list[dict[str, Any]]:
    takes = list(session.scalars(select(Take).order_by(Take.created_at.desc()).limit(limit)))
    jobs = {j.id: j for j in session.scalars(
        select(Job).where(Job.id.in_([t.job_id for t in takes if t.job_id])))}
    return [view(t, jobs.get(t.job_id or "")) for t in takes]


def view(take: Take, job: Job | None) -> dict[str, Any]:
    status, error = state(take, job)
    params = (take.settings or {}).get("params") or {}
    return {
        "id": take.id, "mode": take.mode, "status": status, "error": error,
        "prompt": take.prompt, "final_prompt": take.final_prompt, "camera": take.camera,
        "aspect": (take.settings or {}).get("aspect"),
        "seconds": (take.settings or {}).get("seconds") if take.mode != "image" else None,
        "seed": params.get("SEED"), "width": params.get("WIDTH"), "height": params.get("HEIGHT"),
        "render_on": take.render_on, "rating": take.rating, "workflow": take.workflow,
        "is_video": take.mode != "image",
        "file": f"/ui/create/takes/{take.id}/file" if take.output else None,
        "thumb": f"/ui/create/takes/{take.id}/thumb" if take.thumbnail else None,
        "start": f"/ui/create/takes/{take.id}/start" if take.start_image else None,
        "duration_s": round(take.duration_s, 1) if take.duration_s else None,
        "created_at": take.created_at.isoformat() if take.created_at else None,
    }


def rate(take: Take, value: int) -> None:
    if value not in (-2, -1, 0, 1, 2):
        raise CreateError("A rating is -2, -1, 1 or 2 (0 clears it).")
    take.rating = value or None


def delete(session: Session, settings: Settings, take: Take) -> None:
    """Remove the take, its files and its queued job (a running one stops at its next check)."""
    if take.job_id:
        session.execute(update(Job).where(Job.id == take.job_id,
                                          Job.status.in_(JobStatus.ACTIVE))
                        .values(status=JobStatus.CANCELLED, finished_at=utcnow(),
                                locked_until=None))
    folder = create_dir(settings) / take.id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    session.delete(take)


def video_source(session: Session, settings: Settings, take_id: str) -> dict[str, Any] | None:
    """A finished clip offered as a new video's source on the New video page."""
    take = session.get(Take, take_id)
    if take is None or take.mode == "image" or not take.output:
        return None
    job = session.get(Job, take.job_id) if take.job_id else None
    if state(take, job)[0] != "succeeded":
        return None
    return {"id": take.id, "path": str(path_for(settings, take.output)),
            "prompt": take.prompt, "thumb": f"/ui/create/takes/{take.id}/thumb"}
