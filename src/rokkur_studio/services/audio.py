"""Sound: music and effects generation through ComfyUI, and FFmpeg editing (docs/audio.md).

The dashboard, API and CLI all call ``request_audio`` (generation, queued as an ``audio`` job
rendered by ``pipeline/audio.py``) and ``edit_audio`` (trim, fade, gain, normalise, loop,
speed, mix, join, extract from a video: FFmpeg on the CPU, done at once). Every clip is an
``AudioClip`` row with its file under ``<data_dir>/audio/<id>/clip.flac``. A clip becomes a
video's soundtrack through ``commands.set_soundtrack``, at any stage of that video.
"""

from __future__ import annotations

import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.config import AUDIO_EDITS, AUDIO_OPERATIONS, AudioOperation, Settings
from rokkur_studio.db.models import AudioClip, Job, utcnow
from rokkur_studio.jobs.queue import enqueue
from rokkur_studio.media.ffmpeg import FFmpeg, FFmpegError

AUDIO_EXTS = frozenset({".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus",
                        ".aif", ".aiff", ".wma", ".weba"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
MAX_SEED = 2**53 - 1
INSTRUMENTAL = "[inst]"
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
EditOperation = Literal["trim", "fade", "gain", "normalize", "loop", "speed", "mix", "concat",
                        "extract"]

# What the page says each generation operation is for.
OPERATION_HINTS = {
    "music": "A piece of music from style tags (genre, mood, instruments, tempo) and optional "
             "lyrics with [verse] and [chorus] marks. ACE-Step, up to 4 minutes.",
    "sound": "A sound effect, foley or ambience from a description. Stable Audio Open, up to "
             "47 seconds; weaker for whole songs.",
}


class AudioStore:
    """Files of the sound library, under ``<data_dir>/audio``."""

    def __init__(self, data_dir: Path) -> None:
        self.root = (Path(data_dir) / "audio").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, clip_id: str) -> Path:
        if not _SAFE.match(clip_id):
            raise ValueError(f"unsafe clip id {clip_id!r}")
        path = self.root / clip_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path_for(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if self.root not in path.parents:
            raise ValueError(f"path escapes the sound library: {rel_path!r}")
        return path

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()


class AudioRequest(BaseModel):
    """A generation request. One request makes ``count`` clips in one job."""

    operation: AudioOperation = "music"
    prompt: str = Field("", max_length=4000)       # music: style tags; sound: a description
    lyrics: str = Field("", max_length=8000)       # music only; empty means instrumental
    negative_prompt: str = Field("", max_length=2000)  # sound only
    seconds: float | None = Field(None, gt=0, le=600)
    count: int = Field(1, ge=1, le=4)
    steps: int | None = Field(None, ge=1, le=200)
    cfg: float | None = Field(None, ge=0, le=30)
    seed: int | None = Field(None, ge=0, le=MAX_SEED)
    lyrics_strength: float = Field(0.99, ge=0, le=1)
    profile: str | None = None
    render_on: Literal["local", "cloud"] | None = None
    project_id: str | None = None
    title: str = Field("", max_length=200)


class AudioEdit(BaseModel):
    """An FFmpeg edit: the input clip(s) and the operation's settings."""

    operation: EditOperation
    source_id: str | None = None                   # a library clip
    source_ids: list[str] = Field(default_factory=list)  # mix / concat: two or more clips
    source_path: str | None = None                 # a sound or video file the worker can read
    start: float = Field(0.0, ge=0)                # trim
    end: float | None = Field(None, gt=0)
    fade_in: float = Field(0.0, ge=0, le=60)       # fade
    fade_out: float = Field(0.0, ge=0, le=60)
    db: float = Field(0.0, ge=-60, le=30)          # gain
    lufs: float = Field(-14.0, ge=-40, le=-5)      # normalize
    seconds: float | None = Field(None, gt=0, le=3600)  # loop: target length
    crossfade: float = Field(0.5, ge=0, le=10)
    factor: float = Field(1.0, ge=0.25, le=4)      # speed
    keep_pitch: bool = True
    volumes: list[float] = Field(default_factory=list)  # mix
    offsets: list[float] = Field(default_factory=list)
    duration: Literal["longest", "shortest", "first"] = "longest"
    project_id: str | None = None
    title: str = Field("", max_length=200)
    rights_confirmed: bool = False                 # for a file: you may use and change it
    rights_evidence: str = Field("", max_length=1000)


def _profile_for(request: AudioRequest, settings: Settings) -> str:
    if request.profile:
        return request.profile
    return settings.audio.music_profile if request.operation == "music" else settings.audio.sound_profile


def request_audio(session: Session, settings: Settings, store: AudioStore,
                  request: AudioRequest, *, actor: str = "api") -> list[AudioClip]:
    """Validate, create the rows and queue one ``audio`` job for them."""
    op = request.operation
    if op not in AUDIO_OPERATIONS:
        raise ValueError(f"unknown operation {op!r}")
    prompt = request.prompt.strip()
    if not prompt:
        raise ValueError("Describe the sound: style tags for music, or what the effect is.")
    profile_name = _profile_for(request, settings)
    target = settings.new_project_target(request.render_on)
    if problem := settings.audio_profile_problem(profile_name, target=target):
        raise ValueError(problem)
    profile = settings.audio_profile(profile_name)
    if profile.kind != op:
        raise ValueError(f"{profile_name} makes {profile.kind}, not {op}")
    cap = min(profile.max_seconds, settings.audio.max_seconds)
    seconds = request.seconds or min(settings.audio.default_seconds, cap)
    if seconds > cap:
        raise ValueError(f"{profile_name} makes at most {cap:g} s per clip; ask for less or "
                         "loop or join clips afterwards")
    count = min(request.count, settings.audio.max_batch)
    seed = request.seed if request.seed is not None else random.randint(0, MAX_SEED)
    params: dict[str, Any] = {"SEED": seed, "STEPS": request.steps or profile.steps,
                              "CFG": request.cfg if request.cfg is not None else profile.cfg,
                              "SECONDS": float(seconds)}
    lyrics = request.lyrics.strip()
    if op == "music":
        params.update(TAGS=prompt, LYRICS=lyrics or INSTRUMENTAL,
                      LYRICS_STRENGTH=request.lyrics_strength)
    else:
        params.update(PROMPT=prompt, NEGATIVE_PROMPT=request.negative_prompt.strip())
    rows = [AudioClip(kind=op, status="queued", profile=profile_name, workflow=profile.workflow,
                      prompt=prompt, lyrics=lyrics, params=dict(params),
                      request=request.model_dump(mode="json"), project_id=request.project_id,
                      render_on=target, title=request.title.strip(), seed=seed)
            for _ in range(count)]
    for row in rows:
        session.add(row)
    session.flush()
    payload = {"clip_ids": [r.id for r in rows], "operation": op, "profile": profile_name,
               "workflow": profile.workflow, "params": params, "render_on": target,
               "degrade": list(profile.degrade), "resource_class": profile.resource_class,
               "actor": actor}
    job = enqueue(session, "audio", payload=payload, priority=90,
                  max_attempts=settings.jobs.default_max_attempts)
    assert job is not None
    for row in rows:
        row.job_id = job.id
    session.flush()
    return rows


def _clip_file(session: Session, store: AudioStore, clip_id: str) -> tuple[Path, AudioClip]:
    clip = session.get(AudioClip, clip_id)
    if clip is None or not clip.rel_path:
        raise LookupError(f"clip {clip_id} is not in the library or has no file yet")
    path = store.path_for(clip.rel_path)
    if not path.is_file():
        raise LookupError(f"the file of clip {clip_id} is gone")
    return path, clip


def _inputs(session: Session, store: AudioStore, edit: AudioEdit
            ) -> tuple[list[Path], list[AudioClip]]:
    """The edit's input files and the library clips among them."""
    ids = list(edit.source_ids) or ([edit.source_id] if edit.source_id else [])
    files, parents = [], []
    for clip_id in ids:
        path, clip = _clip_file(session, store, clip_id)
        files.append(path)
        parents.append(clip)
    if edit.source_path:
        path = Path(edit.source_path)
        allowed = VIDEO_EXTS | AUDIO_EXTS if edit.operation == "extract" else AUDIO_EXTS
        if path.suffix.lower() not in allowed or not path.is_file():
            raise ValueError("Pick a sound file the worker can read"
                             + (" (or a video, for extract)." if edit.operation == "extract" else "."))
        if not edit.rights_confirmed or not edit.rights_evidence.strip():
            raise ValueError("Confirm that you may use and change this file and say where it "
                             "comes from (its rights or license).")
        files.append(path)
    if not files:
        raise ValueError("Pick a clip from the library (or a file) to work on.")
    return files, parents


def edit_audio(session: Session, store: AudioStore, ffmpeg: FFmpeg, edit: AudioEdit, *,
               actor: str = "api") -> AudioClip:
    """Run one FFmpeg edit now and put the result in the library as a new clip."""
    op = edit.operation
    if op not in AUDIO_EDITS and op != "concat":
        raise ValueError(f"unknown edit {op!r}")
    files, parents = _inputs(session, store, edit)
    if op in ("mix", "concat") and len(files) < 2:
        raise ValueError(f"{'Mixing' if op == 'mix' else 'Joining'} needs two or more clips.")
    if op not in ("mix", "concat") and len(files) != 1:
        raise ValueError(f"{op} works on one clip at a time.")
    row = AudioClip(kind=op, status="running", prompt="", request=edit.model_dump(mode="json"),
                    project_id=edit.project_id or (parents[0].project_id if parents else None),
                    parent_id=parents[0].id if parents else None, title=edit.title.strip(),
                    render_on="local")
    session.add(row)
    session.flush()
    out = store.dir(row.id) / "clip.flac"
    source = files[0]
    try:
        if op == "trim":
            end = edit.end if edit.end is not None else ffmpeg.audio_info(source).duration
            ffmpeg.trim_audio(source, out, start=edit.start, end=end)
        elif op == "fade":
            if edit.fade_in <= 0 and edit.fade_out <= 0:
                raise ValueError("Set a fade in, a fade out or both.")
            ffmpeg.fade_audio(source, out, fade_in=edit.fade_in, fade_out=edit.fade_out)
        elif op == "gain":
            ffmpeg.gain_audio(source, out, db=edit.db)
        elif op == "normalize":
            ffmpeg.normalize_audio(source, out, lufs=edit.lufs)
        elif op == "loop":
            if not edit.seconds:
                raise ValueError("Say how long the looped clip should be.")
            ffmpeg.loop_audio(source, out, seconds=edit.seconds, crossfade=edit.crossfade)
        elif op == "speed":
            ffmpeg.speed_audio(source, out, factor=edit.factor, keep_pitch=edit.keep_pitch)
        elif op == "mix":
            volumes = list(edit.volumes) + [1.0] * (len(files) - len(edit.volumes))
            offsets = list(edit.offsets) + [0.0] * (len(files) - len(edit.offsets))
            ffmpeg.mix_audios(files, out, volumes=volumes[:len(files)],
                              offsets=offsets[:len(files)], duration=edit.duration)
        elif op == "concat":
            ffmpeg.concat_audio(files, out)
        elif op == "extract":
            ffmpeg.transcode_audio(source, out)
        _finish(ffmpeg, row, store, out)
    except (FFmpegError, ValueError) as exc:
        message = exc.summary if isinstance(exc, FFmpegError) else str(exc)
        session.delete(row)  # nothing to keep: the error goes back to the person
        session.flush()
        shutil.rmtree(store.root / row.id, ignore_errors=True)
        raise ValueError(f"The edit failed: {message}") from exc
    session.flush()
    return row


def _finish(ffmpeg: FFmpeg, row: AudioClip, store: AudioStore, path: Path) -> None:
    info = ffmpeg.audio_info(path)
    row.rel_path = store.rel(path)
    row.duration_s, row.sample_rate, row.channels = round(info.duration, 3), info.sample_rate, info.channels
    row.status, row.error, row.finished_at = "done", None, utcnow()


def import_audio(session: Session, store: AudioStore, ffmpeg: FFmpeg, path: Path, *,
                 kind: str = "upload", title: str = "", project_id: str | None = None,
                 request: dict[str, Any] | None = None) -> AudioClip:
    """Put a sound file (an upload, a video's sound) into the library as FLAC."""
    row = AudioClip(kind=kind, status="running", title=title, project_id=project_id,
                    request=request or {}, render_on="local")
    session.add(row)
    session.flush()
    dest = store.dir(row.id) / "clip.flac"
    try:
        if Path(path).suffix.lower() == ".flac" and not ffmpeg.audio_info(path).has_video:
            shutil.copy2(path, dest)
        else:
            ffmpeg.transcode_audio(path, dest)
        _finish(ffmpeg, row, store, dest)
    except FFmpegError as exc:
        session.delete(row)
        session.flush()
        raise ValueError(f"not a sound the studio can read: {exc.summary}") from exc
    session.flush()
    return row


def list_clips(session: Session, *, limit: int = 60, offset: int = 0, kind: str | None = None,
               project_id: str | None = None, status: str | None = None) -> list[AudioClip]:
    stmt = (select(AudioClip).order_by(AudioClip.created_at.desc(), AudioClip.id.desc())
            .offset(offset).limit(limit))
    if kind:
        stmt = stmt.where(AudioClip.kind == kind)
    if project_id:
        stmt = stmt.where(AudioClip.project_id == project_id)
    if status:
        stmt = stmt.where(AudioClip.status == status)
    return list(session.scalars(stmt))


def get_clip(session: Session, clip_id: str) -> AudioClip:
    clip = session.get(AudioClip, clip_id)
    if clip is None:
        raise LookupError(f"clip {clip_id} not found")
    return clip


def set_verdict(session: Session, clip: AudioClip, value: int | None) -> AudioClip:
    if value is not None and value not in (-2, -1, 1, 2):
        raise ValueError("a verdict is -2, -1, 1 or 2")
    clip.verdict = value
    session.flush()
    return clip


def delete_clip(session: Session, store: AudioStore, clip: AudioClip) -> None:
    """Remove a clip and its files. Clips made from it keep their own files."""
    job = session.get(Job, clip.job_id) if clip.job_id else None
    if clip.status in ("queued", "running") and job is not None and job.status in ("QUEUED", "RETRY_WAIT"):
        job.status = "CANCELLED"
        job.finished_at = utcnow()
    for child in session.scalars(select(AudioClip).where(AudioClip.parent_id == clip.id)):
        child.parent_id = None
    session.delete(clip)
    session.flush()
    folder = store.root / clip.id
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def rights_line(clip: AudioClip) -> str:
    """Where a clip's right to be used comes from, for the soundtrack rights record."""
    evidence = str((clip.request or {}).get("rights_evidence") or "").strip()
    if clip.kind in ("music", "sound"):
        return f"made in the studio with {clip.profile or clip.kind} (clip {clip.id})"
    if evidence:
        return f"{evidence} (clip {clip.id}, {clip.kind})"
    return f"clip {clip.id} ({clip.kind}) from the sound library"


def clip_view(clip: AudioClip) -> dict[str, Any]:
    """What pages and the API show for a clip."""
    return {
        "id": clip.id, "kind": clip.kind, "status": clip.status, "title": clip.title,
        "prompt": clip.prompt, "lyrics": clip.lyrics, "profile": clip.profile,
        "workflow": clip.workflow, "duration_s": clip.duration_s, "sample_rate": clip.sample_rate,
        "channels": clip.channels, "seed": clip.seed, "parent_id": clip.parent_id,
        "project_id": clip.project_id, "render_on": clip.render_on, "verdict": clip.verdict,
        "error": clip.error, "took_s": clip.took_s, "params": clip.params, "request": clip.request,
        "file_url": f"/audio/{clip.id}/file" if clip.rel_path else None,
        "waveform_url": f"/audio/{clip.id}/waveform.png" if clip.rel_path else None,
        "created_at": clip.created_at.isoformat() if isinstance(clip.created_at, datetime) else None,
    }
