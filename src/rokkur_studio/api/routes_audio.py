"""/audio endpoints: the sound library (docs/audio.md)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.db.models import AudioClip
from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import audio as svc
from rokkur_studio.services.audio import AUDIO_EXTS, AudioEdit, AudioRequest, AudioStore, clip_view

router = APIRouter(prefix="/audio", tags=["audio"])
Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]


def _store(ctx: StudioContext) -> AudioStore:
    return AudioStore(ctx.settings.studio.data_dir)


def _clip(session: Session, clip_id: str) -> AudioClip:
    try:
        return svc.get_clip(session, clip_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("", summary="The sound library, newest first")
def list_sounds(session: Db, limit: int = Query(60, le=500), offset: int = 0,
                kind: str | None = None, project_id: str | None = None,
                status: str | None = None) -> list[dict[str, Any]]:
    return [clip_view(c) for c in svc.list_clips(session, limit=limit, offset=offset, kind=kind,
                                                 project_id=project_id, status=status)]


@router.get("/profiles", summary="Audio profiles and whether each can generate now")
def profiles(ctx: Ctx, target: str = "local") -> dict[str, Any]:
    return {name: {"description": profile.description, "kind": profile.kind,
                   "max_seconds": profile.max_seconds,
                   "problem": ctx.settings.audio_profile_problem(name, target=target)}
            for name, profile in ctx.settings.audio_profiles.items()}


@router.post("", status_code=201, summary="Ask for music or a sound; one job makes it")
def create_sounds(body: AudioRequest, ctx: Ctx, session: Db) -> list[dict[str, Any]]:
    try:
        rows = svc.request_audio(session, ctx.settings, _store(ctx), body)
    except (ValueError, LookupError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return [clip_view(r) for r in rows]


@router.post("/edit", status_code=201, summary="Trim, fade, gain, normalise, loop, speed, mix, "
                                               "join or extract; done at once with FFmpeg")
def edit_sound(body: AudioEdit, ctx: Ctx, session: Db) -> dict[str, Any]:
    try:
        return clip_view(svc.edit_audio(session, _store(ctx), ctx.ffmpeg, body))
    except (ValueError, LookupError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/upload", status_code=201, summary="Put your own sound into the library")
def upload_sound(ctx: Ctx, session: Db, file: Annotated[UploadFile, File()],
                 rights_confirmed: Annotated[bool, Form()] = False,
                 rights_evidence: Annotated[str, Form()] = "",
                 title: Annotated[str, Form()] = "") -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(422, "Use a sound file such as MP3, WAV, FLAC, M4A or OGG.")
    if not rights_confirmed or not rights_evidence.strip():
        raise HTTPException(422, "Confirm that you may use and change the sound and say where "
                                 "it comes from.")
    uploads = Path(ctx.settings.studio.data_dir) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{uuid.uuid4().hex[:12]}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        row = svc.import_audio(session, _store(ctx), ctx.ffmpeg, dest, kind="upload",
                               title=title or Path(file.filename or "").stem,
                               request={"rights_confirmed": True,
                                        "rights_evidence": rights_evidence.strip()})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        dest.unlink(missing_ok=True)
    return clip_view(row)


@router.get("/{clip_id}")
def get_sound(clip_id: str, session: Db) -> dict[str, Any]:
    return clip_view(_clip(session, clip_id))


@router.get("/{clip_id}/file")
def sound_file(clip_id: str, ctx: Ctx, session: Db) -> FileResponse:
    clip = _clip(session, clip_id)
    if not clip.rel_path:
        raise HTTPException(404, "this clip has no file yet")
    return FileResponse(_store(ctx).path_for(clip.rel_path), media_type="audio/flac",
                        filename=f"rokkur-{clip.id}.flac")


@router.get("/{clip_id}/waveform.png")
def sound_waveform(clip_id: str, ctx: Ctx, session: Db) -> FileResponse:
    """A picture of the clip's waveform, drawn once and kept next to the clip."""
    clip = _clip(session, clip_id)
    if not clip.rel_path:
        raise HTTPException(404, "this clip has no file yet")
    store = _store(ctx)
    png = store.dir(clip.id) / "waveform.png"
    if not png.is_file():
        try:
            ctx.ffmpeg.waveform(store.path_for(clip.rel_path), png)
        except FFmpegError as exc:
            raise HTTPException(500, f"waveform failed: {exc.summary}") from exc
    return FileResponse(png, media_type="image/png")


class VerdictIn(BaseModel):
    value: int | None = None


@router.post("/{clip_id}/verdict")
def set_sound_verdict(clip_id: str, body: VerdictIn, session: Db) -> dict[str, Any]:
    try:
        return clip_view(svc.set_verdict(session, _clip(session, clip_id), body.value))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/{clip_id}", status_code=204)
def delete_sound(clip_id: str, ctx: Ctx, session: Db) -> None:
    svc.delete_clip(session, _store(ctx), _clip(session, clip_id))
