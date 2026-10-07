"""/youtube endpoints: the channel's playlists and the release schedule."""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.db.models import utcnow
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import publishing
from rokkur_studio.youtube.client import YouTubeError
from rokkur_studio.youtube.oauth import OAuthError

router = APIRouter(prefix="/youtube", tags=["youtube"])
Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]


@router.get("/playlists", summary="Playlists saved by the last refresh")
def playlists(ctx: Ctx) -> dict[str, Any]:
    return publishing.load_playlists(ctx.settings) or {"fetched_at": None, "items": []}


@router.post("/playlists/refresh", summary="Ask YouTube for the channel's playlists (1 quota unit)")
def refresh_playlists(ctx: Ctx) -> dict[str, Any]:
    try:
        with publishing.youtube_client(ctx.settings, ctx.extras.get("youtube_client"),
                                       uploads=False) as yt:
            return publishing.refresh_playlists(ctx.settings, yt)
    except OAuthError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (YouTubeError, httpx.HTTPError) as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/releases", summary="Release times, the next free one, and upcoming releases")
def releases(ctx: Ctx, session: Db) -> dict[str, Any]:
    return release_overview(ctx, session)


def release_overview(ctx: StudioContext, session: Session) -> dict[str, Any]:
    s = ctx.settings
    now = utcnow()
    nxt = publishing.next_release_slot(session, s, now=now)
    upcoming = [{"project_id": pub.project_id, "youtube_video_id": pub.youtube_video_id,
                 "title": pub.request["body"]["snippet"]["title"],
                 "publish_at": publishing.iso(when),
                 "label": publishing.local_label(s, when)}
                for when, pub in publishing.scheduled_releases(session) if when > now]
    return {"timezone": s.youtube.timezone, "release_times": s.youtube.release_times,
            "allow_public": s.youtube.allow_public,
            "min_lead_minutes": s.youtube.min_lead_minutes,
            "next_slot": publishing.iso(nxt),
            "next_slot_label": publishing.local_label(s, nxt) if nxt else None,
            "upcoming": upcoming}
