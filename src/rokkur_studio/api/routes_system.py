"""/jobs, /workers, /gpu, /channels, /approvals, /system endpoints."""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from rokkur_studio import __version__
from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.api.schemas import (
    ApprovalDecision,
    ApprovalOut,
    ChannelIn,
    ChannelOut,
    JobOut,
)
from rokkur_studio.db.models import ApprovalRequest, Channel, GpuLease, Job, utcnow
from rokkur_studio.jobs.queue import JobStatus
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import commands, publishing, taste
from rokkur_studio.youtube.client import YouTubeError
from rokkur_studio.youtube.oauth import OAuthError

router = APIRouter()
Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]


@router.get("/jobs", response_model=list[JobOut], tags=["jobs"])
def list_jobs(session: Db, status: str | None = None, project_id: str | None = None,
              limit: int = Query(200, le=1000)) -> list[Job]:
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Job.status == status)
    if project_id:
        stmt = stmt.where(Job.project_id == project_id)
    return list(session.scalars(stmt))


@router.get("/workers", tags=["jobs"],
            summary="Workers currently holding job leases, and queue depth")
def workers(session: Db) -> dict[str, Any]:
    running = session.execute(
        select(Job.locked_by, func.count(Job.id), func.max(Job.locked_until))
        .where(Job.status == JobStatus.RUNNING).group_by(Job.locked_by)).all()
    depth = dict(session.execute(select(Job.status, func.count(Job.id))
                                 .group_by(Job.status)).all())
    return {"active": [{"worker_id": w, "running_jobs": n, "lease_until": u}
                       for w, n, u in running],
            "queue": depth}


@router.get("/gpu/leases", tags=["system"])
def gpu_leases(ctx: Ctx, session: Db) -> dict[str, Any]:
    active = list(session.scalars(select(GpuLease).where(GpuLease.released_at.is_(None),
                                                         GpuLease.expires_at >= utcnow())))
    return {"vram_gb": ctx.settings.gpu.vram_gb,
            "in_use_gb": sum(lease.vram_gb for lease in active),
            "leases": [{"id": lease.id, "holder": lease.holder, "class": lease.resource_class,
                        "vram_gb": lease.vram_gb, "expires_at": lease.expires_at}
                       for lease in active]}


@router.get("/channels", response_model=list[ChannelOut], tags=["channels"])
def list_channels(session: Db) -> list[Channel]:
    return list(session.scalars(select(Channel).order_by(Channel.created_at)))


@router.post("/channels", response_model=ChannelOut, status_code=201, tags=["channels"])
def create_channel(body: ChannelIn, session: Db) -> Channel:
    channel = Channel(**body.model_dump())
    session.add(channel)
    session.flush()
    return channel


@router.patch("/channels/{channel_id}", response_model=ChannelOut, tags=["channels"])
def update_channel(channel_id: str, body: ChannelIn, session: Db) -> Channel:
    channel = session.get(Channel, channel_id)
    if channel is None:
        raise HTTPException(404, "channel not found")
    for key, value in body.model_dump().items():
        setattr(channel, key, value)
    return channel


@router.get("/approvals", response_model=list[ApprovalOut], tags=["approvals"])
def list_approvals(session: Db, status: str | None = "pending") -> list[ApprovalRequest]:
    stmt = select(ApprovalRequest).order_by(ApprovalRequest.requested_at.desc())
    if status:
        stmt = stmt.where(ApprovalRequest.status == status)
    return list(session.scalars(stmt))


@router.post("/approvals/{approval_id}", response_model=ApprovalOut, tags=["approvals"])
def decide(approval_id: str, body: ApprovalDecision, ctx: Ctx,
           session: Db) -> ApprovalRequest | Response:
    req = session.get(ApprovalRequest, approval_id, with_for_update=True)
    if req is None:
        raise HTTPException(404, "approval not found")
    if req.kind == "publish" and body.approve:  # approving uploads the video as planned
        try:
            with publishing.youtube_client(ctx.settings, ctx.extras.get("youtube_client")) as yt:
                publishing.approve_proposal(session, req, settings=ctx.settings, store=ctx.store,
                                            client=yt, decided_by=body.decided_by,
                                            note=body.note)
        except (publishing.PublishGateError, OAuthError) as exc:
            raise HTTPException(409, str(exc)) from exc
        except YouTubeError as exc:  # returned so the failed attempt's audit trail commits
            return JSONResponse({"detail": str(exc)}, status_code=502)
        return req
    try:
        return commands.decide_approval(session, req, ctx.settings, approve=body.approve,
                                        decided_by=body.decided_by, note=body.note)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _probe(url: str) -> dict[str, Any]:
    try:
        r = httpx.get(url, timeout=3)
        return {"reachable": True, "status": r.status_code}
    except httpx.HTTPError as exc:
        return {"reachable": False, "error": str(exc)}


@router.get("/system", tags=["system"])
def system(ctx: Ctx, session: Db, probe_services: bool = True) -> dict[str, Any]:
    s = ctx.settings
    out: dict[str, Any] = {
        "version": __version__,
        "database": session.execute(text("SELECT 1")).scalar() == 1,
        "ffmpeg": ctx.ffmpeg.available(),
        "renderer": s.render.renderer,
        "agent_provider": s.agents.provider,
        "autonomy_level": s.studio.autonomy_level,
        "default_profile": s.render.default_profile,
        "profiles": sorted(s.profiles),
        "workflow_templates": ctx.registry.names(),
        "youtube_enabled": s.youtube.enabled,
    }
    if probe_services:
        out["comfyui"] = _probe(f"{s.comfyui.url}/system_stats")
        out["ollama"] = _probe(f"{s.ollama.url}/api/version")
    return out


@router.get("/taste", tags=["ratings"],
            summary="What your ratings say works, and the suggestions they support")
def taste_profile(session: Db) -> dict[str, Any]:
    profile = taste.build_profile(session)
    return {**profile, "suggestions": taste.suggestions(profile)}


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
