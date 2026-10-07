"""Small server-rendered control dashboard (Jinja2, no frontend build step)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.api.deps import get_ctx, get_session
from rokkur_studio.api.routes_projects import project_detail
from rokkur_studio.api.routes_system import system as system_info
from rokkur_studio.api.routes_system import workers as workers_info
from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
from rokkur_studio.db.models import ApprovalRequest, Event, Job, Project
from rokkur_studio.domain.rights import RightsCategory
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import commands
from rokkur_studio.services.projects import get_project

router = APIRouter(prefix="/ui", include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["pretty"] = lambda v: json.dumps(v, indent=2, default=str)

Ctx = Annotated[StudioContext, Depends(get_ctx)]
Db = Annotated[Session, Depends(get_session)]

WORKING = {"RIGHTS_PENDING", "ANALYZING", "CREATIVE_PLANNING", "WORKFLOW_COMPILING",
           "RENDER_QUEUED", "RENDERING", "QUALITY_CHECK", "REPAIRING", "EDITING", "PUBLISHING"}


def _page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {"working": WORKING, **ctx})


@router.get("", response_class=HTMLResponse)
def projects_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    projects = list(session.scalars(select(Project).order_by(Project.created_at.desc())
                                    .limit(200)))
    return _page(request, "projects.html", projects=projects,
                 profiles=sorted(ctx.settings.profiles),
                 default_profile=ctx.settings.render.default_profile,
                 categories=[c.value for c in RightsCategory])


@router.post("/projects")
def create_from_form(ctx: Ctx, session: Db, name: Annotated[str, Form()],
                     theme: Annotated[str, Form()], local_path: Annotated[str, Form()],
                     rights_category: Annotated[str, Form()],
                     prompt: Annotated[str, Form()] = "",
                     permission_evidence: Annotated[str, Form()] = "",
                     character_reference_path: Annotated[str, Form()] = "",
                     render_profile: Annotated[str, Form()] = "",
                     target_format: Annotated[str, Form()] = "youtube_short",
                     autostart: Annotated[bool, Form()] = False) -> RedirectResponse:
    body = ProjectCreate(
        name=name, target_format=target_format,  # type: ignore[arg-type]
        render_profile=render_profile or None,
        source=SourceIn(platform="local", local_path=local_path),
        rights=RightsIn(category=RightsCategory(rights_category),
                        permission_evidence=permission_evidence or None),
        creative=CreativeIn(theme=theme, prompt=prompt or None,
                            character_reference_path=character_reference_path or None),
        autostart=autostart)
    project = commands.create_project(session, body, ctx.settings, actor="dashboard")
    return RedirectResponse(f"/ui/projects/{project.id}", status_code=303)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(project_id: str, request: Request, session: Db) -> HTMLResponse:
    try:
        project = get_project(session, project_id)
    except LookupError as exc:
        raise HTTPException(404) from exc
    detail = project_detail(session, project)
    events = list(session.scalars(select(Event).where(Event.project_id == project_id)
                                  .order_by(Event.id.desc()).limit(100)))
    return _page(request, "project.html", d=detail, events=events)


@router.post("/projects/{project_id}/{action}")
def project_action(project_id: str, action: str, ctx: Ctx, session: Db) -> RedirectResponse:
    project = get_project(session, project_id, for_update=True)
    try:
        if action == "start":
            commands.start(session, project, ctx.settings, actor="dashboard")
        elif action == "cancel":
            commands.cancel(session, project, actor="dashboard")
        elif action == "resume":
            commands.resume_project(session, project, ctx.settings, actor="dashboard")
        elif action in ("approve-rights", "reject-rights"):
            commands.decide_rights(session, project, ctx.settings,
                                   approve=action == "approve-rights", decided_by="dashboard",
                                   fields={}, note="decided in dashboard")
        else:
            raise HTTPException(404)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return RedirectResponse(f"/ui/projects/{project_id}", status_code=303)


@router.get("/queue", response_class=HTMLResponse)
def queue_page(request: Request, session: Db) -> HTMLResponse:
    jobs = list(session.scalars(select(Job).order_by(Job.created_at.desc()).limit(200)))
    return _page(request, "queue.html", jobs=jobs, workers=workers_info(session))


@router.get("/approvals", response_class=HTMLResponse)
def approvals_page(request: Request, session: Db) -> HTMLResponse:
    items = list(session.scalars(select(ApprovalRequest)
                                 .order_by(ApprovalRequest.requested_at.desc()).limit(200)))
    return _page(request, "approvals.html", items=items)


@router.get("/system", response_class=HTMLResponse)
def system_page(request: Request, ctx: Ctx, session: Db) -> HTMLResponse:
    return _page(request, "system.html", info=system_info(ctx, session))
