"""Use cases shared by the API and the CLI (create, start, cancel, resume, decide)."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from rokkur_studio.api.schemas import ProjectCreate
from rokkur_studio.config import Settings
from rokkur_studio.db.models import ApprovalRequest, Channel, Job, Project, utcnow
from rokkur_studio.domain.rights import RightsCategory, RightsStatus
from rokkur_studio.domain.states import TERMINAL, InvalidTransition, ProjectStatus
from rokkur_studio.jobs.queue import cancel_project_jobs
from rokkur_studio.pipeline.driver import advance
from rokkur_studio.services.events import EventType, record_event
from rokkur_studio.services.projects import (
    create_source,
    get_project,
    latest_rights,
    record_rights,
    resume,
    transition,
)

S = ProjectStatus


def create_project(session: Session, data: ProjectCreate, settings: Settings,
                   actor: str = "api") -> Project:
    profile = data.render_profile or settings.render.default_profile
    settings.profile(profile)  # validate early
    if data.channel_id and session.get(Channel, data.channel_id) is None:
        raise LookupError(f"channel {data.channel_id} not found")
    project = Project(name=data.name, status=S.DISCOVERED.value, target_format=data.target_format,
                      render_profile=profile, channel_id=data.channel_id,
                      creative_input=data.creative.model_dump(exclude_none=True))
    session.add(project)
    session.flush()
    create_source(session, project, **data.source.model_dump())
    r = data.rights
    record_rights(session, project.id, category=r.category.value, status=RightsStatus.PENDING.value,
                  decided_by=actor, reason="declared at project creation",
                  license=r.license, owner=r.owner, permission_evidence=r.permission_evidence,
                  attribution_required=r.attribution_required, attribution_text=r.attribution_text,
                  allowed_transformations=r.allowed_transformations,
                  commercial_use=r.commercial_use)
    record_event(session, EventType.PROJECT_CREATED, project_id=project.id, actor=actor,
                 to_state=project.status,
                 data={"name": project.name, "profile": profile,
                       "rights_category": r.category.value, "source": data.source.platform})
    session.flush()
    session.refresh(project)
    if data.autostart:
        start(session, project, settings, actor=actor)
    return project


def start(session: Session, project: Project, settings: Settings, actor: str = "api") -> Job | None:
    if project.status in (S.DISCOVERED, S.SCORED):
        transition(session, project, S.RIGHTS_PENDING, actor=actor)
    return advance(session, project, settings, manual=True)


def cancel(session: Session, project: Project, actor: str = "api") -> int:
    if S(project.status) in TERMINAL:
        raise InvalidTransition(S(project.status), S.CANCELLED, "project already finished")
    n = cancel_project_jobs(session, project.id)
    transition(session, project, S.CANCELLED, actor=actor, data={"cancelled_jobs": n})
    return n


def resume_project(session: Session, project: Project, settings: Settings,
                   actor: str = "api") -> Job | None:
    resume(session, project, actor=actor)
    return advance(session, project, settings, manual=True)


def decide_rights(session: Session, project: Project, settings: Settings, *, approve: bool,
                  decided_by: str, fields: dict[str, Any], note: str | None = None) -> Project:
    """A human rights decision: the only way UNKNOWN/ambiguous sources get unblocked."""
    if project.status != S.RIGHTS_PENDING:
        raise InvalidTransition(S(project.status), S.RIGHTS_OK, "project is not awaiting rights")
    previous = latest_rights(session, project.id)
    category = fields.pop("category", None) or (previous.category if previous else "UNKNOWN")
    if approve and category in (RightsCategory.REJECTED, RightsCategory.REFERENCE_ONLY):
        raise ValueError(f"cannot approve ingestion for category {category}")
    record_rights(session, project.id, category=str(category),
                  status=(RightsStatus.APPROVED if approve else RightsStatus.REJECTED).value,
                  decided_by=decided_by, reason=note or "human decision", **fields)
    for req in session.query(ApprovalRequest).filter_by(project_id=project.id,
                                                        kind="rights_ambiguity",
                                                        status="pending"):
        req.status = "approved" if approve else "rejected"
        req.decided_by, req.decided_at, req.note = decided_by, utcnow(), note
    record_event(session, EventType.APPROVAL_DECIDED, project_id=project.id, actor=decided_by,
                 data={"kind": "rights_ambiguity", "approve": approve, "note": note})
    transition(session, project, S.RIGHTS_OK if approve else S.RIGHTS_REJECTED,
               actor=decided_by, reason=note)
    if approve:
        advance(session, project, settings)
    return project


def decide_approval(session: Session, request: ApprovalRequest, settings: Settings, *,
                    approve: bool, decided_by: str, note: str | None) -> ApprovalRequest:
    if request.status != "pending":
        raise ValueError(f"approval {request.id} already {request.status}")
    if request.kind == "publish" and approve:
        # Approving uploads the video; that needs the YouTube client (publishing.approve_proposal).
        raise ValueError("a publish request is approved through publishing.approve_proposal")
    if request.kind == "rights_ambiguity" and request.project_id:
        project = get_project(session, request.project_id, for_update=True)
        decide_rights(session, project, settings, approve=approve, decided_by=decided_by,
                      fields={}, note=note)
        return request
    request.status = "approved" if approve else "rejected"
    request.decided_by, request.decided_at, request.note = decided_by, utcnow(), note
    record_event(session, EventType.APPROVAL_DECIDED, project_id=request.project_id,
                 actor=decided_by, data={"kind": request.kind, "approve": approve, "note": note})
    return request
