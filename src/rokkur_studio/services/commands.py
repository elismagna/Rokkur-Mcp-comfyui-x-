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
    at_budget_limit,
    at_repair_limit,
    create_source,
    get_project,
    latest_document,
    latest_rights,
    record_rights,
    resume,
    save_document,
    transition,
)

S = ProjectStatus


def create_project(session: Session, data: ProjectCreate, settings: Settings,
                   actor: str = "api") -> Project:
    profile = data.render_profile or settings.render.default_profile
    settings.profile(profile)  # validate early
    if problem := settings.profile_problem(profile):
        raise ValueError(problem)
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
    if problem := settings.profile_problem(project.render_profile):
        raise ValueError(problem)
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
    if at_repair_limit(project):
        # Resuming would re-enter repair and stop again at once: a human picks the way out.
        raise InvalidTransition(S.FAILED, S.QUALITY_FAILED,
                                "stopped at the repair limit: allow more repair rounds or keep "
                                "the current renders")
    if at_budget_limit(session, project, settings):
        # The budget check would stop it again before the first render.
        raise InvalidTransition(S.FAILED, S(project.failed_from_state or S.FAILED),
                                "stopped at the render budget: allow more renders first")
    resume(session, project, actor=actor)
    return advance(session, project, settings, manual=True)


def render_grant(settings: Settings) -> int:
    """Renders one "allow more renders" adds by default: half the configured budget."""
    return max(1, settings.render.max_renders_per_project // 2)


def allow_more_renders(session: Session, project: Project, settings: Settings, *, actor: str,
                       renders: int | None = None) -> Job | None:
    """Raise the render budget of a project stopped at it, and resume it.

    GPU minutes grow in proportion, so whichever of the two budgets stopped it is lifted.
    """
    if not at_budget_limit(session, project, settings):
        raise InvalidTransition(S(project.status), S(project.status),
                                "project is not stopped at the render budget")
    base = settings.render.max_renders_per_project
    renders = renders or render_grant(settings)
    if not 1 <= renders <= base:
        raise ValueError(f"allow between 1 and {base} more renders")
    gpu_minutes = round(renders * settings.costs.max_gpu_minutes_per_project / base, 1)
    record_event(session, EventType.BUDGET_EXTENDED, project_id=project.id, actor=actor,
                 data={"renders": renders, "gpu_minutes": gpu_minutes})
    resume(session, project, actor=actor)
    return advance(session, project, settings, manual=True)


def _close_repair_requests(session: Session, project: Project, *, decided_by: str,
                           note: str) -> None:
    for req in session.query(ApprovalRequest).filter_by(project_id=project.id,
                                                        kind="repair_budget", status="pending"):
        req.status = "approved"
        req.decided_by, req.decided_at, req.note = decided_by, utcnow(), note


def repair_more(session: Session, project: Project, settings: Settings, *, actor: str,
                rounds: int | None = None) -> Job | None:
    """Grant more repair rounds to a project stopped at the repair limit, and resume it."""
    if not at_repair_limit(project):
        raise InvalidTransition(S(project.status), S.QUALITY_FAILED,
                                "project is not stopped at the repair limit")
    rounds = rounds or settings.render.max_retries
    if not 1 <= rounds <= 10:
        raise ValueError("grant between 1 and 10 repair rounds")
    record_event(session, EventType.REPAIR_BUDGET_EXTENDED, project_id=project.id, actor=actor,
                 data={"rounds": rounds, "after": project.repair_rounds})
    _close_repair_requests(session, project, decided_by=actor,
                           note=f"{rounds} more repair rounds")
    resume(session, project, actor=actor)
    return advance(session, project, settings, manual=True)


def recheck_quality(session: Session, project: Project, settings: Settings, *,
                    actor: str) -> Job | None:
    """Score the current renders again without rendering, e.g. after QC itself changed."""
    if not at_repair_limit(project):
        raise InvalidTransition(S(project.status), S.QUALITY_CHECK,
                                "project is not stopped at the repair limit")
    _close_repair_requests(session, project, decided_by=actor, note="quality check run again")
    resume(session, project, actor=actor, at=S.QUALITY_CHECK)
    return advance(session, project, settings, manual=True)


def keep_renders(session: Session, project: Project, settings: Settings, *, actor: str,
                 note: str | None = None) -> Job | None:
    """A human keeps the renders QC rejected: the video is edited from them as they are.

    The override is a new QC report version marked PASS that names who decided and what QC
    measured, so the QUALITY_PASSED and publish gates see a recorded human decision.
    """
    if not at_repair_limit(project):
        raise InvalidTransition(S(project.status), S.QUALITY_PASSED,
                                "only a project stopped at the repair limit can keep its renders")
    qc = latest_document(session, project.id, "qc_report")
    if qc is None:
        raise InvalidTransition(S.FAILED, S.QUALITY_PASSED, "no QC report to override")
    override = {"by": actor, "at": utcnow().isoformat(), "note": note,
                "qc_decision": qc.data.get("decision"),
                "failed_shots": qc.data.get("failed_shots", []), "qc_version": qc.version}
    save_document(session, project.id, "qc_report",
                  {**qc.data, "decision": "PASS", "override": override}, created_by=actor)
    record_event(session, EventType.QC_OVERRIDDEN, project_id=project.id, actor=actor,
                 data=override)
    _close_repair_requests(session, project, decided_by=actor, note="kept the current renders")
    resume(session, project, actor=actor)
    transition(session, project, S.QUALITY_PASSED, actor=actor,
               reason="renders kept by a human despite QC")
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
    if request.kind == "repair_budget" and approve and request.project_id:
        project = get_project(session, request.project_id, for_update=True)
        if not at_repair_limit(project):
            raise ValueError("the project is no longer stopped at the repair limit; reject this "
                             "request to clear it")
        repair_more(session, project, settings, actor=decided_by)
        record_event(session, EventType.APPROVAL_DECIDED, project_id=request.project_id,
                     actor=decided_by, data={"kind": request.kind, "approve": True, "note": note})
        return request
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
