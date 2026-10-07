"""Project lifecycle: creation, audited state transitions, gates, documents."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rokkur_studio.db.models import Document, Project, RightsDecision, Source
from rokkur_studio.domain.rights import RightsStatus
from rokkur_studio.domain.states import (
    RESUMABLE,
    InvalidTransition,
    ProjectStatus,
    assert_transition,
)
from rokkur_studio.services.events import EventType, record_event

S = ProjectStatus

# Semantic event emitted in addition to STATE_CHANGED for notable states.
_STATE_EVENTS: dict[ProjectStatus, str] = {
    S.RIGHTS_OK: EventType.RIGHTS_APPROVED,
    S.RIGHTS_REJECTED: EventType.RIGHTS_REJECTED,
    S.DOWNLOADED_OR_INGESTED: EventType.SOURCE_INGESTED,
    S.ANALYZING: EventType.ANALYSIS_STARTED,
    S.ANALYZED: EventType.ANALYSIS_COMPLETED,
    S.QUALITY_PASSED: EventType.QC_PASSED,
    S.QUALITY_FAILED: EventType.QC_FAILED,
    S.REPAIRING: EventType.REPAIR_REQUESTED,
    S.PUBLISHED: EventType.VIDEO_PUBLISHED,
    S.CANCELLED: EventType.PROJECT_CANCELLED,
}


class GateViolation(InvalidTransition):
    """A legal edge whose precondition (rights, QC, approval) is not met."""


class ProjectNotFound(LookupError):
    pass


def get_project(session: Session, project_id: str, *, for_update: bool = False) -> Project:
    stmt = select(Project).where(Project.id == project_id)
    if for_update:
        stmt = stmt.with_for_update()
    project = session.scalars(stmt).one_or_none()
    if project is None:
        raise ProjectNotFound(project_id)
    return project


def latest_rights(session: Session, project_id: str) -> RightsDecision | None:
    return session.scalars(
        select(RightsDecision)
        .where(RightsDecision.project_id == project_id)
        .order_by(RightsDecision.decided_at.desc(), RightsDecision.id.desc())
        .limit(1)
    ).one_or_none()


def rights_approved(session: Session, project_id: str) -> bool:
    decision = latest_rights(session, project_id)
    return decision is not None and decision.status == RightsStatus.APPROVED


def latest_document(session: Session, project_id: str, kind: str) -> Document | None:
    return session.scalars(
        select(Document)
        .where(Document.project_id == project_id, Document.kind == kind)
        .order_by(Document.version.desc())
        .limit(1)
    ).one_or_none()


def save_document(
    session: Session, project_id: str, kind: str, data: dict[str, Any], *, created_by: str,
    schema_version: int = 1,
) -> Document:
    current = session.scalar(
        select(func.max(Document.version)).where(
            Document.project_id == project_id, Document.kind == kind
        )
    )
    doc = Document(
        project_id=project_id,
        kind=kind,
        version=(current or 0) + 1,
        schema_version=schema_version,
        data=data,
        created_by=created_by,
    )
    session.add(doc)
    session.flush()
    return doc


def _check_gates(session: Session, project: Project, target: ProjectStatus) -> None:
    if target in (S.DOWNLOADED_OR_INGESTED, S.PUBLISHING) and not rights_approved(
        session, project.id
    ):
        raise GateViolation(S(project.status), target, "rights are not approved")
    if target is S.PUBLISHING:
        qc = latest_document(session, project.id, "qc_report")
        if qc is None or qc.data.get("decision") != "PASS":
            raise GateViolation(S(project.status), target, "latest QC report did not pass")
    if target is S.RIGHTS_OK and not rights_approved(session, project.id):
        raise GateViolation(S(project.status), target, "no approved rights decision on record")


def transition(
    session: Session,
    project: Project,
    target: ProjectStatus,
    *,
    actor: str,
    reason: str | None = None,
    data: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> Project:
    """The only way a project's status changes. Validates, enforces gates, audits."""
    current = S(project.status)
    assert_transition(current, target)
    _check_gates(session, project, target)
    if target is S.FAILED:
        project.failed_from_state = current.value
    project.status = target.value
    payload = {**(data or {}), **({"reason": reason} if reason else {})}
    record_event(
        session,
        EventType.STATE_CHANGED,
        project_id=project.id,
        actor=actor,
        from_state=current.value,
        to_state=target.value,
        data=payload,
        job_id=job_id,
    )
    if target in _STATE_EVENTS:
        record_event(
            session, _STATE_EVENTS[target], project_id=project.id, actor=actor,
            data=payload, job_id=job_id,
        )
    session.flush()
    return project


def resume(session: Session, project: Project, *, actor: str) -> Project:
    """Return a FAILED project to the state it failed in, so its stage job can run again."""
    if project.status != S.FAILED:
        raise InvalidTransition(S(project.status), S(project.status), "only FAILED projects resume")
    previous = S(project.failed_from_state) if project.failed_from_state else None
    if previous is None or previous not in RESUMABLE:
        raise InvalidTransition(S.FAILED, previous or S.FAILED, "no resumable state recorded")
    # In-progress states are re-entered at the state that schedules their job.
    restart = _RESTART_FROM.get(previous, previous)
    project.status = restart.value
    project.failed_from_state = None
    record_event(
        session, EventType.PROJECT_RESUMED, project_id=project.id, actor=actor,
        from_state=S.FAILED.value, to_state=restart.value,
    )
    session.flush()
    return project


# A project that failed mid-stage restarts from the state that enqueues that stage.
_RESTART_FROM: dict[ProjectStatus, ProjectStatus] = {
    S.ANALYZING: S.DOWNLOADED_OR_INGESTED,
    S.CREATIVE_PLANNING: S.ANALYZED,
    S.WORKFLOW_COMPILING: S.CREATIVE_READY,
    S.RENDER_QUEUED: S.WORKFLOW_READY,
    S.RENDERING: S.WORKFLOW_READY,
    S.REPAIRING: S.QUALITY_FAILED,
    S.EDITING: S.QUALITY_PASSED,
    S.PUBLISHING: S.READY_TO_PUBLISH,
}


def create_source(session: Session, project: Project, **fields: Any) -> Source:
    source = Source(project_id=project.id, **fields)
    session.add(source)
    return source


def record_rights(session: Session, project_id: str, *, category: str, status: str,
                  decided_by: str, reason: str | None = None, **fields: Any) -> RightsDecision:
    decision = RightsDecision(project_id=project_id, category=category, status=status,
                              decided_by=decided_by, reason=reason, **fields)
    session.add(decision)
    session.flush()
    return decision
