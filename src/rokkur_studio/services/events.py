"""Audit/event log: the forensic trail of everything that happens to a project."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from rokkur_studio.db.models import Event

log = logging.getLogger(__name__)


class EventType:
    PROJECT_CREATED = "PROJECT_CREATED"
    STATE_CHANGED = "STATE_CHANGED"
    VIDEO_DISCOVERED = "VIDEO_DISCOVERED"
    RIGHTS_APPROVED = "RIGHTS_APPROVED"
    RIGHTS_REJECTED = "RIGHTS_REJECTED"
    RIGHTS_NEEDS_HUMAN = "RIGHTS_NEEDS_HUMAN"
    SOURCE_INGESTED = "SOURCE_INGESTED"
    ANALYSIS_STARTED = "ANALYSIS_STARTED"
    ANALYSIS_COMPLETED = "ANALYSIS_COMPLETED"
    CREATIVE_BRIEF_CREATED = "CREATIVE_BRIEF_CREATED"
    MANIFEST_CREATED = "MANIFEST_CREATED"
    WORKFLOW_COMPILED = "WORKFLOW_COMPILED"
    RENDER_SUBMITTED = "RENDER_SUBMITTED"
    RENDER_COMPLETED = "RENDER_COMPLETED"
    RENDER_FAILED = "RENDER_FAILED"
    GPU_OOM = "GPU_OOM"
    QC_PASSED = "QC_PASSED"
    QC_FAILED = "QC_FAILED"
    REPAIR_REQUESTED = "REPAIR_REQUESTED"
    REPAIR_BUDGET_EXHAUSTED = "REPAIR_BUDGET_EXHAUSTED"
    REPAIR_BUDGET_EXTENDED = "REPAIR_BUDGET_EXTENDED"
    QC_OVERRIDDEN = "QC_OVERRIDDEN"
    FINAL_ENCODED = "FINAL_ENCODED"
    PUBLISH_DRY_RUN = "PUBLISH_DRY_RUN"
    VIDEO_PUBLISHED = "VIDEO_PUBLISHED"
    PUBLISH_PROPOSAL_SKIPPED = "PUBLISH_PROPOSAL_SKIPPED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_DECIDED = "APPROVAL_DECIDED"
    JOB_FAILED = "JOB_FAILED"
    JOB_RETRY_SCHEDULED = "JOB_RETRY_SCHEDULED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    PROJECT_RESUMED = "PROJECT_RESUMED"
    PROJECT_CANCELLED = "PROJECT_CANCELLED"


def record_event(
    session: Session,
    type_: str,
    *,
    project_id: str | None,
    actor: str,
    data: dict[str, Any] | None = None,
    from_state: str | None = None,
    to_state: str | None = None,
    job_id: str | None = None,
) -> Event:
    event = Event(
        project_id=project_id,
        job_id=job_id,
        type=type_,
        from_state=from_state,
        to_state=to_state,
        actor=actor,
        data=data or {},
    )
    session.add(event)
    log.info(
        "event %s",
        type_,
        extra={"data": {"project_id": project_id, "from": from_state, "to": to_state}},
    )
    return event
