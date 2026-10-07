"""Pipeline driver: which job runs in which state, and how far autonomy lets it go."""

from __future__ import annotations

from sqlalchemy.orm import Session

from rokkur_studio.config import Settings
from rokkur_studio.db.models import Job, Project
from rokkur_studio.domain.states import ProjectStatus
from rokkur_studio.jobs.queue import enqueue

S = ProjectStatus

STAGE_JOBS: dict[ProjectStatus, str] = {
    S.RIGHTS_PENDING: "rights_check",
    S.RIGHTS_OK: "ingest",
    S.DOWNLOADED_OR_INGESTED: "analyze",
    S.ANALYZED: "creative_plan",
    S.CREATIVE_READY: "compile_workflow",
    S.WORKFLOW_READY: "render",
    S.RENDER_QUEUED: "render",
    S.QUALITY_CHECK: "qc",
    S.QUALITY_FAILED: "repair",
    S.QUALITY_PASSED: "edit",
}

# Level 1 ("agents research and draft") stops once a creative brief exists.
_LEVEL1_KINDS = {"rights_check", "ingest", "analyze", "creative_plan"}


def autonomy_level(project: Project, settings: Settings) -> int:
    if project.channel is not None:
        return project.channel.autonomy_level
    return settings.studio.autonomy_level


def next_job_kind(project: Project) -> str | None:
    return STAGE_JOBS.get(S(project.status))


def advance(session: Session, project: Project, settings: Settings, *,
            manual: bool = False) -> Job | None:
    """Enqueue the job for the project's current state if autonomy allows (or ``manual``)."""
    kind = next_job_kind(project)
    if kind is None:
        return None
    level = autonomy_level(project, settings)
    if not manual and (level == 0 or (level == 1 and kind not in _LEVEL1_KINDS)):
        return None
    return enqueue(session, kind, project_id=project.id, stage=project.status,
                   dedupe_key=f"{project.id}:stage", max_attempts=settings.jobs.default_max_attempts)
