"""Durable Postgres job queue: SKIP LOCKED claims, leases, backoff retries, dead-lettering."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from rokkur_studio.config import JobsSection
from rokkur_studio.db.models import Job, new_id, utcnow
from rokkur_studio.services.events import EventType, record_event

log = logging.getLogger(__name__)


class JobStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"  # dead-lettered: attempts exhausted or permanent error
    CANCELLED = "CANCELLED"

    ACTIVE = (QUEUED, RUNNING, RETRY_WAIT)


ACTIVE_PREDICATE = "status IN ('QUEUED','RUNNING','RETRY_WAIT')"


def enqueue(
    session: Session,
    kind: str,
    *,
    project_id: str | None = None,
    payload: dict[str, Any] | None = None,
    stage: str | None = None,
    dedupe_key: str | None = None,
    max_attempts: int = 3,
    priority: int = 100,
    run_after: datetime | None = None,
    agent_id: str | None = None,
) -> Job | None:
    """Insert a job. With ``dedupe_key``, returns None if an active job already has that key."""
    values: dict[str, Any] = dict(
        id=new_id("job"),
        project_id=project_id,
        kind=kind,
        stage=stage,
        agent_id=agent_id,
        status=JobStatus.QUEUED,
        priority=priority,
        payload=payload or {},
        dedupe_key=dedupe_key,
        max_attempts=max_attempts,
        run_after=run_after or utcnow(),
        retry_count=0,
        created_at=utcnow(),
    )
    stmt = insert(Job).values(**values)
    if dedupe_key is not None:
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["dedupe_key"],
            # Literal predicate: must match the partial index text so Postgres can infer it
            # even after psycopg switches to server-side prepared statements.
            index_where=text(ACTIVE_PREDICATE),
        )
    result = session.execute(stmt.returning(Job.id))
    job_id = result.scalar_one_or_none()
    if job_id is None:
        log.info("job deduplicated", extra={"data": {"kind": kind, "dedupe_key": dedupe_key}})
        return None
    return session.get(Job, job_id)


def claim(
    session: Session, worker_id: str, cfg: JobsSection, kinds: list[str] | None = None
) -> Job | None:
    """Atomically claim the next runnable job (or one whose lease expired)."""
    now = utcnow()
    runnable = or_(
        Job.status.in_([JobStatus.QUEUED, JobStatus.RETRY_WAIT]) & (Job.run_after <= now),
        (Job.status == JobStatus.RUNNING) & (Job.locked_until < now),
    )
    stmt = (
        select(Job)
        .where(runnable)
        .order_by(Job.priority, Job.run_after, Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if kinds:
        stmt = stmt.where(Job.kind.in_(kinds))
    job = session.scalars(stmt).one_or_none()
    if job is None:
        return None
    if job.status == JobStatus.RUNNING:
        # The previous worker died holding the lease: that attempt counts.
        job.retry_count += 1
        log.warning("reclaiming expired job lease", extra={"data": {"job_id": job.id,
                    "previous_worker": job.locked_by}})
        if job.retry_count >= job.max_attempts:
            _dead_letter(session, job, {"code": "lease_expired",
                                        "message": "worker lost; attempts exhausted"})
            return None
    job.status = JobStatus.RUNNING
    job.locked_by = worker_id
    job.locked_until = now + timedelta(seconds=cfg.lease_seconds)
    job.started_at = now
    job.finished_at = None
    session.flush()
    return job


def heartbeat(session: Session, job_id: str, worker_id: str, cfg: JobsSection) -> bool:
    """Extend a running job's lease. Returns False if the job was cancelled or stolen."""
    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.locked_by == worker_id, Job.status == JobStatus.RUNNING)
        .values(locked_until=utcnow() + timedelta(seconds=cfg.lease_seconds))
    )
    return bool(result.rowcount)  # type: ignore[attr-defined]


def complete(session: Session, job: Job, result: dict[str, Any] | None = None) -> None:
    job.status = JobStatus.SUCCEEDED
    job.result = result or {}
    job.error = None
    job.finished_at = utcnow()
    job.locked_by = None
    job.locked_until = None


def backoff_seconds(cfg: JobsSection, retry_count: int) -> float:
    return float(min(cfg.backoff_max_s, cfg.backoff_base_s * (2 ** max(0, retry_count - 1))))


def fail(session: Session, job: Job, error: dict[str, Any], *, retryable: bool,
         cfg: JobsSection) -> bool:
    """Record a failed attempt. Returns True if a retry was scheduled."""
    job.retry_count += 1
    job.error = error
    job.finished_at = utcnow()
    job.locked_by = None
    job.locked_until = None
    if retryable and job.retry_count < job.max_attempts:
        delay = backoff_seconds(cfg, job.retry_count)
        job.status = JobStatus.RETRY_WAIT
        job.run_after = utcnow() + timedelta(seconds=delay)
        record_event(session, EventType.JOB_RETRY_SCHEDULED, project_id=job.project_id,
                     job_id=job.id, actor="queue",
                     data={"kind": job.kind, "retry_count": job.retry_count,
                           "delay_s": delay, "error": error})
        return True
    _dead_letter(session, job, error)
    return False


def _dead_letter(session: Session, job: Job, error: dict[str, Any]) -> None:
    job.status = JobStatus.FAILED
    job.error = error
    job.finished_at = utcnow()
    job.locked_by = None
    job.locked_until = None
    record_event(session, EventType.JOB_FAILED, project_id=job.project_id, job_id=job.id,
                 actor="queue", data={"kind": job.kind, "error": error,
                                      "retry_count": job.retry_count})


def cancel_project_jobs(session: Session, project_id: str) -> int:
    result = session.execute(
        update(Job)
        .where(Job.project_id == project_id, Job.status.in_(JobStatus.ACTIVE))
        .values(status=JobStatus.CANCELLED, finished_at=utcnow(), locked_until=None)
    )
    return int(result.rowcount)  # type: ignore[attr-defined]


def is_cancelled(session: Session, job_id: str) -> bool:
    return session.scalar(select(Job.status).where(Job.id == job_id)) == JobStatus.CANCELLED
