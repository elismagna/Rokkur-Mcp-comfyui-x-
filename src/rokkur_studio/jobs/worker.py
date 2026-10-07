"""Worker process: claims jobs, runs stage handlers, records outcomes, advances projects."""

from __future__ import annotations

import logging
import os
import socket
import threading
import traceback
from collections.abc import Mapping
from typing import Any

from rokkur_studio.db.models import Job
from rokkur_studio.domain.states import TERMINAL, ProjectStatus, can_transition
from rokkur_studio.jobs import queue
from rokkur_studio.jobs.errors import JobCancelled, JobError
from rokkur_studio.logging_setup import log_context
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.pipeline.driver import advance
from rokkur_studio.pipeline.stages import HANDLERS, Handler
from rokkur_studio.services.projects import get_project, transition

log = logging.getLogger(__name__)


class _Heartbeat(threading.Thread):
    def __init__(self, ctx: StudioContext, job_id: str, worker_id: str) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{job_id}")
        self.ctx, self.job_id, self.worker_id = ctx, job_id, worker_id
        self.stop = threading.Event()

    def run(self) -> None:
        interval = max(1.0, self.ctx.settings.jobs.lease_seconds / 3)
        while not self.stop.wait(interval):
            try:
                with self.ctx.db.transaction() as s:
                    queue.heartbeat(s, self.job_id, self.worker_id, self.ctx.settings.jobs)
            except Exception:
                log.warning("heartbeat failed", exc_info=True)


class Worker:
    def __init__(self, ctx: StudioContext, *, handlers: Mapping[str, Handler] | None = None,
                 worker_id: str | None = None, kinds: list[str] | None = None) -> None:
        self.ctx = ctx
        self.handlers = dict(handlers or HANDLERS)
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.kinds = kinds or sorted(self.handlers)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        log.info("worker started", extra={"data": {"worker_id": self.worker_id,
                                                   "kinds": self.kinds}})
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:
                log.exception("worker loop error")
                worked = False
            if not worked:
                self._stop.wait(self.ctx.settings.jobs.poll_interval_s)

    def drain(self, max_jobs: int = 1000) -> int:
        """Run jobs until none are runnable now (tests, smoke test, CLI)."""
        count = 0
        while count < max_jobs and self.run_once():
            count += 1
        return count

    def run_once(self) -> bool:
        with self.ctx.db.transaction() as s:
            job = queue.claim(s, self.worker_id, self.ctx.settings.jobs, self.kinds)
            if job is None:
                return False
            s.expunge(job)
        with log_context(job_id=job.id, project_id=job.project_id, stage=job.stage,
                         kind=job.kind, agent_id=job.agent_id or job.kind):
            self._execute(job)
        return True

    def _execute(self, job: Job) -> None:
        handler = self.handlers.get(job.kind)
        beat = _Heartbeat(self.ctx, job.id, self.worker_id)
        beat.start()
        try:
            if handler is None:
                raise JobError("no_handler", f"no handler for job kind {job.kind}")
            log.info("job started")
            result = handler(self.ctx, job)
        except JobCancelled:
            log.info("job cancelled")
            return
        except JobError as exc:
            self._failed(job, exc.to_dict(), exc.retryable)
            return
        except Exception as exc:  # unexpected bug or environment failure: retry, then fail
            self._failed(job, {"code": "unexpected", "message": repr(exc),
                               "traceback": traceback.format_exc()[-4000:]}, True)
            return
        finally:
            beat.stop.set()
        self._succeeded(job, result)

    def _succeeded(self, job: Job, result: dict[str, Any]) -> None:
        with self.ctx.db.transaction() as s:
            row = s.get(Job, job.id, with_for_update=True)
            if row is None or row.status == queue.JobStatus.CANCELLED:
                return
            queue.complete(s, row, result)
            s.flush()
            log.info("job succeeded", extra={"data": {"duration_s": row.duration_s}})
            if row.project_id and not result.get("awaiting_human"):
                project = get_project(s, row.project_id)
                advance(s, project, self.ctx.settings)

    def _failed(self, job: Job, error: dict[str, Any], retryable: bool) -> None:
        with self.ctx.db.transaction() as s:
            row = s.get(Job, job.id, with_for_update=True)
            if row is None or row.status == queue.JobStatus.CANCELLED:
                return
            retried = queue.fail(s, row, error, retryable=retryable, cfg=self.ctx.settings.jobs)
            log.warning("job failed", extra={"data": {"error": error.get("message"),
                                                      "retry_scheduled": retried,
                                                      "retry_count": row.retry_count}})
            if retried or not row.project_id:
                return
            project = get_project(s, row.project_id, for_update=True)
            status = ProjectStatus(project.status)
            if status not in TERMINAL and can_transition(status, ProjectStatus.FAILED):
                transition(s, project, ProjectStatus.FAILED, actor="worker", job_id=row.id,
                           reason=error.get("message"), data={"error_code": error.get("code")})
