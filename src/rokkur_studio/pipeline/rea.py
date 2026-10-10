"""The ``rea`` job: run one REA command and keep its output (docs/rea.md)."""

from __future__ import annotations

from typing import Any

from rokkur_studio.db.models import Job, ReaRun, utcnow
from rokkur_studio.jobs.errors import PermanentJobError
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services.rea import ReaStore, availability, execute


def rea_job(ctx: StudioContext, job: Job) -> dict[str, Any]:
    run_id = str(job.payload["run_id"])
    store = ReaStore(ctx.settings.studio.data_dir)
    with ctx.db.transaction() as s:
        run = s.get(ReaRun, run_id)
        if run is None:
            return {"run": run_id, "skipped": "deleted"}
        avail = availability(ctx.settings, probe=False)
        if not avail.ok:
            run.status, run.finished_at = "failed", utcnow()
            run.error = {"code": "rea_unavailable", "message": avail.reason}
            raise PermanentJobError("rea_unavailable", avail.reason)
        run.status, run.error = "running", None
        s.expunge(run)
    result = execute(ctx.settings, store, run)
    with ctx.db.transaction() as s:
        row = s.get(ReaRun, run_id)
        if row is None:
            return {"run": run_id, "skipped": "deleted"}
        for key, value in result.items():
            setattr(row, key, value)
        row.finished_at = utcnow()
    if result["status"] != "done":
        raise PermanentJobError(result["error"]["code"], result["error"]["message"])
    return {"run": run_id, "exit_code": result["exit_code"], "took_s": result["took_s"]}
