"""Where a project's wall-clock time went: stages, shot renders, repairs, and what they bought.

Read-only. Built from what the studio already records (jobs, renders, QC reports, your
ratings), so it works on projects rendered before it existed. ``studio.ps1 timings`` prints it
to compare settings before and after a change (docs/speed.md).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.db.models import Document, Job, Project, Rating, Render

_RATING = {2: "super-like", 1: "like", -1: "dislike", -2: "super-dislike"}


def _secs(start: datetime | None, end: datetime | None) -> float | None:
    return (end - start).total_seconds() if start and end else None


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{seconds % 60:04.1f}s"
    return f"{seconds:.1f}s"


def latest_project_id(session: Session) -> str | None:
    return session.scalar(select(Project.id).order_by(Project.created_at.desc()).limit(1))


def collect(session: Session, project_id: str) -> dict[str, Any]:
    """The numbers behind :func:`report`, as plain data."""
    project = session.get(Project, project_id)
    if project is None:
        raise LookupError(project_id)
    jobs = session.scalars(select(Job).where(Job.project_id == project_id)
                           .order_by(Job.created_at)).all()
    renders = session.scalars(select(Render).where(Render.project_id == project_id)
                              .order_by(Render.started_at)).all()
    qc: dict[str, float] = {}
    for data in session.scalars(select(Document.data).where(
            Document.project_id == project_id, Document.kind == "qc_report")):
        for shot in data.get("shots", []):
            if shot.get("render_id"):
                qc[shot["render_id"]] = float(shot.get("overall", 0))
    rated = {r.render_id: r.value for r in session.scalars(select(Rating).where(
        Rating.project_id == project_id, Rating.rater == "human",
        Rating.render_id.is_not(None)).order_by(Rating.created_at))}

    stages: dict[str, dict[str, float]] = defaultdict(lambda: {"runs": 0, "seconds": 0.0,
                                                                "waited": 0.0})
    for job in jobs:
        st = stages[job.kind]
        st["runs"] += 1
        st["seconds"] += _secs(job.started_at, job.finished_at) or 0.0
        st["waited"] += _secs(job.run_after or job.created_at, job.started_at) or 0.0

    rows: list[dict[str, Any]] = []
    seen_jobs: set[str | None] = set()
    for r in renders:
        details = (r.params or {}).get("_details") or {}
        execution = details.get("execution_seconds")
        subject = details.get("subject") or {}
        rows.append({
            "shot": r.shot_id, "attempt": r.attempt, "status": r.status, "profile": r.profile,
            "workflow": r.workflow, "size": f"{r.params.get('WIDTH')}x{r.params.get('HEIGHT')}",
            "frames": r.params.get("FRAME_COUNT"), "steps": r.params.get("STEPS"),
            "seconds": r.duration_s, "comfy_seconds": execution,
            "overhead_seconds": (r.duration_s - execution
                                 if r.duration_s is not None and execution is not None else None),
            "mask_seconds": subject.get("seconds"),
            # The first shot of each render job loads the models; later ones should not.
            "first_in_job": r.job_id not in seen_jobs,
            "qc": qc.get(r.id), "rating": rated.get(r.id),
        })
        seen_jobs.add(r.job_id)

    first = jobs[0].created_at if jobs else None
    last = max((j.finished_at for j in jobs if j.finished_at), default=None)
    done = [x for x in rows if x["seconds"] is not None]
    repairs = [x for x in done if x["attempt"] > 1]
    warm = [x["comfy_seconds"] for x in done if x["comfy_seconds"] and not x["first_in_job"]]
    cold = [x["comfy_seconds"] for x in done if x["comfy_seconds"] and x["first_in_job"]]
    return {
        "project": {"id": project.id, "name": project.name, "status": project.status,
                    "profile": project.render_profile, "repair_rounds": project.repair_rounds},
        "total_seconds": _secs(first, last),
        "stages": dict(stages),
        "renders": rows,
        "render_seconds": sum(x["seconds"] for x in done),
        "repair_seconds": sum(x["seconds"] for x in repairs),
        "repair_renders": len(repairs),
        "repair_renders_liked": sum(1 for x in repairs if (x["rating"] or 0) > 0),
        "cold_comfy_avg": sum(cold) / len(cold) if cold else None,
        "warm_comfy_avg": sum(warm) / len(warm) if warm else None,
    }


def report(data: dict[str, Any]) -> str:
    p = data["project"]
    lines = [f"{p['name']} ({p['id']}): {p['status']}, profile {p['profile']}, "
             f"{p['repair_rounds']} repair rounds",
             f"Wall clock, first job queued to last job finished: {_fmt(data['total_seconds'])}",
             "", "Stage            runs   working   waiting in queue"]
    for kind, st in sorted(data["stages"].items(), key=lambda kv: -kv[1]["seconds"]):
        lines.append(f"{kind:<16} {int(st['runs']):>4}   {_fmt(st['seconds']):>8}   "
                     f"{_fmt(st['waited']):>8}")
    lines += ["", "Shot      try  status      size       frames steps     wall   ComfyUI  "
              "other   masks    QC  you"]
    for x in data["renders"]:
        lines.append(
            f"{x['shot']:<9} {x['attempt']:>3}  {x['status']:<10}  {x['size']:<10} "
            f"{x['frames'] or '-':>6} {x['steps'] or '-':>5} {_fmt(x['seconds']):>8} "
            f"{_fmt(x['comfy_seconds']):>8} {_fmt(x['overhead_seconds']):>7} "
            f"{_fmt(x['mask_seconds']):>7} "
            f"{'-' if x['qc'] is None else format(x['qc'], '.2f'):>5}  "
            f"{_RATING.get(x['rating'] or 0, '-')}{'  (loads models)' if x['first_in_job'] else ''}")
    lines += ["", f"Shot renders: {_fmt(data['render_seconds'])} in total; repairs "
              f"{_fmt(data['repair_seconds'])} over {data['repair_renders']} re-renders, "
              f"{data['repair_renders_liked']} of them liked by you."]
    if data["cold_comfy_avg"] is not None and data["warm_comfy_avg"] is not None:
        lines.append(f"ComfyUI time per shot: {_fmt(data['cold_comfy_avg'])} for the first shot "
                     f"of a render run, {_fmt(data['warm_comfy_avg'])} for the rest; the "
                     "difference is mostly model loading.")
    return "\n".join(lines)
