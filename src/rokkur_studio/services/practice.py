"""Night School: the studio practises on finished videos and learns which settings render better.

A practice batch renders one shot of a finished video several times: once as the studio would
render it (the baseline), once with only the seed changed (how much luck alone moves the score),
and once per setting change (control strength, edge detail, CFG, shift, steps, sampler,
stabilizer). Each render is scored for general quality, not for anyone's taste: QC's measured
score (structure, stability, flicker) and the AI picture review (prompt, style, subject,
anatomy, steadiness) count half each, and a render with broken anatomy is capped however steady
it is, because steadier settings can melt bodies (repair notes, 2026-10-09).

Across batches, :func:`lessons` says which changes beat the baseline by more than the seed's
luck, on enough different shots. Lessons are shown on the Practice page; with
``practice.apply_lessons`` new videos start with the proven ones. The renders themselves
(pipeline/practice.py) stay beside the video and never replace its shots.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rokkur_studio.api.schemas import CreativeIn
from rokkur_studio.config import Settings
from rokkur_studio.db.models import Document, Job, Project, utcnow
from rokkur_studio.jobs.queue import JobStatus, enqueue
from rokkur_studio.manifest.schema import ReconstructionManifest
from rokkur_studio.services.projects import latest_document

KIND = "practice"
REPORT = "practice_report"
BASELINE, SEED = "baseline", "another seed"
PRIORITY = 200  # after every real job: a video you started never waits behind practice


@dataclass(frozen=True)
class Variant:
    """One setting change tried against the baseline. ``needs``: the workflow parameter it sets
    (None: it acts outside the workflow, so every profile can try it)."""

    label: str
    overrides: dict[str, Any] = field(default_factory=dict)
    needs: str | None = None


CATALOG: tuple[Variant, ...] = (
    Variant("control 0.85", {"control_strength": 0.85}, "CONTROL_STRENGTH"),
    Variant("control 1.15", {"control_strength": 1.15}, "CONTROL_STRENGTH"),
    Variant("edges 0.4/0.8", {"canny_low": 0.4, "canny_high": 0.8}, "CANNY_LOW"),
    Variant("edges 0.1/0.3", {"canny_low": 0.1, "canny_high": 0.3}, "CANNY_LOW"),
    Variant("cfg 4.5", {"cfg": 4.5}, "CFG"),
    Variant("cfg 7.5", {"cfg": 7.5}, "CFG"),
    Variant("shift 5", {"shift": 5.0}, "SHIFT"),
    Variant("shift 12", {"shift": 12.0}, "SHIFT"),
    Variant("70% steps", {"steps_scale": 0.7}, "STEPS"),
    Variant("sampler euler", {"sampler": "euler"}, "SAMPLER"),
    Variant("stabilizer off", {"stabilize": "off"}),
    Variant("stabilizer strong", {"stabilize": "strong"}),
    Variant("smoothed guide", {"smooth_control": 0.5}),
    # The character reference image Wan gets (manifest identity, not a shot setting). Only
    # tried on videos without an uploaded character reference: that one always wins.
    Variant("reference none", {"reference_mode": "none"}, "REFERENCE_IMAGE"),
    Variant("reference cutout", {"reference_mode": "cutout"}, "REFERENCE_IMAGE"),
    Variant("reference first frame", {"reference_mode": "source"}, "REFERENCE_IMAGE"),
)
MANIFEST_KEYS = ("reference_mode",)  # variant keys set on the manifest, not the shot
BY_LABEL = {v.label: v for v in CATALOG}


def overrides_for(label: str, shot_overrides: dict[str, Any], *, seed: int,
                  steps: int) -> dict[str, Any]:
    """The shot overrides a variant renders with: the shot's own, then the variant's change."""
    if label == BASELINE:
        return dict(shot_overrides)
    if label == SEED:
        return {**shot_overrides, "seed": (int(shot_overrides.get("seed", seed)) + 7919) % 2**32}
    change = dict(BY_LABEL[label].overrides)
    scale = change.pop("steps_scale", None)
    if scale is not None:
        base = int(shot_overrides.get("steps", steps))
        change["steps"] = max(1, round(base * float(scale)))
    return {**shot_overrides, **change}


def effective_reference(mode: str, keep: bool) -> str:
    """The reference a render really gets: ``auto`` is a cutout when the subject is kept."""
    return ("cutout" if keep else "none") if mode == "auto" else mode


def pick_variants(round_no: int, count: int, accepts: dict[str, bool],
                  shot_overrides: dict[str, Any]) -> list[str]:
    """Baseline, another seed, then ``count`` changes the workflow can make, rotating through
    the catalog so a night of batches covers all of them. A change the shot already has (its
    video was made with control 1.15, say) is skipped: it would only repeat the baseline."""
    def fits(v: Variant) -> bool:
        if v.needs is not None and not accepts.get(v.needs, False):
            return False
        fixed = {k: val for k, val in v.overrides.items() if k != "steps_scale"}
        return not fixed or any(shot_overrides.get(k) != val for k, val in fixed.items())

    usable = [v for v in CATALOG if fits(v)]
    if not usable:
        return [BASELINE, SEED]
    start = round_no * count % len(usable)
    chosen = [usable[(start + i) % len(usable)].label for i in range(min(count, len(usable)))]
    return [BASELINE, SEED, *chosen]


# -- general quality score ------------------------------------------------------------------
def quality_score(shot: dict[str, Any], *, anatomy_floor: int = 4,
                  anatomy_cap: float = 5.0) -> dict[str, Any]:
    """One QC'd shot's general quality, 0-10: the mean of what QC measured, the AI review's
    five scores (when the vision model gave one) and the quality judge (when one is trained,
    pipeline/judge.py). ``reviewed`` is False with neither: the score is then QC alone, which
    steadier-but-melted renders can game, so lessons never count it.
    """
    qc = float(shot.get("overall") or 0.0)
    review = shot.get("picture_review")
    ai: float | None = None
    anatomy = None
    if review:
        parts = [shot.get("prompt_adherence"), shot.get("style_consistency"),
                 shot.get("identity"), shot.get("hand_body_deformation"),
                 review.get("steadiness")]
        values = [float(v) for v in parts if v is not None]
        ai = sum(values) / len(values) if values else None
        anatomy = shot.get("hand_body_deformation")
    judge = shot.get("judge")
    parts_used = [qc] + [v for v in (ai, judge) if v is not None]
    score = sum(parts_used) / len(parts_used)
    capped = anatomy is not None and anatomy <= anatomy_floor and score > anatomy_cap
    if capped:  # a broken body is never a good render, however steady
        score = anatomy_cap
    return {"score": round(score, 2), "qc": round(qc, 2),
            "ai": None if ai is None else round(ai, 2), "anatomy": anatomy,
            "judge": judge, "judge_worst": shot.get("judge_worst"),
            "reviewed": ai is not None or judge is not None, "anatomy_capped": capped}


# -- queueing -------------------------------------------------------------------------------
class PracticeError(ValueError):
    """Why a practice batch cannot be queued for this video."""


def project_busy(session: Session, project_id: str) -> bool:
    """A stage job of the video itself is queued or running."""
    return bool(session.scalar(select(func.count(Job.id)).where(
        Job.project_id == project_id, Job.status.in_(JobStatus.ACTIVE))))


def real_work_waiting(session: Session) -> bool:
    """A non-practice job could start now: a practice batch stops early and hands its rest on
    (the queue runs that job first, then the rest of the batch)."""
    return bool(session.scalar(select(func.count(Job.id)).where(
        Job.kind != KIND, Job.status.in_((JobStatus.QUEUED, JobStatus.RETRY_WAIT)),
        Job.run_after <= utcnow())))


def practice_projects(session: Session) -> list[Project]:
    """Videos practice can use: rendered at least once (a manifest exists) and idle."""
    rows = session.scalars(select(Project).join(Document, Document.project_id == Project.id)
                           .where(Document.kind == "manifest").distinct()
                           .order_by(Project.created_at.desc())).all()
    return [p for p in rows if not project_busy(session, p.id)]


def _rounds_so_far(session: Session) -> int:
    return session.scalar(select(func.count(Job.id)).where(Job.kind == KIND)) or 0


def queue_batches(session: Session, settings: Settings, projects: list[Project], *,
                  batches: int, shot_id: str | None = None, profile: str | None = None,
                  actor: str = "dashboard") -> list[Job]:
    """Queue ``batches`` practice batches, cycling through the videos and their shots."""
    if not projects:
        raise PracticeError("No finished video to practise on yet: render one first")
    if batches < 1:
        raise PracticeError("Pick at least one batch")
    name = profile or settings.practice.profile or None
    if name and name not in settings.profiles:
        raise PracticeError(f"Unknown render profile {name}")
    plans: list[tuple[Project, list[str]]] = []
    for project in projects:
        mdoc = latest_document(session, project.id, "manifest")
        if mdoc is None:
            raise PracticeError(f"{project.name} has no shots yet")
        manifest = ReconstructionManifest.model_validate(mdoc.data)
        shots = [s.shot_id for s in manifest.shots]
        if shot_id is not None:
            if shot_id not in shots:
                raise PracticeError(f"{project.name} has no {shot_id}")
            shots = [shot_id]
        problem = settings.profile_problem(name or manifest.render_profile, target="local")
        if problem:
            raise PracticeError(f"Practice renders on this PC: {problem}")
        plans.append((project, shots))
    start = _rounds_so_far(session)
    jobs: list[Job] = []
    for i in range(batches):
        project, shots = plans[i % len(plans)]
        round_no = start + i
        job = enqueue(session, KIND, payload={
            "project_id": project.id, "shot_id": shots[(round_no // len(plans)) % len(shots)],
            "profile": name, "round": round_no, "queued_by": actor,
        }, max_attempts=settings.jobs.default_max_attempts, priority=PRIORITY)
        assert job is not None  # no dedupe key: never deduplicated
        jobs.append(job)
    return jobs


# -- reports and lessons --------------------------------------------------------------------
def reports(session: Session, project_id: str | None = None, limit: int | None = None
            ) -> list[dict[str, Any]]:
    """Finished practice batches, newest first, each with its project's id and name."""
    stmt = (select(Document, Project.name).join(Project, Project.id == Document.project_id)
            .where(Document.kind == REPORT).order_by(Document.created_at.desc()))
    if project_id:
        stmt = stmt.where(Document.project_id == project_id)
    if limit:
        stmt = stmt.limit(limit)
    return [{**doc.data, "project_id": doc.project_id, "project_name": name,
             "created_at": doc.created_at}
            for doc, name in session.execute(stmt).tuples()]


def lessons(session: Session, settings: Settings) -> dict[str, Any]:
    """What practice has learned, per render profile: for each setting change, how its score
    compared with the baseline of the same batch, and whether that is a lesson yet.

    A change counts as ``better`` (or ``worse``) once it was tried in ``practice.min_batches``
    batches on reviewed renders, gained (lost) on average more than both ``practice.min_gain``
    and the seed's average swing, and did so in at least 3 of 4 batches. ``as good, faster``:
    no clear quality difference, at least 15% less render time.
    """
    cfg = settings.practice
    deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    speed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    noise: dict[str, list[float]] = defaultdict(list)
    counted: dict[str, int] = defaultdict(int)
    for rep in reports(session):
        prof = rep.get("profile") or "?"
        rows = {v["label"]: v for v in rep.get("variants", [])}
        base = rows.get(BASELINE)
        if not base or not base.get("reviewed"):
            continue
        counted[prof] += 1
        for label, v in rows.items():
            if label == BASELINE or not v.get("reviewed"):
                continue
            delta = float(v["score"]) - float(base["score"])
            if label == SEED:
                noise[prof].append(abs(delta))
                continue
            deltas[prof][label].append(delta)
            if v.get("seconds") and base.get("seconds"):
                speed[prof][label].append(float(v["seconds"]) / float(base["seconds"]))
    out: dict[str, Any] = {}
    for prof in sorted(set(deltas) | set(noise)):
        swing = sum(noise[prof]) / len(noise[prof]) if noise[prof] else None
        bar = max(cfg.min_gain, swing or 0.0)
        rows_out = []
        for label in [v.label for v in CATALOG if v.label in deltas[prof]]:
            ds = deltas[prof][label]
            mean = sum(ds) / len(ds)
            wins = sum(1 for d in ds if d > 0) / len(ds)
            ratio = (sum(speed[prof][label]) / len(speed[prof][label])
                     if speed[prof][label] else None)
            if len(ds) < cfg.min_batches:
                verdict = f"needs {cfg.min_batches - len(ds)} more"
            elif mean >= bar and wins >= 0.75:
                verdict = "better"
            elif mean <= -bar and wins <= 0.25:
                verdict = "worse"
            elif ratio is not None and ratio <= 0.85 and abs(mean) < bar:
                verdict = "as good, faster"
            else:
                verdict = "no clear difference"
            rows_out.append({"label": label, "batches": len(ds), "mean_delta": round(mean, 2),
                             "wins": round(wins, 2), "time_ratio": round(ratio, 2)
                             if ratio is not None else None, "verdict": verdict,
                             "overrides": BY_LABEL[label].overrides})
        out[prof] = {"batches": counted[prof], "seed_swing": round(swing, 2)
                     if swing is not None else None, "bar": round(bar, 2), "changes": rows_out}
    return out


def proven_overrides(session: Session, settings: Settings, profile: str) -> dict[str, Any]:
    """The shot overrides practice has shown to help on ``profile``: every ``better`` change
    (the larger gain wins where two set the same thing) and every ``as good, faster`` one."""
    learned = lessons(session, settings).get(profile)
    if not learned:
        return {}
    chosen: dict[str, Any] = {}
    gain: dict[str, float] = {}
    keep = [c for c in learned["changes"] if c["verdict"] in ("better", "as good, faster")]
    for change in sorted(keep, key=lambda c: c["mean_delta"]):
        for key, value in change["overrides"].items():
            if change["mean_delta"] >= gain.get(key, float("-inf")):
                chosen[key], gain[key] = value, change["mean_delta"]
    return chosen


def apply_proven(overrides: dict[str, Any], proven: dict[str, Any], steps: int) -> dict[str, Any]:
    """``proven`` added to a shot's overrides. A setting the video was given on purpose (one
    that differs from the New video form's default) always wins over a lesson."""
    defaults = {k: f.default for k, f in CreativeIn.model_fields.items()}
    out = dict(overrides)
    for key, value in proven.items():
        if key == "steps_scale":
            if out.get("steps") is None:
                out["steps"] = max(1, round(steps * float(value)))
        elif out.get(key) is None or out[key] == defaults.get(key):
            out[key] = value
    return out


def report_text(learned: dict[str, Any], recent: list[dict[str, Any]]) -> str:
    """Plain text of :func:`lessons` and the latest batches, for the CLI."""
    lines: list[str] = []
    if not learned:
        lines.append("No practice batches with an AI picture review yet.")
    for prof, data in learned.items():
        swing = "-" if data["seed_swing"] is None else f"{data['seed_swing']:.2f}"
        lines += [f"{prof}: {data['batches']} batches; a new seed alone moves the score "
                  f"{swing} on average, so a lesson needs a gain of {data['bar']:.2f}",
                  "  change              tried  mean gain  wins   time   verdict"]
        for c in data["changes"]:
            ratio = "-" if c["time_ratio"] is None else f"{c['time_ratio']:.0%}"
            lines.append(f"  {c['label']:<19} {c['batches']:>5}  {c['mean_delta']:>+9.2f}  "
                         f"{c['wins']:>4.0%}  {ratio:>5}   {c['verdict']}")
        lines.append("")
    if recent:
        lines.append("Latest batches (score 0-10: half QC, half AI review):")
        for rep in recent:
            ranked = sorted(rep.get("variants", []), key=lambda v: -v["score"])
            top = ", ".join(f"{v['label']} {v['score']:.1f}" for v in ranked[:4])
            flag = "" if rep.get("reviewed") else "  (no AI review: not counted)"
            lines.append(f"  {rep['project_name']} {rep['shot_id']} ({rep['profile']}): "
                         f"{top}{flag}")
    return "\n".join(lines).rstrip()
