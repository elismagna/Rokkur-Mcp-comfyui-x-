"""Your ratings: a verdict on one shot render or a whole video, and what it changes.

Three signals stay separate (docs/ratings.md): your rating (this module), the measured QC
score (pipeline/qc.py), and any model's estimate (``rater="ai"``, not built yet). Learning
(services/taste.py) only reads human ratings.

What a rating changes:
- A render you like counts as accepted when QC runs again, so repairs never redo it.
- Shots you dislike can be redone (commands.redo_shots) with changes chosen from your tags.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rokkur_studio.db.models import Asset, Event, Project, Rating, Render
from rokkur_studio.manifest.schema import ReconstructionManifest
from rokkur_studio.services.events import EventType, record_event
from rokkur_studio.services.projects import latest_document

VIDEO = "video"

VALUES: dict[int, str] = {2: "Super like", 1: "Like", -1: "Dislike", -2: "Super dislike"}

# What a rating is about: in a like, what worked; in a dislike, what went wrong.
TAGS: dict[str, str] = {
    "subject": "Main subject",
    "style": "Style",
    "background": "Background",
    "continuity": "Continuity",
    "motion": "Motion",
    "flicker": "Flicker",
    "detail": "Detail",
    "prompt": "Follows my prompt",
}

# Bounds for a redo's source-structure change, the same as automatic repairs: on a real shot
# 1.15 raised QC's motion score but made anatomy worse (docs/upgrade-2026-10-07.md).
CONTROL_MIN, CONTROL_MAX, CONTROL_STEP = 0.7, 1.0, 0.1
SEED_STEP = 104729  # a prime, so redo seeds never land on an earlier attempt's seed


class RatingIn(BaseModel):
    target: str = Field(pattern=r"^(video|shot_[A-Za-z0-9_]+)$")
    value: Literal[-2, -1, 0, 1, 2]  # 0 removes your rating
    tags: list[str] = Field(default_factory=list, max_length=len(TAGS))
    note: str | None = Field(None, max_length=1000)
    render_id: str | None = None  # the attempt you watched; defaults to the shot's latest

    @field_validator("tags")
    @classmethod
    def _known_tags(cls, tags: list[str]) -> list[str]:
        unknown = [t for t in tags if t not in TAGS]
        if unknown:
            raise ValueError(f"unknown tags: {', '.join(unknown)}")
        return list(dict.fromkeys(tags))

    @field_validator("note")
    @classmethod
    def _strip(cls, note: str | None) -> str | None:
        return (note or "").strip() or None


def latest_render(session: Session, project_id: str, shot_id: str) -> Render | None:
    return session.scalars(
        select(Render).where(Render.project_id == project_id, Render.shot_id == shot_id,
                             Render.status.in_(("succeeded", "superseded")))
        .order_by(Render.attempt.desc()).limit(1)).one_or_none()


def latest_video_asset(session: Session, project_id: str) -> Asset | None:
    """The finished video, or the assembled preview of a project that has none yet."""
    for kind in ("final", "assembled"):
        asset = session.scalars(select(Asset).where(Asset.project_id == project_id,
                                                    Asset.kind == kind)
                                .order_by(Asset.created_at.desc()).limit(1)).one_or_none()
        if asset is not None:
            return asset
    return None


def _brief_shot(session: Session, project_id: str, shot_id: str) -> dict[str, Any]:
    brief = latest_document(session, project_id, "creative_brief")
    plan = (brief.data.get("shot_plan") or []) if brief else []
    return next((s for s in plan if s.get("shot_id") == shot_id), {})


def _qc_for(session: Session, project_id: str, render_id: str) -> dict[str, Any] | None:
    """QC's measurement of this attempt, from the newest report that scored it."""
    from rokkur_studio.db.models import Document

    reports = session.scalars(select(Document).where(Document.project_id == project_id,
                                                     Document.kind == "qc_report")
                              .order_by(Document.version.desc()).limit(20))
    for doc in reports:
        for shot in doc.data.get("shots", []):
            if shot.get("render_id") == render_id:
                return {"overall": shot.get("overall"), "decision": shot.get("decision"),
                        "version": doc.version}
    return None


def _subject_mode(session: Session, project_id: str) -> str | None:
    doc = latest_document(session, project_id, "manifest")
    return (doc.data.get("subject") or {}).get("mode") if doc else None


def shot_snapshot(session: Session, project: Project, render: Render) -> dict[str, Any]:
    """What produced this attempt, frozen at the moment it is rated."""
    params = render.params or {}
    details = params.get("_details") or {}
    plan = _brief_shot(session, project.id, render.shot_id)
    creative = project.creative_input or {}
    canny = [params.get("CANNY_LOW"), params.get("CANNY_HIGH")]
    # Unset sampling options take the workflow's value, which the compiler reports as applied.
    applied = details.get("applied") or {}

    def sampling(key: str) -> Any:
        return params.get(key, applied.get(key))

    return {
        "kind": "shot",
        "shot_id": render.shot_id,
        "attempt": render.attempt,
        "theme": creative.get("theme"),
        "scene_prompt": creative.get("prompt"),
        "profile": render.profile,
        "renderer": render.renderer,
        "workflow": render.workflow,
        "prompt": params.get("STYLE_PROMPT"),
        "negative_prompt": params.get("NEGATIVE_PROMPT"),
        "seed": params.get("SEED"),
        "steps": params.get("STEPS"),
        "cfg": params.get("CFG"),
        "control_strength": params.get("CONTROL_STRENGTH"),
        "canny": canny if any(v is not None for v in canny) else None,
        "shift": sampling("SHIFT"),
        "sampler": sampling("SAMPLER"),
        "scheduler": sampling("SCHEDULER"),
        "stabilize": params.get("_STABILIZE"),
        "smooth_control": params.get("_SMOOTH_CONTROL"),
        "reference": details.get("reference"),
        "subject_mode": _subject_mode(session, project.id),
        "subject_kept": (details.get("subject") or {}).get("kept"),
        "framing": {k: plan.get(k) for k in ("shot_size", "camera_angle", "camera_movement",
                                             "lighting") if plan.get(k)},
        "motion_type": plan.get("motion_type"),
        "qc": _qc_for(session, project.id, render.id),
    }


def video_snapshot(session: Session, project: Project, asset: Asset) -> dict[str, Any]:
    creative = project.creative_input or {}
    renders = session.scalars(select(Render).where(Render.project_id == project.id,
                                                   Render.status == "succeeded")).all()
    qc = latest_document(session, project.id, "qc_report")
    return {
        "kind": "video",
        "asset_kind": asset.kind,
        "theme": creative.get("theme"),
        "scene_prompt": creative.get("prompt"),
        "profile": project.render_profile,
        "workflows": sorted({r.workflow for r in renders if r.workflow}),
        "renders": sorted(r.id for r in renders),
        "subject_mode": _subject_mode(session, project.id),
        "reference_mode": creative.get("reference_mode"),
        "control_strength": creative.get("control_strength"),
        "qc": ({"overall": qc.data.get("overall"), "decision": qc.data.get("decision"),
                "version": qc.version} if qc else None),
    }


def rate(session: Session, project: Project, body: RatingIn, *, actor: str,
         rater: str = "human") -> Rating | None:
    """Set (or with value 0 remove) a rating. One rating per rater per attempt or video."""
    render: Render | None = None
    asset: Asset | None = None
    if body.target == VIDEO:
        asset = latest_video_asset(session, project.id)
        if asset is None:
            raise ValueError("There is no video to rate yet")
    else:
        if body.render_id:
            render = session.get(Render, body.render_id)
            if render is None or render.project_id != project.id or render.shot_id != body.target:
                raise ValueError("That render does not belong to this shot")
        else:
            render = latest_render(session, project.id, body.target)
        if render is None or render.status not in ("succeeded", "superseded"):
            raise ValueError(f"{body.target.replace('shot_', 'Shot ')} has no finished render yet")
    stmt = select(Rating).where(Rating.project_id == project.id, Rating.target == body.target,
                                Rating.rater == rater)
    stmt = (stmt.where(Rating.render_id == render.id) if render is not None
            else stmt.where(Rating.asset_id == (asset.id if asset else None)))
    existing = session.scalars(stmt).one_or_none()
    data = {"target": body.target, "value": body.value, "tags": body.tags,
            "render_id": render.id if render else None, "asset_id": asset.id if asset else None}
    if body.value == 0:
        if existing is not None:
            session.delete(existing)
            record_event(session, EventType.RATING_CLEARED, project_id=project.id, actor=actor,
                         data=data)
        return None
    snapshot = (shot_snapshot(session, project, render) if render is not None
                else video_snapshot(session, project, asset))  # type: ignore[arg-type]
    if existing is None:
        existing = Rating(project_id=project.id, target=body.target, rater=rater,
                          render_id=render.id if render else None,
                          asset_id=asset.id if asset else None, created_by=actor,
                          value=body.value)
        session.add(existing)
    existing.value, existing.tags, existing.note = body.value, body.tags, body.note
    existing.snapshot = snapshot
    record_event(session, EventType.RATING_SET, project_id=project.id, actor=actor, data=data)
    session.flush()
    return existing


def project_ratings(session: Session, project_id: str, *, rater: str = "human") -> list[Rating]:
    return list(session.scalars(select(Rating).where(Rating.project_id == project_id,
                                                     Rating.rater == rater)
                                .order_by(Rating.updated_at)))


def accepted_renders(session: Session, project_id: str) -> set[str]:
    """Renders QC must not send back for repair: ones you liked, and the shots you left as
    they were when you asked for other shots to be redone."""
    liked = set(session.scalars(select(Rating.render_id).where(
        Rating.project_id == project_id, Rating.rater == "human", Rating.value > 0,
        Rating.render_id.is_not(None))))
    kept: set[str] = set()
    for data in session.scalars(select(Event.data).where(
            Event.project_id == project_id, Event.type == EventType.SHOTS_REDO_REQUESTED)):
        kept.update(data.get("accepted_renders") or [])
    return {r for r in liked | kept if r}


def redo_changes(manifest: ReconstructionManifest, shot_id: str, tags: list[str], *,
                 attempt: int, supported: set[str] | None = None
                 ) -> tuple[dict[str, float | int | str | bool], list[str]]:
    """The changes a redo makes to one shot, and why, from what you said went wrong.

    Every redo gets a fresh seed. A motion complaint firms up the source guide; style and
    prompt complaints loosen it; both together, or neither, leave it. Other tags (flicker,
    detail, subject, background, continuity) have no reliable automatic fix yet, so the seed
    is the only change and the reason says so.
    """
    shot = manifest.shot(shot_id)
    seed = int(shot.overrides.get("seed", shot.seed)) + SEED_STEP * max(1, attempt)
    changes: dict[str, float | int | str | bool] = {"seed": seed % 4294967296}
    reasons = ["new seed"]
    firmer = "motion" in tags
    looser = bool({"style", "prompt"} & set(tags))
    if supported is None or "CONTROL_STRENGTH" in supported:
        old = float(shot.overrides.get("control_strength", 1.0))
        new = old
        if firmer and not looser:
            new = round(min(max(old, CONTROL_MAX), old + CONTROL_STEP), 3)
            if new != old:
                reasons.append(f"follows the source more closely ({old:g} → {new:g})")
        elif looser and not firmer:
            new = round(max(min(old, CONTROL_MIN), old - CONTROL_STEP), 3)
            if new != old:
                reasons.append(f"more freedom for the new look ({old:g} → {new:g})")
        if new != old:
            changes["control_strength"] = new
    rest = [TAGS[t].lower() for t in tags
            if t in ("flicker", "detail", "subject", "background", "continuity")]
    if rest:
        reasons.append(f"{', '.join(rest)}: no automatic fix yet, only the seed changes")
    return changes, reasons


_CLOSED = ("CANCELLED", "ARCHIVED", "RIGHTS_REJECTED")


def _unrated(session: Session) -> Any:
    rated = select(Rating.render_id).where(Rating.rater == "human", Rating.render_id.is_not(None))
    return (select(Render, Project.name).join(Project, Project.id == Render.project_id)
            .where(Render.status == "succeeded", Render.output_asset_id.is_not(None),
                   Render.id.not_in(rated), Project.status.not_in(_CLOSED)))


def unrated_renders(session: Session, *, limit: int = 12) -> list[dict[str, Any]]:
    """The newest shot renders you have not rated yet, for a quick rating session."""
    rows = session.execute(_unrated(session).order_by(Render.finished_at.desc().nulls_last())
                           .limit(limit)).all()
    keyframes: dict[tuple[str, str], str] = {}
    if rows:
        for pid, meta, aid in session.execute(select(Asset.project_id, Asset.meta, Asset.id).where(
                Asset.kind == "keyframe", Asset.project_id.in_({r.project_id for r, _ in rows}))):
            keyframes[(pid, str((meta or {}).get("shot_id")))] = aid
    return [{"render_id": r.id, "project_id": r.project_id, "project": name,
             "shot_id": r.shot_id, "attempt": r.attempt, "asset_id": r.output_asset_id,
             "keyframe_id": keyframes.get((r.project_id, r.shot_id))} for r, name in rows]


def unrated_count(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(_unrated(session).subquery())) or 0)


def video_verdicts(session: Session, project_ids: list[str]) -> dict[str, int]:
    """Your latest whole-video rating per project."""
    if not project_ids:
        return {}
    rows = session.execute(select(Rating.project_id, Rating.value).where(
        Rating.project_id.in_(project_ids), Rating.rater == "human", Rating.target == VIDEO)
        .order_by(Rating.updated_at)).all()
    return {pid: value for pid, value in rows}
