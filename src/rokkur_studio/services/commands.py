"""Use cases shared by the API and the CLI (create, start, cancel, resume, decide)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.api.schemas import ProjectCreate
from rokkur_studio.config import Settings, project_target
from rokkur_studio.db.models import ApprovalRequest, Channel, Job, Project, Render, utcnow
from rokkur_studio.domain.rights import RightsCategory, RightsStatus
from rokkur_studio.domain.states import TERMINAL, InvalidTransition, ProjectStatus
from rokkur_studio.jobs.queue import cancel_project_jobs, enqueue
from rokkur_studio.manifest.schema import ReconstructionManifest
from rokkur_studio.pipeline.driver import advance
from rokkur_studio.services import ratings
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
    target = settings.new_project_target(data.creative.render_on)
    if problem := settings.profile_problem(profile, target=target):
        raise ValueError(problem)
    if data.creative.audio_bed_path and (
            not data.creative.audio_bed_rights_confirmed
            or not (data.creative.audio_bed_rights_evidence or "").strip()):
        raise ValueError("Confirm that you can use the added soundtrack and provide its "
                         "permission or license details.")
    if data.channel_id and session.get(Channel, data.channel_id) is None:
        raise LookupError(f"channel {data.channel_id} not found")
    project = Project(name=data.name, status=S.DISCOVERED.value, target_format=data.target_format,
                      render_profile=profile, channel_id=data.channel_id,
                      creative_input={**data.creative.model_dump(exclude_none=True),
                                      "render_on": target})
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
                 data={"name": project.name, "profile": profile, "render_on": target,
                       "rights_category": r.category.value, "source": data.source.platform})
    session.flush()
    session.refresh(project)
    if data.autostart:
        start(session, project, settings, actor=actor)
    return project


def start(session: Session, project: Project, settings: Settings, actor: str = "api") -> Job | None:
    if problem := settings.profile_problem(project.render_profile,
                                           target=project_target(project.creative_input)):
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

    GPU minutes (local and cloud) grow in proportion, so whichever budget stopped it is lifted.
    """
    if not at_budget_limit(session, project, settings):
        raise InvalidTransition(S(project.status), S(project.status),
                                "project is not stopped at the render budget")
    base = settings.render.max_renders_per_project
    renders = renders or render_grant(settings)
    if not 1 <= renders <= base:
        raise ValueError(f"allow between 1 and {base} more renders")
    gpu_minutes = round(renders * settings.costs.max_gpu_minutes_per_project / base, 1)
    cloud_minutes = round(renders * settings.costs.max_cloud_gpu_minutes / base, 1)
    record_event(session, EventType.BUDGET_EXTENDED, project_id=project.id, actor=actor,
                 data={"renders": renders, "gpu_minutes": gpu_minutes,
                       "cloud_minutes": cloud_minutes})
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


def redo_shots(session: Session, project: Project, settings: Settings, *, shot_ids: list[str],
               actor: str, supported: set[str] | None = None) -> Job | None:
    """Render the shots you picked again; every other shot keeps its render as it is.

    Allowed on a finished video and on a project stopped at the repair limit. Each redone shot
    gets a new seed plus the changes its dislike tags call for (ratings.redo_changes); the
    plan is saved as a repair plan so the project page shows what changed. The shots left
    alone are recorded as accepted, so QC never sends them back for repair, and the redone
    shots get the usual automatic repair rounds.
    """
    shot_ids = list(dict.fromkeys(shot_ids))
    if not shot_ids:
        raise ValueError("Pick at least one shot to redo")
    ready = project.status == S.READY_TO_PUBLISH
    if not ready and not at_repair_limit(project):
        raise InvalidTransition(S(project.status), S.REPAIRING,
                                "shots can be redone on a finished video or on a project "
                                "stopped at the repair limit")
    mdoc = latest_document(session, project.id, "manifest")
    if mdoc is None:
        raise ValueError("This project has no shots to redo yet")
    manifest = ReconstructionManifest.model_validate(mdoc.data)
    unknown = [s for s in shot_ids if s not in {shot.shot_id for shot in manifest.shots}]
    if unknown:
        raise ValueError(f"Unknown shots: {', '.join(unknown)}")
    current: dict[str, Render] = {}
    for r in session.scalars(select(Render).where(Render.project_id == project.id,
                                                  Render.status == "succeeded")
                             .order_by(Render.attempt)):
        current[r.shot_id] = r
    verdicts = {r.render_id: r for r in ratings.project_ratings(session, project.id)}
    actions: list[dict[str, Any]] = []
    for sid in shot_ids:
        render = current.get(sid)
        verdict = verdicts.get(render.id) if render is not None else None
        tags = verdict.tags if verdict is not None and verdict.value < 0 else []
        changes, reasons = ratings.redo_changes(manifest, sid, tags,
                                                attempt=render.attempt if render else 1,
                                                supported=supported)
        manifest.shot(sid).overrides.update(changes)
        actions.append({"shot_id": sid, "changes": changes, "reason": "; ".join(reasons),
                        "recommendations": ["CHANGE_SEED"] + (
                            ["ADJUST_CONTROL_STRENGTH"] if "control_strength" in changes else []),
                        "unsupported": [], "tags": tags})
    accepted = sorted(r.id for sid, r in current.items() if sid not in shot_ids)
    if not ready:
        resume(session, project, actor=actor)  # back to QUALITY_FAILED, then repair below
        _close_repair_requests(session, project, decided_by=actor,
                               note=f"redoing {', '.join(shot_ids)}")
    for req in session.query(ApprovalRequest).filter_by(project_id=project.id, kind="publish",
                                                        status="pending"):
        req.status, req.decided_by, req.decided_at = "rejected", actor, utcnow()
        req.note = "superseded: shots are being redone, a new video will be proposed"
    transition(session, project, S.REPAIRING, actor=actor, reason="you asked to redo shots",
               data={"shots": shot_ids})
    save_document(session, project.id, "repair_plan",
                  {"round": project.repair_rounds, "requested_by": actor, "actions": actions},
                  created_by=actor)
    save_document(session, project.id, "manifest", manifest.model_dump(), created_by=actor)
    for r in session.scalars(select(Render).where(Render.project_id == project.id,
                                                  Render.shot_id.in_(shot_ids),
                                                  Render.status == "succeeded")):
        r.status = "superseded"
    record_event(session, EventType.SHOTS_REDO_REQUESTED, project_id=project.id, actor=actor,
                 data={"shots": shot_ids, "changes": {a["shot_id"]: a["changes"] for a in actions},
                       "accepted_renders": accepted})
    # The redone shots get the normal number of automatic repair rounds.
    record_event(session, EventType.REPAIR_BUDGET_EXTENDED, project_id=project.id, actor=actor,
                 data={"rounds": settings.render.max_retries, "after": project.repair_rounds,
                       "reason": "redo"})
    transition(session, project, S.RENDER_QUEUED, actor=actor)
    return advance(session, project, settings, manual=True)


ADJUSTABLE = frozenset({"WORKFLOW_READY", "RENDER_QUEUED", "RENDERING", "QUALITY_CHECK",
                        "QUALITY_FAILED", "REPAIRING"})
SAFE_KEYS = ("prompt_extra", "seed", "steps", "cfg", "control_strength", "canny_low", "canny_high")


def rendered_shots(session: Session, project_id: str) -> set[str]:
    """Shots with a finished render, or one in progress whose settings are already fixed."""
    return set(session.scalars(select(Render.shot_id).where(
        Render.project_id == project_id, Render.status.in_(("succeeded", "running")))))


def adjust_remaining_shots(session: Session, project: Project, *, changes: dict[str, Any],
                           actor: str, reference_image: str | None = None,
                           clear_reference: bool = False) -> dict[str, Any]:
    """Change what is safe while a video renders: only shots without a finished render take
    the new values, and the render loop picks the new manifest up before its next shot.

    ``changes`` holds any of SAFE_KEYS; ``reference_image`` is a store-relative path for the
    appearance reference (every remaining shot then gets it; rendered shots keep theirs).
    """
    if project.status not in ADJUSTABLE:
        raise InvalidTransition(S(project.status), S.RENDERING,
                                "adjustments apply while the shots render; a finished video "
                                "takes a redo instead")
    applied = {k: v for k, v in changes.items() if k in SAFE_KEYS and v is not None}
    if "prompt_extra" in applied and not str(applied["prompt_extra"]).strip():
        applied.pop("prompt_extra")
    if "canny_low" in applied and "canny_high" in applied and applied["canny_low"] >= applied["canny_high"]:
        raise ValueError("the low edge threshold must be below the high one")
    if not applied and reference_image is None and not clear_reference:
        raise ValueError("nothing to change")
    mdoc = latest_document(session, project.id, "manifest")
    if mdoc is None:
        raise ValueError("the shots are not planned yet; adjustments start once the workflow "
                         "is ready")
    manifest = ReconstructionManifest.model_validate(mdoc.data)
    done = rendered_shots(session, project.id)
    remaining = [s.shot_id for s in manifest.shots if s.shot_id not in done]
    if not remaining:
        raise ValueError("every shot has rendered; use Redo on the shots you want changed")
    for shot in manifest.shots:
        if shot.shot_id in remaining:
            shot.overrides.update(applied)
    if clear_reference:
        manifest.identity.reference_image = None
        manifest.identity.reference_mode = "none"
    elif reference_image is not None:
        manifest.identity.reference_image = reference_image
    doc = save_document(session, project.id, "manifest", manifest.model_dump(), created_by=actor)
    record_event(session, EventType.SHOTS_ADJUSTED, project_id=project.id, actor=actor,
                 data={"shots": remaining, "changes": applied, "manifest_version": doc.version,
                       "reference": reference_image, "clear_reference": clear_reference})
    return {"shots": remaining, "changes": applied, "manifest_version": doc.version,
            "rendered_untouched": sorted(done)}


def set_soundtrack(session: Session, project: Project, settings: Settings, *, actor: str,
                   bed_path: str | None, rights: str | None, clip_id: str | None = None,
                   gain: float | None = None, keep_source_audio: bool | None = None
                   ) -> dict[str, Any]:
    """Give a video its soundtrack at any stage (docs/audio.md).

    ``bed_path`` is a sound file (a clip from the sound library or an upload) mixed under the
    original sound in the edit stage, or None to remove the added track; ``rights`` says where
    it comes from. Before the edit stage the choice is saved and used when the edit runs. On a
    finished video the edit runs again now and the new final replaces the old one; the earlier
    final stays in Renders. A published video keeps its sound.
    """
    status = project.status
    if status in (S.PUBLISHED.value, S.MONITORING.value):
        raise InvalidTransition(S(status), S.EDITING,
                                "a published video keeps its sound; make a new video for a "
                                "different soundtrack")
    if status in (S.EDITING.value, S.PUBLISHING.value, S.ARCHIVED.value, S.CANCELLED.value):
        raise InvalidTransition(S(status), S.EDITING, f"the soundtrack cannot change in {status}")
    if bed_path and not (rights or "").strip():
        raise ValueError("Say where the soundtrack comes from (its rights or how it was made).")
    creative = dict(project.creative_input or {})
    changes: dict[str, Any] = {}
    if bed_path:
        changes.update(audio_bed_path=bed_path, audio_bed_rights_confirmed=True,
                       audio_bed_rights_evidence=(rights or "").strip(), audio_clip_id=clip_id)
    else:
        for key in ("audio_bed_path", "audio_bed_rights_confirmed", "audio_bed_rights_evidence",
                    "audio_clip_id"):
            creative.pop(key, None)
        changes["audio_bed_path"] = None
    if gain is not None:
        if not 0 <= gain <= 2:
            raise ValueError("the track level is between 0 and 2")
        changes["audio_bed_gain"] = gain
    if keep_source_audio is not None:
        changes["keep_source_audio"] = keep_source_audio
    creative.update({k: v for k, v in changes.items() if v is not None or k == "audio_bed_path"})
    if creative.get("audio_bed_path") is None:
        creative.pop("audio_bed_path", None)
    project.creative_input = creative
    re_edit = status == S.READY_TO_PUBLISH.value
    job = None
    if re_edit:
        for req in session.query(ApprovalRequest).filter_by(project_id=project.id, kind="publish",
                                                            status="pending"):
            req.status, req.decided_by, req.decided_at = "rejected", actor, utcnow()
            req.note = "superseded: the soundtrack changed, a new video will be proposed"
        transition(session, project, S.EDITING, actor=actor, reason="soundtrack changed",
                   data=changes)
        job = enqueue(session, "edit", project_id=project.id, stage=S.EDITING.value,
                      dedupe_key=f"{project.id}:stage",
                      max_attempts=settings.jobs.default_max_attempts)
    elif status == S.FAILED.value:
        pass  # saved; it is used when the video is resumed and reaches the edit stage
    record_event(session, EventType.SOUNDTRACK_SET, project_id=project.id, actor=actor,
                 job_id=job.id if job else None,
                 data={**changes, "re_edit": re_edit, "state": status})
    return {"changes": changes, "re_edit": re_edit, "job_id": job.id if job else None,
            "state": project.status}


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
