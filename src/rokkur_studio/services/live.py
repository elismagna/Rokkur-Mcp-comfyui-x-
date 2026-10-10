"""The live view of one video: what the studio is doing, what each stage produced and who
produced it, what is waiting for a person, and the evidence to decide on (docs/live.md).

Read-only. Everything here is built from what the studio already records (states, events,
documents, renders, assets, approvals, costs), so nothing is invented and nothing is lost
between the worker and the person: the goal is communication without losing or altering
detail. The dashboard renders ``snapshot`` as a page and polls it as JSON.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.comfyui.client import ComfyError
from rokkur_studio.config import project_target
from rokkur_studio.db.models import ApprovalRequest, Asset, Document, Event, Job, Project, Render
from rokkur_studio.domain.rights import RightsStatus
from rokkur_studio.domain.states import ProjectStatus as S
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import ratings
from rokkur_studio.services.projects import (
    at_budget_limit,
    at_repair_limit,
    budget_usage,
    failure_reason,
    latest_rights,
)

STEP_LABELS = {"rights_check": "Checking rights", "ingest": "Copying the source",
               "analyze": "Analysing the footage", "creative_plan": "Writing the brief",
               "compile_workflow": "Building workflows", "render": "Rendering",
               "qc": "Checking quality", "repair": "Planning repairs",
               "edit": "Editing the final video", "extend": "Extending the video",
               "image": "Making pictures", "audio": "Making sound"}

# The pipeline as a graph: node id, label, the states it covers, and the job that runs it.
NODES: list[tuple[str, str, set[S], str | None]] = [
    ("rights", "Rights", {S.RIGHTS_PENDING, S.RIGHTS_OK}, "rights_check"),
    ("ingest", "Ingest", {S.DOWNLOADED_OR_INGESTED}, "ingest"),
    ("analyse", "Analyse", {S.ANALYZING, S.ANALYZED}, "analyze"),
    ("brief", "Brief", {S.CREATIVE_PLANNING, S.CREATIVE_READY}, "creative_plan"),
    ("workflow", "Workflow", {S.WORKFLOW_COMPILING, S.WORKFLOW_READY}, "compile_workflow"),
    ("render", "Render", {S.RENDER_QUEUED, S.RENDERING}, "render"),
    ("quality", "Quality", {S.QUALITY_CHECK, S.QUALITY_FAILED, S.QUALITY_PASSED}, "qc"),
    ("repair", "Repair", {S.REPAIRING}, "repair"),
    ("edit", "Edit", {S.EDITING}, "edit"),
    ("ready", "Ready", {S.READY_TO_PUBLISH, S.PUBLISHING}, None),
    ("published", "Published", {S.PUBLISHED, S.MONITORING}, None),
]
_ORDER = [n[0] for n in NODES]


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _node_states(project: Project) -> dict[str, str]:
    """done / current / failed / waiting / todo per node, from the project's state."""
    status = S(project.status)
    effective = S(project.failed_from_state) if status == S.FAILED and project.failed_from_state else status
    index = next((i for i, (_, _, states, _) in enumerate(NODES) if effective in states), -1)
    if status in (S.PUBLISHED, S.MONITORING, S.ARCHIVED) and index == -1:
        index = len(NODES) - 1
    out = {}
    for i, (node, _, _, _) in enumerate(NODES):
        if i < index:
            out[node] = "done"
        elif i == index:
            if status in (S.FAILED, S.RIGHTS_REJECTED):
                out[node] = "failed"
            elif status in (S.PUBLISHED, S.MONITORING, S.READY_TO_PUBLISH):
                out[node] = "done"
            else:
                out[node] = "current"
        else:
            out[node] = "todo"
    past_repair = out["edit"] in ("done", "current") or out["ready"] == "done"
    if project.repair_rounds == 0 and out["repair"] != "current" and past_repair:
        out["repair"] = "skipped"
    return out


def _documents(session: Session, project_id: str) -> dict[str, Document]:
    latest: dict[str, Document] = {}
    for doc in session.scalars(select(Document).where(Document.project_id == project_id)
                               .order_by(Document.version)):
        latest[doc.kind] = doc
    return latest


def _by(created_by: str | None) -> str:
    """Who produced a document, in words: ``creative_director:ollama`` → the director on Ollama."""
    if not created_by:
        return "the studio"
    role, _, provider = created_by.partition(":")
    role_text = role.replace("_", " ")
    if created_by in ("dashboard", "api", "cli", "you") or provider == "you":
        return "you"
    if role in ("rule_based", "rules"):
        return "the studio's rules (no model)"
    if provider in ("rule_based", "rules"):
        return f"the {role_text} (rules, no model)"
    if provider:
        return f"the {role_text} on {provider}"
    return f"the {role_text}"


def _stage_trace(session: Session, project: Project, docs: dict[str, Document],
                 renders: Sequence[Render], assets: Sequence[Asset], events: Sequence[Event],
                 states: dict[str, str]) -> list[dict[str, Any]]:
    """What each node produced, who did it, and when: the record a person can check."""
    pid = project.id
    first_at: dict[str, datetime] = {}
    last_at: dict[str, datetime] = {}
    for e in events:
        if e.type == "STATE_CHANGED" and e.to_state:
            node = next((n for n, _, st, _ in NODES if S(e.to_state) in st), None)
            if node:
                first_at.setdefault(node, e.created_at)
                last_at[node] = e.created_at
    out = []
    rights = latest_rights(session, pid)
    source = next((a for a in assets if a.kind == "source"), None)
    analysis = docs.get("analysis")
    brief = docs.get("creative_brief")
    manifest = docs.get("manifest")
    qc = docs.get("qc_report")
    plan = docs.get("repair_plan")
    metadata = docs.get("metadata")
    finals = [a for a in assets if a.kind == "final"]
    succeeded = [r for r in renders if r.status == "succeeded"]
    total_shots = len((manifest.data.get("shots") if manifest else None) or
                      (brief.data.get("shot_plan") if brief else None) or [])
    creative = project.creative_input or {}
    for node, label, _, job_kind in NODES:
        lines: list[str] = []
        by = ""
        if node == "rights" and rights is not None:
            by = ("you" if rights.decided_by in ("dashboard", "api", "cli", "you")
                  else "the rights gate (automatic rules)" if rights.decided_by == "rights_gate"
                  else rights.decided_by or "")
            lines.append(f"{rights.category.replace('_', ' ').lower()} · {rights.status.lower()}"
                         + (f" · {rights.permission_evidence}" if rights.permission_evidence else ""))
            if rights.reason:
                lines.append(rights.reason)
        elif node == "ingest" and source is not None:
            probe = (source.meta or {}).get("probe") or {}
            if probe:
                lines.append(f"{probe.get('width')}×{probe.get('height')} · {probe.get('fps')} fps · "
                             f"{float(probe.get('duration', 0)):.1f} s"
                             + (" · with sound" if probe.get("has_audio") else " · silent"))
            lines.append(source.rel_path)
        elif node == "analyse" and analysis is not None:
            d = analysis.data
            lines.append(f"{len(d.get('shots', []))} shots from {len(d.get('scene_cuts', []))} "
                         f"scene cuts · {float(d.get('duration', 0)):.1f} s")
            lines.append("measured: " + ", ".join(d.get("signals_computed", [])))
            by = "FFmpeg scene detection and motion measurement"
        elif node == "brief" and brief is not None:
            d = brief.data
            by = _by(brief.created_by)
            if d.get("prompt"):
                lines.append(d["prompt"])
            if d.get("subject"):
                lines.append(f"follows: {d['subject']}")
            lines.append(f"{len(d.get('shot_plan', []))} shots planned"
                         + (" · Stable mode" if creative.get("stable") else ""))
        elif node == "workflow" and manifest is not None:
            d = manifest.data
            video = d.get("video", {})
            lines.append(f"{d.get('render_profile')} · {video.get('width')}×{video.get('height')} · "
                         f"{video.get('fps')} fps · manifest v{manifest.version}")
            subject = d.get("subject") or {}
            if subject:
                lines.append(f"subject {subject.get('mode', '?')}: {subject.get('reason', '')}")
            identity = d.get("identity") or {}
            lines.append(f"reference {identity.get('reference_mode', 'none')}")
            by = _by(manifest.created_by)
        elif node == "render":
            done_ids = {r.shot_id for r in succeeded}
            lines.append(f"{len(done_ids)} of {total_shots or '?'} shots rendered · "
                         f"{len(renders)} attempts")
            if renders:
                lines.append(f"{renders[0].renderer} · {renders[-1].workflow or renders[-1].profile}")
                by = "ComfyUI" if renders[-1].renderer == "comfyui" else "FFmpeg preview"
        elif node == "quality" and qc is not None:
            d = qc.data
            lines.append(f"{float(d.get('overall') or 0):.2f} of 10 · pass mark "
                         f"{d.get('threshold')} · {str(d.get('decision', '')).lower()}")
            if d.get("failed_shots"):
                lines.append("failing: " + ", ".join(s.replace("shot_", "") for s in d["failed_shots"]))
            if d.get("override"):
                lines.append("kept by you")
            by = "measured metrics only (temporal, motion, structure)"
        elif node == "repair":
            if project.repair_rounds:
                lines.append(f"{project.repair_rounds} round{'s' if project.repair_rounds != 1 else ''}")
            if plan is not None:
                for a in plan.data.get("actions", [])[:4]:
                    changes = ", ".join(f"{k} {v}" for k, v in (a.get("changes") or {}).items())
                    lines.append(f"{a['shot_id'].replace('shot_', 'shot ')}: {changes or 'new seed'}"
                                 + (f" ({a['reason']})" if a.get("reason") else ""))
                by = _by(plan.created_by)
        elif node == "edit" and finals:
            final = finals[-1]
            probe = (final.meta or {}).get("probe") or {}
            lines.append(f"{final.rel_path} · {float(probe.get('duration', 0)):.1f} s"
                         + (" · extended" if (final.meta or {}).get("extended") else ""))
            if creative.get("audio_bed_path"):
                lines.append(f"soundtrack mixed at {float(creative.get('audio_bed_gain', 0.25)):.0%}"
                             + ("" if creative.get("keep_source_audio", True) else ", footage muted"))
            by = "FFmpeg"
        elif node == "ready" and metadata is not None:
            d = metadata.data
            lines.append(d.get("title", ""))
            by = "you" if d.get("text_by") == "you" else _by(metadata.created_by)
        elif node == "published":
            pubs = [e for e in events if e.type == "VIDEO_PUBLISHED"]
            if pubs:
                lines.append(f"uploaded · {pubs[-1].data.get('privacy', 'private')}")
        out.append({"id": node, "label": label, "state": states[node], "by": by, "lines": lines,
                    "started_at": _iso(first_at.get(node)), "updated_at": _iso(last_at.get(node)),
                    "job": job_kind})
    return out


def _shots(session: Session, project: Project, docs: dict[str, Document],
           renders: Sequence[Render], assets: Sequence[Asset]) -> list[dict[str, Any]]:
    pid = project.id
    brief = docs.get("creative_brief")
    manifest = docs.get("manifest")
    qc = docs.get("qc_report")
    analysis = docs.get("analysis")
    analysed = {s["shot_id"]: s for s in (analysis.data.get("shots", []) if analysis else [])}
    planned = {s["shot_id"]: s for s in (brief.data.get("shot_plan", []) if brief else [])}
    timing = {s["shot_id"]: s for s in (manifest.data.get("shots", []) if manifest else [])}
    qc_by_render = {r.get("render_id"): r for r in (qc.data.get("shots", []) if qc else [])}
    keyframes = {a.meta.get("shot_id"): a for a in assets if a.kind == "keyframe"}
    verdicts = {r.render_id: r for r in ratings.project_ratings(session, pid) if r.render_id}
    order = list(timing) or list(planned) or list(analysed)
    for r in renders:
        if r.shot_id not in order:
            order.append(r.shot_id)
    out = []
    for sid in order:
        mine = [r for r in renders if r.shot_id == sid]
        seen = analysed.get(sid, {})
        latest = next((r for r in reversed(mine) if r.status == "succeeded"), None)
        running = next((r for r in mine if r.status == "running"), None)
        failed = next((r for r in reversed(mine) if r.status in ("failed", "oom")), None) \
            if latest is None and running is None else None
        asset = session.get(Asset, latest.output_asset_id) if latest and latest.output_asset_id else None
        result = qc_by_render.get(latest.id) if latest else None
        rating = verdicts.get(latest.id) if latest else None
        if running:
            state = "rendering"
        elif latest and result and result.get("decision") != "PASS" and not result.get("accepted_by"):
            state = "needs repair"
        elif latest:
            state = "rendered"
        elif failed:
            state = "failed"
        else:
            state = "planned"
        plan = {**seen, **planned.get(sid, {})}
        spec = timing.get(sid, {})
        current = running or latest or failed
        motion = f"{plan['motion_type']} motion" if plan.get("motion_type") else ""
        out.append({
            "id": sid, "label": sid.replace("shot_", "Shot "), "state": state,
            "start": spec.get("start", plan.get("start")), "end": spec.get("end", plan.get("end")),
            "intent": plan.get("intent") or motion,
            "prompt": plan.get("prompt", ""),
            "framing": [t for t in (plan.get("shot_size"), plan.get("camera_angle"),
                                    plan.get("camera_movement"), plan.get("lighting")) if t],
            "keyframe_url": (f"/projects/{pid}/assets/{keyframes[sid].id}/file"
                             if sid in keyframes else None),
            "render_url": f"/projects/{pid}/assets/{asset.id}/file" if asset else None,
            "attempts": len(mine), "attempt": current.attempt if current else 0,
            "running_since": _iso(running.started_at) if running else None,
            "qc": ({"overall": result.get("overall"), "decision": result.get("decision"),
                    "issues": result.get("issues", []), "accepted_by": result.get("accepted_by")}
                   if result else None),
            "rating": rating.value if rating else None,
            "overrides": {k: v for k, v in (spec.get("overrides") or {}).items()
                          if k in ("seed", "steps", "control_strength", "cfg", "prompt_extra")},
            "error": (failed.error or {}).get("message") if failed else None,
        })
    return out


def _now(session: Session, ctx: StudioContext, project: Project, renders: Sequence[Render]
         ) -> dict[str, Any]:
    """The running job, the shot in progress and ComfyUI's queue, right now."""
    job = session.scalars(select(Job).where(Job.project_id == project.id, Job.status == "RUNNING")
                          .order_by(Job.started_at.desc()).limit(1)).one_or_none()
    queued = session.scalars(select(Job).where(Job.project_id == project.id,
                                               Job.status.in_(("QUEUED", "RETRY_WAIT")))
                             .order_by(Job.created_at)).all()
    running = next((r for r in renders if r.status == "running"), None)
    out: dict[str, Any] = {
        "job": ({"id": job.id, "kind": job.kind, "label": STEP_LABELS.get(job.kind, job.kind),
                 "started_at": _iso(job.started_at), "worker": job.locked_by} if job else None),
        "queued": [{"id": j.id, "kind": j.kind, "label": STEP_LABELS.get(j.kind, j.kind),
                    "status": j.status, "run_after": _iso(j.run_after),
                    "error": (j.error or {}).get("message")} for j in queued],
        "shot": ({"id": running.shot_id, "label": running.shot_id.replace("shot_", "Shot "),
                  "attempt": running.attempt, "started_at": _iso(running.started_at),
                  "workflow": running.workflow, "renderer": running.renderer}
                 if running else None),
        "comfyui": None,
    }
    if running is not None and ctx.settings.render.renderer == "comfyui":
        out["comfyui"] = _comfy_queue(ctx, project_target(project.creative_input))
    return out


def _comfy_queue(ctx: StudioContext, target: str) -> dict[str, Any]:
    """ComfyUI's queue as it is now: how many prompts run and wait, and whether one of them is
    the studio's (its inputs sit under a ``rokkur/`` folder)."""
    client = None
    try:
        client = ctx.comfy_for(target)
        data = client.queue()
    except (ComfyError, httpx.HTTPError, OSError):
        return {"reachable": False, "running": 0, "pending": 0, "ours": None}
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
    running = data.get("queue_running") or []
    pending = data.get("queue_pending") or []

    def ours(entry: Any) -> bool:
        try:
            graph = entry[2]
        except (IndexError, TypeError):
            return False
        return any("rokkur/" in str(v) for n in graph.values() for v in n.get("inputs", {}).values()
                   if isinstance(v, str))

    position = None
    if any(ours(e) for e in running):
        position = 0
    else:
        position = next((i + 1 for i, e in enumerate(pending) if ours(e)), None)
    return {"reachable": True, "running": len(running), "pending": len(pending), "ours": position}


def _decisions(session: Session, ctx: StudioContext, project: Project, docs: dict[str, Document],
               shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What waits for a person, each with the evidence to look at and the actions it offers.
    Actions are the project page's own forms, so deciding here is deciding there."""
    pid = project.id
    base = f"/ui/projects/{pid}"
    out: list[dict[str, Any]] = []
    status = S(project.status)
    rights = latest_rights(session, pid)
    if status == S.RIGHTS_PENDING and rights is not None and rights.status == RightsStatus.NEEDS_HUMAN:
        out.append({
            "kind": "rights", "title": "May the studio use this footage?",
            "why": (rights.reason if rights and rights.reason else
                    "The rights you declared could not be confirmed automatically."),
            "evidence": [{"kind": "image", "url": s["keyframe_url"], "caption": s["label"]}
                         for s in shots if s["keyframe_url"]][:6],
            "actions": [{"label": "Approve rights", "url": f"{base}/approve-rights", "style": ""},
                        {"label": "Reject", "url": f"{base}/reject-rights", "style": "secondary"}],
        })
    if at_repair_limit(project):
        qc = docs.get("qc_report")
        failing = set((qc.data.get("failed_shots") if qc else None) or [])
        evidence = []
        for s in shots:
            if s["id"] in failing or s["state"] == "needs repair":
                if s["render_url"]:
                    score = (s.get("qc") or {}).get("overall")
                    evidence.append({"kind": "video", "url": s["render_url"],
                                     "caption": f"{s['label']} · render"
                                     + (f" · {score:.1f}" if score is not None else ""),
                                     "issues": (s.get("qc") or {}).get("issues", [])})
                if s["keyframe_url"]:
                    evidence.append({"kind": "image", "url": s["keyframe_url"],
                                     "caption": f"{s['label']} · original"})
        out.append({
            "kind": "repair_limit",
            "title": f"Stopped after {project.repair_rounds} repair round"
                     f"{'' if project.repair_rounds == 1 else 's'}",
            "why": "The quality check still fails on "
                   + (", ".join(s.replace("shot_", "shot ") for s in sorted(failing)) or "some shots")
                   + ". Compare each failing render with its original frame: like the ones that "
                     "look right (a liked render passes), redo the others, or keep everything.",
            "evidence": evidence,
            "actions": [
                {"label": f"Try {ctx.settings.render.max_retries} more repairs",
                 "url": f"{base}/repair-more", "style": ""},
                {"label": "Check quality again", "url": f"{base}/recheck-quality", "style": "secondary"},
                {"label": "Keep these renders", "url": f"{base}/keep-renders", "style": "secondary",
                 "confirm": "Keep these renders and finish the video from them?"},
                {"label": "Rate and redo shots on the project page", "href": f"{base}#shots",
                 "style": "ghost"}],
        })
    elif at_budget_limit(session, project, ctx.settings) and status == S.FAILED:
        usage = budget_usage(session, pid, ctx.settings)
        out.append({
            "kind": "budget", "title": "Stopped at the render budget",
            "why": f"{usage['renders']} of {usage['max_renders']} renders and {usage['gpu']:.1f} of "
                   f"{usage['max_gpu']:g} GPU minutes used. Look at the renders before allowing more.",
            "evidence": [{"kind": "video", "url": s["render_url"], "caption": s["label"]}
                         for s in shots if s["render_url"]][:6],
            "actions": [{"label": "Allow more renders and continue",
                         "url": f"{base}/allow-more-renders", "style": ""}],
        })
    elif status == S.FAILED:
        out.append({
            "kind": "failed", "title": "Stopped",
            "why": f"During {(project.failed_from_state or '').replace('_', ' ').lower()}: "
                   f"{failure_reason(session, project) or 'see the events below'}. Fix the cause, "
                   "then resume.",
            "evidence": [],
            "actions": [{"label": "Resume", "url": f"{base}/resume", "style": ""}],
        })
    for req in session.scalars(select(ApprovalRequest).where(ApprovalRequest.project_id == pid,
                                                              ApprovalRequest.status == "pending")
                               .order_by(ApprovalRequest.requested_at)):
        if (req.kind == "repair_budget" and at_repair_limit(project)) or (
                req.kind == "rights_ambiguity" and any(d["kind"] == "rights" for d in out)):
            continue  # shown above with its evidence
        finals = [{"kind": "video", "url": f"/projects/{pid}/assets/{a.id}/file", "caption": "final video"}
                  for a in session.scalars(select(Asset).where(Asset.project_id == pid,
                                                               Asset.kind == "final")
                                           .order_by(Asset.created_at.desc()).limit(1))]
        out.append({
            "kind": req.kind, "title": "Upload to YouTube?" if req.kind == "publish"
            else req.kind.replace("_", " ").capitalize(),
            "why": req.summary, "evidence": finals if req.kind == "publish" else [],
            "actions": [{"label": "Approve and upload" if req.kind == "publish" else "Approve",
                         "url": f"/ui/approvals/{req.id}/approve", "style": "",
                         "confirm": "Upload this video to YouTube now, as described?"
                         if req.kind == "publish" else None},
                        {"label": "Reject", "url": f"/ui/approvals/{req.id}/reject",
                         "style": "secondary"}],
        })
    return out


def _state_words(value: str | None) -> str:
    return (value or "").replace("_", " ").lower()


def describe_event(e: Event) -> str:
    """One readable line per event, with the numbers that matter."""
    d = e.data or {}
    t = e.type
    shot = str(d.get("shot_id", "")).replace("shot_", "shot ")
    if t == "STATE_CHANGED":
        text = f"{_state_words(e.from_state)} → {_state_words(e.to_state)}"
        return text + (f" ({d['reason']})" if d.get("reason") else "")
    if t == "RENDER_SUBMITTED":
        return f"rendering {shot} attempt {d.get('attempt', '?')}"
    if t == "RENDER_COMPLETED":
        secs = d.get("seconds") or d.get("duration_s")
        return f"{shot} rendered" + (f" in {float(secs):.0f} s" if secs else "")
    if t == "RENDER_FAILED":
        return f"{shot or 'a shot'} failed: {d.get('error') or d.get('message') or ''}"
    if t == "GPU_OOM":
        return f"CUDA out of memory on {shot}; stepping down ({d.get('recovery_step', '')})"
    if t in ("QC_PASSED", "QC_FAILED"):
        score = d.get("overall")
        return (f"quality {float(score):.2f} of 10 · " if score is not None else "quality · ") + \
            ("pass" if t == "QC_PASSED" else "fail: " + ", ".join(
                s.replace("shot_", "shot ") for s in d.get("failed_shots", [])))
    if t == "REPAIR_REQUESTED":
        shots = ", ".join(s.replace("shot_", "shot ") for s in d.get("shots", []))
        if d.get("round") is None and not shots:
            return "repair requested" + (f" ({d['reason']})" if d.get("reason") else "")
        return f"repair round {d.get('round', '?')}" + (f" for {shots}" if shots else "")
    if t == "SHOTS_ADJUSTED":
        return "you adjusted " + ", ".join(s.replace("shot_", "shot ") for s in d.get("shots", [])) \
            + ": " + ", ".join(f"{k} {v}" for k, v in (d.get("changes") or {}).items())
    if t == "SHOTS_REDO_REQUESTED":
        return "you asked to redo " + ", ".join(s.replace("shot_", "shot ") for s in d.get("shots", []))
    if t == "SOUNDTRACK_SET":
        return "soundtrack " + ("changed" if d.get("audio_bed_path") else "removed") + \
            (" · the video is edited again" if d.get("re_edit") else "")
    if t == "EXTENSION_REQUESTED":
        return f"extension of {d.get('seconds')} s requested"
    if t == "VIDEO_EXTENDED":
        return f"video extended by {d.get('seconds')} s"
    if t == "FINAL_ENCODED":
        return f"final video encoded · {float(d.get('duration', 0)):.1f} s"
    if t == "RATING_SET":
        return f"you rated {str(d.get('target', '')).replace('shot_', 'shot ')}: {d.get('value')}"
    if t in ("REPAIR_BUDGET_EXHAUSTED", "BUDGET_EXCEEDED"):
        return t.replace("_", " ").lower() + (f": {d['reason']}" if d.get("reason") else "")
    if t == "JOB_FAILED":
        return f"job failed: {d.get('error') or d.get('message') or ''}"
    return t.replace("_", " ").lower()


def snapshot(session: Session, ctx: StudioContext, project: Project, *, events_limit: int = 60
             ) -> dict[str, Any]:
    """Everything the live view shows, as plain data."""
    pid = project.id
    docs = _documents(session, pid)
    renders = session.scalars(select(Render).where(Render.project_id == pid)
                              .order_by(Render.started_at, Render.attempt)).all()
    assets = session.scalars(select(Asset).where(Asset.project_id == pid)
                             .order_by(Asset.created_at)).all()
    events = session.scalars(select(Event).where(Event.project_id == pid)
                             .order_by(Event.id)).all()
    states = _node_states(project)
    shots = _shots(session, project, docs, renders, assets)
    decisions = _decisions(session, ctx, project, docs, shots)
    if any(d["kind"] == "rights" for d in decisions):
        states["rights"] = "waiting"  # the check ran and left the question to a person
    qc = docs.get("qc_report")
    if (qc is not None and qc.data.get("decision") != "PASS" and not qc.data.get("override")
            and states["quality"] == "done" and states["edit"] == "todo"):
        states["quality"] = "failed"  # the last check failed; repair is where it stopped
    stages = _stage_trace(session, project, docs, renders, assets, events, states)
    now = _now(session, ctx, project, renders)
    usage = budget_usage(session, pid, ctx.settings)
    recent = events[-events_limit:]
    return {
        "project": {"id": pid, "name": project.name, "status": project.status,
                    "failed_from": project.failed_from_state, "profile": project.render_profile,
                    "render_on": project_target(project.creative_input),
                    "repair_rounds": project.repair_rounds,
                    "theme": (project.creative_input or {}).get("theme", "")},
        "working": bool(now["job"] or now["queued"]) or project.status in {s.value for s in (
            S.RIGHTS_OK, S.DOWNLOADED_OR_INGESTED, S.ANALYZING, S.ANALYZED, S.CREATIVE_PLANNING,
            S.CREATIVE_READY, S.WORKFLOW_COMPILING, S.WORKFLOW_READY, S.RENDER_QUEUED, S.RENDERING,
            S.QUALITY_CHECK, S.QUALITY_FAILED, S.REPAIRING, S.QUALITY_PASSED, S.EDITING, S.PUBLISHING)},
        "nodes": [{"id": n, "label": label, "state": states[n]} for n, label, _, _ in NODES],
        "stages": stages,
        "now": now,
        "shots": shots,
        "decisions": decisions,
        "events": [{"id": e.id, "at": _iso(e.created_at), "type": e.type, "actor": e.actor,
                    "text": describe_event(e), "data": e.data} for e in reversed(recent)],
        "usage": {"renders": usage["renders"], "max_renders": usage["max_renders"],
                  "gpu_minutes": round(usage["gpu"], 1), "max_gpu_minutes": usage["max_gpu"],
                  "cloud_minutes": round(usage["cloud"], 1), "usd": round(usage["usd"], 2)},
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
