"""The ``practice`` job: one Night School batch (services/practice.py, docs/practice.md).

Renders one shot of a finished video as the baseline, with another seed and with each picked
setting change, on this PC, in one GPU batch so the models load once. Then scores every render
like QC does, with the AI picture review always on, and saves a ``practice_report`` on the
video. The video itself is never touched: no state change, its shots keep their renders, and
practice renders do not count against its render budget.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from sqlalchemy import select

from rokkur_studio.db.models import Asset, Job, Project, Render
from rokkur_studio.jobs.errors import JobError, PermanentJobError
from rokkur_studio.jobs.queue import enqueue
from rokkur_studio.manifest.schema import ReconstructionManifest, ShotSpec
from rokkur_studio.pipeline import qc as qc_mod
from rokkur_studio.pipeline import stages, vision
from rokkur_studio.pipeline.context import StudioContext
from rokkur_studio.services import practice
from rokkur_studio.services.assets import register_asset
from rokkur_studio.services.projects import latest_document, save_document


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower().replace("%", "pct")).strip("_")


def _load(ctx: StudioContext, job: Job) -> tuple[Project, ReconstructionManifest]:
    payload = job.payload or {}
    pid = str(payload.get("project_id") or "")
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        if project is None:
            raise PermanentJobError("practice_no_project", f"no video {pid!r} to practise on")
        if practice.project_busy(s, pid):  # its own stages first; try again after them
            raise JobError("project_busy", f"{project.name} is being worked on right now")
        mdoc = latest_document(s, pid, "manifest")
        if mdoc is None:
            raise PermanentJobError("practice_no_shots", f"{project.name} has no shots yet")
        manifest = ReconstructionManifest.model_validate(mdoc.data)
    return project, manifest


def run_batch(ctx: StudioContext, job: Job) -> dict[str, Any]:
    payload = dict(job.payload or {})
    batch_id = str(payload.get("continues") or job.id)
    project, manifest = _load(ctx, job)
    pid = project.id
    try:
        shot = manifest.shot(str(payload.get("shot_id")))
    except KeyError as exc:
        raise PermanentJobError("practice_no_shot", f"{project.name} has no shot {exc}") from exc
    name = payload.get("profile") or ctx.settings.practice.profile or manifest.render_profile
    problem = ctx.settings.profile_problem(name, target="local")
    if problem:
        raise PermanentJobError("practice_profile", problem)
    profile = ctx.settings.profile(name)
    renderer = stages._renderer(ctx, job, "local")
    if renderer.name == "comfyui":
        live = stages._live_problem(ctx, profile.workflow)
        if live:
            raise PermanentJobError("practice_profile", f"{profile.workflow}: {live}")
    accepts = {v.needs: stages._accepts(ctx, profile.workflow, v.needs)
               for v in practice.CATALOG if v.needs}
    labels: list[str] = payload.get("variants") or practice.pick_variants(
        int(payload.get("round", 0)), ctx.settings.practice.variants, accepts, shot.overrides)

    # What this batch (or the batch it continues, or an earlier try of this job) rendered.
    done: dict[str, str] = dict(payload.get("done") or {})
    with ctx.db.session() as s:
        for r in s.scalars(select(Render).where(Render.job_id == job.id,
                                                Render.status == "practice")):
            label = (r.params or {}).get("_PRACTICE")
            if label:
                done.setdefault(label, r.id)

    # The seed variant first: the first render of a run also loads the models, so the
    # baseline's time, which the others are compared with, is a warm one.
    order = sorted(labels, key=lambda lb: lb != practice.SEED)
    masker = (stages._subject_masker(ctx)
              if manifest.subject.mode == "keep" or manifest.identity.reference_mode == "cutout"
              else None)
    batch = (ctx.gpu.heavy_batch(job.id) if renderer.name != "ffmpeg_preview"
             else contextlib.nullcontext())
    first = True
    with batch:
        for label in order:
            if label in done:
                continue
            if done and _hand_on(ctx, job, payload, labels, done):
                return {"handed_on": True, "rendered": sorted(done)}
            variant = shot.model_copy(deep=True)
            variant.overrides = {**practice.overrides_for(label, shot.overrides, seed=shot.seed,
                                                          steps=profile.steps),
                                 "profile": profile.name}
            render_id = stages._render_shot(ctx, job, manifest, variant, profile, renderer,
                                            masker=masker, target="local", practice=batch_id)
            with ctx.db.transaction() as s:
                row = s.get(Render, render_id)
                assert row is not None
                row.params = {**row.params, "_PRACTICE": label, "_PRACTICE_COLD": first,
                              "_PRACTICE_OVERRIDES": {
                                  k: v for k, v in variant.overrides.items()
                                  if k != "profile" and shot.overrides.get(k) != v}}
            done[label] = render_id
            first = False

    report = {"batch": batch_id, "shot_id": shot.shot_id, "profile": profile.name,
              "round": payload.get("round"),
              **_score(ctx, project, manifest, shot, labels, done, batch_id)}
    with ctx.db.transaction() as s:
        doc = save_document(s, pid, practice.REPORT, report, created_by="practice")
    return {"report_version": doc.version, "best": report["best"],
            "variants": len(report["variants"])}


def _hand_on(ctx: StudioContext, job: Job, payload: dict[str, Any], labels: list[str],
             done: dict[str, str]) -> bool:
    """A real job is waiting: queue the rest of this batch behind it and stop here."""
    with ctx.db.transaction() as s:
        if not practice.real_work_waiting(s):
            return False
        enqueue(s, practice.KIND, payload={**payload, "variants": labels, "done": done,
                                           "continues": job.id},
                max_attempts=ctx.settings.jobs.default_max_attempts,
                priority=practice.PRIORITY)  # the queue runs the real job first
    return True


def _score(ctx: StudioContext, project: Project, manifest: ReconstructionManifest,
           shot: ShotSpec, labels: list[str], done: dict[str, str],
           batch_id: str) -> dict[str, Any]:
    """Every variant scored like QC scores a shot, plus the general quality score."""
    pid, fps = project.id, manifest.video.fps
    detail_w, detail_h = qc_mod.detail_size(manifest.video.width, manifest.video.height)
    sheet_w, sheet_h = vision.frame_size(manifest.video.width, manifest.video.height)
    columns = ctx.settings.quality.contact_sheet_columns
    clip = stages._shot_clip(ctx, manifest, shot)
    src = ctx.ffmpeg.read_gray_frames(clip, 64, 64, fps=fps)
    src_detail = ctx.ffmpeg.read_gray_frames(clip, detail_w, detail_h, fps=fps)
    src_rgb = ctx.ffmpeg.read_rgb_frames(clip, sheet_w, sheet_h, fps=fps)
    # Ollama loads after the batch freed ComfyUI: the 8 GB card never holds both.
    checker = stages._picture_checker(ctx, project, force=True)
    if checker is not None:
        checker.max_calls = len(labels)
    batch_dir = ctx.store.project_dir(pid, f"practice/{batch_id}")
    rows: list[dict[str, Any]] = []
    for label in labels:
        with ctx.db.session() as s:
            render = s.get(Render, done[label])
            asset = s.get(Asset, render.output_asset_id) if render is not None else None
        if render is None or asset is None:
            continue
        rendered = ctx.store.path_for(asset.rel_path)
        out = ctx.ffmpeg.read_gray_frames(rendered, 64, 64, fps=fps)
        result = qc_mod.score_shot(
            src, out, threshold=ctx.settings.quality.pass_threshold, shot_id=shot.shot_id,
            source_detail=src_detail,
            render_detail=ctx.ffmpeg.read_gray_frames(rendered, detail_w, detail_h, fps=fps))
        result, sheet_png = vision.check_shot(
            checker, result, src_rgb,
            ctx.ffmpeg.read_rgb_frames(rendered, sheet_w, sheet_h, fps=fps),
            theme=manifest.style.theme, prompt=shot.prompt or manifest.style.prompt,
            columns=columns)
        sheet = batch_dir / f"sheet_{_slug(label)}.png"
        sheet.write_bytes(sheet_png)
        with ctx.db.transaction() as s:
            sheet_asset = register_asset(s, ctx.store, pid, "practice_sheet", sheet,
                                         {"shot_id": shot.shot_id, "variant": label})
        params = render.params or {}
        rows.append({
            "label": label, "overrides": params.get("_PRACTICE_OVERRIDES") or {},
            "render_id": render.id, "asset_id": asset.id, "sheet_asset_id": sheet_asset.id,
            "seconds": None if params.get("_PRACTICE_COLD") else render.duration_s,
            **practice.quality_score(result),
            "decision": result.get("decision"), "issues": result.get("issues") or [],
            "stability": result.get("stability"), "flicker": result.get("flicker"),
            "review": result.get("picture_review"),
            "review_skipped": result.get("picture_review_skipped"),
        })
    ranked = sorted(rows, key=lambda r: -r["score"])
    base = next((r for r in rows if r["label"] == practice.BASELINE), None)
    for r in rows:
        r["delta"] = round(r["score"] - base["score"], 2) if base else None
    return {"variants": rows, "ranking": [r["label"] for r in ranked],
            "best": ranked[0]["label"] if ranked else None,
            "reviewed": all(r["reviewed"] for r in rows) if rows else False}
