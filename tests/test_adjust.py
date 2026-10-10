"""Changing the workload while a video renders, and Stable mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from rokkur_studio.db.models import Document, Event, Render
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.manifest.schema import ReconstructionManifest
from rokkur_studio.pipeline.renderers import RenderOutcome
from rokkur_studio.services import commands
from rokkur_studio.services.projects import get_project, latest_document
from tests.test_dashboard import client_for
from tests.test_images import with_comfy
from tests.test_pipeline import create, run, status


def manifest_of(ctx, pid: str) -> ReconstructionManifest:
    with ctx.db.session() as s:
        doc = latest_document(s, pid, "manifest")
        assert doc is not None
        return ReconstructionManifest.model_validate(doc.data)


def render_params(ctx, pid: str) -> dict[str, dict[str, Any]]:
    with ctx.db.session() as s:
        rows = s.scalars(select(Render).where(Render.project_id == pid, Render.status == "succeeded")
                         .order_by(Render.attempt)).all()
        return {r.shot_id: r.params for r in rows}


class AdjustingRenderer:
    """The preview renderer, but the first shot adjusts the rest while it renders."""

    name = "ffmpeg_preview"

    def __init__(self, ctx, inner, changes):
        self.ctx, self.inner, self.changes, self.calls = ctx, inner, changes, 0

    def render_shot(self, *, clip: Path, params: dict[str, Any], workflow: str, out: Path,
                    on_progress=None) -> RenderOutcome:
        self.calls += 1
        if self.calls == 1:
            with self.ctx.db.transaction() as s:
                project = get_project(s, params["_PID"], for_update=True)
                commands.adjust_remaining_shots(s, project, changes=self.changes, actor="you")
        return self.inner.render_shot(clip=clip, params=params, workflow=workflow, out=out)


def test_adjusting_while_rendering_changes_only_the_shots_still_to_come(ctx, sample_video):
    from rokkur_studio.pipeline import stages
    from rokkur_studio.pipeline.renderers import FFmpegPreviewRenderer

    pid = create(ctx, sample_video)
    original = stages.shot_params

    def tagged(manifest, shot, profile):  # the renderer needs to know which project it serves
        return {**original(manifest, shot, profile), "_PID": manifest.project_id}

    stages.shot_params = tagged
    try:
        ctx.extras["renderer"] = AdjustingRenderer(ctx, FFmpegPreviewRenderer(ctx.ffmpeg),
                                                   {"prompt_extra": "warmer light", "seed": 77,
                                                    "steps": 12})
        run(ctx)
    finally:
        stages.shot_params = original
        ctx.extras.pop("renderer")
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    params = render_params(ctx, pid)
    first, rest = params["shot_001"], [params[k] for k in sorted(params) if k != "shot_001"]
    assert rest, "the fixture has two shots"
    assert "warmer light" not in first["STYLE_PROMPT"] and first["SEED"] != 77
    assert all(p["STYLE_PROMPT"].endswith(", warmer light") and p["SEED"] == 77 and p["STEPS"] == 12
               for p in rest)
    manifest = manifest_of(ctx, pid)
    assert manifest.shot("shot_001").overrides.get("seed") is None
    with ctx.db.session() as s:
        event = s.scalars(select(Event).where(Event.project_id == pid,
                                              Event.type == "SHOTS_ADJUSTED")).one()
        assert event.data["shots"] == [sh.shot_id for sh in manifest.shots
                                       if sh.shot_id != "shot_001"]


def test_adjust_is_refused_when_nothing_is_left_or_the_video_is_finished(ctx, sample_video):
    pid = create(ctx, sample_video)
    run(ctx)
    with ctx.db.transaction() as s, pytest.raises(Exception, match="redo"):
        commands.adjust_remaining_shots(s, get_project(s, pid), changes={"seed": 1}, actor="t")
    pid2 = create(ctx, sample_video)
    Worker(ctx, worker_id="t").drain(max_jobs=5)  # rights, ingest, analyze, plan, compile
    assert status(ctx, pid2) in ("WORKFLOW_READY", "RENDER_QUEUED")
    with ctx.db.transaction() as s:
        with pytest.raises(ValueError, match="nothing to change"):
            commands.adjust_remaining_shots(s, get_project(s, pid2), changes={}, actor="t")
        with pytest.raises(ValueError, match="below the high"):
            commands.adjust_remaining_shots(s, get_project(s, pid2), actor="t",
                                            changes={"canny_low": 0.8, "canny_high": 0.3})
        result = commands.adjust_remaining_shots(s, get_project(s, pid2), actor="t",
                                                 changes={"control_strength": 1.2, "bogus": 1},
                                                 clear_reference=True)
    assert result["changes"] == {"control_strength": 1.2} and result["rendered_untouched"] == []
    m = manifest_of(ctx, pid2)
    assert m.identity.reference_mode == "none" and all(
        sh.overrides["control_strength"] == 1.2 for sh in m.shots)
    run(ctx)
    assert all(p["CONTROL_STRENGTH"] == 1.2 for p in render_params(ctx, pid2).values())


def test_adjust_from_the_page_and_the_api_with_a_library_reference(ctx, sample_video):
    from tests.test_images import request as request_images

    with_comfy(ctx)
    ctx.settings.render.renderer = "ffmpeg_preview"  # the video keeps the preview renderer
    pid = create(ctx, sample_video)
    Worker(ctx, worker_id="t").drain(max_jobs=5)
    c = client_for(ctx)
    page = c.get(f"/ui/projects/{pid}").text
    assert "Adjust the remaining shots" in page
    r = c.post(f"/ui/projects/{pid}/adjust", data={"prompt_extra": "softer colours", "steps": "10"},
               follow_redirects=False)
    assert r.status_code == 303 and "msg=Applied%20to%20the" in r.headers["location"]
    ctx.settings.render.renderer = "comfyui"
    picture = request_images(ctx, prompt="a reference look", seed=1)[0]
    Worker(ctx, kinds=["image"], worker_id="pictures").drain()  # the video stays unrendered
    ctx.settings.render.renderer = "ffmpeg_preview"
    r = c.post(f"/projects/{pid}/adjust", json={"reference_image_id": picture.id, "cfg": 5.5})
    assert r.status_code == 200, r.text
    assert r.json()["changes"] == {"cfg": 5.5}
    m = manifest_of(ctx, pid)
    assert m.identity.reference_image and m.identity.reference_image.endswith(f"reference_{picture.id}.png")
    assert c.post(f"/projects/{pid}/adjust", json={}).status_code == 409
    run(ctx)
    params = render_params(ctx, pid)
    assert all(p["STYLE_PROMPT"].endswith(", softer colours") and p["STEPS"] == 10 and p["CFG"] == 5.5
               for p in params.values())
    assert all(p["REFERENCE_IMAGE"].endswith(f"reference_{picture.id}.png") for p in params.values())
    with ctx.db.session() as s:
        versions = s.scalars(select(Document.version).where(Document.project_id == pid,
                                                            Document.kind == "manifest")).all()
    assert max(versions) >= 3


def test_stable_mode_locks_seed_reference_steps_and_raises_the_bar(ctx, sample_video):
    pid = create(ctx, sample_video, stable=True)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    m = manifest_of(ctx, pid)
    seeds = {sh.overrides["seed"] for sh in m.shots}
    assert len(seeds) == 1 and all(sh.overrides["steps"] >= 20 for sh in m.shots)
    assert all(sh.overrides["control_strength"] == 1.0 for sh in m.shots)
    assert m.identity.reference_mode == "cutout"
    with ctx.db.session() as s:
        brief = latest_document(s, pid, "creative_brief")
        qc = latest_document(s, pid, "qc_report")
    assert brief is not None and qc is not None
    plan = brief.data["shot_plan"]
    assert all(not p.get("shot_size") for p in plan)  # no per-shot framing in stable mode
    assert qc.data["threshold"] == ctx.settings.quality.pass_threshold + 1.0
    params = render_params(ctx, pid)
    assert len({p["SEED"] for p in params.values()}) == 1
    loose = create(ctx, sample_video, stable=False, seed=5)
    run(ctx)
    assert {p["SEED"] for p in render_params(ctx, loose).values()} == {5}
