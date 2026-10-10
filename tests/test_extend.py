"""Extending a clip before a video is made, and a finished video after its render."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.db.models import Asset, CostEntry, Event, GpuLease, Job
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.pipeline.extend import OVERLAP_FRAMES, ExtendRequest, plan_frames, request_extension
from tests.conftest import ROOT
from tests.test_dashboard import client_for
from tests.test_images import with_comfy
from tests.test_pipeline import create, run, status


def test_the_extension_template_compiles_and_frames_are_planned_in_fours():
    t = TemplateRegistry(ROOT / "workflows").get("v2v_3070_extend")
    c = compile_workflow(t, {"STYLE_PROMPT": "on it goes", "INPUT_VIDEO": "c.mp4", "MASK_VIDEO": "m.mp4",
                             "WIDTH": 480, "HEIGHT": 832, "FRAME_COUNT": 57, "SEED": 1, "STEPS": 20,
                             "CFG": 6, "FPS": 16, "CANNY_LOW": 0.2})
    assert "Canny" not in {n["class_type"] for n in c.workflow.values()}
    vace = next(n for n in c.workflow.values() if n["class_type"] == "WanVaceToVideo")
    assert vace["inputs"]["control_video"] == ["12", 0] and vace["inputs"]["control_masks"] == ["34", 0]
    assert c.ignored == {"CANNY_LOW": 0.2}
    assert plan_frames(3, 16, 81, 4) == (57, 48)
    assert plan_frames(8, 16, 81, 4) == (81, 72)   # capped by the profile
    assert plan_frames(0.1, 16, 33, 4) == (13, 4)


def test_extending_needs_comfyui_rights_and_a_finished_video(ctx, sample_video):
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="ComfyUI renderer"):
        request_extension(s, ctx.settings, ExtendRequest(source_path=str(sample_video), prompt="x"))
    with_comfy(ctx)
    with ctx.db.transaction() as s:
        with pytest.raises(ValueError, match="Confirm"):
            request_extension(s, ctx.settings, ExtendRequest(source_path=str(sample_video), prompt="x"))
        with pytest.raises(ValueError, match="Choose"):
            request_extension(s, ctx.settings, ExtendRequest(prompt="x"))
    pid = create(ctx, sample_video, autostart=False)
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="no finished render"):
        request_extension(s, ctx.settings, ExtendRequest(project_id=pid, prompt="x"))


def test_a_media_clip_is_extended_next_to_the_original(ctx, sample_video, tmp_path):
    fake = with_comfy(ctx)
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "harbour.mp4"
    clip.write_bytes(sample_video.read_bytes())
    before = ctx.ffmpeg.probe(clip)
    with ctx.db.transaction() as s:
        job = request_extension(s, ctx.settings, ExtendRequest(
            source_path=str(clip), seconds=2, prompt="the boats drift on", rights_confirmed=True,
            rights_evidence="mine"))
        job_id = job.id
    assert Worker(ctx, worker_id="t").drain() == 1
    with ctx.db.session() as s:
        done = s.get(Job, job_id)
        assert done.status == "SUCCEEDED", done.error
        result = done.result
        assert all(lease.released_at is not None for lease in s.scalars(select(GpuLease)))
        assert s.scalars(select(CostEntry).where(CostEntry.job_id == job_id)).one().kind == "gpu_minutes"
    out = Path(result["path"])
    assert out.parent == media and out.name.startswith("harbour_extended_")
    after = ctx.ffmpeg.probe(out)
    assert after.width == before.width and after.height == before.height
    total, added = plan_frames(2, 16, 33, 4)  # the PREVIEW profile caps VACE at 33 frames
    assert added == 24 and result["frames_added"] == added and result["width"] % 16 == 0
    assert abs(after.duration - (before.duration + added / 16)) < 0.1 and after.has_audio
    # the fake echoed the control video: last frames real, then white, so the tail is white
    tail = ctx.ffmpeg.read_rgb_frames(out, 16, 16)[-1]
    assert tail.min() > 200
    sent = next(iter(fake.prompts.values()))
    masks = [n for n in sent.values() if n["class_type"] == "LoadVideo"]
    assert len(masks) == 2 and sent["14"]["inputs"]["length"] == total
    uploaded = {name: data for name, data in fake.uploads.items()}
    control = next(v for k, v in uploaded.items() if "control" in k)
    control_path = tmp_path / "control.mp4"
    control_path.write_bytes(control)
    frames = ctx.ffmpeg.read_rgb_frames(control_path, 8, 8)
    assert frames[0].mean() < 250 and frames[-1].min() > 200 and len(frames) == sent["14"]["inputs"]["length"]
    mask = next(v for k, v in uploaded.items() if "mask" in k)
    mask_path = tmp_path / "mask.mp4"
    mask_path.write_bytes(mask)
    mframes = ctx.ffmpeg.read_gray_frames(mask_path, 8, 8)
    assert mframes[0].max() < 60 and mframes[OVERLAP_FRAMES].min() > 200


def test_a_finished_video_is_extended_into_a_new_final_from_the_page_and_the_api(ctx, sample_video):
    fake = with_comfy(ctx)
    ctx.settings.render.renderer = "ffmpeg_preview"
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    c = client_for(ctx)
    assert "Extend the video" not in c.get(f"/ui/projects/{pid}").text  # preview renderer
    ctx.settings.render.renderer = "comfyui"
    page = c.get(f"/ui/projects/{pid}").text
    assert "Extend the video" in page and "What happens next" in page
    r = c.post(f"/ui/projects/{pid}/extend", data={"seconds": "1.5"}, follow_redirects=False)
    assert r.status_code == 303 and "Extension%20of%201.5" in r.headers["location"]
    with ctx.db.session() as s:
        job = s.scalars(select(Job).where(Job.project_id == pid, Job.kind == "extend")).one()
        assert job.payload["prompt"]  # the brief's scene prompt
        before = [a for a in s.scalars(select(Asset).where(Asset.project_id == pid, Asset.kind == "final"))]
    assert "Extending now" in c.get(f"/ui/projects/{pid}").text
    Worker(ctx, worker_id="t", kinds=["extend"]).drain()
    with ctx.db.session() as s:
        finals = [a for a in s.scalars(select(Asset).where(Asset.project_id == pid, Asset.kind == "final")
                                       .order_by(Asset.created_at))]
        assert len(finals) == len(before) + 1 and finals[-1].meta["extended"] is True
        assert finals[-1].meta["seconds"] == 1.5
        kinds = {e.type for e in s.scalars(select(Event).where(Event.project_id == pid))}
        assert {"EXTENSION_REQUESTED", "VIDEO_EXTENDED"} <= kinds
        longer = ctx.ffmpeg.probe(ctx.store.path_for(finals[-1].rel_path))
        shorter = ctx.ffmpeg.probe(ctx.store.path_for(before[-1].rel_path))
    assert longer.duration > shorter.duration + 1.0
    page = c.get(f"/ui/projects/{pid}").text
    assert "extended by 1.5 s" in page
    r = c.post(f"/projects/{pid}/extend", json={"seconds": 1, "prompt": "and then the rain"})
    assert r.status_code == 202 and r.json()["kind"] == "extend"
    assert c.post("/projects/nope/extend", json={"seconds": 1}).status_code == 404
    assert len(fake.prompts) == 1
    assert isinstance(np.zeros(1), np.ndarray)
