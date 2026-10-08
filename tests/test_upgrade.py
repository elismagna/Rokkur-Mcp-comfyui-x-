"""Regression coverage for source guidance, timing, usable repairs and review UX."""
from pathlib import Path

import numpy as np
import pytest

from rokkur_studio.agents.providers import OllamaProvider, RuleBasedProvider
from rokkur_studio.agents.roles import CreativeDirector, DirectorOfPhotography, RepairPlanner
from rokkur_studio.agents.schemas import ShotFraming
from rokkur_studio.comfyui.client import ComfyClient, ComfyError
from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.config import DirectorSection
from rokkur_studio.director.assets import AssetTracker
from rokkur_studio.director.passes import direct
from rokkur_studio.jobs.errors import JobCancelled
from rokkur_studio.manifest.builder import build_manifest, shot_params
from rokkur_studio.pipeline.qc import score_shot
from rokkur_studio.pipeline.renderers import ComfyUIRenderer
from rokkur_studio.services.projects import get_project
from tests.fakes import FakeComfyUI, ollama_transport
from tests.test_dashboard import client_for
from tests.test_manifest_agents import ANALYSIS, brief
from tests.test_pipeline import create, run, status

ROOT = Path(__file__).resolve().parents[1]


def test_reference_node_is_optional_and_connected_when_present():
    template = TemplateRegistry(ROOT / "workflows").get("v2v_3070_quality")
    values = {"STYLE_PROMPT": "clay monkey", "INPUT_VIDEO": "clip.mp4"}
    absent = compile_workflow(template, values).workflow
    assert "20" not in absent and "reference_image" not in absent["14"]["inputs"]
    present = compile_workflow(template, {**values, "REFERENCE_IMAGE": "guide.png"}).workflow
    assert present["20"]["inputs"]["image"] == "guide.png"
    assert present["14"]["inputs"]["reference_image"] == ["20", 0]
    assert "20" in template.workflow  # compiling without a reference is not destructive


def test_wan_frame_padding_and_profile_degradation_keep_shot_duration(settings):
    profile = settings.profile("RTX3070_QUALITY")
    manifest = build_manifest(project_id="test", source_asset="source", analysis=ANALYSIS,
                              brief=brief(), profile=profile, target_format="youtube_video")
    shot = manifest.shots[0]
    shot.end = 2.434
    params = shot_params(manifest, shot, profile)
    assert params["FRAME_COUNT"] == 41 and params["_OUTPUT_FRAMES"] == 39
    assert abs(params["_OUTPUT_FRAMES"] / params["FPS"] - shot.duration) < 1 / params["FPS"]
    shot.end = 8.0
    fallback = settings.profile("PREVIEW")
    params = shot_params(manifest, shot, fallback)
    assert params["WIDTH"] <= fallback.max_width and params["HEIGHT"] <= fallback.max_height
    assert params["FRAME_COUNT"] <= fallback.max_frames
    assert (params["FRAME_COUNT"] - 1) % 4 == 0
    assert params["_OUTPUT_FRAMES"] / params["FPS"] == pytest.approx(shot.duration)


@pytest.mark.parametrize("mode,custom,expected", [
    ("source", False, "source first frame"),
    ("none", False, "none"),
    ("none", True, "uploaded"),
])
def test_renderer_uploads_appearance_reference_and_trims_padding(
        ffmpeg, sample_video, tmp_path, mode, custom, expected):
    fake = FakeComfyUI()
    client = ComfyClient("http://comfy", transport=fake.transport())
    params = {"STYLE_PROMPT": "clay", "WIDTH": 320, "HEIGHT": 560, "FPS": 16,
              "FRAME_COUNT": 41, "_OUTPUT_FRAMES": 39, "_REFERENCE_MODE": mode}
    if custom:
        params["REFERENCE_IMAGE"] = str(ffmpeg.thumbnail(sample_video, tmp_path / "custom.png", at=0))
    out = ComfyUIRenderer(client, TemplateRegistry(ROOT / "workflows"), ffmpeg,
                         timeout_s=10, poll_s=0).render_shot(
        clip=sample_video, params=params, workflow="v2v_3070_quality", out=tmp_path / "out.mp4")
    workflow = next(iter(fake.prompts.values()))
    assert out.details["reference"] == expected
    info = ffmpeg.probe(out.path)
    assert info.frame_count == 39 and info.fps == 16
    assert info.duration == pytest.approx(39 / 16, abs=0.001)
    assert len(ffmpeg.read_gray_frames(out.path)) == 39
    if expected == "none":
        assert "reference_image" not in workflow["14"]["inputs"]
    else:
        image = workflow["20"]["inputs"]["image"]
        assert image in fake.uploads
        assert fake.uploads[image].startswith(b"\x89PNG")
        assert workflow["14"]["inputs"]["reference_image"] == ["20", 0]


def test_cancelled_comfy_render_is_a_cancelled_job(ffmpeg, sample_video, tmp_path, monkeypatch):
    client = ComfyClient("http://comfy", transport=FakeComfyUI().transport())

    def cancelled(*args, **kwargs):
        raise ComfyError("cancelled", "cancelled")

    monkeypatch.setattr(client, "wait", cancelled)
    renderer = ComfyUIRenderer(client, TemplateRegistry(ROOT / "workflows"), ffmpeg,
                              timeout_s=10, poll_s=0)
    with pytest.raises(JobCancelled):
        renderer.render_shot(clip=sample_video, workflow="v2v_3070_quality", out=tmp_path / "x.mp4",
                             params={"STYLE_PROMPT": "clay", "WIDTH": 320, "HEIGHT": 560,
                                     "FPS": 16, "FRAME_COUNT": 17, "_REFERENCE_MODE": "none"})


def test_wan_repairs_only_change_mapped_controls():
    report = {"shots": [{"shot_id": "a", "decision": "FAIL", "issues": ["layout drift"],
                         "recommendations": ["REDUCE_STYLE_STRENGTH", "ADD_DEPTH_CONTROL"]}]}
    plan = RepairPlanner().plan(report, 1, {"a": {"seed": 12, "control_strength": 1}},
                                 supported={"SEED", "CONTROL_STRENGTH"})
    action = plan.actions[0]
    assert set(action.changes) == {"seed", "control_strength"}
    assert action.changes["control_strength"] == 1.15
    assert set(action.unsupported) == {"style_strength", "depth"}
    assert "ADJUST_CONTROL_STRENGTH" in action.recommendations
    assert not RepairPlanner().plan(report, 1, {}, supported=set()).actions


def test_story_is_merged_by_shot_id_and_reads_source_images():
    planned = brief()
    planned.shot_plan[0].subject = "brown monkey"
    planned.shot_plan[1].subject = "green frog"
    planned.shot_plan.reverse()
    calls = []
    provider = OllamaProvider("http://o", "m", transport=ollama_transport([planned.model_dump_json()], calls))
    result, by = CreativeDirector(provider).run({}, ANALYSIS, "youtube_short",
                                                keyframes={"shot_001": b"image"})
    assert by == "ollama"
    assert [s.subject for s in result.shot_plan] == ["brown monkey", "green frog"]
    assert [s.start for s in result.shot_plan] == [0, 2]
    assert calls[0]["messages"][1]["images"]


def test_vision_observation_is_not_primed_with_story_guesses():
    calls = []

    class VisualProvider:
        name = "test_vision"

        def generate(self, role, instructions, payload, schema, images=None):
            calls.append(payload)
            assert {"observed_subject", "observed_background"} <= set(schema.model_json_schema()["required"])
            return ShotFraming(shot_size="Medium Shot", camera_angle="Eye-level",
                camera_movement="Static", lighting="High-key overhead",
                observed_subject="frog held by a hand", observed_background="bubbles")

    original = brief()
    original.shot_plan[0].subject = "invented shell"
    result, _ = DirectorOfPhotography(VisualProvider(), vision=True).run(original,
        keyframes={s.shot_id: b"frame" for s in original.shot_plan})
    assert "subject" not in calls[0]["shot"] and "intent" not in calls[0]["shot"]
    assert "observed_subject" not in calls[1]["previous_shot"]
    assert result.shot_plan[0].subject == "frog held by a hand"


def test_custom_negative_prompt_survives_director_disabled():
    result, _ = direct(RuleBasedProvider(), {"theme": "clay", "negative_prompt": "wireframe"},
                       ANALYSIS, "youtube_short", tracker=AssetTracker(),
                       settings=DirectorSection(enabled=False))
    assert result.negative_prompt == "wireframe"


def test_nearly_static_source_does_not_fail_due_to_uncorrelated_codec_noise():
    rng = np.random.default_rng(34)
    background = np.full((1, 64, 64), 100.0)
    background[:, 20:40, 20:40] = 160
    a = np.clip(background + rng.normal(0, 0.25, (30, 64, 64)), 0, 255).astype(np.uint8)
    b = np.clip(background + rng.normal(0, 0.25, (30, 64, 64)), 0, 255).astype(np.uint8)
    result = score_shot(a, b, threshold=6.5, shot_id="static")
    assert result["motion_method"] == "low-motion difference"
    assert result["motion"] > 9 and result["decision"] == "PASS"


def test_edges_are_not_mistaken_for_a_detailed_finished_image():
    a = np.full((10, 64, 64), 160, dtype=np.uint8)
    a[:, 16:48, 16:48] = 110
    b = np.zeros_like(a)
    b[:, 16:48, 16] = b[:, 16:48, 47] = 255
    b[:, 16, 16:48] = b[:, 47, 16:48] = 255
    result = score_shot(a, b, threshold=6.5, shot_id="edges")
    assert result["visual_review_required"] and result["decision"] == "FAIL"


def test_truncated_static_render_cannot_pass_quality():
    frames = np.full((30, 64, 64), 160, dtype=np.uint8)
    assert score_shot(frames, frames[:10], threshold=6.5, shot_id="short")["decision"] == "FAIL"


def test_frozen_render_does_not_get_a_perfect_motion_score():
    frames = np.full((30, 64, 64), 100, dtype=np.uint8)
    for i, frame in enumerate(frames):
        frame[:, i:i + 16] = 220
    frozen = np.repeat(frames[:1], len(frames), axis=0)
    assert score_shot(frames, frozen, threshold=6.5, shot_id="frozen")["motion"] == 0


def test_custom_reference_reaches_the_workflow_through_the_full_pipeline(ctx, sample_video, tmp_path):
    ctx.settings.render.renderer = "comfyui"
    fake = FakeComfyUI()
    ctx.comfy_factory = lambda: ComfyClient("http://comfy", transport=fake.transport())
    ref = ctx.ffmpeg.thumbnail(sample_video, tmp_path / "reference.png", at=0)
    pid = create(ctx, sample_video, character_reference_path=str(ref),
                 control_strength=0.75, seed=23, steps=22, negative_prompt="edge map")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert len(fake.prompts) == 2
    for workflow in fake.prompts.values():
        assert workflow["14"]["inputs"]["strength"] == 0.75
        assert workflow["15"]["inputs"]["seed"] == 23
        assert workflow["15"]["inputs"]["steps"] == 22
        assert workflow["20"]["inputs"]["image"] == "character.png"
        assert "edge map" in workflow["6"]["inputs"]["text"]
    page = client_for(ctx).get(f"/ui/projects/{pid}").text
    assert "Applied render settings" in page and "uploaded" in page


def test_unavailable_profile_is_refused_before_creating_a_project(ctx, sample_video):
    ctx.settings.render.renderer = "comfyui"
    c = client_for(ctx)
    response = c.post("/ui/projects", data={"theme": "clay", "render_profile": "FUTURE_24GB",
        "rights_category": "USER_OWNED", "local_path": str(sample_video)}, follow_redirects=False)
    assert response.status_code == 303 and "err=" in response.headers["location"]
    assert "unavailable" in c.get("/ui/new").text


def test_reference_upload_and_advanced_controls_survive_creation(ctx, sample_video, ffmpeg, tmp_path):
    c = client_for(ctx)
    ref = ffmpeg.thumbnail(sample_video, tmp_path / "guide.png", at=0)
    response = c.post("/ui/projects", data={"theme": "clay", "rights_category": "USER_OWNED",
        "local_path": str(sample_video), "control_strength": "0.75", "seed": "13", "steps": "22",
        "cfg": "5.5", "reference_mode": "none", "negative_prompt": "wireframe"},
        files={"reference_file": ("guide.png", ref.read_bytes(), "image/png")}, follow_redirects=False)
    assert response.status_code == 303
    pid = response.headers["location"].split("/")[-1].split("?")[0]
    with ctx.db.session() as session:
        creative = get_project(session, pid).creative_input
    assert creative["control_strength"] == 0.75 and creative["seed"] == 13
    assert creative["steps"] == 22 and creative["cfg"] == 5.5
    assert creative["reference_mode"] == "none" and creative["negative_prompt"] == "wireframe"
    assert creative["character_reference_path"].endswith(".png")
