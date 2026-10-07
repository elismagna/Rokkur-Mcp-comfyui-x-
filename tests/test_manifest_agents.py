import json

import pytest
from pydantic import ValidationError

from rokkur_studio.agents.providers import (
    AgentOutputError,
    AgentUnavailable,
    OllamaProvider,
    RokkurCollectiveProvider,
    RuleBasedProvider,
)
from rokkur_studio.agents.roles import CreativeDirector, RepairPlanner
from rokkur_studio.agents.schemas import CreativeBrief
from rokkur_studio.config import RenderProfile
from rokkur_studio.manifest.builder import build_manifest, fit_within, shot_params
from rokkur_studio.manifest.schema import ReconstructionManifest
from tests.fakes import ollama_transport

ANALYSIS = {"duration": 4.0, "fps": 24.0, "width": 360, "height": 640, "source_asset": "p/s.mp4",
            "shots": [{"shot_id": "shot_001", "start": 0, "end": 2, "motion_intensity": 0.3,
                       "motion_type": "moderate"},
                      {"shot_id": "shot_002", "start": 2, "end": 4, "motion_intensity": 0.1,
                       "motion_type": "gentle"}]}
PROFILE = RenderProfile(name="PREVIEW", workflow="v2v_preview", max_width=384, max_height=672,
                        fps=12, max_frames=48, controls={"pose": True})


def brief():
    b, _ = CreativeDirector(RuleBasedProvider()).run({"theme": "retro clay sci-fi"}, ANALYSIS,
                                                     "youtube_short")
    return b


def test_rule_based_brief_is_structured():
    b = brief()
    assert b.target_aspect_ratio == "9:16" and len(b.shot_plan) == 2
    assert "retro clay sci-fi" in b.prompt


def test_manifest_is_versioned_and_profile_aware():
    m = build_manifest(project_id="p", source_asset="p/s.mp4", analysis=ANALYSIS, brief=brief(),
                       profile=PROFILE, target_format="youtube_short")
    assert m.version == 1 and m.video.fps == 12
    assert (m.video.width, m.video.height) == (352, 640)  # rounded to multiples of 16
    assert m.shots[0].controls.pose is True
    again = ReconstructionManifest.model_validate(json.loads(m.model_dump_json()))
    assert again == m


def test_manifest_rejects_overlapping_or_inverted_shots():
    m = build_manifest(project_id="p", source_asset="s", analysis=ANALYSIS, brief=brief(),
                       profile=PROFILE, target_format="youtube_short").model_dump()
    m["shots"][1]["start"] = 1.0
    with pytest.raises(ValidationError, match="overlap"):
        ReconstructionManifest.model_validate(m)
    m["shots"][1].update(start=3, end=2)
    with pytest.raises(ValidationError):
        ReconstructionManifest.model_validate(m)


def test_fit_within_keeps_aspect_and_multiple_of_8():
    assert fit_within(1080, 1920, 576, 1024) == (576, 1024)
    w, h = fit_within(1920, 1080, 576, 1024)
    assert w <= 576 and w % 8 == 0 and h % 8 == 0


def test_shot_params_apply_overrides():
    m = build_manifest(project_id="p", source_asset="s", analysis=ANALYSIS, brief=brief(),
                       profile=PROFILE, target_format="youtube_short")
    shot = m.shots[0]
    base = shot_params(m, shot, PROFILE)
    shot.overrides.update(seed=5, style_strength=0.4, resolution_scale=0.5, fps=8)
    p = shot_params(m, shot, PROFILE)
    assert p["SEED"] == 5 and p["DENOISE"] < base["DENOISE"]
    assert p["WIDTH"] < base["WIDTH"] and p["FRAME_COUNT"] == 16


def test_ollama_provider_retries_malformed_output_then_succeeds():
    good = brief().model_dump_json()
    calls = []
    provider = OllamaProvider("http://ollama", "m", transport=ollama_transport(
        ['{"style": 1}', good], calls))
    out = provider.generate("creative_director", "x", {}, CreativeBrief)
    assert isinstance(out, CreativeBrief) and len(calls) == 2
    assert calls[0]["format"]["title"] == "CreativeBrief"  # JSON schema constrained output
    assert "invalid" in calls[1]["messages"][-1]["content"]


def test_ollama_provider_gives_up_on_persistently_bad_json():
    provider = OllamaProvider("http://ollama", "m", max_retries=1,
                              transport=ollama_transport(["nope", "{}"]))
    with pytest.raises(AgentOutputError):
        provider.generate("r", "x", {}, CreativeBrief)


def test_llm_brief_cannot_move_shot_boundaries():
    llm = brief().model_copy(deep=True)
    llm.shot_plan[0].end = 3.3
    llm.shot_plan[0].intent = "hero walk"
    provider = OllamaProvider("http://o", "m", transport=ollama_transport([llm.model_dump_json()]))
    b, by = CreativeDirector(provider).run({"theme": "t"}, ANALYSIS, "youtube_short")
    assert by == "ollama" and b.shot_plan[0].end == 2 and b.shot_plan[0].intent == "hero walk"


def test_ollama_unload_uses_keep_alive_zero():
    calls = []
    provider = OllamaProvider("http://o", "m", transport=ollama_transport([], calls))
    assert provider.unload_all() == ["qwen2.5:7b-instruct"]
    assert calls[-1] == {"model": "qwen2.5:7b-instruct", "keep_alive": 0}


def test_collective_provider_refuses_instead_of_faking():
    with pytest.raises(AgentUnavailable, match="not implemented"):
        RokkurCollectiveProvider().generate("r", "x", {}, CreativeBrief)


def test_repair_planner_changes_only_failing_shots():
    report = {"shots": [{"shot_id": "a", "decision": "PASS"},
                        {"shot_id": "b", "decision": "FAIL", "issues": ["flicker"],
                         "recommendations": ["CHANGE_SEED", "REDUCE_STYLE_STRENGTH",
                                             "ADD_POSE_CONTROL"]}]}
    plan = RepairPlanner().plan(report, 1, {"b": {"seed": 10, "style_strength": 0.7}})
    assert [a.shot_id for a in plan.actions] == ["b"]
    ch = plan.actions[0].changes
    assert ch["seed"] != 10 and ch["style_strength"] == 0.55 and ch["pose"] is True


def test_fit_within_gives_sizes_video_models_accept():
    # 360x640 source into the PREVIEW box used to give 320x568, which Wan rejects.
    w, h = fit_within(360, 640, 320, 576)
    assert (w % 16, h % 16) == (0, 0) and h <= 576
