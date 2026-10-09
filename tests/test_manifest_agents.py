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
from rokkur_studio.manifest.builder import build_manifest, fit_within, negative_prompt, shot_params
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
    assert w <= 1024 and h <= 576 and w % 8 == 0 and h % 8 == 0


def test_fit_within_turns_the_box_and_caps_the_area_at_480p():
    # The ape clip is 1920x1080: the portrait box used to squeeze it to 576x320.
    assert fit_within(1920, 1080, 576, 1024) == (1024, 576)
    assert fit_within(1920, 1080, 576, 1024, max_pixels=480 * 832) == (832, 464)
    assert fit_within(1080, 1920, 576, 1024, max_pixels=480 * 832) == (464, 832)
    assert fit_within(320, 240, 576, 1024, max_pixels=480 * 832) == (320, 240)  # no upscaling


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


def test_creative_director_falls_back_to_rules_on_bad_model_output():
    provider = OllamaProvider("http://o", "m", max_retries=0,
                              transport=ollama_transport(["not json"]))
    b, by = CreativeDirector(provider).run({"theme": "t"}, ANALYSIS, "youtube_short")
    assert by == "rule_based" and len(b.shot_plan) == 2


def test_channel_manager_drafts_metadata_and_rule_based_skips():
    from rokkur_studio.agents.roles import ChannelManager

    assert ChannelManager(RuleBasedProvider()).run({}, {}, "youtube_short", 4) == (None, "rule_based")
    reply = json.dumps({"title": "Clay walk", "description": "A walk, in clay.",
                        "tags": ["Clay", "walk"]})
    calls = []
    provider = OllamaProvider("http://o", "m", transport=ollama_transport([reply], calls))
    draft, by = ChannelManager(provider).run({"theme": "clay"}, brief().model_dump(),
                                             "youtube_short", 4.0)
    assert by == "ollama" and draft is not None and draft.title == "Clay walk"
    assert calls[0]["think"] is False and calls[0]["format"]["title"] == "MetadataDraft"


def test_ollama_check_reports_missing_model():
    import httpx

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
        return httpx.Response(404)

    assert OllamaProvider("http://o", "qwen3.5:9b", transport=httpx.MockTransport(handle)
                          ).check() is None
    assert "ollama pull other" in (OllamaProvider("http://o", "other",
                                                  transport=httpx.MockTransport(handle)).check()
                                   or "")
    assert "not reachable" in (OllamaProvider("http://o", "x", transport=httpx.MockTransport(
        lambda r: httpx.Response(500))).check() or "")


def test_agent_check_runs_both_roles(capsys):
    import httpx

    from rokkur_studio.cli import run_agent_check
    from rokkur_studio.config import load_settings

    framing = json.dumps({"shot_size": "Full Shot", "camera_angle": "Eye-level",
                          "camera_movement": "Static", "lighting": "Golden hour diffusion"})
    replies = [brief().model_dump_json(), framing, framing,
               json.dumps({"title": "Test", "description": "D", "tags": ["a"]})]

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": replies.pop(0)}})
        return httpx.Response(404)

    settings = load_settings(None)
    settings.ollama.model = "qwen3.5:9b"
    assert run_agent_check(settings, theme="t", transport=httpx.MockTransport(handle)) == 0
    out = capsys.readouterr().out
    assert "[ok] creative_director" in out and "[ok] channel_manager" in out
    assert "[ok] director_of_photography" in out and "Full Shot · Eye-level" in out
    assert "prompt: Cinematic film still, (full shot:1.3)" in out and "schedule: " in out


def test_free_idle_comfyui_only_when_queue_empty(settings):
    from rokkur_studio.pipeline import context

    calls = []

    class Fake:
        def __init__(self, url):
            pass

        def queue(self):
            return {"queue_running": calls[0] if calls else [], "queue_pending": []}

        def free(self):
            calls.append("free")

        def close(self):
            pass

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(context, "ComfyClient", Fake)
    try:
        hook = context.free_idle_comfyui(settings)
        hook()
        assert calls == ["free"]
        calls[0] = [["busy"]]  # now something is "running"
        hook()
        assert calls == [[["busy"]]]
    finally:
        mp.undo()


def test_before_generate_failure_does_not_block_the_agent():
    good = brief().model_dump_json()

    def boom():
        raise RuntimeError("comfy down")

    provider = OllamaProvider("http://o", "m", transport=ollama_transport([good]),
                              before_generate=boom)
    assert isinstance(provider.generate("r", "x", {}, CreativeBrief), CreativeBrief)


def test_ollama_format_has_no_length_caps_but_output_is_still_validated():
    from rokkur_studio.agents.providers import grammar_schema
    from rokkur_studio.agents.schemas import MetadataDraft

    sent = json.dumps(grammar_schema(MetadataDraft.model_json_schema()))
    assert "maxLength" not in sent and "minLength" not in sent and "maxItems" not in sent
    too_long = json.dumps({"title": "t" * 150, "description": "d", "tags": []})
    ok = json.dumps({"title": "Fine", "description": "d", "tags": []})
    calls = []
    provider = OllamaProvider("http://o", "m", transport=ollama_transport([too_long, ok], calls))
    assert provider.generate("r", "x", {}, MetadataDraft).title == "Fine" and len(calls) == 2


def test_ollama_http_error_includes_body():
    import httpx

    def handle(request):
        return httpx.Response(400, json={"error": "bad grammar"})

    provider = OllamaProvider("http://o", "m", transport=httpx.MockTransport(handle))
    with pytest.raises(AgentUnavailable, match="bad grammar"):
        provider.generate("r", "x", {}, CreativeBrief)


def test_negative_prompt_starts_from_wans_own_and_fits_the_look():
    base = "过曝，静态，风格，作品，画作，画面，最差质量"
    m = build_manifest(project_id="p", source_asset="s", analysis=ANALYSIS, brief=brief(),
                       profile=PROFILE, target_format="youtube_short")
    m.style.negative_prompt = "watermark"
    assert negative_prompt("", m) == "watermark"  # profiles without a base are unchanged
    m.style.theme, m.style.prompt = "luxury spa bathroom", "marble, warm light"
    photo = negative_prompt(base, m)
    assert photo.startswith(base) and "3d render, cgi" in photo and photo.endswith("watermark")
    m.style.theme = "1970s claymation"
    stylized = negative_prompt(base, m)
    # "style, artwork, painting, picture" would fight a stylized look; CGI may be the look.
    assert stylized == "过曝，静态，最差质量, watermark"


def test_quality_profile_renders_landscape_at_480p(settings):
    profile = settings.profile("RTX3070_QUALITY")
    analysis = {**ANALYSIS, "width": 1920, "height": 1080}
    m = build_manifest(project_id="p", source_asset="s", analysis=analysis, brief=brief(),
                       profile=profile, target_format="youtube_video")
    p = shot_params(m, m.shots[0], profile)
    assert (p["WIDTH"], p["HEIGHT"]) == (832, 464)
    assert p["NEGATIVE_PROMPT"].startswith("过曝，静态")
