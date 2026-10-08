"""Director passes: vocabulary, diffusion parsing rules, asset tracker, DP pass, schedule."""

from __future__ import annotations

import base64
import json
import re

import httpx
import pytest
from pydantic import ValidationError

from rokkur_studio.agents.providers import OllamaProvider, RuleBasedProvider
from rokkur_studio.agents.roles import CreativeDirector, DirectorOfPhotography
from rokkur_studio.agents.schemas import CreativeBrief, ShotFraming
from rokkur_studio.config import DirectorSection
from rokkur_studio.director.assets import (
    AssetTracker,
    TrackerError,
    load_tracker,
    resolve_character,
    save_tracker,
    tracker_path,
)
from rokkur_studio.director.passes import direct
from rokkur_studio.director.prompts import (
    Weights,
    build_schedule,
    compile_brief,
    merge_negatives,
    preview,
)
from rokkur_studio.director.rules import clean_phrase, strip_actions, weighted
from rokkur_studio.director.vocabulary import PHRASES, VOCABULARY, lighting_for
from tests.fakes import ollama_transport
from tests.test_manifest_agents import ANALYSIS

FRAMING = {"shot_size": "Close-up", "camera_angle": "Low-angle",
           "camera_movement": "Slow push-in", "lighting": "Moody neon rim lighting"}


def fizz_parse(text: str) -> dict[str, str]:
    """FizzNodes' own parser (ScheduleFuncs.process_input_text)."""
    t = "{" + text.replace("\n", "") + "}"
    return json.loads(re.sub(r",\s*}", "}", t).strip())


def rule_brief(**ci) -> CreativeBrief:
    b, _ = CreativeDirector(RuleBasedProvider()).run({"theme": "retro clay sci-fi", **ci},
                                                     ANALYSIS, "youtube_short")
    return b


# -- vocabulary -----------------------------------------------------------------------------
def test_every_allowed_term_has_a_prompt_phrase_and_others_are_rejected():
    for terms in VOCABULARY.values():
        assert all(t in PHRASES for t in terms)
    assert "Rembrandt lighting" in VOCABULARY["Lighting styles"]
    ShotFraming(**FRAMING)
    with pytest.raises(ValidationError):
        ShotFraming(**{**FRAMING, "shot_size": "Super wide hero shot"})
    schema = ShotFraming.model_json_schema()
    assert schema["properties"]["camera_angle"]["enum"] == list(VOCABULARY["Camera angles"])


def test_rule_based_lighting_follows_the_theme():
    assert lighting_for("cyberpunk city at night") == "Cyberpunk bi-color hue"
    assert lighting_for("film noir detective") == "Chiaroscuro high-contrast"
    assert lighting_for("misty forest") == "Volumetric god rays"
    assert lighting_for("claymation") == "Golden hour diffusion"  # default
    assert lighting_for("swarm of bees") == "Golden hour diffusion"  # "warm" is a whole word


# -- rules ----------------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("is walking", "in mid-stride"),
    ("suddenly runs toward the door", "in full sprint toward the door"),
    ("thinking about the past", "focused expression"),
    ("is walking, suddenly runs, thinking about the past",
     "in mid-stride, in full sprint, focused expression"),
    ("He is walking through the rain while thinking about home",
     "in mid-stride through the rain, focused expression"),
    ("she turns and runs", "head turned sharply, in full sprint"),
    ("starts to run across the bridge", "in full sprint across the bridge"),
    ("looking at the camera", "gaze fixed on the camera"),
    ("reaching for the sword", "arm outstretched toward the sword"),
    ("feeling sad because she lost", "sad expression"),
    ("a man is waiting for the bus", "a man waiting for the bus"),
    # adjectives and nouns that only look like verbs stay
    ("running water, waving flag, a falling star", "running water, waving flag, a falling star"),
    ("the waves crash on rocks", "the waves crash on rocks"),
])
def test_action_strip_out(text, expected):
    assert strip_actions(text) == expected


def test_clean_phrase_removes_weights_brackets_and_sentence_breaks():
    assert clean_phrase('Hero stands. Rain falls! (red scarf:1.2) "neon" [x] --neg ugly') == \
        "Hero stands, Rain falls, red scarf neon x ugly"
    assert clean_phrase("shot on f/1.8 at 16:9") == "shot on f/1.8 at 16:9"
    assert weighted("close-up shot", 1.3) == "(close-up shot:1.3)"
    assert weighted("close-up shot", 1.0) == "close-up shot"


def test_merge_negatives_drops_terms_the_theme_asks_for():
    neg, dropped = merge_negatives(["blurry, cartoon, illustration", "Blurry, text"],
                                   wanted="cartoon noir")
    assert neg == "blurry, illustration, text" and dropped == ["cartoon"]


# -- asset tracker --------------------------------------------------------------------------
def test_tracker_defaults_round_trip_and_validation(tmp_path):
    t = AssetTracker()
    data = t.to_json()
    assert set(data) == {"PROMPT_PREFIX", "GLOBAL_STYLE_MODIFIERS", "GLOBAL_NEGATIVE_PROMPT",
                         "CHARACTERS"}
    assert data["CHARACTERS"]["NEO"].startswith("a 20s athletic male")
    path = tracker_path(tmp_path)
    assert load_tracker(path) == t  # nothing saved yet: defaults
    save_tracker(path, AssetTracker.model_validate({**data, "CHARACTERS": {"old man": "grey"}}))
    assert load_tracker(path).characters == {"OLD_MAN": "grey"}
    assert load_tracker(path).character("Old Man") == "grey"
    with pytest.raises(ValidationError):
        AssetTracker.model_validate({"CHARACTERS": {"???": "x"}})
    path.write_text("{not json")
    with pytest.raises(TrackerError):
        load_tracker(path)


def test_resolve_character():
    t = AssetTracker()
    assert resolve_character({"character_key": "neo"}, t)[:2] == ("NEO", t.characters["NEO"])
    key, anchor, warning = resolve_character({"character_key": "ZED",
                                              "character_description": "a robot"}, t)
    assert (key, anchor) == (None, "a robot") and "ZED" in (warning or "")
    assert resolve_character({}, t) == (None, None, None)


# -- prompt compiler ------------------------------------------------------------------------
def framed_brief(**ci) -> CreativeBrief:
    b = rule_brief(**ci)
    b.shot_plan[0].subject = "is walking through the rain, thinking about home"
    b.shot_plan[0].background = "neon alley. wet asphalt"
    out, _ = DirectorOfPhotography(RuleBasedProvider()).run(b)
    out.shot_plan[0] = out.shot_plan[0].model_copy(update=FRAMING)
    return out


def test_compiled_prompt_follows_the_rule_of_nouns():
    t = AssetTracker()
    b = compile_brief(framed_brief(), creative_input={"theme": "retro clay sci-fi"}, tracker=t,
                      anchor=t.characters["NEO"], character_key="NEO", weights=Weights())
    p = b.shot_plan[0].prompt
    assert p.startswith("Cinematic film still, (close-up shot:1.3), (low-angle shot:1.25), "
                        "slow push-in, a 20s athletic male, wearing a tattered black hooded "
                        "jacket, dark denim jeans, intense pale features, in mid-stride through "
                        "the rain, focused expression, moody neon rim lighting, neon alley, "
                        "wet asphalt, retro clay sci-fi")
    assert p.endswith("depth of field")
    assert "walking" not in p and "thinking" not in p
    # second shot keeps the same anchor (character consistency) with its own framing
    assert t.characters["NEO"] in b.shot_plan[1].prompt
    assert b.shot_plan[1].prompt.startswith("Cinematic film still, (medium shot:1.3)")
    assert "distorted hands" in b.negative_prompt
    assert b.director is not None and b.director.character_key == "NEO"


def test_global_look_can_be_switched_off_and_weights_disabled():
    b = compile_brief(framed_brief(), creative_input={"use_global_look": False},
                      tracker=AssetTracker(), anchor=None, character_key=None,
                      weights=Weights(framing=1.0, angle=1.0))
    p = b.shot_plan[0].prompt
    assert p.startswith("close-up shot, low-angle shot, slow push-in, in mid-stride")
    assert "35mm" not in p and "distorted hands" not in b.negative_prompt


def test_preview_runs_the_same_rules():
    out = preview(AssetTracker(), theme="cartoon noir", subject="he suddenly jumps",
                  shot_size="Full Shot", character_key="NEO")
    assert out["subject_state"] == "airborne mid-jump"
    assert out["prompt"].startswith("Cinematic film still, (full shot:1.3), a 20s athletic")
    assert out["dropped_negatives"] == ["cartoon"]


# -- schedule -------------------------------------------------------------------------------
def test_schedule_is_valid_fizznodes_text_with_hard_cuts():
    b = compile_brief(framed_brief(), creative_input={}, tracker=AssetTracker(), anchor=None,
                      character_key=None, weights=Weights())
    b.shot_plan[1].start = b.shot_plan[0].end = 2.3
    sched = build_schedule(b.shot_plan, b.negative_prompt, duration=4.0)
    parsed = fizz_parse(sched.text)
    assert list(parsed) == ["0", "24", "48", "54", "55", "72"]  # 24-frame grid + both cut sides
    assert sched.max_frames == 96
    pos, neg = parsed["54"].split(" --neg ")
    assert pos == b.shot_plan[0].prompt and neg == b.negative_prompt
    assert parsed["55"].startswith(b.shot_plan[1].prompt)
    assert [k["shot_id"] for k in sched.keyframes][-2:] == ["shot_002", "shot_002"]
    plain = build_schedule(b.shot_plan, "x", duration=4.0, fps=16, interval=16,
                           inline_negative=False)
    assert "--neg" not in plain.text and list(fizz_parse(plain.text))[:3] == ["0", "16", "32"]


# -- model passes ---------------------------------------------------------------------------
def test_dp_pass_uses_the_model_and_falls_back_per_shot():
    calls: list[dict] = []
    provider = OllamaProvider("http://o", "m", max_retries=0, transport=ollama_transport(
        [json.dumps(FRAMING), json.dumps({**FRAMING, "lighting": "Disco strobe"})], calls))
    out, by = DirectorOfPhotography(provider).run(rule_brief())
    assert by == "mixed"
    assert out.shot_plan[0].shot_size == "Close-up" and out.shot_plan[0].framing_by == "ollama"
    assert out.shot_plan[1].framing_by == "rule_based"  # invented lighting term was rejected
    assert calls[0]["format"]["title"] == "ShotFraming"
    assert "images" not in calls[0]["messages"][1]


def test_dp_pass_sends_keyframes_to_a_vision_model():
    calls: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision"]})
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps({**FRAMING,
            "observed_subject": "a figure in mid-stride", "observed_background": "forest"})}})

    provider = OllamaProvider("http://o", "m", transport=httpx.MockTransport(handle))
    assert provider.supports_images()
    brief, _ = direct(provider, {"theme": "t", "character_key": "NEO"}, ANALYSIS,
                      "youtube_short", tracker=AssetTracker(), settings=DirectorSection(),
                      keyframes={"shot_001": b"jpeg-1", "shot_002": b"jpeg-2"})
    # story pass fell back (it got framing JSON), the DP saw one frame per shot
    sent = [c["messages"][1].get("images") for c in calls
            if c["format"]["title"] == "ObservedShotFraming"]
    assert sent == [[base64.b64encode(b"jpeg-1").decode()], [base64.b64encode(b"jpeg-2").decode()]]
    assert brief.director is not None and brief.director.vision
    assert all(s.framing_by == "ollama+vision" for s in brief.shot_plan)
    assert brief.shot_plan[0].prompt.startswith("Cinematic film still, (close-up shot:1.3)")


def test_dp_stops_asking_a_model_that_is_down():
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    provider = OllamaProvider("http://o", "m", transport=httpx.MockTransport(handle))
    out, by = DirectorOfPhotography(provider).run(rule_brief())
    assert by == "rule_based" and calls == 1
    assert out.shot_plan[0].camera_movement == "Tracking pan"  # moderate source motion


def test_story_pass_keeps_subjects_and_the_full_loop_runs():
    story = rule_brief().model_dump()
    story["shot_plan"][0].update(subject="is walking, suddenly turns", background="rooftop")
    replies = [json.dumps(story), json.dumps(FRAMING), json.dumps(FRAMING)]
    calls: list[dict] = []
    provider = OllamaProvider("http://o", "m", transport=ollama_transport(replies, calls))
    brief, by = direct(provider, {"theme": "retro clay sci-fi", "character_key": "NEO"},
                       ANALYSIS, "youtube_short", tracker=AssetTracker(),
                       settings=DirectorSection())
    assert by == "ollama" and brief.director is not None
    assert brief.director.story_by == "ollama" and brief.director.framing_by == "ollama"
    assert calls[0]["format"]["title"] == "StoryBrief"
    assert "character_anchor" in calls[0]["messages"][1]["content"]
    assert "do not restate their clothing" in calls[0]["messages"][0]["content"]
    p = brief.shot_plan[0].prompt
    assert "in mid-stride, head turned sharply, moody neon rim lighting, rooftop" in p


def test_director_can_be_switched_off():
    brief, _ = direct(RuleBasedProvider(), {"theme": "t"}, ANALYSIS, "youtube_short",
                      tracker=AssetTracker(), settings=DirectorSection(enabled=False))
    assert all(not s.prompt and s.shot_size is None for s in brief.shot_plan)
