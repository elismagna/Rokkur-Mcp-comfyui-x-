"""More render options end to end (API, New video form, CLI, shot parameters, workflows) and the
cloud-only Wan VACE 14B profile."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from rokkur_studio.api.schemas import EDGE_PRESETS, SAMPLERS, SCHEDULERS, CreativeIn
from rokkur_studio.cli import _profile_fits, build_parser, render_creative
from rokkur_studio.comfyui.client import ComfyClient
from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.db.models import Render
from rokkur_studio.manifest.builder import build_manifest, shot_params
from rokkur_studio.services import ratings
from rokkur_studio.services.projects import get_project
from tests.fakes import FakeComfyUI
from tests.test_dashboard import client_for
from tests.test_manifest_agents import ANALYSIS, brief
from tests.test_pipeline import create, run

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"
REGISTRY = TemplateRegistry(WF)
WAN = ["v2v_3070_quality", "v2v_3070_keep", "v2v_3070_depth", "v2v_3070_depth_keep",
       "v2v_cloud_14b", "v2v_cloud_14b_keep"]
DRAFTS = ["v2v_3070_draft", "v2v_3070_draft_keep"]
BASE = {"STYLE_PROMPT": "clay", "INPUT_VIDEO": "clip.mp4", "MASK_VIDEO": "mask.mp4",
        "WIDTH": 480, "HEIGHT": 832, "FRAME_COUNT": 81, "FPS": 16.0, "STEPS": 20, "SEED": 7}


# -- API schema ---------------------------------------------------------------------------
def test_creative_defaults_keep_the_workflow_and_turn_tuning_on():
    c = CreativeIn(theme="x")
    assert (c.shift, c.sampler, c.scheduler, c.resolution_scale) == (None, None, None, None)
    assert (c.stabilize, c.smooth_control, c.min_stability) == ("auto", 0.0, 0.0)
    assert c.auto_tune is True and c.picture_review is True
    # Stored as model_dump(exclude_none=True): unset sampling options never reach the overrides.
    stored = c.model_dump(exclude_none=True)
    assert not {"shift", "sampler", "scheduler", "resolution_scale"} & set(stored)
    assert stored["stabilize"] == "auto" and stored["auto_tune"] is True


@pytest.mark.parametrize("field,value", [
    ("shift", 1), ("shift", 20), ("sampler", "res_multistep"), ("scheduler", "sgm_uniform"),
    ("stabilize", "strong"), ("smooth_control", 1), ("resolution_scale", 0.5),
    ("resolution_scale", 1.0), ("min_stability", 10), ("auto_tune", False),
])
def test_creative_accepts_options_within_bounds(field, value):
    assert getattr(CreativeIn(theme="x", **{field: value}), field) == value


@pytest.mark.parametrize("field,value", [
    ("shift", 0.5), ("shift", 21), ("sampler", "lcm"), ("sampler", "karras"),
    ("scheduler", "karras"), ("stabilize", "max"), ("smooth_control", -0.1),
    ("smooth_control", 1.5), ("resolution_scale", 0.4), ("resolution_scale", 1.1),
    ("min_stability", -1), ("min_stability", 10.5),
])
def test_creative_rejects_options_out_of_bounds(field, value):
    with pytest.raises(ValidationError):
        CreativeIn(theme="x", **{field: value})


def test_offered_samplers_and_schedulers_are_comfyui_names():
    # ComfyUI core's KSampler lists (comfy/samplers.py KSAMPLER_NAMES / SCHEDULER_NAMES).
    assert set(SAMPLERS) <= {"euler", "uni_pc", "uni_pc_bh2", "dpmpp_2m", "res_multistep"}
    assert set(SCHEDULERS) <= {"simple", "beta", "normal", "sgm_uniform"}
    assert EDGE_PRESETS == {"default": None, "calm": (0.4, 0.8), "tight": (0.1, 0.3)}


# -- New video form -----------------------------------------------------------------------
def _created(ctx, response) -> dict:
    assert response.status_code == 303, response.headers.get("location")
    pid = response.headers["location"].split("/")[-1].split("?")[0]
    with ctx.db.session() as s:
        return dict(get_project(s, pid).creative_input)


def test_form_carries_the_new_options_into_the_project(ctx, sample_video):
    c = client_for(ctx)
    page = c.get("/ui/new").text
    for name in ("shift", "sampler", "scheduler", "edge_detail", "resolution_scale", "stabilize",
                 "smooth_control", "auto_tune", "min_stability", "picture_review"):
        assert f'name="{name}"' in page, name
    edge_values = re.findall(r'<option value="([^"]*)"', page.split('id="edge_detail"')[1]
                             .split("</select>")[0])
    assert edge_values and set(edge_values) <= set(EDGE_PRESETS)
    assert "Stability and tuning" in page and "res_multistep" in page and "sgm_uniform" in page
    creative = _created(ctx, c.post("/ui/projects", data={
        "theme": "clay", "rights_category": "USER_OWNED", "local_path": str(sample_video),
        "shift": "5", "sampler": "euler", "scheduler": "beta", "edge_detail": "calm",
        "resolution_scale": "0.67", "stabilize": "strong", "smooth_control": "0.3",
        "min_stability": "6.5", "picture_review": "true"}, follow_redirects=False))
    assert (creative["shift"], creative["sampler"], creative["scheduler"]) == (5.0, "euler", "beta")
    assert (creative["canny_low"], creative["canny_high"]) == (0.4, 0.8)
    assert creative["resolution_scale"] == 0.67 and creative["stabilize"] == "strong"
    assert creative["smooth_control"] == 0.3 and creative["min_stability"] == 6.5
    # A cleared checkbox sends nothing, so auto_tune is off here and picture review on.
    assert creative["auto_tune"] is False and creative["picture_review"] is True


def test_form_defaults_leave_the_workflow_settings_alone(ctx, sample_video):
    c = client_for(ctx)
    creative = _created(ctx, c.post("/ui/projects", data={
        "theme": "clay", "rights_category": "USER_OWNED", "local_path": str(sample_video),
        "edge_detail": "default", "stabilize": "auto", "smooth_control": "0",
        "min_stability": "0", "auto_tune": "true", "picture_review": "true"},
        follow_redirects=False))
    assert not {"shift", "sampler", "scheduler", "canny_low", "canny_high",
                "resolution_scale"} & set(creative)
    assert creative["stabilize"] == "auto" and creative["smooth_control"] == 0.0
    assert creative["auto_tune"] is True and creative["picture_review"] is True
    tight = _created(ctx, c.post("/ui/projects", data={
        "theme": "clay", "rights_category": "USER_OWNED", "local_path": str(sample_video),
        "edge_detail": "tight"}, follow_redirects=False))
    assert (tight["canny_low"], tight["canny_high"]) == (0.1, 0.3)


@pytest.mark.parametrize("field,value", [("edge_detail", "wild"), ("sampler", "lcm"),
                                         ("shift", "50"), ("resolution_scale", "2"),
                                         ("stabilize", "max"), ("smooth_control", "3")])
def test_form_refuses_invalid_options(ctx, sample_video, field, value):
    r = client_for(ctx).post("/ui/projects", data={
        "theme": "clay", "rights_category": "USER_OWNED", "local_path": str(sample_video),
        field: value}, follow_redirects=False)
    assert r.status_code == 303 and "err=" in r.headers["location"]


# -- ratings remember the settings --------------------------------------------------------
def test_a_rating_remembers_the_sampling_and_stability_settings(ctx, sample_video):
    pid = create(ctx, sample_video)
    run(ctx)
    with ctx.db.transaction() as s:
        render = s.scalars(select(Render).where(Render.project_id == pid,
                                                Render.shot_id == "shot_001")).one()
        # As the ComfyUI renderer records them: SHIFT set by the video, the rest the
        # workflow's own values reported as applied.
        render.params = {**render.params, "SHIFT": 5.0, "_STABILIZE": "light",
                         "_SMOOTH_CONTROL": 0.3, "_details": {
                             **render.params.get("_details", {}),
                             "applied": {"SHIFT": 5.0, "SAMPLER": "uni_pc",
                                         "SCHEDULER": "simple"}}}
        rating = ratings.rate(s, get_project(s, pid),
                              ratings.RatingIn(target="shot_001", value=1), actor="test")
        assert rating is not None
        snap = rating.snapshot
    assert (snap["shift"], snap["sampler"], snap["scheduler"]) == (5.0, "uni_pc", "simple")
    assert snap["stabilize"] == "light" and snap["smooth_control"] == 0.3


# -- CLI ----------------------------------------------------------------------------------
RENDER = ["render", "clip.mp4", "--theme", "clay", "--rights", "USER_OWNED", "--evidence", "mine"]


def test_cli_render_flags_reach_the_creative_input():
    args = build_parser().parse_args(RENDER + [
        "--shift", "5", "--sampler", "dpmpp_2m", "--scheduler", "beta", "--stabilize", "light",
        "--smooth-control", "0.6", "--scale", "0.75", "--no-auto-tune", "--min-stability", "7",
        "--no-picture-review", "--canny", "0.4", "0.8"])
    c = render_creative(args)
    assert (c.shift, c.sampler, c.scheduler) == (5.0, "dpmpp_2m", "beta")
    assert (c.stabilize, c.smooth_control, c.resolution_scale) == ("light", 0.6, 0.75)
    assert c.auto_tune is False and c.min_stability == 7 and c.picture_review is False
    assert (c.canny_low, c.canny_high) == (0.4, 0.8)


def test_cli_render_defaults_match_the_api():
    c = render_creative(build_parser().parse_args(RENDER))
    assert c == CreativeIn(theme="clay", subject="auto", reference_mode="auto")


def test_cli_rejects_unknown_names_and_out_of_range_values(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(RENDER + ["--sampler", "lcm"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(RENDER + ["--stabilize", "max"])
    with pytest.raises(ValidationError):
        render_creative(build_parser().parse_args(RENDER + ["--scale", "2"]))


# -- shot parameters ----------------------------------------------------------------------
def _manifest(settings, profile_name: str = "RTX3070_QUALITY", analysis: dict | None = None):
    profile = settings.profile(profile_name)
    manifest = build_manifest(project_id="test", source_asset="source",
                              analysis=analysis or ANALYSIS, brief=brief(), profile=profile,
                              target_format="youtube_short")
    return manifest, profile


def _workflow_values(params: dict) -> dict:
    """What the ComfyUI renderer hands the compiler: never the underscore keys."""
    return {**BASE, **{k: v for k, v in params.items() if not k.startswith("_")}}


def test_shot_params_send_sampling_options_only_when_set(settings):
    manifest, profile = _manifest(settings)
    shot = manifest.shots[0]
    params = shot_params(manifest, shot, profile)
    assert not {"SHIFT", "SAMPLER", "SCHEDULER"} & set(params)
    assert params["_STABILIZE"] == "auto" and params["_SMOOTH_CONTROL"] == 0.0
    template = REGISTRY.get("v2v_3070_quality")
    wf = compile_workflow(template, _workflow_values(params)).workflow
    assert wf["4"]["inputs"]["shift"] == 8.0
    assert (wf["15"]["inputs"]["sampler_name"], wf["15"]["inputs"]["scheduler"]) == ("uni_pc",
                                                                                     "simple")
    shot.overrides.update(shift=5, sampler="euler", scheduler="beta", stabilize="strong",
                          smooth_control=0.3, cfg=5.0, steps=24)
    params = shot_params(manifest, shot, profile)
    assert (params["SHIFT"], params["SAMPLER"], params["SCHEDULER"]) == (5.0, "euler", "beta")
    assert params["_STABILIZE"] == "strong" and params["_SMOOTH_CONTROL"] == 0.3
    compiled = compile_workflow(template, _workflow_values(params))
    wf = compiled.workflow
    assert wf["4"]["inputs"]["shift"] == 5.0
    assert (wf["15"]["inputs"]["sampler_name"], wf["15"]["inputs"]["scheduler"]) == ("euler",
                                                                                     "beta")
    assert (wf["15"]["inputs"]["cfg"], wf["15"]["inputs"]["steps"]) == (5.0, 24)
    assert not any(k.startswith("_") for k in compiled.ignored)


def test_draft_workflow_takes_the_shift_and_reports_the_sampler_ignored(settings):
    manifest, profile = _manifest(settings, "RTX3070_DRAFT")
    shot = manifest.shots[0]
    shot.overrides.update(shift=6, sampler="euler", scheduler="beta")
    compiled = compile_workflow(REGISTRY.get("v2v_3070_draft"),
                                _workflow_values(shot_params(manifest, shot, profile)))
    assert compiled.workflow["4"]["inputs"]["shift"] == 6.0
    assert (compiled.workflow["15"]["inputs"]["sampler_name"],
            compiled.workflow["15"]["inputs"]["scheduler"]) == ("lcm", "simple")
    assert {"SAMPLER", "SCHEDULER"} <= set(compiled.ignored)


# -- workflow templates -------------------------------------------------------------------
@pytest.mark.parametrize("name", REGISTRY.names())
def test_every_template_loads_and_compiles_with_the_new_params(name):
    template = REGISTRY.get(name)
    raw = template.workflow
    plain = compile_workflow(template, BASE).workflow
    sampler_nodes = [k for k, n in raw.items() if n["class_type"] == "KSampler"]
    for node in sampler_nodes:  # defaults are the graph's own: existing renders are unchanged
        for key in ("sampler_name", "scheduler"):
            assert plain[node]["inputs"][key] == raw[node]["inputs"][key]
    compiled = compile_workflow(template, {**BASE, "SHIFT": 4.0, "SAMPLER": "euler",
                                           "SCHEDULER": "beta"})
    params = template.spec.parameters
    if name in WAN + DRAFTS:
        assert params["SHIFT"].targets[0].node == "4"
        assert raw["4"]["class_type"] == "ModelSamplingSD3"
        assert plain["4"]["inputs"]["shift"] == raw["4"]["inputs"]["shift"] == 8.0
        assert compiled.workflow["4"]["inputs"]["shift"] == 4.0
    if name in WAN:
        assert {params["SAMPLER"].targets[0].node, params["SCHEDULER"].targets[0].node} == {"15"}
        assert raw["15"]["class_type"] == "KSampler"
        assert compiled.workflow["15"]["inputs"]["sampler_name"] == "euler"
        assert compiled.workflow["15"]["inputs"]["scheduler"] == "beta"
    else:
        assert {"SAMPLER", "SCHEDULER"} <= set(compiled.ignored)
    for p in params.values():  # every target exists (load_template checks), every bound holds
        if p.default is not None and p.type in ("int", "float"):
            assert (p.min is None or p.default >= p.min) and (p.max is None or p.default <= p.max)


@pytest.mark.parametrize(("big", "base"), [("v2v_cloud_14b", "v2v_3070_quality"),
                                           ("v2v_cloud_14b_keep", "v2v_3070_keep")])
def test_14b_workflows_are_the_proven_graphs_with_the_14b_model(big, base):
    b14, b13 = REGISTRY.get(big), REGISTRY.get(base)
    assert {k: v for k, v in b14.workflow.items() if k != "1"} == \
        {k: v for k, v in b13.workflow.items() if k != "1"}
    assert b14.workflow["1"] == {"class_type": "UNETLoader", "inputs": {
        "unet_name": "wan2.1_vace_14B_fp16.safetensors", "weight_dtype": "fp8_e4m3fn"}}
    assert set(b14.spec.parameters) == set(b13.spec.parameters) | {"WEIGHT_DTYPE"}
    # No CausVid (CC-BY-NC) and no LoRA at all: the template's own non-LoRA sampler settings.
    text = (WF / big / "workflow.json").read_text() + (WF / big / "params.yaml").read_text()
    assert "LoraLoader" not in text and "Wan21_CausVid" not in text
    ks = b14.workflow["15"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (20, 6.0, "uni_pc",
                                                                             "simple")
    assert b14.workflow["4"]["inputs"]["shift"] == 8.0 and b14.workflow["18"]["inputs"]["fps"] == 16.0


# -- CLOUD_14B profile --------------------------------------------------------------------
def test_cloud_14b_profile_is_cloud_only(settings):
    profile = settings.profile("CLOUD_14B")
    assert (profile.workflow, profile.keep_workflow) == ("v2v_cloud_14b", "v2v_cloud_14b_keep")
    assert (profile.fps, profile.max_frames, profile.frame_multiple, profile.steps) == (16, 81, 4, 20)
    assert profile.max_pixels == 1280 * 720 and profile.min_vram_gb == 24
    assert profile.location == "local" and profile.negative_base.startswith("过曝")
    assert not any(step.startswith("switch_profile") for step in profile.degrade)
    settings.render.renderer = "comfyui"
    assert "Needs 24 GB" in settings.profile_problem("CLOUD_14B")
    settings.cloud.enabled, settings.cloud.url = True, "https://gpu.example"
    assert settings.profile_problem("CLOUD_14B", target="cloud") is None
    # comfy-check: the PC's ComfyUI need not have the 14B model; the cloud server must.
    assert not _profile_fits(settings, profile, cloud=False)
    assert _profile_fits(settings, profile, cloud=True)
    assert _profile_fits(settings, settings.profile("RTX3070_QUALITY"), cloud=False)
    assert not _profile_fits(settings, settings.profile("HYBRID_MAX"), cloud=False)


def test_new_page_lists_cloud_14b_as_cloud_only(ctx):
    ctx.settings.render.renderer = "comfyui"
    down = FakeComfyUI()
    down.down = True  # this PC's ComfyUI is off: only the static checks apply
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=down.transport())
    ctx.settings.cloud.enabled, ctx.settings.cloud.url = True, "http://127.0.0.1:9"
    page = client_for(ctx).get("/ui/new").text
    assert re.search(r'value="CLOUD_14B"[^>]*>CLOUD 14B · cloud only<', page)


@pytest.mark.parametrize(("size", "expected"), [((1920, 1080), (1280, 720)),
                                                ((1080, 1920), (720, 1280)),
                                                ((3840, 2160), (1280, 720))])
def test_cloud_14b_compiles_at_720p(settings, size, expected):
    analysis = {**ANALYSIS, "width": size[0], "height": size[1], "fps": 30.0}
    manifest, profile = _manifest(settings, "CLOUD_14B", analysis)
    params = shot_params(manifest, manifest.shots[0], profile)
    assert (params["WIDTH"], params["HEIGHT"]) == expected
    assert params["STEPS"] == 20 and params["CFG"] == 6.0 and params["FPS"] == 16.0
    assert (params["FRAME_COUNT"] - 1) % 4 == 0 and params["FRAME_COUNT"] <= 81
    for name in ("v2v_cloud_14b", "v2v_cloud_14b_keep"):
        compiled = compile_workflow(REGISTRY.get(name), {
            **{k: v for k, v in params.items() if not k.startswith("_")},
            "INPUT_VIDEO": "clip.mp4", "MASK_VIDEO": "mask.mp4"})
        wf = compiled.workflow
        assert (wf["14"]["inputs"]["width"], wf["14"]["inputs"]["height"]) == expected
        assert (wf["12"]["inputs"]["width"], wf["12"]["inputs"]["height"]) == expected
        assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"
        assert compiled.applied["DIFFUSION_MODEL"] == "wan2.1_vace_14B_fp16.safetensors"
    # A smaller render size (the form's 67%) brings it to about 480P.
    manifest.shots[0].overrides["resolution_scale"] = 0.67
    smaller = shot_params(manifest, manifest.shots[0], profile)
    assert smaller["WIDTH"] * smaller["HEIGHT"] <= 480 * 864


def test_14b_params_header_records_the_verified_model_size():
    header = (WF / "v2v_cloud_14b" / "params.yaml").read_text()
    assert "34,675,323,640 bytes" in header and "wan2.1_vace_14B_fp16.safetensors" in header
    assert json.loads((WF / "v2v_cloud_14b" / "workflow.json").read_text())["1"]["inputs"][
        "unet_name"] == "wan2.1_vace_14B_fp16.safetensors"
