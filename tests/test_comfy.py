import json
from pathlib import Path

import pytest

from rokkur_studio.comfyui.client import (
    ComfyClient,
    ComfyExecutionError,
    ComfyUnavailable,
    ComfyValidationError,
    parse_history_entry,
)
from rokkur_studio.comfyui.compiler import (
    TemplateError,
    TemplateRegistry,
    compile_workflow,
    load_template,
    validate_against_object_info,
)
from tests.fakes import FakeComfyUI

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"


def client(fake):
    return ComfyClient("http://comfy:8188", transport=fake.transport())


def test_submit_wait_download_roundtrip(tmp_path):
    fake = FakeComfyUI()
    c = client(fake)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video-bytes")
    name = c.upload_input(src)
    assert fake.uploads[name] == b"video-bytes"
    wf = compile_workflow(TemplateRegistry(WF).get("v2v_preview"),
                          {"STYLE_PROMPT": "clay", "INPUT_VIDEO": name}).workflow
    pid = c.submit(wf)
    result = c.wait(pid, poll_s=0)
    assert result.status == "success" and result.execution_seconds == 2.5
    out = c.download(result.outputs[0], tmp_path / "out.mp4")
    assert out.read_bytes() == b"video-bytes"


def test_oom_is_classified():
    fake = FakeComfyUI(behaviours=["oom"])
    c = client(fake)
    pid = c.submit({"1": {"class_type": "LoadVideo", "inputs": {"file": "x"}}})
    with pytest.raises(ComfyExecutionError) as exc:
        c.wait(pid, poll_s=0)
    assert exc.value.is_oom and exc.value.details["node_type"] == "KSampler"


def test_node_error_is_not_oom():
    fake = FakeComfyUI(behaviours=["node_error"])
    c = client(fake)
    pid = c.submit({"1": {"class_type": "LoadVideo", "inputs": {"file": "x"}}})
    with pytest.raises(ComfyExecutionError) as exc:
        c.wait(pid, poll_s=0)
    assert not exc.value.is_oom


def test_validation_error_surfaces_details():
    fake = FakeComfyUI(object_info={"LoadVideo": {}})
    with pytest.raises(ComfyValidationError):
        client(fake).submit({"1": {"class_type": "NotInstalled", "inputs": {}}})


def test_unreachable_and_vanished_prompts():
    fake = FakeComfyUI()
    fake.down = True
    with pytest.raises(ComfyUnavailable):
        client(fake).system_stats()
    fake.down = False
    with pytest.raises(ComfyUnavailable, match="vanished"):
        client(fake).wait("never-submitted", poll_s=0)


def test_cancel_and_free():
    fake = FakeComfyUI()
    c = client(fake)
    c.cancel("abc")
    c.free()
    assert fake.deleted == ["abc"] and fake.freed == 1


def test_parse_history_collects_all_output_kinds():
    r = parse_history_entry("p", {"outputs": {"9": {"images": [
        {"filename": "a.png", "subfolder": "", "type": "output"}], "gifs": [
        {"filename": "b.mp4", "subfolder": "s", "type": "output"}]}},
        "status": {"status_str": "success", "messages": []}})
    assert [(o.filename, o.kind) for o in r.outputs] == [("a.png", "images"), ("b.mp4", "gifs")]


# -- compiler -----------------------------------------------------------------------------
def test_compiler_substitutes_semantic_params_without_mutating_template():
    t = TemplateRegistry(WF).get("v2v_preview")
    c = compile_workflow(t, {"STYLE_PROMPT": "clay robot", "INPUT_VIDEO": "in.mp4",
                             "SEED": "42", "DENOISE": 0.5, "WIDTH": 384, "HEIGHT": 672,
                             "IDENTITY_STRENGTH": 0.9})
    assert c.workflow["6"]["inputs"]["text"] == "clay robot"
    assert c.workflow["3"]["inputs"]["seed"] == 42
    assert c.workflow["10"]["inputs"]["file"] == "in.mp4"
    assert c.ignored == {"IDENTITY_STRENGTH": 0.9}  # reported, not silently dropped
    assert t.workflow["6"]["inputs"]["text"] == "STYLE PROMPT"


def test_compiler_rejects_missing_required_and_out_of_range():
    t = TemplateRegistry(WF).get("v2v_preview")
    with pytest.raises(TemplateError, match="STYLE_PROMPT"):
        compile_workflow(t, {"INPUT_VIDEO": "x"})
    with pytest.raises(TemplateError, match="maximum"):
        compile_workflow(t, {"STYLE_PROMPT": "a", "INPUT_VIDEO": "x", "DENOISE": 2})


def test_template_loader_catches_bad_node_references(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "workflow.json").write_text(json.dumps(
        {"1": {"class_type": "KSampler", "inputs": {"seed": 0}}}))
    (d / "params.yaml").write_text("name: bad\nversion: 1\nparameters:\n"
                                   "  SEED: {node: '2', input: seed, type: int}\n")
    with pytest.raises(TemplateError, match="missing node"):
        load_template(d)


def test_missing_template_explains_how_to_install():
    with pytest.raises(TemplateError, match="API format"):
        TemplateRegistry(WF).get("hybrid_quality")


def test_validate_against_object_info_reports_missing_nodes_and_options():
    t = TemplateRegistry(WF).get("v2v_preview")
    info = {cls: {"input": {"required": {k: [["x"]] if k == "ckpt_name" else ["INT"]
                                         for k in node["inputs"]}}}
            for cls, node in ((n["class_type"], n) for n in t.workflow.values())}
    info.pop("LoadVideo")
    problems = validate_against_object_info(t, info)
    assert any("LoadVideo not installed" in p for p in problems)
    assert any("ckpt_name" in p for p in problems)


def test_validate_flags_loader_with_no_files_installed():
    t = TemplateRegistry(WF).get("v2v_preview")
    info = {cls: {"input": {"required": {k: [[]] if k == "ckpt_name" else ["INT"]
                                         for k in node["inputs"]}}}
            for cls, node in ((n["class_type"], n) for n in t.workflow.values())}
    problems = validate_against_object_info(t, info)
    assert any("ckpt_name" in p and "none installed" in p for p in problems)


def test_v2v_3070_quality_compiles_shot_params():
    t = TemplateRegistry(WF).get("v2v_3070_quality")
    compiled = compile_workflow(t, {"STYLE_PROMPT": "oil painting", "INPUT_VIDEO": "clip.mp4",
                                    "WIDTH": 480, "HEIGHT": 832, "FRAME_COUNT": 81, "FPS": 16.0,
                                    "STEPS": 20, "SEED": 7, "DENOISE": 0.6})
    wf = compiled.workflow
    assert wf["12"]["inputs"]["width"] == wf["14"]["inputs"]["width"] == 480
    assert wf["14"]["inputs"]["length"] == 81 and wf["15"]["inputs"]["seed"] == 7
    assert "DENOISE" in compiled.ignored


def test_validate_skips_inputs_filled_at_render_time():
    t = TemplateRegistry(WF).get("v2v_3070_quality")
    info = {cls: {"input": {"required": {k: [[]] if k == "file" else ["INT"] for k in node["inputs"]}}}
            for cls, node in ((n["class_type"], n) for n in t.workflow.values())}
    assert validate_against_object_info(t, info) == []


def test_depth_workflow_matches_the_canny_one_except_the_control_node():
    canny = TemplateRegistry(WF).get("v2v_3070_quality")
    depth = TemplateRegistry(WF).get("v2v_3070_depth")
    assert set(depth.spec.parameters) == \
        set(canny.spec.parameters) - {"CANNY_LOW", "CANNY_HIGH"} | {"DEPTH_MODEL"}
    assert {k: v for k, v in depth.workflow.items() if k != "13"} == \
        {k: v for k, v in canny.workflow.items() if k != "13"}
    node = depth.workflow["13"]
    assert node["class_type"] == "DepthAnythingV2Preprocessor"
    # Small is the only Depth Anything V2 size licensed for commercial use (Apache-2.0).
    assert node["inputs"]["ckpt_name"] == "depth_anything_v2_vits.pth"
    compiled = compile_workflow(depth, {"STYLE_PROMPT": "clay", "INPUT_VIDEO": "clip.mp4",
                                        "WIDTH": 576, "HEIGHT": 320, "FRAME_COUNT": 41,
                                        "FPS": 16.0, "STEPS": 20, "SEED": 7})
    assert compiled.workflow["14"]["inputs"]["control_video"] == ["13", 0]
    assert compiled.workflow["12"]["inputs"]["width"] == 576


@pytest.mark.parametrize(("keep", "base", "extra"), [
    ("v2v_3070_keep", "v2v_3070_quality", set()),
    ("v2v_3070_depth_keep", "v2v_3070_depth", {"37"}),
])
def test_keep_workflows_are_their_base_plus_the_subject_mask(keep, base, extra):
    k, b = TemplateRegistry(WF).get(keep), TemplateRegistry(WF).get(base)
    assert set(k.spec.parameters) == set(b.spec.parameters) | {"MASK_VIDEO"}
    mask_nodes = {"30", "31", "32", "33", "34", "35", "36"} | extra
    assert set(k.workflow) == set(b.workflow) | mask_nodes
    for node, spec in b.workflow.items():
        if node != "14":
            assert k.workflow[node] == spec
    vace = dict(k.workflow["14"]["inputs"])
    assert vace.pop("control_video") == ["36", 0] and vace.pop("control_masks") == ["35", 0]
    assert {**vace, "control_video": b.workflow["14"]["inputs"]["control_video"]} == \
        b.workflow["14"]["inputs"]


def test_depth_workflow_reports_a_missing_add_on():
    t = TemplateRegistry(WF).get("v2v_3070_depth")
    info = {n["class_type"]: {"input": {"required": {k: ["INT"] for k in n["inputs"]}}}
            for n in t.workflow.values() if n["class_type"] != "DepthAnythingV2Preprocessor"}
    assert validate_against_object_info(t, info) == [
        "node 13: class DepthAnythingV2Preprocessor not installed"]
    # As controlnet_aux declares it: image required, the rest optional.
    info["DepthAnythingV2Preprocessor"] = {"input": {
        "required": {"image": ["IMAGE"]},
        "optional": {"ckpt_name": [["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth",
                                    "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"]],
                     "resolution": ["INT", {"default": 512}]}}}
    assert validate_against_object_info(t, info) == []


@pytest.mark.parametrize(("draft", "base"), [("v2v_3070_draft", "v2v_3070_quality"),
                                             ("v2v_3070_draft_keep", "v2v_3070_keep")])
def test_draft_workflows_add_the_self_forcing_lora_and_fix_the_sampler(draft, base):
    d, b = TemplateRegistry(WF).get(draft), TemplateRegistry(WF).get(base)
    assert set(d.workflow) == set(b.workflow) | {"7"}
    assert d.workflow["7"]["class_type"] == "LoraLoaderModelOnly"
    assert d.workflow["7"]["inputs"]["model"] == ["1", 0] and d.workflow["4"]["inputs"]["model"] == ["7", 0]
    ks = d.workflow["15"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (4, 1.0, "lcm", "simple")
    assert set(d.spec.parameters) == set(b.spec.parameters) - {"STEPS", "CFG"} | {"LORA", "LORA_STRENGTH"}
    compiled = compile_workflow(d, {"STYLE_PROMPT": "spa", "INPUT_VIDEO": "c.mp4", "MASK_VIDEO": "m.mp4",
                                    "STEPS": 20, "CFG": 6.0})
    assert compiled.workflow["15"]["inputs"]["steps"] == 4 and {"STEPS", "CFG"} <= set(compiled.ignored)


def test_draft_profile_is_loadable(settings):
    profile = settings.profile("RTX3070_DRAFT")
    assert profile.workflow == "v2v_3070_draft" and profile.keep_workflow == "v2v_3070_draft_keep"
    assert settings.profile_problem("RTX3070_DRAFT") is None
