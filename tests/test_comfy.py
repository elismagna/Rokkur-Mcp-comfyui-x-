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
