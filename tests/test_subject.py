"""Main-subject handling: the decision, the masks and the composite over the render."""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from rokkur_studio.api.app import create_app
from rokkur_studio.api.schemas import CreativeIn, ProjectCreate, RightsIn, SourceIn
from rokkur_studio.comfyui.client import ComfyClient
from rokkur_studio.config import SubjectSection
from rokkur_studio.db.models import Asset, Event, Project, Render
from rokkur_studio.domain.rights import RightsCategory
from rokkur_studio.pipeline import subject as subject_mod
from rokkur_studio.pipeline.subject import (
    MaskerUnavailable,
    MaskModel,
    OnnxSubjectMasker,
    cutout_reference,
    decide_subject,
    feather,
    fetch_model,
    grow,
    keep_subject,
    resize,
    shot_masks,
    smooth_in_time,
    vace_mask_video,
)
from rokkur_studio.services import commands
from rokkur_studio.services.projects import get_project, latest_document
from tests.fakes import FakeComfyUI
from tests.test_pipeline import create, run, status

SPA = "luxury spa bathroom, marble, warm light"


class BoxMasker:
    """A stand-in for the model: the middle quarter of every frame is the subject."""

    name = "box"
    size = 64

    def __init__(self, value: float | None = None) -> None:
        self.value, self.calls = value, 0

    def predict(self, frames: np.ndarray) -> np.ndarray:
        self.calls += 1
        probs = np.zeros((len(frames), self.size, self.size), np.float32)
        if self.value is None:
            probs[:, 16:48, 16:48] = 1.0
        else:
            probs[:] = self.value
        return probs


# -- deciding -----------------------------------------------------------------------------
@pytest.mark.parametrize(("creative", "mode", "by"), [
    ({"theme": SPA}, "keep", "auto"),
    ({"theme": "cyberpunk neon city at night"}, "keep", "auto"),
    ({"theme": "spa", "prompt": "turn the bathroom into a jungle"}, "keep", "auto"),
    ({"theme": "spa", "prompt": "make the room look like a temple"}, "keep", "auto"),
    ({"theme": "spa", "prompt": "replace the background with a beach"}, "keep", "auto"),
    ({"theme": "anime", "prompt": "keep the ape real, only the background changes"}, "keep", "auto"),
    ({"theme": "1970s claymation"}, "restyle", "auto"),
    ({"theme": "Warm retro science-fiction animation",
      "prompt": "Keep the monkey's face and original movement."}, "restyle", "auto"),
    ({"theme": "medieval castle", "prompt": "turn the ape into a robot"}, "restyle", "auto"),
    ({"theme": "medieval castle", "prompt": "turn her into a knight"}, "restyle", "auto"),
    ({"theme": "spa", "prompt": "the ape wearing a tuxedo"}, "restyle", "auto"),
    ({"theme": "spa", "character_key": "NEO"}, "restyle", "auto"),
    ({"theme": "anime", "subject": "keep"}, "keep", "you"),
    ({"theme": SPA, "subject": "restyle"}, "restyle", "you"),
])
def test_subject_decision_follows_the_prompt_unless_the_person_chose(creative, mode, by):
    decision = decide_subject(creative)
    assert (decision.mode, decision.decided_by) == (mode, by)
    assert decision.reason


def test_decision_reason_quotes_what_decided_it():
    assert '"claymation"' in decide_subject({"theme": "claymation"}).reason
    assert '"turn the ape into"' in decide_subject(
        {"theme": "x", "prompt": "turn the ape into a robot"}).reason


# -- mask helpers -------------------------------------------------------------------------
def test_grow_feather_resize_and_smoothing():
    m = np.zeros((20, 30), np.float32)
    m[10, 15] = 1
    g = grow(m, 2)
    assert g[8:13, 13:18].min() == 1 and g.sum() == 25
    f = feather(g, 4)
    assert 0 < f[10, 15] <= 1 and f[10, 22] < f[10, 17] and f.shape == m.shape
    assert resize(np.ones((4, 4), np.float32), 9, 7).shape == (9, 7)
    r = resize(np.array([[0, 1]], np.float32), 1, 4)
    assert r[0, 0] == 0 and r[0, -1] == 1 and np.all(np.diff(r[0]) >= 0)
    p = np.zeros((5, 2, 2), np.float32)
    p[2] = 1
    s = smooth_in_time(p)
    assert np.allclose(s[:, 0, 0], [0, 1 / 3, 1 / 3, 1 / 3, 0])


# -- compositing --------------------------------------------------------------------------
@pytest.fixture
def shot(tmp_path, ffmpeg):
    clip = ffmpeg.make_test_video(tmp_path / "clip.mp4", seconds=1, width=160, height=288,
                                  fps=16, with_audio=False, scene_cut=False)
    render = ffmpeg.filter_video(clip, tmp_path / "render.mp4", "fps=16,scale=160:288,negate",
                                 fps=16)
    return clip, render


def test_keep_subject_puts_the_source_subject_over_the_render(tmp_path, ffmpeg, shot):
    clip, render = shot
    settings = SubjectSection(harmonize=0)
    masker = BoxMasker()
    out = tmp_path / "out.mp4"
    result = keep_subject(ffmpeg, masker, settings, clip=clip, render=render, fps=16, out=out,
                          cache=tmp_path / "masks" / "shot_001")
    assert result["kept"] and result["coverage"] == 0.25 and result["model"] == "box"
    got = ffmpeg.read_rgb_frames(out, 160, 288).astype(int)
    src = ffmpeg.read_rgb_frames(clip, 160, 288, fps=16, frames=len(got)).astype(int)
    rendered = ffmpeg.read_rgb_frames(render, 160, 288).astype(int)
    assert len(got) == len(rendered) == 16
    inside, corner = np.s_[:, 100:188, 60:100], np.s_[:, 0:40, 0:20]
    assert np.abs(got[inside] - src[inside]).mean() < 12
    assert np.abs(got[corner] - rendered[corner]).mean() < 12
    assert np.abs(src[corner] - rendered[corner]).mean() > 60  # the two really differ there
    assert Path(result["mask_video"]).is_file()
    # A repair round re-renders the shot: the masks come from the cache.
    keep_subject(ffmpeg, masker, settings, clip=clip, render=render, fps=16,
                 out=tmp_path / "out2.mp4", cache=tmp_path / "masks" / "shot_001")
    assert masker.calls == 1


def test_colour_shift_moves_the_subject_toward_the_new_room(tmp_path, ffmpeg, shot):
    clip, _ = shot
    darker = ffmpeg.filter_video(clip, tmp_path / "dark.mp4", "fps=16,eq=brightness=-0.3",
                                 fps=16)
    result = keep_subject(ffmpeg, BoxMasker(), SubjectSection(harmonize=1), clip=clip,
                          render=darker, fps=16, out=tmp_path / "out.mp4",
                          cache=tmp_path / "masks" / "s")
    assert all(v < -15 for v in result["colour_shift"])


def test_vace_mask_and_cutout_reference_follow_the_subject(tmp_path, ffmpeg, shot):
    clip, _ = shot
    masker = BoxMasker()
    probs = shot_masks(ffmpeg, masker, clip=clip, fps=16, frames=17, aspect=(160, 288),
                       cache=tmp_path / "masks" / "s")
    assert probs.shape == (17, 64, 64)
    mask = ffmpeg.read_gray_frames(vace_mask_video(ffmpeg, probs, tmp_path / "vace.mp4",
                                                   width=160, height=288, fps=16), 160, 288)
    assert len(mask) == 17
    assert mask[:, 100:188, 60:100].mean() > 230 and mask[:, 0:40, 0:20].mean() < 25
    ref = cutout_reference(ffmpeg, probs, clip=clip, fps=16, width=160, height=288,
                           out=tmp_path / "cutout.png")
    got = ffmpeg.read_rgb_frames(ref, 160, 288)[0].astype(int)
    src = ffmpeg.read_rgb_frames(clip, 160, 288, fps=16, frames=1)[0].astype(int)
    assert got[0:40, 0:20].min() > 245  # plain white around the subject
    assert np.abs(got[100:188, 60:100] - src[100:188, 60:100]).mean() < 12
    assert masker.calls == 1 and shot_masks(  # read back from the cache, also for fewer frames
        ffmpeg, masker, clip=clip, fps=16, frames=9, aspect=(160, 288),
        cache=tmp_path / "masks" / "s").shape[0] == 9 and masker.calls == 1


@pytest.mark.parametrize(("value", "reason"), [(0.0, "no clear subject"),
                                               (1.0, "fills most of the frame")])
def test_shots_without_a_usable_subject_keep_the_render(tmp_path, ffmpeg, shot, value, reason):
    clip, render = shot
    out = tmp_path / "out.mp4"
    result = keep_subject(ffmpeg, BoxMasker(value), SubjectSection(), clip=clip, render=render,
                          fps=16, out=out, cache=tmp_path / "masks" / "s")
    assert not result["kept"] and reason in result["reason"] and not out.exists()


@pytest.mark.skipif(not os.environ.get("ROKKUR_TEST_MASK_MODEL_DIR"),
                    reason="set ROKKUR_TEST_MASK_MODEL_DIR to a folder holding u2net.onnx")
def test_real_u2net_finds_a_bright_subject(tmp_path, ffmpeg):
    pytest.importorskip("onnxruntime")
    clip = tmp_path / "subject.mp4"
    ffmpeg._run(ffmpeg._ff("-f", "lavfi", "-i", "color=c=0x303830:s=160x288:r=16:d=1",
                           "-vf", "drawbox=x=50:y=90:w=60:h=110:color=0xffcc66:t=fill,noise="
                           "alls=12:allf=t", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           str(clip)))
    render = ffmpeg.filter_video(clip, tmp_path / "render.mp4", "fps=16,hue=h=120", fps=16)
    settings = SubjectSection(model_dir=Path(os.environ["ROKKUR_TEST_MASK_MODEL_DIR"]),
                              download=False)
    result = keep_subject(ffmpeg, OnnxSubjectMasker(settings, tmp_path), settings, clip=clip,
                          render=render, fps=16, out=tmp_path / "out.mp4",
                          cache=tmp_path / "masks" / "s")
    assert result["kept"] and 0.05 < result["coverage"] < 0.4


# -- the model file -----------------------------------------------------------------------
def test_missing_model_without_downloads_is_reported_not_fetched(tmp_path):
    masker = OnnxSubjectMasker(SubjectSection(download=False), tmp_path)
    with pytest.raises(MaskerUnavailable, match="download is off"):
        masker.predict(np.zeros((1, 320, 320, 3), np.uint8))
    with pytest.raises(MaskerUnavailable):  # remembered: no second attempt per job
        masker.predict(np.zeros((1, 320, 320, 3), np.uint8))


def test_download_is_kept_only_when_its_checksum_matches(tmp_path, monkeypatch):
    payload = b"onnx bytes"

    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self, size: int):
            yield payload

    @contextlib.contextmanager
    def stream(*args, **kwargs):
        yield Response()

    monkeypatch.setattr(subject_mod.httpx, "stream", stream)
    good = MaskModel("https://example.invalid/m.onnx", hashlib.sha256(payload).hexdigest(), 320,
                     (0, 0, 0), (1, 1, 1))
    path = fetch_model(good, tmp_path / "models" / "m.onnx")
    assert path.read_bytes() == payload
    bad = MaskModel(good.url, "0" * 64, 320, (0, 0, 0), (1, 1, 1))
    with pytest.raises(MaskerUnavailable, match="checksum"):
        fetch_model(bad, tmp_path / "models" / "n.onnx")
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["m.onnx"]


# -- in the pipeline ----------------------------------------------------------------------
def _renders(ctx, pid):
    with ctx.db.session() as s:
        return s.scalars(select(Render).where(Render.project_id == pid)
                         .order_by(Render.shot_id, Render.attempt)).all()


def create_spa(ctx, video: Path) -> str:
    with ctx.db.transaction() as s:
        return commands.create_project(s, ProjectCreate(
            name="spa", source=SourceIn(local_path=str(video)),
            rights=RightsIn(category=RightsCategory.USER_OWNED, permission_evidence="mine"),
            creative=CreativeIn(theme=SPA), autostart=True), ctx.settings).id


def test_pipeline_keeps_the_real_subject_for_a_place_prompt(ctx, sample_video):
    masker = BoxMasker()
    ctx.extras["subject_masker"] = masker
    pid = create_spa(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        manifest = latest_document(s, pid, "manifest").data
        assert manifest["subject"]["mode"] == "keep"
        assert manifest["subject"]["decided_by"] == "auto"
        raw = s.scalars(select(Asset).where(Asset.project_id == pid,
                                            Asset.kind == "render_raw")).all()
        done = s.scalars(select(Event).where(Event.project_id == pid,
                                             Event.type == "RENDER_COMPLETED")).all()
    renders = _renders(ctx, pid)
    assert renders and all(r.params["_details"]["subject"]["kept"] for r in renders)
    assert len(raw) == len(renders) and all(e.data["subject"] == "kept" for e in done)
    assert masker.calls == len(renders)


def test_pipeline_falls_back_to_the_render_when_masks_are_unavailable(ctx, sample_video):
    # No model in the test data folder and downloads are off: the render stands, with a reason.
    pid = create_spa(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    for r in _renders(ctx, pid):
        subject = r.params["_details"]["subject"]
        assert not subject["kept"] and "unavailable" in subject["reason"]


def test_stylized_projects_restyle_the_subject_without_masks(ctx, sample_video):
    masker = BoxMasker()
    ctx.extras["subject_masker"] = masker
    pid = create(ctx, sample_video)  # "retro clay sci-fi"
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH" and masker.calls == 0
    with ctx.db.session() as s:
        assert latest_document(s, pid, "manifest").data["subject"]["mode"] == "restyle"
    assert all("subject" not in r.params["_details"] for r in _renders(ctx, pid))


def _comfy(ctx):
    fake = FakeComfyUI()
    ctx.settings.render.renderer = "comfyui"
    ctx.settings.comfyui.poll_interval_s = 0
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    return fake


def test_missing_live_custom_nodes_disable_profiles_before_project_creation(ctx, sample_video):
    object_info = {}
    for name in ctx.registry.names():
        for node in ctx.registry.get(name).workflow.values():
            entry = object_info.setdefault(node["class_type"],
                                           {"input": {"required": {}, "optional": {}}})
            entry["input"]["required"].update({key: ["STRING"] for key in node["inputs"]})
    object_info.pop("DepthAnythingV2Preprocessor", None)
    fake = FakeComfyUI(object_info=object_info)
    ctx.settings.render.renderer = "comfyui"
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    client = TestClient(create_app(ctx=ctx))

    page = client.get("/ui/new").text
    options = [part.split("</option>", 1)[0] for part in page.split("<option") if "</option>" in part]
    depth_option = next(option for option in options if 'value="RTX3070_DEPTH"' in option)
    quality_option = next(option for option in options if 'value="RTX3070_QUALITY"' in option)
    assert "disabled" in depth_option
    assert "disabled" not in quality_option

    before = len(client.get("/projects").json())
    payload = ProjectCreate(name="depth", render_profile="RTX3070_DEPTH",
        source=SourceIn(platform="local", local_path=str(sample_video)),
        rights=RightsIn(category=RightsCategory.USER_OWNED, permission_evidence="mine"),
        creative=CreativeIn(theme=SPA)).model_dump(mode="json")
    response = client.post("/projects", json=payload)
    assert response.status_code == 422
    assert "DepthAnythingV2Preprocessor not installed" in response.json()["detail"]
    assert len(client.get("/projects").json()) == before

    form = client.post("/ui/projects", data={"theme": SPA, "rights_category": "USER_OWNED",
        "local_path": str(sample_video), "render_profile": "RTX3070_DEPTH"},
        follow_redirects=False)
    assert form.status_code == 303 and "DepthAnythingV2Preprocessor" in form.headers["location"]
    assert len(client.get("/projects").json()) == before


def test_kept_subject_renders_with_the_vace_keep_workflow_and_a_cutout(ctx, sample_video):
    fake = _comfy(ctx)
    masker = BoxMasker()
    ctx.extras["subject_masker"] = masker
    pid = create_spa(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    renders = _renders(ctx, pid)
    assert len(fake.prompts) == len(renders) == 2
    for wf in fake.prompts.values():
        assert wf["14"]["inputs"]["control_masks"] == ["35", 0]
        assert wf["14"]["inputs"]["control_video"] == ["36", 0]
        assert wf["30"]["inputs"]["file"] in fake.uploads  # the subject mask video
        assert fake.uploads[wf["20"]["inputs"]["image"]].startswith(b"\x89PNG")  # the cutout
    for r in renders:
        details = r.params["_details"]
        assert r.workflow == "v2v_3070_keep" and details["subject_mask"]
        assert details["reference"] == "subject cutout"
        assert details["guidance"] == {"keep_workflow": "v2v_3070_keep",
                                       "reference": "subject cutout"}
        assert details["subject"]["kept"]  # the exact subject still goes back over the render
    assert masker.calls == 2  # one mask pass per shot, shared by Wan and the composite
    page = TestClient(create_app(ctx=ctx)).get(f"/ui/projects/{pid}").text
    assert "v2v_3070_keep · Wan kept the real subject and redrew the room" in page
    assert "subject cutout" in page and "Canny 0.2 / 0.5" in page


def test_unusable_masks_leave_the_plain_workflow_and_no_reference(ctx, sample_video):
    fake = _comfy(ctx)
    ctx.extras["subject_masker"] = BoxMasker(0.0)
    pid = create_spa(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    for wf in fake.prompts.values():
        assert "30" not in wf and "control_masks" not in wf["14"]["inputs"]
        assert "reference_image" not in wf["14"]["inputs"]
    for r in _renders(ctx, pid):
        assert r.workflow == "v2v_3070_quality"
        assert r.params["_details"]["guidance"] == {"masks": "no clear subject in this shot"}
        assert r.params["_details"]["reference"] == "none"


def test_a_missing_keep_workflow_still_renders_with_the_cutout(ctx, sample_video):
    fake = _comfy(ctx)
    ctx.extras["subject_masker"] = BoxMasker()
    for profile in ctx.settings.profiles.values():
        profile.keep_workflow = "no_such_workflow"
    pid = create_spa(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert all("30" not in wf and "20" in wf for wf in fake.prompts.values())
    r = _renders(ctx, pid)[0]
    assert r.workflow == ctx.settings.profile(r.profile).workflow
    assert r.params["_details"]["guidance"]["keep_workflow"] == "no_such_workflow is not available"


def test_repair_rounds_reuse_the_shot_masks(ctx, sample_video):
    masker = BoxMasker()
    ctx.extras["subject_masker"] = masker
    pid = create(ctx, sample_video, subject="keep",
                 test_faults={"shot_002": {"kind": "black", "attempts": [1]}})
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    renders = _renders(ctx, pid)
    assert [(r.shot_id, r.attempt) for r in renders] == [
        ("shot_001", 1), ("shot_002", 1), ("shot_002", 2)]
    assert masker.calls == 2  # one per shot, not one per attempt


# -- dashboard ----------------------------------------------------------------------------
def test_form_explains_and_records_the_subject_choice(ctx, settings, sample_video, tmp_path):
    c = TestClient(create_app(ctx=ctx))
    d = c.get("/ui/subject-decision", params={"theme": SPA}).json()
    assert d["mode"] == "keep" and d["decided_by"] == "auto"
    d = c.get("/ui/subject-decision", params={"theme": SPA, "reference": "true"}).json()
    assert d["mode"] == "restyle"
    assert 'name="subject"' in c.get("/ui/new").text
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(sample_video.read_bytes())
    settings.studio.media_dir = media
    r = c.post("/ui/projects", data={"theme": "anime", "rights_category": "USER_OWNED",
                                     "media_file": str(media / "clip.mp4"), "subject": "keep"},
               follow_redirects=False)
    pid = r.headers["location"].split("/ui/projects/")[1].split("?")[0]
    with ctx.db.session() as s:
        assert get_project(s, pid).creative_input["subject"] == "keep"
    r = c.post("/ui/projects", data={"theme": "x", "rights_category": "USER_OWNED",
                                     "media_file": str(media / "clip.mp4"), "subject": "maybe"},
               follow_redirects=False)
    assert "err=" in r.headers["location"]


def test_project_page_shows_the_subject_plan_and_each_shot(ctx, sample_video):
    ctx.extras["subject_masker"] = BoxMasker()
    pid = create(ctx, sample_video, subject="keep", autostart=False)
    c = TestClient(create_app(ctx=ctx))
    c.post(f"/ui/projects/{pid}/start")
    run(ctx, rounds=4)  # rights, ingest, analysis, brief: no manifest yet
    page = c.get(f"/ui/projects/{pid}").text
    assert "Main subject" in page and "(planned)" in page
    run(ctx)
    page = c.get(f"/ui/projects/{pid}").text
    assert "Kept real" in page and "(planned)" not in page
    assert "Real subject kept (25% of the frame)" in page and "restyled subject" in page


def test_render_command_passes_the_comparison_options(ctx, settings, sample_video, monkeypatch):
    from rokkur_studio import cli

    monkeypatch.setattr(cli, "_settings", lambda args: settings)
    monkeypatch.setattr("rokkur_studio.pipeline.context.build_context", lambda s: ctx)
    monkeypatch.setattr(cli, "_drive", lambda *args, **kwargs: 0)
    assert cli.main(["render", str(sample_video), "--theme", SPA, "--rights", "USER_OWNED",
                     "--evidence", "mine", "--renderer", "ffmpeg_preview", "--reference",
                     "cutout", "--canny", "0.4", "0.8", "--seed", "7"]) == 0
    with ctx.db.session() as s:
        project = s.scalars(select(Project)).one()
        creative = project.creative_input
    assert creative["reference_mode"] == "cutout" and creative["seed"] == 7
    assert (creative["canny_low"], creative["canny_high"]) == (0.4, 0.8)


def test_keep_workflow_with_nodes_missing_from_comfyui_falls_back(ctx, sample_video):
    fake = _comfy(ctx)
    plain = ctx.registry.get("v2v_3070_quality").workflow.values()
    fake.object_info_data = {n["class_type"]: {"input": {"required": {
        k: ["*"] for k in n["inputs"]}}} for n in plain}  # no ThresholdMask & co.
    ctx.extras["subject_masker"] = BoxMasker()
    pid = create_spa(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    r = _renders(ctx, pid)[0]
    assert r.workflow == "v2v_3070_quality"
    assert "not installed" in r.params["_details"]["guidance"]["keep_workflow"]
    assert r.params["_details"]["subject"]["kept"]  # the composite still keeps the ape
