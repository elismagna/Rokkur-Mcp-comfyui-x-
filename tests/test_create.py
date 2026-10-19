"""The Create page: prompts with Cinema choices, the three workflows, and takes end to end."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from rokkur_studio.api.app import create_app
from rokkur_studio.comfyui.client import ComfyClient
from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.db.models import CostEntry, Job, Take
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.services import create
from tests.fakes import FakeComfyUI

ROOT = Path(__file__).resolve().parents[1]


def test_cinema_choices_go_into_the_prompt_in_order():
    camera = create.clean_camera({"shot": "Close-up", "move": "Orbit", "lens": "Classic anamorphic",
                                  "aperture": "f/1.4", "focal": "nonsense"}, "video")
    assert camera == {"shot": "Close-up", "move": "Orbit", "lens": "Classic anamorphic",
                      "aperture": "f/1.4"}
    text = create.compose_prompt("  a fox   in the snow. ", camera)
    assert text.startswith("close-up shot, camera orbits around the subject. a fox in the snow. ")
    assert text.endswith("anamorphic lens, oval bokeh, horizontal lens flares, "
                         "f/1.4, shallow depth of field, creamy background blur")
    # a still cannot move, and no choices leaves the words alone
    assert "move" not in create.clean_camera({"move": "Orbit"}, "image")
    assert create.compose_prompt("a fox", {}) == "a fox"


def test_clip_lengths_are_wan_frame_counts():
    assert [create.frames_for(s, 16, 81) for s in create.SECONDS] == [33, 49, 81]
    assert create.frames_for(10, 16, 81) == 81
    assert all((n - 1) % 4 == 0 for n in (create.frames_for(s, 16, 161) for s in (2, 3, 5, 7)))
    for w, h in [*create.IMAGE_SIZES.values(), *create.VIDEO_SIZES.values()]:
        assert w % 16 == 0 and h % 16 == 0
    assert all(w * h <= 832 * 480 for w, h in create.VIDEO_SIZES.values())


@pytest.mark.parametrize("name", ["t2i_zimage_turbo", "t2v_3070_vace", "i2v_3070_vace"])
def test_create_workflows_compile_with_every_link_resolved(name):
    template = TemplateRegistry(ROOT / "workflows").get(name)
    params = {"PROMPT": "a fox", "WIDTH": 832, "HEIGHT": 480, "SEED": 7}
    if name != "t2i_zimage_turbo":
        params |= {"FRAME_COUNT": 81, "FPS": 16}
    if name == "i2v_3070_vace":
        params |= {"START_IMAGE": "start.png", "UNKNOWN_FRAMES": 80}
    compiled = compile_workflow(template, params)
    graph = compiled.workflow
    for node_id, node in graph.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list):
                assert str(value[0]) in graph, f"{name}: {node_id}.{key} -> missing {value[0]}"
    assert not compiled.ignored
    if name == "t2v_3070_vace":  # no reference image: the node and its link are dropped
        assert "20" not in graph and "reference_image" not in graph["14"]["inputs"]
    if name == "i2v_3070_vace":
        assert graph["23"]["inputs"]["amount"] == 80
        assert {graph[n]["inputs"]["width"] for n in ("11", "14", "21", "25")} == {832}


def _done(ctx):
    Worker(ctx).drain()


def test_preview_takes_render_pictures_and_clips(ctx, ffmpeg):
    with ctx.db.transaction() as s:
        image = create.submit(s, ctx.settings, create.TakeRequest(
            mode="image", prompt="a fox", aspect="16:9", seed=5, count=2))
        clip = create.submit(s, ctx.settings, create.TakeRequest(
            mode="video", prompt="a fox runs", aspect="9:16", seconds=2,
            camera={"move": "Crane up"}))[0]
        ids = [t.id for t in image] + [clip.id]
    _done(ctx)
    with ctx.db.session() as s:
        takes = {t.id: t for t in s.scalars(select(Take))}
        assert all(takes[i].status == "succeeded" for i in ids)
        first, second = (takes[t] for t in ids[:2])
        assert (first.settings["params"]["SEED"], second.settings["params"]["SEED"]) == (5, 6)
        assert first.workflow == "t2i_zimage_turbo" and first.negative_prompt == ""
        made = create.path_for(ctx.settings, first.output)
        info = ffmpeg.probe(made)
        assert (info.width, info.height) == (1344, 768)
        video = takes[ids[2]]
        assert video.final_prompt.startswith("crane shot rising upward. a fox runs")
        assert video.negative_prompt  # clips get Wan's negative prompt
        probe = ffmpeg.probe(create.path_for(ctx.settings, video.output))
        assert (probe.width, probe.height, probe.frame_count) == (480, 832, 33)
        assert create.path_for(ctx.settings, video.thumbnail).is_file()


def test_comfyui_take_animates_a_start_picture_under_the_gpu_lease(ctx, ffmpeg, tmp_path):
    fake = FakeComfyUI()
    clip = ffmpeg.test_pattern(tmp_path / "clip.mp4", width=832, height=480, frames=33)
    fake.generated_video = clip.read_bytes()
    start = ffmpeg.test_pattern(tmp_path / "start.png", width=640, height=480)
    ctx.settings.render.renderer = "comfyui"
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    with ctx.db.transaction() as s:
        take = create.submit(s, ctx.settings, create.TakeRequest(
            mode="animate", prompt="the fox looks up", seconds=2, seed=11,
            start_image=start))[0]
        take_id = take.id
    _done(ctx)
    (workflow,) = fake.prompts.values()
    assert workflow["5"]["inputs"]["text"] == "the fox looks up"
    assert workflow["14"]["inputs"]["length"] == 33 and workflow["23"]["inputs"]["amount"] == 32
    assert workflow["15"]["inputs"]["seed"] == 11
    assert workflow["10"]["inputs"]["image"] in fake.uploads
    with ctx.db.session() as s:
        row = s.get(Take, take_id)
        assert row.status == "succeeded" and row.remote_id in fake.prompts
        assert create.path_for(ctx.settings, row.output).read_bytes() == clip.read_bytes()
        assert s.scalar(select(CostEntry.kind).where(CostEntry.job_id == row.job_id)) == "gpu_minutes"
    assert not ctx.gpu.active()  # the lease was given back


def test_out_of_memory_fails_the_take_with_a_plain_reason(ctx):
    fake = FakeComfyUI(behaviours=["oom"])
    ctx.settings.render.renderer = "comfyui"
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    with ctx.db.transaction() as s:
        take_id = create.submit(s, ctx.settings, create.TakeRequest(
            mode="image", prompt="a fox"))[0].id
    _done(ctx)
    with ctx.db.session() as s:
        take = s.get(Take, take_id)
        job = s.get(Job, take.job_id)
        assert create.state(take, job) == ("failed", take.error)
        assert "ran out of memory" in take.error and job.status == "FAILED"


def test_requests_the_page_cannot_run_are_refused(ctx, tmp_path):
    with ctx.db.transaction() as s:
        for req, words in [
            (create.TakeRequest(mode="image", prompt="   "), "Write what"),
            (create.TakeRequest(mode="animate", prompt="a fox"), "start picture"),
            (create.TakeRequest(mode="video", prompt="a fox", seconds=9), "seconds long"),
            (create.TakeRequest(mode="image", prompt="a fox", render_on="cloud"), "cloud"),
        ]:
            with pytest.raises(create.CreateError, match=words):
                create.submit(s, ctx.settings, req)
        with pytest.raises(create.CreateError):
            create.path_for(ctx.settings, "../../etc/passwd")


def test_create_page_flow(ctx, ffmpeg):
    c = TestClient(create_app(ctx=ctx))
    for tab in ("image", "video", "cinema"):
        page = c.get(f"/ui/create?tab={tab}")
        assert page.status_code == 200 and "Make a picture or a clip" in page.text
    r = c.post("/ui/create", data={"tab": "cinema", "output": "still", "prompt": "a lighthouse",
                                   "aspect": "1:1", "count": "1", "cam_lens": "Fisheye",
                                   "cam_move": "Orbit"}, follow_redirects=False)
    assert r.status_code == 303 and "Queued+1+picture" in r.headers["location"].replace("%20", "+")
    _done(ctx)
    with ctx.db.session() as s:
        picture = s.scalars(select(Take)).one()
        assert picture.camera == {"lens": "Fisheye"}  # a still drops the camera move
        pid = picture.id
    page = c.get("/ui/create").text
    assert f"/ui/create/takes/{pid}/file" in page and "Animate" in page
    assert c.get(f"/ui/create/takes/{pid}/file").status_code == 200
    assert c.post(f"/ui/create/takes/{pid}/rate", data={"value": "2"}).json() == {"rating": 2}
    # animate the picture, then hand the clip to a new video project
    c.post("/ui/create", data={"tab": "video", "prompt": "waves roll in", "start_take": pid,
                               "seconds": "2", "aspect": "16:9"})
    _done(ctx)
    with ctx.db.session() as s:
        clip = s.scalars(select(Take).where(Take.mode == "animate")).one()
        assert clip.status == "succeeded" and clip.parent_id == pid and clip.start_image
        cid = clip.id
    assert c.get(f"/ui/create/takes/{cid}/start").status_code == 200
    new = c.get(f"/ui/new?take={cid}").text
    assert "Using your Create clip" in new and f"take {cid}" in new
    assert c.post(f"/ui/create/takes/{cid}/again", follow_redirects=False).status_code == 303
    assert c.post(f"/ui/create/takes/{pid}/delete", follow_redirects=False).status_code == 303
    with ctx.db.session() as s:
        assert s.get(Take, pid) is None
    assert not (create.create_dir(ctx.settings) / pid).exists()


def test_create_api(ctx):
    c = TestClient(create_app(ctx=ctx))
    assert "Orbit" in c.get("/create/options").json()["cinema"]["move"]
    made = c.post("/create/takes", json={"mode": "video", "prompt": "rain on a window",
                                         "seconds": 3, "camera": {"shot": "Close-up"}})
    assert made.status_code == 201
    (take,) = made.json()
    assert take["status"] == "queued" and take["final_prompt"].startswith("close-up shot. ")
    _done(ctx)
    assert c.get(f"/create/takes/{take['id']}").json()["status"] == "succeeded"
    assert c.post("/create/takes", json={"prompt": "x", "start_take": "nope",
                                         "mode": "video"}).status_code == 422
