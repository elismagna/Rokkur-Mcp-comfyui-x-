"""Still pictures: generate, edit, inpaint, outpaint, upscale and vary through a fake ComfyUI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from rokkur_studio.comfyui.client import ComfyClient
from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.db.models import CostEntry, GpuLease, Image, Job
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.services import images as svc
from rokkur_studio.services.images import ImageRequest, ImageStore, pad_values, snap_size
from tests.conftest import ROOT
from tests.fakes import PNG_32, FakeComfyUI


def with_comfy(ctx, behaviours=None) -> FakeComfyUI:
    ctx.settings.render.renderer = "comfyui"
    ctx.settings.comfyui.poll_interval_s = 0
    fake = FakeComfyUI(behaviours)
    ctx.comfy_factory = lambda: ComfyClient("http://comfy:8188", transport=fake.transport())
    return fake


def store_for(ctx) -> ImageStore:
    return ImageStore(ctx.settings.studio.data_dir)


def request(ctx, **fields) -> list[Image]:
    with ctx.db.transaction() as s:
        rows = svc.request_images(s, ctx.settings, store_for(ctx), ctx.ffmpeg,
                                  ImageRequest(**fields))
        return [s.get(Image, r.id) for r in rows]


def run(ctx) -> int:
    return Worker(ctx, worker_id="test").drain()


def rows(ctx, ids: list[str]) -> list[Image]:
    with ctx.db.session() as s:
        return [s.get(Image, i) for i in ids]


def picture(ffmpeg, path: Path, width: int = 200, height: int = 120) -> Path:
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    frame[:, :, 2] = 200
    return ffmpeg.write_image(frame, path)


def alpha_of(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgba", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 4)[:, 3]


def test_every_image_template_compiles_with_the_studio_parameters():
    registry = TemplateRegistry(ROOT / "workflows")
    base = {"PROMPT": "a lantern", "SEED": 7, "STEPS": 4, "CFG": 1.0, "BATCH": 2,
            "OUTPUT_PREFIX": "rokkur/images/test"}
    cases = {
        "img_klein_t2i": {"WIDTH": 1024, "HEIGHT": 768},
        "img_klein_edit": {"SOURCE_IMAGE": "x/source.png", "MEGAPIXELS": 1.0},
        "img_klein_inpaint": {"SOURCE_IMAGE": "x/source.png", "MASK_GROW": 12},
        "img_klein_outpaint": {"SOURCE_IMAGE": "x/source.png", "PAD_LEFT": 256, "PAD_RIGHT": 256,
                               "PAD_TOP": 0, "PAD_BOTTOM": 0, "FEATHER": 24, "MASK_GROW": 0},
        "img_zimage_t2i": {"WIDTH": 1024, "HEIGHT": 1024},
        "img_upscale": {"SOURCE_IMAGE": "x/source.png", "SCALE_BY": 0.5},
    }
    for name, extra in cases.items():
        compiled = compile_workflow(registry.get(name), {**base, **extra})
        assert "NEGATIVE_PROMPT" not in compiled.applied, name  # distilled models take none
        assert all(isinstance(n["inputs"], dict) for n in compiled.workflow.values())
        save = [n for n in compiled.workflow.values() if n["class_type"] == "SaveImage"]
        assert len(save) == 1 and save[0]["inputs"]["filename_prefix"] == "rokkur/images/test"
    assert "img_klein_t2i" in registry.names()


def test_sizes_snap_to_multiples_of_16_within_the_area_cap():
    assert snap_size(1024, 1024, 1048576) == (1024, 1024)
    assert snap_size(1920, 1080, 1048576) == (1360, 768)
    assert snap_size(100, 3000, 1048576) == (96, 2992)
    pads = pad_values({"left": 100, "right": 0}, 1024, 768)
    assert (1024 + pads["left"] + pads["right"]) % 16 == 0 and pads["left"] == 100
    with pytest.raises(ValueError, match="at least one side"):
        pad_values({}, 1024, 768)


def test_images_need_the_comfyui_renderer_and_a_prompt(ctx):
    with pytest.raises(ValueError, match="ComfyUI renderer"):
        request(ctx, prompt="a cat")
    with_comfy(ctx)
    with pytest.raises(ValueError, match="Write a prompt"):
        request(ctx, prompt="  ")
    with pytest.raises(ValueError, match="cannot upscale"):
        request(ctx, operation="upscale", profile="KLEIN_4B", source_path="/nowhere.png")
    with pytest.raises(ValueError, match="needs a picture"):
        request(ctx, operation="edit", prompt="make it blue")


def test_generate_makes_a_batch_through_comfyui_with_a_gpu_lease(ctx):
    fake = with_comfy(ctx)
    made = request(ctx, prompt="a red lantern in fog", size="landscape", count=3, seed=42,
                   title="Lantern")
    assert [m.status for m in made] == ["queued"] * 3 and {m.seed for m in made} == {42}
    assert made[0].params["WIDTH"] == 1216 and made[0].params["HEIGHT"] == 832
    assert run(ctx) == 1
    done = rows(ctx, [m.id for m in made])
    assert [d.status for d in done] == ["done"] * 3
    assert all(d.rel_path and store_for(ctx).path_for(d.rel_path).read_bytes() == PNG_32 for d in done)
    assert all((d.width, d.height) == (32, 32) for d in done)  # what the fake draws
    prompt = next(iter(fake.prompts.values()))
    latent = next(n for n in prompt.values() if n["class_type"] == "EmptyFlux2LatentImage")
    assert latent["inputs"]["batch_size"] == 3 and latent["inputs"]["width"] == 1216
    text = next(n for n in prompt.values() if n["class_type"] == "CLIPTextEncode")
    assert text["inputs"]["text"] == "a red lantern in fog"
    with ctx.db.session() as s:
        costs = s.scalars(select(CostEntry).where(CostEntry.job_id == done[0].job_id)).all()
        assert len(costs) == 3 and {c.kind for c in costs} == {"gpu_minutes"}
        assert all(lease.released_at is not None for lease in s.scalars(select(GpuLease)))
        assert all(lease.resource_class == "GPU_HEAVY" for lease in s.scalars(select(GpuLease)))
    assert fake.freed == 0  # test settings turn the after-heavy hook off


def test_edit_inpaint_outpaint_upscale_and_variation_prepare_their_inputs(ctx, tmp_path):
    fake = with_comfy(ctx)
    source = picture(ctx.ffmpeg, tmp_path / "photo.png", 1000, 600)
    with pytest.raises(ValueError, match="Confirm"):
        request(ctx, operation="edit", prompt="make it night", source_path=str(source))
    edited = request(ctx, operation="edit", prompt="make it night", source_path=str(source),
                     rights_confirmed=True, rights_evidence="my own photo")[0]
    prepared = store_for(ctx).path_for(edited.source_rel_path)
    assert ctx.ffmpeg.image_size(prepared) == (992, 592)  # multiples of 16, never upscaled
    assert run(ctx) == 1
    edited = rows(ctx, [edited.id])[0]
    assert edited.status == "done" and edited.parent_id is None
    assert edited.width == 992  # the fake echoes the uploaded picture
    sent = fake.prompts[edited.remote_id]
    assert any(n["class_type"] == "ReferenceLatent" for n in sent.values())
    assert sent["80"]["inputs"]["image"].endswith("source.png")

    mask = tmp_path / "mask.png"
    m = np.zeros((600, 1000, 3), np.uint8)
    m[100:300, 200:500] = 255
    ctx.ffmpeg.write_image(m, mask)
    painted = request(ctx, operation="inpaint", prompt="a hat", source_id=edited.id,
                      mask_path=str(mask), mask_grow=20)[0]
    masked = store_for(ctx).path_for(painted.source_rel_path)
    alpha = alpha_of(masked).reshape(592, 992)
    assert alpha[200, 350] == 0 and alpha[10, 10] == 255  # transparent where painted
    assert painted.parent_id == edited.id and painted.params["MASK_GROW"] == 20
    run(ctx)
    sent = fake.prompts[rows(ctx, [painted.id])[0].remote_id]
    assert sent["85"]["inputs"]["expand"] == 20 and sent["90"]["inputs"]["noise_mask"] is True

    extended = request(ctx, operation="outpaint", prompt="more sea", source_id=edited.id,
                       pad={"left": 200, "bottom": 100})[0]
    assert extended.params["PAD_LEFT"] == 200 and extended.params["PAD_BOTTOM"] == 112
    assert (992 + 200 + extended.params["PAD_RIGHT"]) % 16 == 0

    big = request(ctx, operation="upscale", source_id=edited.id, scale=2)[0]
    assert big.profile == "UPSCALE" and big.params["SCALE_BY"] == 0.5 and big.prompt == ""

    varied = request(ctx, operation="variation", source_id=edited.id, count=2)
    assert len(varied) == 2 and varied[0].prompt.startswith("Create a new variation")
    assert varied[0].workflow == "img_klein_edit"
    run(ctx)
    assert {r.status for r in rows(ctx, [extended.id, big.id] + [v.id for v in varied])} == {"done"}


def test_oom_walks_the_ladder_and_a_rejected_prompt_fails_the_pictures(ctx):
    fake = with_comfy(ctx, ["oom", "oom", "ok", "ok"])
    made = request(ctx, prompt="a long bridge", count=2, seed=3)
    run(ctx)
    done = rows(ctx, [m.id for m in made])
    assert [d.status for d in done] == ["done", "done"]
    assert done[0].params["_oom_steps"] == ["clear_cache", "single_image"]
    assert fake.freed >= 1 and len(fake.prompts) == 4  # 2 failed, then one prompt per picture
    seeds = sorted(n["inputs"]["noise_seed"] for p in list(fake.prompts.values())[2:]
                   for n in p.values() if n["class_type"] == "RandomNoise")
    assert seeds == [3, 4]
    fake.behaviours = ["node_error"]
    failed = request(ctx, prompt="x", count=1)[0]
    run(ctx)
    failed = rows(ctx, [failed.id])[0]
    assert failed.status == "failed" and failed.error["code"] == "render_rejected"
    with ctx.db.session() as s:
        job = s.get(Job, failed.job_id)
        assert job.status == "FAILED"


def test_unreachable_comfyui_leaves_the_pictures_queued_for_a_retry(ctx):
    fake = with_comfy(ctx)
    fake.down = True
    made = request(ctx, prompt="a quiet lake")[0]
    assert Worker(ctx, worker_id="test").run_once()
    again = rows(ctx, [made.id])[0]
    assert again.status == "queued" and again.error["code"] == "unavailable"
    with ctx.db.session() as s:
        assert s.get(Job, made.job_id).status == "RETRY_WAIT"
    fake.down = False
    run(ctx)
    assert rows(ctx, [made.id])[0].status == "done"


def test_cloud_pictures_use_the_cloud_server_and_record_its_minutes(ctx):
    local = with_comfy(ctx)
    cloud = FakeComfyUI()
    ctx.settings.cloud.enabled, ctx.settings.cloud.url = True, "http://127.0.0.1:9"
    ctx.settings.cloud.price_per_hour_usd = 1.2
    ctx.cloud_factory = lambda: ComfyClient("http://cloud:8188", transport=cloud.transport())
    made = request(ctx, prompt="northern lights", render_on="cloud")[0]
    run(ctx)
    assert rows(ctx, [made.id])[0].status == "done" and not local.prompts and cloud.prompts
    with ctx.db.session() as s:
        cost = s.scalars(select(CostEntry)).one()
        assert cost.kind == "cloud_gpu_minutes" and cost.usd > 0
        assert s.scalars(select(GpuLease)).all() == []


def test_library_import_verdict_and_delete(ctx, tmp_path):
    with_comfy(ctx)
    path = picture(ctx.ffmpeg, tmp_path / "ref.png", 64, 48)
    with ctx.db.transaction() as s:
        imported = svc.import_image(s, store_for(ctx), ctx.ffmpeg, path, kind="upload",
                                    title="Reference")
        assert imported.status == "done" and (imported.width, imported.height) == (64, 48)
        svc.set_verdict(s, imported, 2)
        with pytest.raises(ValueError):
            svc.set_verdict(s, imported, 3)
        child = svc.request_images(s, ctx.settings, store_for(ctx), ctx.ffmpeg, ImageRequest(
            operation="edit", prompt="brighter", source_id=imported.id))[0]
        imported_id, child_id = imported.id, child.id
    with ctx.db.transaction() as s:
        assert svc.list_images(s)[0].id == child_id and svc.get_image(s, imported_id).verdict == 2
        folder = store_for(ctx).root / imported_id
        assert folder.is_dir()
        svc.delete_image(s, store_for(ctx), svc.get_image(s, imported_id))
    with ctx.db.session() as s:
        assert s.get(Image, imported_id) is None and s.get(Image, child_id).parent_id is None
        assert s.get(Image, child_id).status == "queued"
    assert not folder.exists()


def test_pictures_page_api_and_cli_cover_the_whole_flow(ctx, tmp_path, sample_video):
    from tests.test_dashboard import client_for
    from tests.test_pipeline import create as create_project
    from tests.test_pipeline import run as run_project
    from tests.test_subject import BoxMasker

    fake = with_comfy(ctx)
    c = client_for(ctx)
    page = c.get("/ui/images").text
    assert "RÖKKUR STUDIO" in page and "No pictures yet" in page and "Pictures" in page
    r = c.post("/ui/images", data={"operation": "generate", "prompt": "a lighthouse at dusk",
                                   "size": "short", "count": "2", "seed": "9"},
               follow_redirects=False)
    assert r.status_code == 303 and "image=img_" in r.headers["location"]
    first = r.headers["location"].split("image=")[1].split("&")[0].split("#")[0]
    assert "msg=2%20pictures%20queued" in r.headers["location"]
    assert "queued" in c.get(f"/ui/images?image={first}").text
    run(ctx)
    page = c.get(f"/ui/images?image={first}").text
    assert "<code>9</code>" in page and "Use as a video reference" in page and "Repaint an area" in page
    assert "32 × 32" in page  # the size of the file the fake drew, not the size asked for
    assert c.get(f"/images/{first}/file").content == PNG_32
    r = c.post(f"/ui/images/{first}/verdict", data={"value": "2"},
               headers={"Accept": "application/json"})
    assert r.json() == {"verdict": 2}
    assert c.post(f"/ui/images/{first}/verdict", data={"value": "2"},
                  headers={"Accept": "application/json"}).json() == {"verdict": None}
    # inpaint through the form with a painted mask and the subject model
    ctx.extras["subject_masker"] = BoxMasker()
    auto = c.get(f"/ui/images/{first}/automask")
    assert auto.status_code == 200 and auto.content[:8] == b"\x89PNG\r\n\x1a\n"
    mask_png = (tmp_path / "m.png")
    m = np.zeros((32, 32, 3), np.uint8)
    m[8:24, 8:24] = 255
    ctx.ffmpeg.write_image(m, mask_png)
    import base64
    data = "data:image/png;base64," + base64.b64encode(mask_png.read_bytes()).decode()
    r = c.post("/ui/images", data={"operation": "inpaint", "prompt": "a red door", "source_id": first,
                                   "mask_data": data, "mask_grow": "4"}, follow_redirects=False)
    assert r.status_code == 303 and "image=img_" in r.headers["location"]
    r = c.post("/ui/images", data={"operation": "inpaint", "prompt": "x", "source_id": first},
               follow_redirects=False)
    assert "Paint%20the%20area" in r.headers["location"]
    # the reference hand-off to New video and the thumbnail hand-off to a finished video
    assert "Appearance reference from Pictures" in c.get(f"/ui/new?reference_image={first}").text
    pid = create_project(ctx, sample_video)
    run_project(ctx)
    r = c.post(f"/ui/images/{first}/thumbnail", data={"project_id": pid}, follow_redirects=False)
    assert r.status_code == 303 and f"/ui/projects/{pid}" in r.headers["location"]
    assert "Thumbnail" in c.get(f"/ui/projects/{pid}").text
    # the API
    r = c.post("/images", json={"operation": "variation", "source_id": first, "count": 1})
    assert r.status_code == 201 and r.json()[0]["parent_id"] == first
    assert c.post("/images", json={"operation": "edit", "prompt": "x"}).status_code == 422
    listed = c.get("/images?limit=5").json()
    assert listed[0]["id"] == r.json()[0]["id"] and listed[0]["status"] == "queued"
    photo = picture(ctx.ffmpeg, tmp_path / "mine.png", 48, 48)
    with photo.open("rb") as fh:
        r = c.post("/images/upload", files={"file": ("mine.png", fh, "image/png")},
                   data={"rights_confirmed": "true", "rights_evidence": "my photo", "title": "Mine"})
    assert r.status_code == 201 and r.json()["kind"] == "upload" and r.json()["title"] == "Mine"
    with photo.open("rb") as fh:
        assert c.post("/images/upload", files={"file": ("mine.png", fh, "image/png")}).status_code == 422
    profiles = c.get("/images/profiles").json()
    assert profiles["KLEIN_4B"]["operations"]["inpaint"] is None
    assert "cannot" not in (profiles["UPSCALE"]["operations"]["upscale"] or "")
    r = c.delete(f"/images/{r.json()['id']}")
    assert r.status_code == 204
    # the CLI
    ctx.extras.pop("subject_masker")
    assert fake.prompts  # the worker above used the fake


def test_cli_queues_and_renders_a_picture(ctx, monkeypatch, capsys):
    from rokkur_studio import cli
    from rokkur_studio.pipeline import context as context_mod

    fake = with_comfy(ctx)
    monkeypatch.setattr(cli, "_settings", lambda args: ctx.settings)
    monkeypatch.setattr(context_mod, "build_context", lambda settings, db=None: ctx)
    assert cli.main(["image", "a harbour at night", "--size", "landscape", "--seed", "5", "--wait"]) == 0
    out = capsys.readouterr().out
    assert "queued 1 picture" in out and "done" in out
    assert cli.main(["image-list"]) == 0
    assert "generate" in capsys.readouterr().out and len(fake.prompts) == 1
    assert cli.main(["image", "--op", "edit", "x", "--source", "/nowhere.png", "--rights", "mine"]) == 1
