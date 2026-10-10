"""The 3D studio: mesh files, edits, reliefs, scans, reconstruction jobs, page, API, CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.db.models import CostEntry, GpuLease, Model3D
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.mesh import ops
from rokkur_studio.mesh.io import MeshError, glb_bytes, read_mesh, read_points, write_mesh
from rokkur_studio.services import models3d as svc
from rokkur_studio.services.models3d import EditRequest, ModelStore, ReconRequest
from tests.conftest import ROOT
from tests.test_dashboard import client_for
from tests.test_images import picture, with_comfy

FAKE_MESHROOM = r'''
import sys, os
args = sys.argv[1:]
src, out = args[args.index("--input") + 1], args[args.index("--output") + 1]
photos = sorted(os.listdir(src))
print(f"meshroom: {len(photos)} images")
if len(photos) < 3:
    print("not enough images"); sys.exit(2)
os.makedirs(os.path.join(out, "texturing"), exist_ok=True)
with open(os.path.join(out, "texturing", "texturedMesh.obj"), "w") as fh:
    fh.write("v 0 0 0\nv 4 0 0\nv 0 4 0\nv 0 0 4\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 1 4 3\n")
'''


def store_for(ctx) -> ModelStore:
    return ModelStore(ctx.settings.studio.data_dir)


def stl(tmp_path: Path, mesh=None, name: str = "part.stl") -> Path:
    return write_mesh(mesh or ops.box((10, 20, 30)), tmp_path / name)


def load(ctx, model_id: str) -> Model3D:
    with ctx.db.session() as s:
        return s.get(Model3D, model_id)


def test_formats_round_trip_and_measurements_are_exact(tmp_path):
    box = ops.box((10, 20, 30))
    for ext in ("stl", "obj", "ply", "glb"):
        back = read_mesh(write_mesh(box, tmp_path / f"b.{ext}"))
        stats = ops.measure(back)
        assert stats["volume"] == 6000 and stats["area"] == 2200 and stats["closed"], ext
    ascii_ = read_mesh(write_mesh(box, tmp_path / "a.stl", ascii_stl=True))
    assert ascii_.meta["encoding"] == "ascii" and ops.measure(ascii_)["volume"] == 6000
    assert (tmp_path / "b.glb").read_bytes() == glb_bytes(box)
    open_top = ops.box((1, 1, 1))
    open_top.faces = open_top.faces[2:]  # drop the bottom: two triangles, four open edges
    assert ops.measure(open_top)["open_edges"] == 4 and not ops.measure(open_top)["closed"]
    with pytest.raises(MeshError, match="use an STL"):
        read_mesh(tmp_path / "x.fbx")
    (tmp_path / "pts.xyz").write_text("x y z\n0 0 0\n1 0 0, \n0 1 0 255 255 255\n1 1 1\n")
    assert read_points(tmp_path / "pts.xyz").shape == (4, 3)


def test_edits_keep_the_model_printable(tmp_path):
    box = ops.box((10, 20, 30))
    assert ops.measure(ops.fit(box, 60))["size"] == [20, 40, 60]
    assert ops.measure(ops.scale(box, 25.4))["size"][0] == 254
    turned = ops.measure(ops.rotate(box, "z", 90))
    assert turned["size"] == [20, 10, 30] and turned["closed"] and not turned["inverted"]
    mirrored = ops.measure(ops.mirror(box, "x"))
    assert mirrored["volume"] == 6000 and not mirrored["inverted"]  # winding fixed with the mirror
    assert ops.measure(ops.flip_normals(box))["inverted"]
    floor = ops.measure(ops.center(ops.translate(box, (5, 5, 5)), on_floor=True))
    assert floor["min"] == [-5, -10, 0]
    soup = ops.box((1, 1, 1))
    soup.vertices = soup.vertices + np.array([[0, 0, 1e-7]] * 8) * np.arange(8)[:, None]
    assert ops.measure(ops.weld(ops.combine([soup, soup]), 1e-3))["vertices"] == 8
    relief = ops.relief_from_gray(np.tile(np.linspace(0, 255, 30), (12, 1)), width_mm=58,
                                  depth_mm=4, base_mm=1)
    r = ops.measure(relief)
    assert r["closed"] and r["size"][0] == 58 and r["size"][2] == 5
    litho = ops.measure(ops.relief_from_gray(np.full((5, 5), 255.0), width_mm=10, depth_mm=3,
                                             base_mm=0.5, invert=True))
    assert litho["size"][2] == 0.5  # white is thin in a lithophane
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 20, (5000, 3))
    pts[:, 2] = 3 + np.cos(pts[:, 0] / 3)
    solid = ops.measure(ops.relief_from_points(pts, cell=1.0, base=1.0))
    assert solid["closed"] and solid["size"][2] == pytest.approx(3.0, abs=0.05)  # 2 of relief + 1 base
    holey = pts[(pts[:, 0] < 8) | (pts[:, 0] > 12)]  # a gap four cells wide is not filled
    assert ops.measure(ops.relief_from_points(holey, cell=1.0, fill=1))["closed"]
    with pytest.raises(MeshError, match="too fine"):
        ops.relief_from_points(pts, cell=0.001)
    svg = ops.preview_svg(box)
    assert svg.startswith("<svg") and svg.count("<polygon") >= 3


def test_import_edit_combine_export_and_delete(ctx, tmp_path):
    store = store_for(ctx)
    with ctx.db.transaction() as s:
        part = svc.import_model(s, store, stl(tmp_path), rights_evidence="my design")
        assert part.status == "done" and part.stats["volume"] == 6000 and part.source_rel_path
        bigger = svc.edit_model(s, store, EditRequest(operation="fit", source_id=part.id, size=60))
        assert bigger.parent_id == part.id and bigger.stats["size"] == [20, 40, 60]
        other = svc.make_box(s, store, (5, 5, 5))
        both = svc.edit_model(s, store, EditRequest(operation="combine",
                                                    source_ids=[part.id, other.id]))
        assert both.kind == "combine" and both.stats["triangles"] == 24 and both.parent_id == part.id
        inside_out = svc.edit_model(s, store, EditRequest(operation="flip", source_id=part.id))
        assert inside_out.stats["inverted"]
        fixed = svc.edit_model(s, store, EditRequest(operation="repair", source_id=inside_out.id))
        assert not fixed.stats["inverted"] and fixed.stats["closed"]
        with pytest.raises(ValueError, match="two or more"):
            svc.edit_model(s, store, EditRequest(operation="combine", source_ids=[part.id]))
        with pytest.raises(ValueError, match="Pick the model"):
            svc.edit_model(s, store, EditRequest(operation="scale"))
        part_id, bigger_id = part.id, bigger.id
    with ctx.db.session() as s:
        row = s.get(Model3D, bigger_id)
        for fmt in ("obj", "ply", "glb", "stl"):
            out = svc.export_model(store, row, fmt)
            assert ops.measure(read_mesh(out))["volume"] == pytest.approx(48000, rel=1e-4)
        with pytest.raises(ValueError, match="Export as"):
            svc.export_model(store, row, "fbx")
        view = svc.view_data(store, row, max_faces=6)
        assert view["thinned"] and len(view["faces"]) == 18 and view["total_faces"] == 12
    cloud = tmp_path / "cloud.ply"
    write_mesh(ops.Mesh(np.random.default_rng(2).uniform(0, 1, (50, 3)), np.zeros((0, 3))), cloud)
    with ctx.db.transaction() as s:
        with pytest.raises(ValueError, match="point cloud"):
            svc.import_model(s, store, cloud)
        svc.delete_model(s, store, svc.get_model(s, part_id))
    with ctx.db.session() as s:
        assert s.get(Model3D, part_id) is None and s.get(Model3D, bigger_id).parent_id is None
    assert not (store.root / part_id).exists()


def test_relief_from_a_library_picture_and_a_lidar_scan(ctx, tmp_path):
    from tests.test_images import request as request_images

    store = store_for(ctx)
    photo = picture(ctx.ffmpeg, tmp_path / "photo.png", 200, 100)
    with ctx.db.transaction() as s:
        with pytest.raises(ValueError, match="Confirm"):
            svc.make_relief(s, ctx.settings, store, ctx.ffmpeg,
                            svc.ReliefRequest(image_path=str(photo)))
        relief = svc.make_relief(s, ctx.settings, store, ctx.ffmpeg, svc.ReliefRequest(
            image_path=str(photo), width_mm=80, depth_mm=2, base_mm=1, resolution=40,
            rights_confirmed=True, rights_evidence="mine"))
        assert relief.stats["closed"] and relief.stats["size"][0] == 80
        assert relief.stats["size"][1] == pytest.approx(40, abs=2.1)  # the picture is 2:1
        assert relief.stats["size"][2] <= 3.0 and relief.method == "relief"
    fake = with_comfy(ctx)
    pic = request_images(ctx, prompt="a lantern", seed=1)[0]
    Worker(ctx, kinds=["image"], worker_id="p").drain()
    with ctx.db.transaction() as s:
        litho = svc.make_relief(s, ctx.settings, store, ctx.ffmpeg, svc.ReliefRequest(
            image_id=pic.id, invert=True, resolution=16, width_mm=50))
        assert litho.method == "lithophane" and "library" in litho.request["rights_evidence"]
    assert fake.prompts
    scan = tmp_path / "scan.xyz"
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 10, (3000, 3))
    pts[:, 2] = 1 + pts[:, 0] / 10
    scan.write_text("\n".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in pts))
    with ctx.db.transaction() as s:
        solid = svc.points_to_model(s, store, svc.PointsRequest(
            path=str(scan), rights_confirmed=True, rights_evidence="my scan"))
        assert solid.stats["closed"] and solid.params["points"] == 3000 and solid.params["cell"] > 0
        with pytest.raises(ValueError, match="Confirm"):
            svc.points_to_model(s, store, svc.PointsRequest(path=str(scan)))


def test_picture_to_3d_runs_hunyuan3d_through_comfyui_with_a_lease(ctx, tmp_path):
    t = TemplateRegistry(ROOT / "workflows").get("mesh_hunyuan3d_i2m")
    c = compile_workflow(t, {"SOURCE_IMAGE": "x/source.png", "SEED": 5, "OCTREE": 128,
                             "OUTPUT_PREFIX": "rokkur/3d/t"})
    assert c.workflow["61"]["inputs"]["octree_resolution"] == 128 and c.workflow["3"]["inputs"]["cfg"] == 8.0
    store = store_for(ctx)
    status = svc.availability(ctx.settings)
    assert status["image"] and "ComfyUI" in status["image"] and status["relief"] is None
    fake = with_comfy(ctx)
    assert svc.availability(ctx.settings)["image"] is None
    photo = picture(ctx.ffmpeg, tmp_path / "toy.png", 300, 200)
    with ctx.db.transaction() as s:
        row = svc.request_reconstruction(s, ctx.settings, store, ctx.ffmpeg, ReconRequest(
            method="image", image_path=str(photo), seed=7, octree=128, rights_confirmed=True,
            rights_evidence="my photo"))
        assert row.status == "queued" and row.params == {"SEED": 7, "OCTREE": 128}
        row_id = row.id
    assert Worker(ctx, worker_id="t").drain() == 1
    done = load(ctx, row_id)
    assert done.status == "done", done.error
    assert done.stats["size"] == [12, 8, 5] and done.stats["closed"] and done.remote_id
    sent = fake.prompts[done.remote_id]
    assert sent["56"]["inputs"]["image"].endswith("source.png") and sent["3"]["inputs"]["seed"] == 7
    assert sent["82"]["inputs"]["filename_prefix"].startswith("rokkur/3d/")
    with ctx.db.session() as s:
        assert s.scalars(select(CostEntry).where(CostEntry.job_id == done.job_id)).one().kind == "gpu_minutes"
        assert all(lease.released_at is not None for lease in s.scalars(select(GpuLease)))
    fake.behaviours = ["oom"]
    with ctx.db.transaction() as s:
        row_id = svc.request_reconstruction(s, ctx.settings, store, ctx.ffmpeg, ReconRequest(
            method="image", image_path=str(photo), rights_confirmed=True, rights_evidence="x")).id
    Worker(ctx, worker_id="t").drain()
    failed = load(ctx, row_id)
    assert failed.status == "failed" and "6 GB" in failed.error["message"]


def test_photos_and_video_go_through_meshroom_when_it_is_installed(ctx, tmp_path, sample_video):
    store = store_for(ctx)
    photos = [picture(ctx.ffmpeg, tmp_path / f"p{i}.jpg", 64, 48) for i in range(4)]
    ctx.settings.three.photogrammetry_command = "no-such-meshroom"
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="not on the worker's PATH"):
        svc.request_reconstruction(s, ctx.settings, store, ctx.ffmpeg, ReconRequest(
            method="photos", photo_paths=[str(p) for p in photos], rights_confirmed=True,
            rights_evidence="mine"))
    script = tmp_path / "fake_meshroom.py"
    script.write_text(FAKE_MESHROOM)
    ctx.settings.three.photogrammetry_command = f"{sys.executable} {script}"
    assert svc.availability(ctx.settings)["photos"] is None
    with ctx.db.transaction() as s:
        with pytest.raises(ValueError, match="at least 3"):
            svc.request_reconstruction(s, ctx.settings, store, ctx.ffmpeg, ReconRequest(
                method="photos", photo_paths=[str(photos[0])], rights_confirmed=True,
                rights_evidence="mine"))
        from_photos = svc.request_reconstruction(s, ctx.settings, store, ctx.ffmpeg, ReconRequest(
            method="photos", photo_paths=[str(p) for p in photos], rights_confirmed=True,
            rights_evidence="mine")).id
        from_video = svc.request_reconstruction(s, ctx.settings, store, ctx.ffmpeg, ReconRequest(
            method="video", video_path=str(sample_video), rights_confirmed=True,
            rights_evidence="mine")).id
    ctx.settings.three.video_fps = 4
    Worker(ctx, worker_id="t").drain()
    a, b = load(ctx, from_photos), load(ctx, from_video)
    assert a.status == "done" and a.stats["triangles"] == 4 and a.stats["closed"]
    assert b.status == "done" and b.params["frames"] >= 3
    log = (store.root / a.id / "work" / "tool.log").read_text()
    assert "meshroom: 4 images" in log and "--input" in log
    frames = list((store.root / b.id / "work" / "frames").glob("frame_*.jpg"))
    assert len(frames) == b.params["frames"]
    assert svc.availability(ctx.settings)["poisson"]  # Open3D is not installed here


def test_three_page_api_and_cli(ctx, tmp_path, monkeypatch, capsys):
    from rokkur_studio import cli

    c = client_for(ctx)
    page = c.get("/ui/3d").text
    assert "No models yet" in page and "Picture relief" in page and "LiDAR / points" in page
    assert "Unavailable" in page  # the AI and photogrammetry tools are not set up here
    part = stl(tmp_path)
    with part.open("rb") as fh:
        r = c.post("/ui/3d/upload", files={"model_file": ("part.stl", fh, "model/stl")},
                   data={"title": "Bracket"}, follow_redirects=False)
    assert r.status_code == 303 and "model=m3d_" in r.headers["location"]
    model_id = r.headers["location"].split("model=")[1].split("&")[0].split("#")[0]
    page = c.get(f"/ui/3d?model={model_id}").text
    assert "Bracket" in page and "closed, watertight" in page and "10 × 20 × 30" in page
    assert 'id="m3d-canvas"' in page
    view = c.get(f"/models3d/{model_id}/view.json").json()
    assert len(view["faces"]) == 36 and not view["thinned"]
    assert c.get(f"/models3d/{model_id}/preview.svg").text.startswith("<svg")
    r = c.post("/ui/3d/edit", data={"operation": "rotate", "source_id": model_id, "axis": "z",
                                    "degrees": "90"}, follow_redirects=False)
    assert "msg=Rotate%20done" in r.headers["location"]
    r = c.post("/ui/3d/box", data={"x": "5", "y": "5", "z": "2"}, follow_redirects=False)
    box_id = r.headers["location"].split("model=")[1].split("&")[0].split("#")[0]
    r = c.post("/ui/3d/edit", data={"operation": "combine", "source_id": model_id,
                                    "source_ids": [model_id, box_id]}, follow_redirects=False)
    assert "msg=Combine%20done" in r.headers["location"]
    scan = tmp_path / "scan.xyz"
    pts = np.random.default_rng(4).uniform(0, 5, (800, 3))
    scan.write_text("\n".join(" ".join(f"{v:.3f}" for v in p) for p in pts))
    with scan.open("rb") as fh:
        r = c.post("/ui/3d/points", files={"points_file": ("scan.xyz", fh, "text/plain")},
                   data={"rights_confirmed": "true", "rights_evidence": "mine", "cell": "0.5"},
                   follow_redirects=False)
    assert "Solid%20made" in r.headers["location"]
    r = c.post("/ui/3d/reconstruct", data={"method": "photos"}, follow_redirects=False)
    assert "err=" in r.headers["location"]
    file = c.get(f"/models3d/{model_id}/file?format=obj")
    assert file.status_code == 200 and file.content.startswith(b"# Rokkur Studio")
    assert "Bracket.obj" in file.headers["content-disposition"]
    assert c.get(f"/models3d/{model_id}/file?ascii=1").content.startswith(b"solid")
    methods = c.get("/models3d/methods").json()
    assert methods["relief"]["problem"] is None and methods["image"]["problem"]
    r = c.post("/models3d/edit", json={"operation": "fit", "source_id": model_id, "size": 15})
    assert r.status_code == 201 and r.json()["stats"]["size"] == [5, 10, 15]
    assert c.post("/models3d/edit", json={"operation": "fit"}).status_code == 422
    assert c.post("/models3d/box", json={"size": [1, 2, 0]}).status_code == 422
    listed = c.get("/models3d?limit=50").json()
    assert {m["kind"] for m in listed} >= {"upload", "edit", "combine", "primitive", "points"}
    assert c.delete(f"/models3d/{r.json()['id']}").status_code == 204
    # the CLI, which needs no database for files
    out = tmp_path / "out.ply"
    assert cli.main(["mesh-convert", str(part), str(out), "--fit", "60", "--floor"]) == 0
    assert "closed" in capsys.readouterr().out
    assert cli.main(["mesh-info", str(out)]) == 0
    assert "20 x 40 x 60" in capsys.readouterr().out
    assert cli.main(["mesh-info", str(tmp_path / "nope.stl")]) == 1
    monkeypatch.setattr(cli, "_settings", lambda args: ctx.settings)
    assert cli.main(["model3d-list"]) == 0 and "Bracket" in capsys.readouterr().out
