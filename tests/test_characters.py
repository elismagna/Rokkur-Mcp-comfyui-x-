"""Characters found in a clip before prompting, and storyboard stills per shot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rokkur_studio.config import SubjectSection
from rokkur_studio.db.models import Image
from rokkur_studio.services import characters as svc
from rokkur_studio.services.characters import crop_box, find_characters, sample_times, signature
from tests.test_dashboard import client_for
from tests.test_images import run, with_comfy
from tests.test_pipeline import create
from tests.test_pipeline import run as run_project
from tests.test_subject import BoxMasker


def test_sample_times_cover_the_clip_and_its_scenes(ffmpeg, sample_video):
    times = sample_times(ffmpeg, sample_video)
    assert 3 <= len(times) <= 8 and times == sorted(times)
    assert any(abs(t - 1.0) < 0.6 for t in times) and any(abs(t - 3.0) < 0.6 for t in times)


def test_find_characters_writes_a_cutout_and_merges_repeat_sightings(ffmpeg, sample_video, tmp_path):
    found = find_characters(ffmpeg, BoxMasker(), sample_video, tmp_path / "chars",
                            settings=SubjectSection(download=False))
    assert 1 <= len(found) <= 4
    best = found[0]
    assert Path(best["path"]).is_file() and Path(best["crop"]).is_file()
    assert best["seen"] >= 1 and 0 < best["share"] < 0.75
    top, bottom, left, right = best["box"]
    width, height = ffmpeg.image_size(Path(best["path"]))
    assert bottom - top < height and right - left < width  # cropped to the box, with a margin
    assert ffmpeg.image_size(Path(best["crop"])) == (right - left, bottom - top)
    cutout = ffmpeg.read_rgb_frames(Path(best["path"]), 60, 60)[0]
    assert cutout[2, 2].min() > 240  # white away from the subject
    assert sum(c["seen"] for c in found) >= len(found)


def test_signature_and_crop_box_helpers():
    frame = np.zeros((20, 20, 3), np.uint8)
    frame[5:15, 5:15] = (250, 10, 10)
    alpha = np.zeros((20, 20), np.float32)
    alpha[5:15, 5:15] = 1
    a = signature(frame, alpha)
    assert abs(float(np.linalg.norm(a)) - 1) < 1e-5 and a.argmax() == 3 * 16
    assert float(np.dot(a, signature(frame, alpha))) > 0.999
    assert crop_box(alpha) == (4, 16, 4, 16)
    assert crop_box(np.zeros((20, 20), np.float32)) == (0, 20, 0, 20)
    assert "exactly as in the reference" in svc.describe("NEO", "clip", 0.4)


def test_new_video_finds_characters_and_hands_one_to_the_form(ctx, settings, sample_video, tmp_path):
    with_comfy(ctx)
    ctx.extras["subject_masker"] = BoxMasker()
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(sample_video.read_bytes())
    settings.studio.media_dir = media
    c = client_for(ctx)
    assert c.post("/ui/new/characters", data={}).status_code == 422
    r = c.post("/ui/new/characters", data={"media_file": str(media / "clip.mp4")})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["characters"] and data["characters"][0]["title"].startswith("Subject 1 from clip")
    first = data["characters"][0]
    assert Path(first["path"]).is_file() and first["description"].startswith("Subject 1")
    assert c.get(first["crop_url"]).status_code == 200
    with ctx.db.session() as s:
        row = s.get(Image, first["id"])
        assert row.kind == "character" and row.status == "done"
    page = c.get(f"/ui/new?reference_image={first['id']}").text
    assert "Appearance reference from Pictures" in page and first["path"] in page
    with (tmp_path / "up.mp4").open("wb") as fh:
        fh.write(sample_video.read_bytes())
    with (tmp_path / "up.mp4").open("rb") as fh:
        r = c.post("/ui/new/characters", files={"source_file": ("up.mp4", fh, "video/mp4")})
    assert r.status_code == 200 and r.json()["characters"][0]["uploaded"].endswith("up.mp4")
    assert "Characters" in c.get("/ui/images?kind=character").text


def test_storyboard_stills_follow_the_brief_prompts(ctx, sample_video):
    fake = with_comfy(ctx)
    pid = create(ctx, sample_video)
    run_project(ctx)
    c = client_for(ctx)
    page = c.get(f"/ui/projects/{pid}").text
    assert "Preview the look as stills" in page
    r = c.post(f"/ui/projects/{pid}/storyboard", follow_redirects=False)
    assert r.status_code == 303 and "storyboard" in r.headers["location"]
    with ctx.db.session() as s:
        stills = list(s.query(Image).filter(Image.project_id == pid, Image.kind == "storyboard"))
    assert len(stills) >= 2
    assert {s.shot_id for s in stills} == {f"shot_{i:03d}" for i in range(1, len(stills) + 1)}
    assert all(s.params["WIDTH"] == 768 and s.params["HEIGHT"] == 1344 for s in stills)
    run(ctx)
    with ctx.db.session() as s:
        assert {r.status for r in s.query(Image).filter(Image.kind == "storyboard")} == {"done"}
    page = c.get(f"/ui/projects/{pid}").text
    assert "Make the stills again" in page and "Shot 001" in page and len(fake.prompts) >= 2
