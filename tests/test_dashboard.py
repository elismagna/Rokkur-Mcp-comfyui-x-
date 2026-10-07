"""The dashboard pages render and their actions go through the same gates as the API."""

import httpx
from fastapi.testclient import TestClient

from rokkur_studio.api.app import create_app
from rokkur_studio.dashboard.views import percent, stage_progress
from tests.fakes_youtube import FakeGoogle
from tests.test_pipeline import create, run, status


def client_for(ctx):
    return TestClient(create_app(ctx=ctx))


def test_stage_progress_maps_states():
    steps = stage_progress("RENDERING")
    assert [s["state"] for s in steps][:6] == ["done"] * 5 + ["current"]
    assert stage_progress("PUBLISHED")[-1]["state"] == "done" and percent("PUBLISHED") == 100
    failed = stage_progress("FAILED", "QUALITY_CHECK")
    assert failed[6] == {"label": "Quality", "state": "failed"} and failed[5]["state"] == "done"
    assert percent("DISCOVERED") == 0


def test_every_page_renders(ctx, sample_video):
    c = client_for(ctx)
    pid = create(ctx, sample_video)
    run(ctx)
    for path in ("/ui", "/ui/projects", "/ui/projects?group=ready&q=t", "/ui/new",
                 f"/ui/projects/{pid}", "/ui/queue", "/ui/approvals", "/ui/youtube",
                 "/ui/agents", "/ui/system"):
        r = c.get(path)
        assert r.status_code == 200, path
        assert "RÖKKUR STUDIO" in r.text
    page = c.get(f"/ui/projects/{pid}").text
    assert "Final video" in page and "Creative brief" in page and "Dry run" in page
    assert "Not signed in" in page or "uploads are off" in page.lower()


def test_new_from_media_folder_and_upload(ctx, settings, sample_video, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(sample_video.read_bytes())
    (media / "notes.txt").write_text("x")
    settings.studio.media_dir = media
    c = client_for(ctx)
    page = c.get("/ui/new").text
    assert "clip.mp4" in page and "notes.txt" not in page
    r = c.post("/ui/projects", data={"theme": "clay", "rights_category": "USER_OWNED",
                                     "media_file": str(media / "clip.mp4")},
               follow_redirects=False)
    assert r.status_code == 303 and "/ui/projects/proj_" in r.headers["location"]
    r = c.post("/ui/projects", data={"theme": "clay", "rights_category": "USER_OWNED",
                                     "media_file": "/etc/passwd"}, follow_redirects=False)
    assert "err=" in r.headers["location"]
    with sample_video.open("rb") as fh:
        r = c.post("/ui/projects", data={"theme": "clay", "rights_category": "USER_OWNED"},
                   files={"source_file": ("mine.mp4", fh, "video/mp4")}, follow_redirects=False)
    assert r.status_code == 303 and "/ui/projects/proj_" in r.headers["location"]
    assert any((settings.studio.data_dir / "uploads").glob("*_mine.mp4"))
    r = c.post("/ui/projects", data={"theme": "clay", "rights_category": "USER_OWNED"},
               follow_redirects=False)
    assert "err=" in r.headers["location"]


def test_edit_metadata_and_publish_from_dashboard(ctx, settings, sample_video, tmp_path):
    from rokkur_studio.youtube.client import YouTubeClient
    from tests.test_youtube import CLIENT, enable_youtube, signed_in_store

    pid = create(ctx, sample_video)
    run(ctx)
    c = client_for(ctx)
    url = f"/ui/projects/{pid}"
    r = c.post(f"{url}/metadata", data={"title": "My clay walk #shorts",
                                        "description": "A walk.", "tags": "clay, walk"},
               follow_redirects=False)
    assert "msg=" in r.headers["location"]
    r = c.post(f"{url}/metadata", data={"title": "x" * 120, "description": "d"},
               follow_redirects=False)
    assert "err=" in r.headers["location"]
    # uploads are off by default: the button is disabled and the server refuses too
    r = c.post(f"{url}/publish", data={"mode": "upload", "privacy": "private"},
               follow_redirects=False)
    assert "err=" in r.headers["location"] and status(ctx, pid) == "READY_TO_PUBLISH"
    r = c.post(f"{url}/publish", data={"mode": "dry", "privacy": "private"},
               follow_redirects=False)
    assert "Dry+run+OK" in r.headers["location"] or "Dry%20run%20OK" in r.headers["location"]
    enable_youtube(settings, tmp_path)
    google = FakeGoogle()
    ctx.extras["youtube_client"] = YouTubeClient(
        CLIENT, signed_in_store(tmp_path), http=httpx.Client(transport=google.transport))
    r = c.post(f"{url}/publish", data={"mode": "upload", "privacy": "public"},
               follow_redirects=False)
    assert "err=" in r.headers["location"]  # public blocked without allow_public
    r = c.post(f"{url}/publish", data={"mode": "upload", "privacy": "unlisted"},
               follow_redirects=False)
    assert "youtube.com" in r.headers["location"], r.headers["location"]
    assert status(ctx, pid) == "PUBLISHED"
    sent = google.uploads[0]["body"]
    assert sent["snippet"]["title"] == "My clay walk #shorts"
    assert sent["snippet"]["tags"] == ["clay", "walk"]
    assert sent["status"]["privacyStatus"] == "unlisted"
    page = c.get(url).text
    assert "watch?v=vid123" in page
    assert "watch?v=vid123" in c.get("/ui/youtube").text
    r = c.post("/ui/youtube/check", follow_redirects=False)
    assert "R%C3%B6kkur" in r.headers["location"] or "Rökkur" in r.headers["location"]
