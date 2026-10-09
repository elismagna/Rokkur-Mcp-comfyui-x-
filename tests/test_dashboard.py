"""The dashboard pages render and their actions go through the same gates as the API."""

from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from rokkur_studio.api.app import create_app
from rokkur_studio.dashboard.views import percent, stage_progress
from rokkur_studio.db.models import Project
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
                 "/ui/agents", "/ui/director", "/ui/system"):
        r = c.get(path)
        assert r.status_code == 200, path
        assert "RÖKKUR STUDIO" in r.text
    page = c.get(f"/ui/projects/{pid}").text
    assert "Final video" in page and "Creative brief" in page and "Dry run" in page
    assert "Prompt schedule" in page and "(medium shot:1.3)" in page
    assert 'class="kf"' in page  # each shot shows its middle frame
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


def test_prompt_editor_returns_reviewable_draft_without_creating_project(ctx):
    class FakeProvider:
        name = "ollama"

        def __init__(self):
            self.calls = []

        def generate(self, role, instructions, payload, schema, images=None):
            self.calls.append((role, payload))
            return schema(theme="Soft clay animation", prompt="A monkey explores a bright room.",
                          changes="Kept the original subject and action.")

        def supports_images(self):
            return False

    ctx.settings.agents.provider = "ollama"
    provider = FakeProvider()
    ctx.provider = provider
    with ctx.db.session() as session:
        before = session.scalar(select(func.count(Project.id))) or 0

    response = client_for(ctx).post("/ui/prompt-enhance", json={
        "theme": "soft clay", "prompt": "monkey in room", "subject": "keep",
        "character": "small brown monkey", "target_format": "youtube_video"})

    assert response.status_code == 200
    assert response.json() == {"theme": "Soft clay animation",
                               "prompt": "A monkey explores a bright room.",
                               "changes": "Kept the original subject and action."}
    assert provider.calls == [("Prompt Editor", {"theme": "soft clay",
        "prompt": "monkey in room", "subject": "keep", "character": "small brown monkey",
        "target_format": "youtube_video"})]
    with ctx.db.session() as session:
        assert (session.scalar(select(func.count(Project.id))) or 0) == before
    assert ctx.gpu.active() == []


def test_prompt_editor_does_not_call_model_while_render_holds_gpu(ctx):
    class FakeProvider:
        name = "ollama"
        called = False

        def generate(self, role, instructions, payload, schema, images=None):
            self.called = True
            return schema(theme="unexpected", prompt="", changes="")

        def supports_images(self):
            return False

    ctx.settings.agents.provider = "ollama"
    provider = FakeProvider()
    ctx.provider = provider
    lease = ctx.gpu.try_acquire("active-render", "GPU_HEAVY")
    assert lease is not None
    try:
        response = client_for(ctx).post("/ui/prompt-enhance", json={"theme": "clay"})
    finally:
        ctx.gpu.release(lease.id)
    assert response.status_code == 409
    assert not provider.called
    assert "unchanged" in response.json()["detail"]


def test_new_video_uploads_optional_sound_bed_and_preserves_source_audio(ctx, sample_video,
                                                                        tmp_path):
    bed = tmp_path / "bed.wav"
    ctx.ffmpeg._run(ctx.ffmpeg._ff("-f", "lavfi", "-i", "sine=frequency=660:duration=1",
                                   "-c:a", "pcm_s16le", str(bed)))
    client = client_for(ctx)
    with bed.open("rb") as fh:
        rejected = client.post("/ui/projects", data={
            "theme": "clay", "rights_category": "USER_OWNED", "local_path": str(sample_video)},
            files={"audio_bed_file": ("bed.wav", fh, "audio/wav")}, follow_redirects=False)
    assert "permission" in rejected.headers["location"].lower()
    assert not list((ctx.settings.studio.data_dir / "uploads").glob("*_soundtrack.wav"))
    with bed.open("rb") as fh:
        response = client.post("/ui/projects", data={
            "theme": "clay", "rights_category": "USER_OWNED", "local_path": str(sample_video),
            "audio_bed_gain": "0.3", "audio_bed_rights_confirmed": "true",
            "audio_bed_rights_evidence": "Created with a licensed tool"},
            files={"audio_bed_file": ("bed.wav", fh, "audio/wav")}, follow_redirects=False)
    assert response.status_code == 303
    project_id = response.headers["location"].split("/ui/projects/")[-1].split("?")[0]
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        assert project is not None
        assert project.creative_input["keep_source_audio"] is True
        assert project.creative_input["audio_bed_gain"] == 0.3
        assert project.creative_input["audio_bed_rights_confirmed"] is True
        sound_path = project.creative_input["audio_bed_path"]
        assert sound_path.endswith("_soundtrack.wav") and "uploads" in sound_path
        assert Path(sound_path).is_file()


def test_new_video_prompt_workbench_is_outside_create_form(ctx):
    page = client_for(ctx).get("/ui/new").text
    assert page.index("</form>") < page.index('<dialog id="prompt-editor"')
    assert "Use these prompts" in page and "Your form is unchanged" in page
    assert "AI suggestions unavailable" in page


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


def test_director_page_edits_the_tracker_and_previews(ctx, settings):
    c = client_for(ctx)
    r = c.post("/ui/director/look", data={"prompt_prefix": "Stop-motion film still",
                                          "style_modifiers": "visible fingerprints",
                                          "negative_prompt": "blurry"})
    assert r.status_code == 200 and "Saved" in r.text and "visible fingerprints" in r.text
    r = c.post("/ui/director/characters", data={"key": "old man", "description": "grey beard"})
    assert "OLD_MAN" in r.text
    assert "OLD_MAN" in c.get("/ui/new").text  # pickable for a new video
    r = c.post("/ui/director/characters", data={"key": "x", "description": "y" * 700})
    assert "under 600 characters" in r.text
    r = c.post("/ui/director/characters/OLD_MAN/delete")
    assert "OLD_MAN" not in r.text
    page = c.get("/ui/director", params={"run": "1", "subject": "he is walking",
                                         "shot_size": "Close-up", "global_look": "on"}).text
    assert "in mid-stride" in page and "Stop-motion film still, (close-up shot:1.3)" in page
    page = c.get("/ui/director", params={"run": "1", "subject": "x"}).text  # box unticked
    assert "Stop-motion film still" not in page.split("<h3>Prompt</h3>")[1]
    (settings.studio.data_dir / "director" / "asset_tracker.json").write_text("{bad")
    assert "not a valid asset tracker" in c.get("/ui/director").text


def test_new_video_with_a_tracker_character(ctx, settings, sample_video):
    c = client_for(ctx)
    r = c.post("/ui/projects", data={"theme": "clay", "rights_category": "USER_OWNED",
                                     "local_path": str(sample_video), "character_key": "NEO",
                                     "use_global_look": "true", "autostart": "true"})
    assert r.status_code == 200
    pid = str(r.url).rstrip("/").split("/")[-1].split("?")[0]
    run(ctx)
    page = c.get(f"/ui/projects/{pid}").text
    assert "<b>NEO</b>" in page and "tattered black hooded jacket" in page


def test_repair_limit_page_offers_more_repairs_or_keeping_the_renders(ctx, sample_video):
    ctx.settings.render.max_retries = 1
    c = client_for(ctx)
    pid = create(ctx, sample_video,
                 test_faults={"shot_002": {"kind": "black", "attempts": [1, 2, 3]}})
    run(ctx)
    assert status(ctx, pid) == "FAILED"
    page = c.get(f"/ui/projects/{pid}").text
    assert "Stopped after 1 repair round" in page and "fails on 002" in page
    assert "Keep these renders" in page and "Try 1 more repairs" in page
    assert "Check quality again" in page
    assert ">Resume<" not in page
    assert "Repair 1 more times" in c.get("/ui/approvals").text
    r = c.post(f"/ui/projects/{pid}/resume", follow_redirects=False)
    assert "repair+limit" in r.headers["location"] or "repair%20limit" in r.headers["location"]
    c.post(f"/ui/projects/{pid}/keep-renders")
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    page = c.get(f"/ui/projects/{pid}").text
    assert "kept by dashboard" in page and "these renders were kept anyway" in page


def test_render_budget_stop_offers_more_renders_instead_of_resume(ctx, sample_video):
    ctx.settings.render.max_renders_per_project = 1
    c = client_for(ctx)
    pid = create(ctx, sample_video)
    run(ctx)
    page = c.get(f"/ui/projects/{pid}").text
    assert "Stopped at this project's render budget: 1 of 1 renders" in page
    assert "Allow 1 more renders and continue" in page and ">Resume<" not in page
    r = c.post(f"/ui/projects/{pid}/allow-more-renders", follow_redirects=False)
    assert "Allowed%20more%20renders" in r.headers["location"]
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
