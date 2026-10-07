from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from rokkur_studio.api.app import create_app
from rokkur_studio.jobs.worker import Worker
from rokkur_studio.services.publishing import (
    PublishGateError,
    build_insert_request,
    validate_metadata,
)


@pytest.fixture
def client(ctx):
    return TestClient(create_app(ctx=ctx))


def payload(path, **kw):
    return {"name": "Clay walk", "source": {"platform": "local", "local_path": str(path)},
            "rights": {"category": "USER_OWNED", "permission_evidence": "I filmed it"},
            "creative": {"theme": "1970s stop-motion sci-fi"}, **kw}


def test_create_persists_and_lists(client, sample_video):
    r = client.post("/projects", json=payload(sample_video))
    assert r.status_code == 201
    pid = r.json()["id"]
    assert r.json()["status"] == "DISCOVERED" and r.json()["render_profile"] == "PREVIEW"
    assert [p["id"] for p in client.get("/projects").json()] == [pid]
    detail = client.get(f"/projects/{pid}").json()
    assert detail["rights"]["category"] == "USER_OWNED" and detail["next_job"] is None
    assert client.get(f"/projects/{pid}/events").json()[0]["type"] == "PROJECT_CREATED"


def test_validation_errors(client, sample_video):
    assert client.post("/projects", json={"name": "x"}).status_code == 422
    assert client.post("/projects", json=payload(sample_video, render_profile="NOPE")
                       ).status_code == 422
    assert client.get("/projects/proj_missing").status_code == 404


def test_full_flow_over_http(client, ctx, sample_video):
    pid = client.post("/projects", json=payload(sample_video)).json()["id"]
    assert client.post(f"/projects/{pid}/start").json()["status"] == "RIGHTS_PENDING"
    assert client.post(f"/projects/{pid}/publish", json={}).status_code == 409  # gates
    Worker(ctx).drain()
    detail = client.get(f"/projects/{pid}").json()
    assert detail["project"]["status"] == "READY_TO_PUBLISH"
    assert {"analysis", "creative_brief", "prompt_schedule", "manifest", "qc_report",
            "metadata"} <= set(detail["documents"])
    keyframes = [a for a in detail["assets"] if a["kind"] == "keyframe"]
    assert len(keyframes) == len(detail["documents"]["creative_brief"]["data"]["shot_plan"])
    schedule = client.get(f"/projects/{pid}/prompt-schedule")
    assert schedule.status_code == 200 and schedule.text.startswith('"0": "Cinematic film still')
    final = next(a for a in detail["assets"] if a["kind"] == "final")
    video = client.get(f"/projects/{pid}/assets/{final['id']}/file")
    assert video.status_code == 200 and len(video.content) == final["size_bytes"]
    r = client.post(f"/projects/{pid}/publish", json={"dry_run": False})
    assert r.status_code == 409 and "disabled" in r.json()["detail"]  # youtube.enabled false
    at = (datetime.now(UTC) + timedelta(days=30)).replace(hour=18, minute=0, second=0,
                                                          microsecond=0)
    body = {"privacy": "private", "publish_at": at.isoformat()}
    r = client.post(f"/projects/{pid}/publish", json=body)
    assert r.status_code == 409 and "allow_public" in r.json()["detail"]  # release goes public
    ctx.settings.youtube.allow_public = True
    pub = client.post(f"/projects/{pid}/publish", json=body).json()
    assert pub["dry_run"] and pub["request"]["body"]["status"]["publishAt"] == \
        at.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert client.get("/jobs", params={"project_id": pid}).json()
    assert "queue" in client.get("/workers").json()


def test_director_endpoints(client, settings):
    assert "Rembrandt lighting" in client.get("/director/vocabulary").json()["Lighting styles"]
    tracker = client.get("/director/assets").json()
    assert "NEO" in tracker["CHARACTERS"]
    tracker["CHARACTERS"]["mara"] = "a tall woman, red raincoat"
    saved = client.put("/director/assets", json=tracker).json()
    assert saved["CHARACTERS"]["MARA"] == "a tall woman, red raincoat"
    assert (settings.studio.data_dir / "director" / "asset_tracker.json").is_file()
    assert client.put("/director/assets", json={"CHARACTERS": {"!": "x"}}).status_code == 422
    out = client.post("/director/preview", json={
        "subject": "is running", "shot_size": "Close-up", "character_key": "MARA"}).json()
    assert out["prompt"].startswith("Cinematic film still, (close-up shot:1.3), a tall woman")
    assert client.post("/director/preview", json={"shot_size": "Giant"}).status_code == 422
    assert client.get("/projects/nope/prompt-schedule").status_code == 404


def test_upload_source_cancel_and_rights_decision(client, sample_video):
    pid = client.post("/projects", json=payload(
        "/nowhere.mp4", rights={"category": "UNKNOWN"})).json()["id"]
    with sample_video.open("rb") as fh:
        r = client.post(f"/projects/{pid}/source", files={"file": ("clip.mp4", fh, "video/mp4")})
    assert r.status_code == 200 and r.json()["kind"] == "source"
    client.post(f"/projects/{pid}/start")
    r = client.post(f"/projects/{pid}/rights", json={"approve": True, "category": "USER_OWNED",
                                                     "note": "my footage"})
    assert r.status_code == 200 and r.json()["status"] == "RIGHTS_OK"
    assert client.post(f"/projects/{pid}/cancel").json()["status"] == "CANCELLED"
    assert client.post(f"/projects/{pid}/cancel").status_code == 409


def test_channels_and_system(client):
    ch = client.post("/channels", json={"name": "Rökkur", "autonomy_level": 1}).json()
    assert ch["autonomy_level"] == 1
    assert client.get("/channels").json()[0]["id"] == ch["id"]
    info = client.get("/system", params={"probe_services": False}).json()
    assert info["database"] and "v2v_preview" in info["workflow_templates"]
    assert client.get("/gpu/leases").json()["vram_gb"] == 8
    assert client.get("/approvals").json() == []


def test_openapi_and_dashboard_render(client, sample_video):
    assert "/projects" in client.get("/openapi.json").json()["paths"]
    r = client.post("/ui/projects", data={"name": "ui", "theme": "clay", "local_path":
                                          str(sample_video), "rights_category": "USER_OWNED"},
                    follow_redirects=False)
    assert r.status_code == 303
    for path in ("/ui", r.headers["location"], "/ui/queue", "/ui/approvals", "/ui/system"):
        assert client.get(path).status_code == 200, path


def test_metadata_validation_and_scheduling_rules():
    meta = {"title": "a" * 101, "description": "<b>", "tags": ["x" * 501],
            "made_for_kids": False, "contains_synthetic_media": True, "category_id": "1"}
    errors = validate_metadata(meta)
    assert len(errors) == 3
    with pytest.raises(PublishGateError, match="private"):
        build_insert_request(meta, privacy="public", publish_at=datetime.now(UTC))


def test_apply_draft_keeps_policy_parts():
    from rokkur_studio.services import publishing

    base = {"title": "x #shorts", "description": "x.\n\nMade with AI", "tags": ["x"],
            "made_for_kids": False, "contains_synthetic_media": True}
    out = publishing.apply_draft(base, {"title": "A <b> walk", "description": "Nice.\n\nhttp://x",
                                        "tags": ["Clay ", "clay", ""]},
                                 target_format="youtube_short", rights=None)
    assert out["title"] == "A b walk #shorts" and out["tags"] == ["clay"]
    assert "Rökkur Studio" in out["description"] and out["made_for_kids"] is False
    too_long = publishing.apply_draft(base, {"title": "t" * 100}, target_format="youtube_short",
                                      rights=None)
    assert too_long == base  # a draft over YouTube's title limit is dropped
