import json
import os
import stat
import time
from pathlib import Path

import httpx
import pytest

from rokkur_studio.services import publishing
from rokkur_studio.services.projects import get_project
from rokkur_studio.youtube import client as yt
from rokkur_studio.youtube import oauth
from tests.fakes_youtube import FakeGoogle
from tests.test_pipeline import create, events, run, status

CLIENT = oauth.OAuthClient("cid", "csecret")


def write_client_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"installed": {"client_id": "cid", "client_secret": "csecret",
                                              "token_uri": oauth.TOKEN_URL}}))
    return path


def signed_in_store(tmp_path: Path) -> oauth.TokenStore:
    store = oauth.TokenStore(tmp_path / "secrets" / "youtube_token.json")
    store.save(oauth.Token(refresh_token="rt-1", access_token="at-1",
                           expires_at=time.time() + 3600))
    return store


# -- OAuth ----------------------------------------------------------------------------------

def test_client_file_loads_and_rejects_garbage(tmp_path):
    path = write_client_file(tmp_path / "s" / "c.json")
    assert oauth.OAuthClient.load(path).client_id == "cid"
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(oauth.OAuthError, match="not a Google OAuth client"):
        oauth.OAuthClient.load(tmp_path / "bad.json")
    with pytest.raises(oauth.OAuthError, match="not found"):
        oauth.OAuthClient.load(tmp_path / "missing.json")


def test_token_store_is_owner_only_and_round_trips(tmp_path):
    store = oauth.TokenStore(tmp_path / "deep" / "tok.json")
    assert not store.exists()
    with pytest.raises(oauth.OAuthError, match="youtube-auth"):
        store.load()
    store.save(oauth.Token(refresh_token="rt", access_token="at", expires_at=5.0))
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    assert store.load() == oauth.Token(refresh_token="rt", access_token="at", expires_at=5.0)
    assert store.load().expired()
    store.delete()
    assert not store.exists()


def test_auth_url_has_offline_access_and_pkce():
    url = oauth.build_auth_url(CLIENT, "http://127.0.0.1:8401/", state="st", code_challenge="ch")
    assert url.startswith(oauth.AUTH_URL + "?")
    for part in ("access_type=offline", "prompt=consent", "code_challenge=ch",
                 "code_challenge_method=S256", "state=st", "youtube.upload"):
        assert part in url


def test_parse_redirect_checks_state_and_errors():
    assert oauth.parse_redirect("http://127.0.0.1:8401/?state=s&code=abc", expected_state="s") == "abc"
    assert oauth.parse_redirect("state=s&code=abc&scope=x", expected_state="s") == "abc"
    with pytest.raises(oauth.OAuthError, match="state mismatch"):
        oauth.parse_redirect("?state=other&code=abc", expected_state="s")
    with pytest.raises(oauth.OAuthError, match="denied"):
        oauth.parse_redirect("?state=s&error=access_denied", expected_state="s")


def test_installed_flow_with_paste_saves_refresh_token(tmp_path):
    google = FakeGoogle()
    store = oauth.TokenStore(tmp_path / "tok.json")
    shown: list[str] = []
    with httpx.Client(transport=google.transport) as http:
        def paste() -> str:
            state = dict(p.split("=") for p in shown[0].split("?")[1].split("&"))["state"]
            return f"http://127.0.0.1:8401/?state={state}&code=the-code"

        token = oauth.installed_flow(CLIENT, store, port=8401, http=http, open_url=shown.append,
                                     paste=True, read_line=paste)
    assert token.refresh_token == "rt-1" and store.load().refresh_token == "rt-1"
    form = google.tokens[0]
    assert form["grant_type"] == "authorization_code" and form["code"] == "the-code"
    assert "code_verifier" in form and form["client_secret"] == "csecret"


def test_installed_flow_via_loopback_listener(tmp_path):
    import threading
    import urllib.request

    google = FakeGoogle()
    store = oauth.TokenStore(tmp_path / "tok.json")
    port = 18401

    def browser(url: str) -> None:
        state = dict(p.split("=") for p in url.split("?")[1].split("&"))["state"]

        def hit() -> None:
            for _ in range(50):
                try:
                    urllib.request.urlopen(  # noqa: S310 - local test server
                        f"http://127.0.0.1:{port}/?state={state}&code=c2", timeout=2).read()
                    return
                except OSError:
                    time.sleep(0.05)
        threading.Thread(target=hit, daemon=True).start()

    with httpx.Client(transport=google.transport) as http:
        token = oauth.installed_flow(CLIENT, store, port=port, http=http, open_url=browser,
                                     timeout_s=5)
    assert token.refresh_token == "rt-1" and google.tokens[0]["code"] == "c2"


def test_exchange_without_refresh_token_is_an_error():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "at"})
    with httpx.Client(transport=httpx.MockTransport(handle)) as http, \
            pytest.raises(oauth.OAuthError, match="no refresh token"):
        oauth.exchange_code(http, CLIENT, code="c", redirect_uri="r", code_verifier="v")


# -- Data API client ------------------------------------------------------------------------

def test_client_refreshes_expired_token_and_uploads_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(yt, "CHUNK", 1024)
    google = FakeGoogle(chunk_308=True)
    store = oauth.TokenStore(tmp_path / "tok.json")
    store.save(oauth.Token(refresh_token="rt-1", access_token="", expires_at=0))
    video = tmp_path / "v.mp4"
    payload = os.urandom(2500)
    video.write_bytes(payload)
    api = yt.YouTubeClient(CLIENT, store, http=httpx.Client(transport=google.transport))
    assert api.my_channel()["title"] == "Rökkur"
    assert google.refreshes == 1 and store.load().access_token == "at-refreshed"
    body = {"snippet": {"title": "t"}, "status": {"privacyStatus": "private"}}
    assert api.upload_video(video, body)["id"] == "vid123"
    assert bytes(google.received) == payload and google.uploads[0]["length"] == 2500
    api.set_thumbnail("vid123", video)
    assert google.thumbnails == ["vid123"]
    assert api.quota_used == 1 + 1600 + 50


def test_client_surfaces_api_errors_with_reason(tmp_path):
    google = FakeGoogle(fail_upload=0)
    api = yt.YouTubeClient(CLIENT, signed_in_store(tmp_path),
                           http=httpx.Client(transport=google.transport))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 10)
    with pytest.raises(yt.YouTubeError) as exc:
        api.upload_video(video, {"snippet": {}, "status": {}})
    assert exc.value.status == 403 and exc.value.reason == "quotaExceeded"


# -- publishing service ---------------------------------------------------------------------

def enable_youtube(settings, tmp_path, *, allow_public=False):
    settings.youtube.enabled = True
    settings.youtube.allow_public = allow_public
    settings.youtube.secrets_dir = tmp_path / "secrets"
    write_client_file(settings.youtube.client_secret_path)
    signed_in_store(tmp_path)


def test_upload_publishes_project_and_audits(ctx, settings, sample_video, tmp_path):
    enable_youtube(settings, tmp_path)
    pid = create(ctx, sample_video)
    run(ctx)
    google = FakeGoogle()
    api = yt.YouTubeClient(CLIENT, signed_in_store(tmp_path),
                           http=httpx.Client(transport=google.transport))
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        pub = publishing.upload(s, project, settings=settings, store=ctx.store, client=api,
                                actor="test")
        assert pub.status == "uploaded" and pub.youtube_video_id == "vid123"
        assert pub.dry_run is False and pub.error is None
    assert status(ctx, pid) == "PUBLISHED"
    sent = google.uploads[0]["body"]
    assert sent["status"]["privacyStatus"] == "private"
    assert sent["status"]["containsSyntheticMedia"] is True
    assert sent["snippet"]["title"].endswith("#shorts")
    assert google.thumbnails == ["vid123"]
    ev = events(ctx, pid)
    assert "VIDEO_PUBLISHED" in ev
    with ctx.db.session() as s:
        from sqlalchemy import select

        from rokkur_studio.db.models import CostEntry
        quota = s.scalars(select(CostEntry).where(CostEntry.kind == "youtube_quota")).one()
        assert quota.amount == 1650


def test_upload_failure_returns_project_to_ready(ctx, settings, sample_video, tmp_path):
    enable_youtube(settings, tmp_path)
    pid = create(ctx, sample_video)
    run(ctx)
    google = FakeGoogle(fail_upload=1)
    api = yt.YouTubeClient(CLIENT, signed_in_store(tmp_path),
                           http=httpx.Client(transport=google.transport))
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        with pytest.raises(yt.YouTubeError):
            publishing.upload(s, project, settings=settings, store=ctx.store, client=api)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        from sqlalchemy import select

        from rokkur_studio.db.models import Publication
        pub = s.scalars(select(Publication).where(Publication.project_id == pid)).one()
        assert pub.status == "failed" and pub.error["status"] == 500


def test_upload_gates(ctx, settings, sample_video, tmp_path):
    pid = create(ctx, sample_video)
    run(ctx)
    google = FakeGoogle()
    api = yt.YouTubeClient(CLIENT, signed_in_store(tmp_path),
                           http=httpx.Client(transport=google.transport))
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        with pytest.raises(publishing.PublishGateError, match="disabled"):
            publishing.upload(s, project, settings=settings, store=ctx.store, client=api)
        enable_youtube(settings, tmp_path)
        with pytest.raises(publishing.PublishGateError, match="public uploads are disabled"):
            publishing.resolve_privacy(settings, project, "public")
        settings.youtube.allow_public = True
        settings.youtube.default_privacy = "public"
        with pytest.raises(publishing.PublishGateError, match="explicitly"):
            publishing.resolve_privacy(settings, project, None)
        assert publishing.resolve_privacy(settings, project, "public") == "public"
        assert publishing.resolve_privacy(settings, project, "unlisted") == "unlisted"
    assert google.uploads == []  # nothing left the building
    assert status(ctx, pid) == "READY_TO_PUBLISH"


def test_make_client_needs_sign_in(settings, tmp_path):
    settings.youtube.secrets_dir = tmp_path / "secrets"
    with pytest.raises(publishing.PublishGateError, match="disabled"):
        publishing.make_client(settings)
    settings.youtube.enabled = True
    with pytest.raises(oauth.OAuthError, match="not found"):
        publishing.make_client(settings)
    write_client_file(settings.youtube.client_secret_path)
    with pytest.raises(oauth.OAuthError, match="youtube-auth"):
        publishing.make_client(settings)
    signed_in_store(tmp_path)
    publishing.make_client(settings).close()


def test_api_real_publish_uses_injected_client(ctx, settings, sample_video, tmp_path):
    from fastapi.testclient import TestClient

    from rokkur_studio.api.app import create_app

    enable_youtube(settings, tmp_path)
    pid = create(ctx, sample_video)
    run(ctx)
    google = FakeGoogle()
    ctx.extras["youtube_client"] = yt.YouTubeClient(
        CLIENT, signed_in_store(tmp_path), http=httpx.Client(transport=google.transport))
    client = TestClient(create_app(ctx=ctx))
    r = client.post(f"/projects/{pid}/publish", json={"dry_run": False, "privacy": "public"})
    assert r.status_code == 409 and "public uploads are disabled" in r.json()["detail"]
    r = client.post(f"/projects/{pid}/publish", json={"dry_run": False, "privacy": "unlisted"})
    assert r.status_code == 200, r.text
    assert r.json()["youtube_video_id"] == "vid123" and r.json()["dry_run"] is False
    assert google.uploads[0]["body"]["status"]["privacyStatus"] == "unlisted"
    assert status(ctx, pid) == "PUBLISHED"
