"""Scheduled releases, playlists and publish approvals (autonomy level 3)."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from rokkur_studio.api.app import create_app
from rokkur_studio.config import YoutubeSection
from rokkur_studio.db.models import ApprovalRequest, CostEntry, Event, Publication
from rokkur_studio.services import commands, publishing
from rokkur_studio.services.projects import get_project, latest_document, save_document
from rokkur_studio.youtube import client as yt
from tests.fakes_youtube import FakeGoogle
from tests.test_pipeline import create, events, run, status
from tests.test_youtube import CLIENT, enable_youtube, signed_in_store

OSLO = ZoneInfo("Europe/Oslo")


def fake_client(tmp_path, google: FakeGoogle) -> yt.YouTubeClient:
    return yt.YouTubeClient(CLIENT, signed_in_store(tmp_path),
                            http=httpx.Client(transport=google.transport))


def schedule_settings(settings, tmp_path, *, times=("18:00",)):
    enable_youtube(settings, tmp_path, allow_public=True)
    settings.youtube.timezone = "Europe/Oslo"
    settings.youtube.release_times = list(times)


def finished(ctx, sample_video) -> str:
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    return pid


def quota(ctx) -> float:
    with ctx.db.session() as s:
        return sum(c.amount for c in s.scalars(select(CostEntry)
                                               .where(CostEntry.kind == "youtube_quota")))


# -- settings and client ----------------------------------------------------------------------

def test_release_settings_are_validated():
    assert YoutubeSection(release_times=[1080, "9:30", "18:00"]).release_times == ["09:30", "18:00"]
    assert YoutubeSection(release_times="21:00, 7:05").release_times == ["07:05", "21:00"]
    for bad in ({"timezone": "Mars/Olympus"}, {"release_times": ["25:00"]},
                {"default_playlist_id": "not a playlist"}):
        with pytest.raises(ValueError):
            YoutubeSection(**bad)


def test_client_pages_through_playlists_and_adds_items(tmp_path):
    google = FakeGoogle(playlist_page_size=2)
    api = fake_client(tmp_path, google)
    lists = api.my_playlists()
    assert [p["title"] for p in lists] == ["claymation", "Drafts", "Shorts"]  # A to Z
    assert lists[2] == {"id": "PLshorts0001", "title": "Shorts", "videos": 12,
                        "privacy": "public"}
    assert google.playlist_pages == 2
    api.add_to_playlist("PLshorts0001", "vid9")
    assert google.playlist_items == [{"playlistId": "PLshorts0001", "resourceId": {
        "kind": "youtube#video", "videoId": "vid9"}}]
    assert api.quota_used == 2 + 50


# -- schedule rules ---------------------------------------------------------------------------

def test_scheduled_release_rules(settings, tmp_path):
    now = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
    later = now + timedelta(hours=3, seconds=42)
    with pytest.raises(publishing.PublishGateError, match="allow_public"):
        publishing.resolve_schedule(settings, "private", later, now=now)
    settings.youtube.allow_public = True
    with pytest.raises(publishing.PublishGateError, match="time zone"):
        publishing.resolve_schedule(settings, "private", later.replace(tzinfo=None), now=now)
    with pytest.raises(publishing.PublishGateError, match="choose private"):
        publishing.resolve_schedule(settings, "unlisted", later, now=now)
    with pytest.raises(publishing.PublishGateError, match="at least 30 minutes"):
        publishing.resolve_schedule(settings, "private", now + timedelta(minutes=20), now=now)
    when = publishing.resolve_schedule(settings, "private", later.astimezone(OSLO), now=now)
    assert when == datetime(2026, 10, 7, 15, 0, tzinfo=UTC) and when.tzinfo == UTC
    assert publishing.resolve_schedule(settings, "public", None) is None


def test_times_read_in_the_studio_zone(settings):
    settings.youtube.timezone = "Europe/Oslo"
    when = publishing.parse_when("2026-10-09 18:00", settings)
    assert publishing.iso(when) == "2026-10-09T16:00:00Z"  # CEST is UTC+2
    assert publishing.iso(publishing.parse_when("2026-10-09T18:00Z", settings)) == \
        "2026-10-09T18:00:00Z"
    assert publishing.local_label(settings, when) == "Fri 9 Oct 18:00 CEST"
    with pytest.raises(publishing.PublishGateError, match="YYYY-MM-DD"):
        publishing.parse_when("tomorrow evening", settings)


def test_next_release_slot_skips_taken_and_too_soon(ctx, settings, sample_video, tmp_path):
    schedule_settings(settings, tmp_path, times=("09:00", "18:00"))
    pid = create(ctx, sample_video, autostart=False)
    now = datetime(2026, 10, 7, 15, 45, tzinfo=UTC)  # 17:45 in Oslo
    with ctx.db.transaction() as s:
        # 18:00 today is only 15 minutes away (lead is 30), so tomorrow 09:00 comes next
        assert publishing.next_release_slot(s, settings, now=now) == \
            datetime(2026, 10, 8, 7, 0, tzinfo=UTC)
        s.add(ApprovalRequest(kind="publish", summary="x", requested_by="t",
                              payload={"publish_at": "2026-10-08T07:00:00Z"}))
        s.flush()
        assert publishing.next_release_slot(s, settings, now=now) == \
            datetime(2026, 10, 8, 16, 0, tzinfo=UTC)
        # a video's own waiting request does not block the time it was offered
        s.add(ApprovalRequest(project_id=pid, kind="publish", summary="y", requested_by="t",
                              payload={"publish_at": "2026-10-08T16:00:00Z"}))
        s.flush()
        assert publishing.next_release_slot(s, settings, now=now) == \
            datetime(2026, 10, 9, 7, 0, tzinfo=UTC)
        assert publishing.next_release_slot(s, settings, now=now, for_project=pid) == \
            datetime(2026, 10, 8, 16, 0, tzinfo=UTC)
        # winter time from 25 October: 18:00 in Oslo is 17:00 UTC
        late = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
        assert publishing.next_release_slot(s, settings, now=late) == \
            datetime(2026, 10, 30, 17, 0, tzinfo=UTC)
        settings.youtube.allow_public = False
        assert publishing.next_release_slot(s, settings, now=now) is None


# -- uploads ----------------------------------------------------------------------------------

def test_scheduled_upload_into_a_playlist(ctx, settings, sample_video, tmp_path):
    schedule_settings(settings, tmp_path)
    pid = finished(ctx, sample_video)
    google = FakeGoogle()
    when = datetime.now(UTC) + timedelta(days=1)
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        pub = publishing.upload(s, project, settings=settings, store=ctx.store,
                                client=fake_client(tmp_path, google), publish_at=when,
                                playlist_id="PLshorts0001", actor="test")
        assert pub.status == "uploaded" and pub.error is None
    sent = google.uploads[0]["body"]["status"]
    assert sent["privacyStatus"] == "private"
    assert sent["publishAt"] == publishing.iso(when.replace(second=0, microsecond=0))
    assert google.playlist_items[0]["resourceId"]["videoId"] == "vid123"
    assert status(ctx, pid) == "PUBLISHED"
    assert quota(ctx) == 1600 + 50 + 50
    with ctx.db.session() as s:
        ev = s.scalars(select(Event).where(Event.type == "VIDEO_PUBLISHED")).one()
        assert ev.data["publish_at"] == sent["publishAt"]
        assert ev.data["playlist_id"] == "PLshorts0001" and ev.data["playlist_error"] is None


def test_playlist_failure_is_only_a_warning(ctx, settings, sample_video, tmp_path):
    enable_youtube(settings, tmp_path)
    pid = finished(ctx, sample_video)
    google = FakeGoogle(fail_playlist_item=True)
    with ctx.db.transaction() as s:
        project = get_project(s, pid, for_update=True)
        pub = publishing.upload(s, project, settings=settings, store=ctx.store,
                                client=fake_client(tmp_path, google),
                                playlist_id="PLgone000000")
        assert pub.status == "uploaded" and pub.youtube_video_id == "vid123"
        assert pub.error["message"].startswith("not added to the playlist: ")
        assert pub.error["playlist"]["reason"] == "playlistNotFound"
    assert status(ctx, pid) == "PUBLISHED"


def test_api_scheduling_counts_as_public(ctx, settings, sample_video, tmp_path):
    """A scheduled release goes public, so it is refused while public uploads are off."""
    enable_youtube(settings, tmp_path)
    pid = finished(ctx, sample_video)
    google = FakeGoogle()
    ctx.extras["youtube_client"] = fake_client(tmp_path, google)
    c = TestClient(create_app(ctx=ctx))
    at = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    for dry in (True, False):
        r = c.post(f"/projects/{pid}/publish", json={"dry_run": dry, "privacy": "private",
                                                     "publish_at": at})
        assert r.status_code == 409 and "allow_public" in r.json()["detail"], r.text
    assert google.uploads == [] and status(ctx, pid) == "READY_TO_PUBLISH"
    settings.youtube.allow_public = True
    r = c.post(f"/projects/{pid}/publish", json={"dry_run": False, "publish_at": at,
                                                 "playlist_id": "PLclaymation1"})
    assert r.status_code == 200, r.text
    assert google.uploads[0]["body"]["status"]["publishAt"].endswith("Z")
    assert google.playlist_items[0]["playlistId"] == "PLclaymation1"


def test_api_upload_failure_keeps_the_audit_trail(ctx, settings, sample_video, tmp_path):
    enable_youtube(settings, tmp_path)
    pid = finished(ctx, sample_video)
    ctx.extras["youtube_client"] = fake_client(tmp_path, FakeGoogle(fail_upload=1))
    r = TestClient(create_app(ctx=ctx)).post(f"/projects/{pid}/publish",
                                             json={"dry_run": False})
    assert r.status_code == 502 and "backend" in r.json()["detail"]
    with ctx.db.session() as s:
        pub = s.scalars(select(Publication).where(Publication.project_id == pid)).one()
        assert pub.status == "failed" and pub.error["status"] == 500
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    assert events(ctx, pid).count("STATE_CHANGED") > 0 and quota(ctx) == 1600


# -- proposals (autonomy level 3) -------------------------------------------------------------

def pending(ctx, pid) -> list[ApprovalRequest]:
    with ctx.db.session() as s:
        return list(s.scalars(select(ApprovalRequest).where(
            ApprovalRequest.project_id == pid, ApprovalRequest.kind == "publish")))


def test_level3_proposes_and_approval_uploads_as_planned(ctx, settings, sample_video, tmp_path):
    settings.studio.autonomy_level = 3
    schedule_settings(settings, tmp_path)
    settings.youtube.default_playlist_id = "PLshorts0001"
    google = FakeGoogle()
    publishing.refresh_playlists(settings, fake_client(tmp_path, google))
    pid = finished(ctx, sample_video)
    [req] = pending(ctx, pid)
    assert req.status == "pending" and req.requested_by == "channel_manager"
    plan = req.payload
    assert plan["privacy"] == "private" and plan["playlist_title"] == "Shorts"
    at = publishing.parse_iso(plan["publish_at"])
    assert at.astimezone(OSLO).strftime("%H:%M") == "18:00" and at > datetime.now(UTC)
    assert "goes public" in req.summary and "Shorts" in req.summary
    assert "APPROVAL_REQUESTED" in events(ctx, pid) and google.uploads == []
    with ctx.db.session() as s:  # the plan was checked with a dry run before asking
        assert s.get(Publication, plan["publication_id"]).dry_run is True

    ctx.extras["youtube_client"] = fake_client(tmp_path, google)
    c = TestClient(create_app(ctx=ctx))
    page = c.get("/ui/approvals").text
    assert "Approve and upload" in page and "goes public" in page
    r = c.post(f"/ui/approvals/{req.id}/approve", follow_redirects=False)
    assert r.status_code == 303 and "msg=Uploaded" in r.headers["location"]
    assert google.uploads[0]["body"]["status"]["publishAt"] == plan["publish_at"]
    assert google.playlist_items[0]["playlistId"] == "PLshorts0001"
    assert status(ctx, pid) == "PUBLISHED"
    [req] = pending(ctx, pid)
    assert req.status == "approved" and req.decided_by == "dashboard"
    assert "watch?v=vid123" in req.note
    # the next finished video gets the next free slot: this one is taken now
    with ctx.db.session() as s:
        nxt = publishing.next_release_slot(s, settings).astimezone(OSLO)
    assert nxt.date() == at.astimezone(OSLO).date() + timedelta(days=1)
    assert nxt.strftime("%H:%M") == "18:00"


def test_rejecting_a_proposal_leaves_the_video_ready(ctx, settings, sample_video, tmp_path):
    settings.studio.autonomy_level = 3
    enable_youtube(settings, tmp_path)  # no release times: a plain private upload is proposed
    pid = finished(ctx, sample_video)
    [req] = pending(ctx, pid)
    assert req.payload["publish_at"] is None and req.payload["privacy"] == "private"
    c = TestClient(create_app(ctx=ctx))
    r = c.post(f"/approvals/{req.id}", json={"approve": False, "note": "not this one"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.transaction() as s, pytest.raises(ValueError, match="approve_proposal"):
        fresh = ApprovalRequest(project_id=pid, kind="publish", summary="x", requested_by="t")
        s.add(fresh)
        s.flush()  # the generic approve path would mark it approved without uploading
        commands.decide_approval(s, fresh, settings, approve=True, decided_by="t", note=None)


def test_api_approval_uploads_and_manual_upload_supersedes(ctx, settings, sample_video,
                                                            tmp_path):
    settings.studio.autonomy_level = 3
    enable_youtube(settings, tmp_path)
    google = FakeGoogle()
    ctx.extras["youtube_client"] = fake_client(tmp_path, google)
    c = TestClient(create_app(ctx=ctx))
    first, second = finished(ctx, sample_video), finished(ctx, sample_video)
    [req1], [req2] = pending(ctx, first), pending(ctx, second)
    r = c.post(f"/approvals/{req1.id}", json={"approve": True, "decided_by": "elis"})
    assert r.status_code == 200 and r.json()["status"] == "approved", r.text
    assert status(ctx, first) == "PUBLISHED"
    # publishing the second one by hand closes its proposal instead of leaving it dangling
    r = c.post(f"/ui/projects/{second}/publish", data={"mode": "upload", "privacy": "unlisted"},
               follow_redirects=False)
    assert "msg=Uploaded" in r.headers["location"]
    [req2] = pending(ctx, second)
    assert req2.status == "superseded" and len(google.uploads) == 2
    r = c.post(f"/approvals/{req2.id}", json={"approve": True})
    assert r.status_code == 409 and "already superseded" in r.json()["detail"]


def test_no_proposal_when_metadata_needs_a_decision(ctx, settings, sample_video, tmp_path):
    enable_youtube(settings, tmp_path)
    pid = finished(ctx, sample_video)  # level 2: nothing proposed
    assert pending(ctx, pid) == []
    with ctx.db.transaction() as s:
        meta = latest_document(s, pid, "metadata").data
        save_document(s, pid, "metadata", {**meta, "warnings": ["too long for a Short"]},
                      created_by="test")
        with pytest.raises(publishing.PublishGateError, match="too long for a Short"):
            publishing.propose(s, get_project(s, pid), settings)


def test_a_stale_release_time_is_not_approved(ctx, settings, sample_video, tmp_path):
    settings.studio.autonomy_level = 3
    schedule_settings(settings, tmp_path)
    pid = finished(ctx, sample_video)
    [req] = pending(ctx, pid)
    google = FakeGoogle()
    with ctx.db.transaction() as s:
        r = s.get(ApprovalRequest, req.id)
        r.payload = {**r.payload, "publish_at": publishing.iso(datetime.now(UTC)
                                                               - timedelta(hours=1))}
        with pytest.raises(publishing.PublishGateError, match="has passed"):
            publishing.approve_proposal(s, r, settings=settings, store=ctx.store,
                                        client=fake_client(tmp_path, google), decided_by="t")
    assert google.uploads == [] and status(ctx, pid) == "READY_TO_PUBLISH"


# -- dashboard --------------------------------------------------------------------------------

def test_dashboard_schedules_in_the_studio_zone_without_script(ctx, settings, sample_video,
                                                               tmp_path):
    schedule_settings(settings, tmp_path)
    pid = finished(ctx, sample_video)
    google = FakeGoogle()
    ctx.extras["youtube_client"] = fake_client(tmp_path, google)
    c = TestClient(create_app(ctx=ctx))
    r = c.post("/ui/youtube/playlists", follow_redirects=False)
    assert "msg=Loaded+3+playlists" in r.headers["location"].replace("%20", "+")
    page = c.get(f"/ui/projects/{pid}").text
    assert 'name="release" value="schedule"' in page and "Next free release time" in page
    assert '<option value="PLshorts0001"' in page
    yt_page = c.get("/ui/youtube").text
    assert "Release times" in yt_page and "Europe/Oslo" in yt_page and "claymation" in yt_page
    local = (datetime.now(OSLO) + timedelta(days=3)).replace(hour=20, minute=30)
    r = c.post(f"/ui/projects/{pid}/publish", follow_redirects=False, data={
        "mode": "upload", "release": "schedule", "publish_local": local.strftime("%Y-%m-%dT%H:%M"),
        "playlist_id": "PLclaymation1"})
    assert "msg=Uploaded" in r.headers["location"], r.headers["location"]
    sent = google.uploads[0]["body"]["status"]
    assert sent["privacyStatus"] == "private"
    assert sent["publishAt"] == publishing.iso(local.replace(second=0, microsecond=0))
    page = c.get(f"/ui/projects/{pid}").text
    assert "goes public" in page and "20:30" in page
    assert "20:30" in c.get("/ui/youtube").text  # listed under Scheduled


def test_dashboard_refuses_schedule_while_public_is_off(ctx, settings, sample_video, tmp_path):
    enable_youtube(settings, tmp_path)
    pid = finished(ctx, sample_video)
    google = FakeGoogle()
    ctx.extras["youtube_client"] = fake_client(tmp_path, google)
    c = TestClient(create_app(ctx=ctx))
    page = c.get(f"/ui/projects/{pid}").text
    assert 'value="schedule" disabled' in page and "public uploads are off" in page
    when = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    r = c.post(f"/ui/projects/{pid}/publish", follow_redirects=False,
               data={"mode": "upload", "release": "schedule", "publish_at": when})
    assert "err=" in r.headers["location"] and google.uploads == []


def test_a_failed_proposal_does_not_fail_the_video(ctx, settings, sample_video, tmp_path,
                                                   monkeypatch):
    settings.studio.autonomy_level = 3
    enable_youtube(settings, tmp_path)

    def refuse(*a, **kw):
        raise publishing.PublishGateError("no free slot")

    monkeypatch.setattr(publishing, "propose", refuse)
    pid = finished(ctx, sample_video)
    assert pending(ctx, pid) == []
    with ctx.db.session() as s:
        ev = s.scalars(select(Event).where(Event.type == "PUBLISH_PROPOSAL_SKIPPED")).one()
        assert ev.project_id == pid and ev.data == {"reason": "no free slot"}
