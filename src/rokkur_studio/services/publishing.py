"""Publication metadata, the YouTube ``videos.insert`` dry-run, and the real upload.

Field names and limits follow the YouTube Data API v3 ``videos`` resource documentation.
A real upload only happens when someone asks for it explicitly (CLI ``publish``, the API
with ``dry_run: false``, the dashboard's Upload button, or approving a publish request);
nothing in the pipeline uploads on its own. At autonomy level 3+ the pipeline only *proposes*
an upload (an approval request with the full plan); a person approves it.

A scheduled release uploads the video as private with ``status.publishAt``; YouTube makes it
public at that time, so scheduling counts as a public upload for ``youtube.allow_public``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.config import PLAYLIST_ID, Settings
from rokkur_studio.db.models import (
    ApprovalRequest,
    CostEntry,
    Project,
    Publication,
    RightsDecision,
    utcnow,
)
from rokkur_studio.domain.states import ProjectStatus
from rokkur_studio.services.assets import project_assets
from rokkur_studio.services.events import EventType, record_event
from rokkur_studio.services.projects import (
    get_project,
    latest_document,
    rights_approved,
    transition,
)
from rokkur_studio.storage.base import AssetStore
from rokkur_studio.youtube.client import YouTubeClient, YouTubeError, video_url
from rokkur_studio.youtube.oauth import OAuthClient, OAuthError, TokenStore

log = logging.getLogger(__name__)

TITLE_MAX = 100
DESCRIPTION_MAX_BYTES = 5000
TAGS_MAX_CHARS = 500
SHORTS_MAX_SECONDS = 180
CATEGORY_FILM_ANIMATION = "1"


class PublishGateError(ValueError):
    pass


def _clean(text: str) -> str:
    return text.replace("<", "").replace(">", "").strip()


def draft_metadata(project: Project, brief: dict[str, Any], rights: RightsDecision | None,
                   *, duration: float) -> dict[str, Any]:
    theme = brief.get("theme") or project.creative_input.get("theme") or project.name
    title = _clean(project.creative_input.get("title") or f"{project.name} | {theme}")
    if project.target_format == "youtube_short" and "#shorts" not in title.lower():
        title = f"{title} #shorts"
    lines = [project.creative_input.get("description") or f"{theme}.",
             "", "Made with AI-assisted video generation (Rökkur Studio)."]
    if rights is not None and rights.attribution_required and rights.attribution_text:
        lines += ["", f"Source: {rights.attribution_text}"]
    tags = list(dict.fromkeys(
        [t.strip() for t in (project.creative_input.get("tags") or []) if t.strip()]
        + [w for w in theme.lower().replace(",", " ").split() if len(w) > 3][:8]))
    warnings = []
    if project.target_format == "youtube_short" and duration > SHORTS_MAX_SECONDS:
        warnings.append(f"duration {duration:.1f}s exceeds the {SHORTS_MAX_SECONDS}s Shorts limit")
    return {
        "title": title[:TITLE_MAX],
        "description": "\n".join(lines),
        "tags": tags,
        "category_id": CATEGORY_FILM_ANIMATION,
        "made_for_kids": bool(project.creative_input.get("made_for_kids", False)),
        "contains_synthetic_media": True,
        "default_language": project.creative_input.get("language", "en"),
        "warnings": warnings,
    }


def apply_draft(meta: dict[str, Any], draft: dict[str, Any], *, target_format: str,
                rights: RightsDecision | None) -> dict[str, Any]:
    """Overlay a model-written draft on rule-based metadata, keeping the parts policy fixes."""
    title = _clean(draft.get("title") or meta["title"])
    if target_format == "youtube_short" and "#shorts" not in title.lower():
        title = f"{title} #shorts"
    lines = [_clean(line) for line in (draft.get("description") or "").splitlines()]
    body = "\n".join(line for line in lines if line) or meta["description"].split("\n")[0]
    lines = [body, "", "Made with AI-assisted video generation (Rökkur Studio)."]
    if rights is not None and rights.attribution_required and rights.attribution_text:
        lines += ["", f"Source: {rights.attribution_text}"]
    tags = list(dict.fromkeys(_clean(t).lower() for t in draft.get("tags") or [] if _clean(t)))
    out = {**meta, "title": title, "description": "\n".join(lines), "tags": tags or meta["tags"]}
    if validate_metadata(out):
        return meta  # a draft that breaks YouTube's limits is dropped, not trimmed blindly
    return out


def validate_metadata(meta: dict[str, Any]) -> list[str]:
    errors = []
    if not meta.get("title"):
        errors.append("title is required")
    elif len(meta["title"]) > TITLE_MAX:
        errors.append(f"title longer than {TITLE_MAX} characters")
    if "<" in meta.get("title", "") + meta.get("description", "") or ">" in meta.get(
            "title", "") + meta.get("description", ""):
        errors.append("title/description may not contain < or >")
    if len(meta.get("description", "").encode("utf-8")) > DESCRIPTION_MAX_BYTES:
        errors.append(f"description longer than {DESCRIPTION_MAX_BYTES} bytes")
    tag_chars = sum(len(t) + (2 if " " in t else 0) for t in meta.get("tags", [])) + max(
        0, len(meta.get("tags", [])) - 1)
    if tag_chars > TAGS_MAX_CHARS:
        errors.append(f"tags exceed {TAGS_MAX_CHARS} characters")
    return errors


def build_insert_request(meta: dict[str, Any], *, privacy: str,
                         publish_at: datetime | None = None,
                         playlist_id: str | None = None) -> dict[str, Any]:
    if publish_at is not None and privacy != "private":
        raise PublishGateError("scheduled publishing requires privacyStatus=private")
    status: dict[str, Any] = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": meta["made_for_kids"],
        "containsSyntheticMedia": meta["contains_synthetic_media"],
    }
    if publish_at is not None:
        status["publishAt"] = publish_at.isoformat().replace("+00:00", "Z")
    request: dict[str, Any] = {
        "method": "videos.insert",
        "part": "snippet,status",
        "uploadType": "resumable",
        "body": {
            "snippet": {
                "title": meta["title"],
                "description": meta["description"],
                "tags": meta["tags"],
                "categoryId": meta["category_id"],
                "defaultLanguage": meta.get("default_language", "en"),
            },
            "status": status,
        },
    }
    if playlist_id:
        request["then"] = [{"method": "playlistItems.insert", "part": "snippet",
                            "body": {"snippet": {"playlistId": playlist_id,
                                                 "resourceId": {"kind": "youtube#video"}}}}]
    request["then_thumbnail"] = {"method": "thumbnails.set"}
    return request


def check_publish_gates(session: Session, project: Project) -> None:
    if project.status != ProjectStatus.READY_TO_PUBLISH:
        raise PublishGateError(f"project is {project.status}, not READY_TO_PUBLISH")
    if not rights_approved(session, project.id):
        raise PublishGateError("rights are not approved")
    qc = latest_document(session, project.id, "qc_report")
    if qc is None or qc.data.get("decision") != "PASS":
        raise PublishGateError("latest QC report did not pass")


def check_playlist_id(playlist_id: str | None) -> str | None:
    playlist_id = (playlist_id or "").strip() or None
    if playlist_id and not PLAYLIST_ID.fullmatch(playlist_id):
        raise PublishGateError(f"{playlist_id!r} is not a YouTube playlist id")
    return playlist_id


def resolve_schedule(settings: Settings, privacy: str, publish_at: datetime | None, *,
                     now: datetime | None = None) -> datetime | None:
    """Check a scheduled release: private until ``publish_at``, then public on YouTube's side."""
    if publish_at is None:
        return None
    if publish_at.tzinfo is None:
        raise PublishGateError("the release time needs a time zone")
    if privacy != "private":
        raise PublishGateError("a scheduled release uploads as private and YouTube makes it "
                               "public at the release time; choose private")
    if not settings.youtube.allow_public:
        raise PublishGateError("a scheduled release makes the video public, and public uploads "
                               "are disabled (youtube.allow_public is false)")
    when = publish_at.astimezone(UTC).replace(second=0, microsecond=0)
    lead = settings.youtube.min_lead_minutes
    if when < (now or utcnow()) + timedelta(minutes=lead):
        raise PublishGateError(f"schedule the release at least {lead} minutes from now "
                               f"(asked for {local_label(settings, when)})")
    return when


def dry_run(session: Session, project: Project, *, privacy: str | None = None,
            publish_at: datetime | None = None, playlist_id: str | None = None,
            actor: str = "api", settings: Settings | None = None,
            now: datetime | None = None) -> Publication:
    """Validate everything a real upload needs and record the exact request, without uploading.

    A release time is checked only with ``settings`` (the schedule gates live there).
    """
    check_publish_gates(session, project)
    playlist_id = check_playlist_id(playlist_id)
    meta_doc = latest_document(session, project.id, "metadata")
    if meta_doc is None:
        raise PublishGateError("no metadata drafted")
    errors = validate_metadata(meta_doc.data)
    if errors:
        raise PublishGateError("; ".join(errors))
    finals = project_assets(session, project.id, "final")
    thumbs = project_assets(session, project.id, "thumbnail")
    if not finals:
        raise PublishGateError("no final render")
    privacy = privacy or (project.channel.default_privacy if project.channel else "private")
    if publish_at is not None:
        if settings is None:
            raise ValueError("dry_run needs settings to check a release time")
        publish_at = resolve_schedule(settings, privacy, publish_at, now=now)
    request = build_insert_request(meta_doc.data, privacy=privacy, publish_at=publish_at,
                                   playlist_id=playlist_id)
    pub = Publication(project_id=project.id, dry_run=True, status="prepared", request=request,
                      video_asset_id=finals[-1].id,
                      thumbnail_asset_id=thumbs[-1].id if thumbs else None)
    session.add(pub)
    session.flush()
    record_event(session, EventType.PUBLISH_DRY_RUN, project_id=project.id, actor=actor,
                 data={"publication_id": pub.id, "privacy": privacy,
                       "channel": project.channel_id,
                       "publish_at": iso(publish_at), "playlist_id": playlist_id,
                       "warnings": meta_doc.data.get("warnings", [])})
    return pub


def resolve_privacy(settings: Settings, project: Project, requested: str | None) -> str:
    """Private unless asked otherwise; public needs both the ask and ``youtube.allow_public``."""
    privacy = requested or (project.channel.default_privacy if project.channel
                            else settings.youtube.default_privacy)
    if privacy == "public" and not settings.youtube.allow_public:
        raise PublishGateError("public uploads are disabled (youtube.allow_public is false); "
                               "upload as private or unlisted, or enable it in config/studio.yaml")
    if privacy == "public" and requested != "public":
        raise PublishGateError("a public upload must be requested explicitly (privacy=public)")
    return privacy


def make_client(settings: Settings, *, uploads: bool = True) -> YouTubeClient:
    """Raises OAuthError when the OAuth client file or the sign-in token is missing.

    ``uploads=False`` (reading playlists) works while uploads are still switched off.
    """
    yt = settings.youtube
    if uploads and not yt.enabled:
        raise PublishGateError("YouTube publishing is disabled (youtube.enabled is false)")
    client = OAuthClient.load(yt.client_secret_path)
    store = TokenStore(yt.token_path)
    if not store.exists():
        raise OAuthError(f"not signed in to YouTube ({yt.token_path} missing); "
                         "run: rokkur-studio youtube-auth")
    return YouTubeClient(client, store)


@contextmanager
def youtube_client(settings: Settings, injected: Any = None, *,
                   uploads: bool = True) -> Iterator[YouTubeClient]:
    """The injected client (tests, dashboard extras) as is, or a fresh one that is closed after."""
    if injected is not None:
        yield injected
        return
    client = make_client(settings, uploads=uploads)
    try:
        yield client
    finally:
        client.close()


def upload(session: Session, project: Project, *, settings: Settings, store: AssetStore,
           client: YouTubeClient, privacy: str | None = None,
           publish_at: datetime | None = None, playlist_id: str | None = None,
           actor: str = "api") -> Publication:
    """The real thing: gates, then ``videos.insert`` + ``thumbnails.set`` (+ the playlist).

    The project moves READY_TO_PUBLISH -> PUBLISHING -> PUBLISHED, or back to
    READY_TO_PUBLISH with the error recorded on the publication and in an event.
    A thumbnail or playlist failure is a warning on the publication: the video is up.
    The session is flushed, not committed; the caller owns the transaction (and must commit
    on a YouTubeError too, or the failed attempt's audit trail is lost).
    """
    if not settings.youtube.enabled:
        raise PublishGateError("YouTube publishing is disabled (youtube.enabled is false)")
    check_publish_gates(session, project)
    meta_doc = latest_document(session, project.id, "metadata")
    if meta_doc is None:
        raise PublishGateError("no metadata drafted")
    errors = validate_metadata(meta_doc.data)
    if errors:
        raise PublishGateError("; ".join(errors))
    if meta_doc.data.get("warnings"):
        raise PublishGateError("metadata has warnings that need a decision: "
                               + "; ".join(meta_doc.data["warnings"]))
    finals = project_assets(session, project.id, "final")
    thumbs = project_assets(session, project.id, "thumbnail")
    if not finals:
        raise PublishGateError("no final render")
    privacy = resolve_privacy(settings, project, privacy)
    publish_at = resolve_schedule(settings, privacy, publish_at)
    playlist_id = check_playlist_id(playlist_id)
    request = build_insert_request(meta_doc.data, privacy=privacy, publish_at=publish_at,
                                   playlist_id=playlist_id)
    video_path = store.path_for(finals[-1].rel_path)
    if not video_path.is_file():
        raise PublishGateError(f"final render missing on disk: {video_path}")
    pub = Publication(project_id=project.id, dry_run=False, status="uploading", request=request,
                      video_asset_id=finals[-1].id,
                      thumbnail_asset_id=thumbs[-1].id if thumbs else None)
    session.add(pub)
    transition(session, project, ProjectStatus.PUBLISHING, actor=actor,
               data={"publication_id": pub.id, "privacy": privacy})
    session.flush()
    try:
        result = client.upload_video(video_path, request["body"])
    except YouTubeError as exc:
        pub.status = "failed"
        pub.error = exc.to_dict()
        _spend_quota(session, project.id, client)
        transition(session, project, ProjectStatus.READY_TO_PUBLISH, actor=actor,
                   reason=str(exc), data={"publication_id": pub.id})
        session.flush()
        raise
    video_id = result["id"]
    pub.youtube_video_id = video_id
    # The video is up from here on; what follows can only add warnings.
    warnings: dict[str, dict[str, Any]] = {}
    if thumbs:
        try:
            client.set_thumbnail(video_id, store.path_for(thumbs[-1].rel_path))
        except YouTubeError as exc:
            warnings["thumbnail"] = exc.to_dict()
    if playlist_id:
        try:
            client.add_to_playlist(playlist_id, video_id)
        except YouTubeError as exc:
            warnings["playlist"] = exc.to_dict()
    if warnings:
        log.warning("published with warnings", extra={"data": warnings})
    pub.status = "uploaded"
    pub.error = ({"message": "; ".join(f"{_WARNING_LABEL[k]}: {v['message']}"
                                       for k, v in warnings.items()), **warnings}
                 if warnings else None)
    _spend_quota(session, project.id, client)
    transition(session, project, ProjectStatus.PUBLISHED, actor=actor, data={
        "publication_id": pub.id, "youtube_video_id": video_id, "privacy": privacy,
        "url": video_url(video_id), "publish_at": iso(publish_at), "playlist_id": playlist_id,
        "thumbnail_error": warnings.get("thumbnail"), "playlist_error": warnings.get("playlist"),
        "quota_units": client.quota_used})
    for other in pending_proposals(session, project.id):
        other.status, other.decided_by, other.decided_at = "superseded", actor, utcnow()
        other.note = f"published by {actor}"
    session.flush()
    return pub


_WARNING_LABEL = {"thumbnail": "thumbnail not set", "playlist": "not added to the playlist"}


def _spend_quota(session: Session, project_id: str, client: YouTubeClient) -> None:
    if client.quota_used:
        session.add(CostEntry(project_id=project_id, kind="youtube_quota",
                              amount=float(client.quota_used), unit="units"))
        client.quota_used = 0


def publication_path(store: AssetStore, rel: str) -> Path:
    return store.path_for(rel)


# -- release times -------------------------------------------------------------------------
def iso(when: datetime | None) -> str | None:
    return when.astimezone(UTC).isoformat().replace("+00:00", "Z") if when else None


def parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    when = datetime.fromisoformat(text)
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def zone(settings: Settings) -> ZoneInfo:
    return ZoneInfo(settings.youtube.timezone)


def local_label(settings: Settings, when: datetime) -> str:
    """``Thu 9 Oct 18:00 CEST`` in the studio's release time zone."""
    t = when.astimezone(zone(settings))
    return f"{t:%a} {t.day} {t:%b %H:%M} {t.tzname()}"


def parse_when(text: str, settings: Settings) -> datetime:
    """``2026-10-09 18:00`` (studio time zone) or ISO 8601 with an offset or ``Z``."""
    raw = text.strip().replace(" ", "T", 1)
    try:
        when = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise PublishGateError(f"cannot read the time {text!r}; use YYYY-MM-DD HH:MM") from exc
    return when if when.tzinfo else when.replace(tzinfo=zone(settings))


def scheduled_releases(session: Session) -> list[tuple[datetime, Publication]]:
    """Uploads that carry a release time, soonest first (past ones included)."""
    out = []
    for pub in session.scalars(select(Publication).where(Publication.dry_run.is_(False),
                                                         Publication.status == "uploaded")):
        when = parse_iso(pub.request.get("body", {}).get("status", {}).get("publishAt"))
        if when is not None:
            out.append((when, pub))
    return sorted(out, key=lambda x: x[0])


def pending_proposals(session: Session, project_id: str | None = None) -> list[ApprovalRequest]:
    stmt = select(ApprovalRequest).where(ApprovalRequest.kind == "publish",
                                         ApprovalRequest.status == "pending")
    if project_id is not None:
        stmt = stmt.where(ApprovalRequest.project_id == project_id)
    return list(session.scalars(stmt.order_by(ApprovalRequest.requested_at)))


def next_release_slot(session: Session, settings: Settings, *, now: datetime | None = None,
                      for_project: str | None = None) -> datetime | None:
    """The first ``youtube.release_times`` slot that is far enough ahead and not yet taken by
    a scheduled upload or a pending proposal (``for_project``'s own proposal doesn't count).
    None without release times or allow_public."""
    yt = settings.youtube
    if not yt.release_times or not yt.allow_public:
        return None
    now = now or utcnow()
    taken = {when for when, _ in scheduled_releases(session)}
    taken |= {w for r in pending_proposals(session)
              if (for_project is None or r.project_id != for_project)
              and (w := parse_iso(r.payload.get("publish_at"))) is not None}
    earliest = now + timedelta(minutes=yt.min_lead_minutes)
    tz = zone(settings)
    first: date = now.astimezone(tz).date()
    for offset in range(120):
        day = first + timedelta(days=offset)
        for hhmm in yt.release_times:
            h, m = (int(x) for x in hhmm.split(":"))
            when = datetime.combine(day, time(h, m), tzinfo=tz).astimezone(UTC)
            if when >= earliest and when not in taken:
                return when
    return None


# -- playlists -----------------------------------------------------------------------------
def playlists_path(settings: Settings) -> Path:
    return Path(settings.studio.data_dir) / "youtube" / "playlists.json"


def load_playlists(settings: Settings) -> dict[str, Any] | None:
    """The playlists saved by the last refresh: ``{"fetched_at", "items": [...]}``."""
    path = playlists_path(settings)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("items"), list) else None


def refresh_playlists(settings: Settings, client: YouTubeClient) -> dict[str, Any]:
    """Ask YouTube for the channel's playlists and keep the list (ids and titles only)."""
    data = {"fetched_at": iso(utcnow()), "items": client.my_playlists()}
    path = playlists_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return data


def playlist_title(settings: Settings, playlist_id: str | None) -> str | None:
    if not playlist_id:
        return None
    cached = load_playlists(settings) or {"items": []}
    return next((p["title"] for p in cached["items"] if p.get("id") == playlist_id), None)


def find_playlist(settings: Settings, text: str) -> str:
    """A playlist id, or the id of the saved playlist with exactly this title."""
    text = text.strip()
    cached = load_playlists(settings) or {"items": []}
    matches = [p["id"] for p in cached["items"] if p.get("title", "").lower() == text.lower()]
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise PublishGateError(f"more than one playlist is called {text!r}; use its id")
    if PLAYLIST_ID.fullmatch(text):
        return text
    raise PublishGateError(f"no saved playlist called {text!r}; run youtube-playlists first "
                           "or pass the playlist id")


# -- publish proposals (autonomy level 3+) ------------------------------------------------
def propose(session: Session, project: Project, settings: Settings, *,
            actor: str = "channel_manager", now: datetime | None = None) -> ApprovalRequest:
    """Put a finished video up for approval with a complete upload plan; uploads nothing.

    The plan: the channel's default visibility (never public directly), the next free release
    time when release times are set and public uploads are allowed, and the default playlist.
    Raises PublishGateError when the video could not be uploaded as planned.
    """
    existing = pending_proposals(session, project.id)
    if existing:
        return existing[0]
    now = now or utcnow()
    yt = settings.youtube
    meta = latest_document(session, project.id, "metadata")
    if meta is not None and meta.data.get("warnings"):
        raise PublishGateError("metadata has warnings that need a decision: "
                               + "; ".join(meta.data["warnings"]))
    default = project.channel.default_privacy if project.channel else yt.default_privacy
    privacy = "unlisted" if default == "unlisted" else "private"
    publish_at = next_release_slot(session, settings, now=now) if privacy == "private" else None
    playlist_id = yt.default_playlist_id or None
    pub = dry_run(session, project, privacy=privacy, publish_at=publish_at,
                  playlist_id=playlist_id, actor=actor, settings=settings, now=now)
    title = pub.request["body"]["snippet"]["title"]
    ptitle = playlist_title(settings, playlist_id)
    summary = f"Upload \u201c{title}\u201d as {privacy}"
    if publish_at is not None:
        summary += f"; it goes public {local_label(settings, publish_at)}"
    if playlist_id:
        summary += f"; add it to the playlist \u201c{ptitle or playlist_id}\u201d"
    req = ApprovalRequest(project_id=project.id, kind="publish", summary=summary,
                          requested_by=actor, payload={
                              "privacy": privacy, "publish_at": iso(publish_at),
                              "playlist_id": playlist_id, "playlist_title": ptitle,
                              "title": title, "publication_id": pub.id})
    session.add(req)
    session.flush()
    record_event(session, EventType.APPROVAL_REQUESTED, project_id=project.id, actor=actor,
                 data={"kind": "publish", "approval_id": req.id, **req.payload})
    return req


def approve_proposal(session: Session, req: ApprovalRequest, *, settings: Settings,
                     store: AssetStore, client: YouTubeClient, decided_by: str,
                     note: str | None = None) -> Publication:
    """Approving a publish request uploads the video exactly as planned."""
    if req.kind != "publish" or req.project_id is None:
        raise PublishGateError("not a publish request")
    if req.status != "pending":
        raise PublishGateError(f"this request is already {req.status}")
    plan = req.payload or {}
    publish_at = parse_iso(plan.get("publish_at"))
    lead = timedelta(minutes=settings.youtube.min_lead_minutes)
    if publish_at is not None and publish_at < utcnow() + lead:
        raise PublishGateError(f"the planned release time ({local_label(settings, publish_at)}) "
                               "is too close or has passed; reject this request and schedule "
                               "the video from its page")
    project = get_project(session, req.project_id, for_update=True)
    pub = upload(session, project, settings=settings, store=store, client=client,
                 privacy=plan.get("privacy"), publish_at=publish_at,
                 playlist_id=plan.get("playlist_id"), actor=decided_by)
    req.status, req.decided_by, req.decided_at = "approved", decided_by, utcnow()
    req.note = note or f"uploaded: {video_url(pub.youtube_video_id or '')}"
    record_event(session, EventType.APPROVAL_DECIDED, project_id=req.project_id,
                 actor=decided_by, data={"kind": "publish", "approve": True,
                                         "approval_id": req.id, "publication_id": pub.id})
    session.flush()
    return pub
