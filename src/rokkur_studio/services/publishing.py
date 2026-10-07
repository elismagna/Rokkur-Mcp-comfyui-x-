"""Publication metadata and the YouTube ``videos.insert`` dry-run.

Real uploads (OAuth + resumable upload) are Phase 6 and deliberately not implemented here.
Field names and limits follow the YouTube Data API v3 ``videos`` resource documentation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from rokkur_studio.db.models import Project, Publication, RightsDecision
from rokkur_studio.domain.states import ProjectStatus
from rokkur_studio.services.assets import project_assets
from rokkur_studio.services.events import EventType, record_event
from rokkur_studio.services.projects import latest_document, rights_approved

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


def dry_run(session: Session, project: Project, *, privacy: str | None = None,
            publish_at: datetime | None = None, playlist_id: str | None = None,
            actor: str = "api") -> Publication:
    """Validate everything a real upload needs and record the exact request, without uploading."""
    check_publish_gates(session, project)
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
                       "warnings": meta_doc.data.get("warnings", [])})
    return pub
