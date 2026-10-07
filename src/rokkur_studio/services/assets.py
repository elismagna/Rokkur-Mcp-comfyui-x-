"""Register files in the asset store as Asset rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.db.models import Asset
from rokkur_studio.storage.base import AssetStore
from rokkur_studio.storage.local import guess_mime, sha256_file


def register_asset(
    session: Session,
    store: AssetStore,
    project_id: str,
    kind: str,
    path: Path,
    meta: dict[str, Any] | None = None,
) -> Asset:
    """Record a file that already lives inside the store."""
    asset = Asset(
        project_id=project_id,
        kind=kind,
        rel_path=store.rel(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        mime=guess_mime(path),
        meta=meta or {},
    )
    session.add(asset)
    session.flush()
    return asset


def import_file(
    session: Session, store: AssetStore, project_id: str, kind: str, area: str, src: Path,
    name: str | None = None, meta: dict[str, Any] | None = None,
) -> Asset:
    rel = store.put_file(project_id, area, src, name)
    return register_asset(session, store, project_id, kind, store.path_for(rel), meta)


def project_assets(session: Session, project_id: str, kind: str | None = None) -> list[Asset]:
    stmt = select(Asset).where(Asset.project_id == project_id).order_by(Asset.created_at)
    if kind:
        stmt = stmt.where(Asset.kind == kind)
    return list(session.scalars(stmt))
