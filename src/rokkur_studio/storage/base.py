"""Asset store interface. Local filesystem now; S3/MinIO can implement the same protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

PROJECT_DIRS = (
    "source", "references", "analysis", "controls", "manifests", "renders",
    "qc", "final", "thumbnails", "logs", "work", "prompts",
)


class AssetStore(Protocol):
    def project_dir(self, project_id: str, area: str) -> Path:
        """Local working directory for ``area`` (created on demand)."""

    def put_file(self, project_id: str, area: str, src: Path, name: str | None = None) -> str:
        """Copy ``src`` into the store; return its store-relative path."""

    def path_for(self, rel_path: str) -> Path:
        """Resolve a store-relative path to a local file path."""

    def rel(self, path: Path) -> str:
        """Store-relative path of a file that already lives in the store."""

    def exists(self, rel_path: str) -> bool: ...
