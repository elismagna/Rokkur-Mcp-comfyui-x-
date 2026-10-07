"""Local filesystem asset store rooted at ``<data_dir>/projects``."""

from __future__ import annotations

import hashlib
import mimetypes
import re
import shutil
from pathlib import Path

from rokkur_studio.storage.base import PROJECT_DIRS

_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guess_mime(path: Path) -> str | None:
    return mimetypes.guess_type(path.name)[0]


class LocalAssetStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = (Path(data_dir) / "projects").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _check(self, *parts: str) -> None:
        for part in parts:
            if not _SAFE.match(part):
                raise ValueError(f"unsafe path component {part!r}")

    def project_dir(self, project_id: str, area: str) -> Path:
        self._check(project_id)
        if area.split("/")[0] not in PROJECT_DIRS:
            raise ValueError(f"unknown asset area {area!r}")
        self._check(*area.split("/"))
        path = self.root / project_id / area
        path.mkdir(parents=True, exist_ok=True)
        return path

    def put_file(self, project_id: str, area: str, src: Path, name: str | None = None) -> str:
        name = name or src.name
        self._check(name)
        dest = self.project_dir(project_id, area) / name
        if Path(src).resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return self.rel(dest)

    def path_for(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if self.root not in path.parents:
            raise ValueError(f"path escapes asset store: {rel_path!r}")
        return path

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def exists(self, rel_path: str) -> bool:
        return self.path_for(rel_path).exists()
