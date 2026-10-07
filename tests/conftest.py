from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import text

from rokkur_studio.config import Settings, load_settings
from rokkur_studio.db.models import Base
from rokkur_studio.db.session import Database
from rokkur_studio.media.ffmpeg import FFmpeg
from rokkur_studio.pipeline.context import StudioContext, build_context

ROOT = Path(__file__).resolve().parents[1]
TEST_DB = os.environ.get("TEST_DATABASE_URL",
                         "postgresql+psycopg://postgres@127.0.0.1:5432/rokkur_test")


@pytest.fixture(scope="session")
def database() -> Iterator[Database]:
    db = Database(TEST_DB)
    try:
        with db.engine.connect() as c:
            c.execute(text("select 1"))
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.exit(f"PostgreSQL test database not reachable at {TEST_DB}: {exc}", 2)
    Base.metadata.drop_all(db.engine)
    Base.metadata.create_all(db.engine)
    yield db
    db.dispose()


@pytest.fixture
def db(database: Database) -> Database:
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    with database.engine.begin() as c:
        c.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    return database


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = load_settings(ROOT / "config", environ={
        "STUDIO_STUDIO__DATA_DIR": str(tmp_path / "data"),
        "STUDIO_WORKFLOWS_DIR": str(ROOT / "workflows"),
        "STUDIO_JOBS__BACKOFF_BASE_S": "0",
        "STUDIO_JOBS__BACKOFF_MAX_S": "0",
        "STUDIO_GPU__UNLOAD_OLLAMA_BEFORE_HEAVY": "false",
        "STUDIO_GPU__FREE_COMFYUI_AFTER_HEAVY": "false",
        "STUDIO_RENDER__DEFAULT_PROFILE": "PREVIEW",
    })
    s.database.url = TEST_DB
    return s


@pytest.fixture
def ctx(settings: Settings, db: Database) -> StudioContext:
    return build_context(settings, db)


@pytest.fixture(scope="session")
def ffmpeg() -> FFmpeg:
    f = FFmpeg()
    if not f.available():  # pragma: no cover
        pytest.skip("ffmpeg not installed")
    return f


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory, ffmpeg: FFmpeg) -> Path:
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    return ffmpeg.make_test_video(out, seconds=4)
