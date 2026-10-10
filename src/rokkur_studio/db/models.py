"""SQLAlchemy models. Operational state lives in columns; agent outputs are versioned documents."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JsonType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Time-sortable id like ``proj_0192f3a1b2c3_8f3a9c1d``."""
    return f"{prefix}_{int(time.time() * 1000):012x}_{secrets.token_hex(4)}"


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JsonType, list[Any]: JsonType}


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("chan"))
    name: Mapped[str] = mapped_column(String(200))
    youtube_channel_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    autonomy_level: Mapped[int] = mapped_column(Integer, default=2)
    default_privacy: Mapped[str] = mapped_column(String(16), default="private")
    guidelines: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("proj"))
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), index=True)
    failed_from_state: Mapped[str | None] = mapped_column(String(32))
    channel_id: Mapped[str | None] = mapped_column(ForeignKey("channels.id"))
    target_format: Mapped[str] = mapped_column(String(32), default="youtube_short")
    render_profile: Mapped[str] = mapped_column(String(32))
    creative_input: Mapped[dict[str, Any]] = mapped_column(default=dict)
    repair_rounds: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    version: Mapped[int] = mapped_column(Integer, default=1)

    __mapper_args__ = {"version_id_col": version}

    source: Mapped[Source | None] = relationship(back_populates="project", uselist=False)
    channel: Mapped[Channel | None] = relationship()


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("src"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), unique=True)
    platform: Mapped[str] = mapped_column(String(32))  # youtube | local | upload | url
    video_id: Mapped[str | None] = mapped_column(String(64), index=True)
    url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    creator: Mapped[str | None] = mapped_column(String(200))
    local_path: Mapped[str | None] = mapped_column(Text)  # path supplied by the user, pre-ingest
    asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"))
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)

    project: Mapped[Project] = relationship(back_populates="source")


class RightsDecision(Base):
    __tablename__ = "rights_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("rgt"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    category: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(200))
    owner: Mapped[str | None] = mapped_column(String(200))
    permission_evidence: Mapped[str | None] = mapped_column(Text)
    attribution_required: Mapped[bool] = mapped_column(Boolean, default=False)
    attribution_text: Mapped[str | None] = mapped_column(Text)
    allowed_transformations: Mapped[list[Any]] = mapped_column(default=list)
    commercial_use: Mapped[bool | None] = mapped_column(Boolean)
    decided_by: Mapped[str] = mapped_column(String(64))
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ast"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # source, reference, render, final, thumbnail…
    rel_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime: Mapped[str | None] = mapped_column(String(100))
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    """Versioned structured output: analysis, creative_brief, manifest, qc_report, metadata…"""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("project_id", "kind", "version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("doc"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    data: Mapped[dict[str, Any]] = mapped_column()
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Render(Base):
    __tablename__ = "renders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("rnd"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(64))
    shot_id: Mapped[str] = mapped_column(String(32))
    attempt: Mapped[int] = mapped_column(Integer)
    profile: Mapped[str] = mapped_column(String(32))
    renderer: Mapped[str] = mapped_column(String(32))
    workflow: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))  # running | succeeded | failed | superseded
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    remote_id: Mapped[str | None] = mapped_column(String(64))  # ComfyUI prompt_id
    output_asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"))
    error: Mapped[dict[str, Any] | None] = mapped_column()
    duration_s: Mapped[float | None] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Rating(Base):
    """A judgement of one render attempt (``target`` = shot id) or a whole video ("video").

    ``value``: -2 super dislike, -1 dislike, 1 like, 2 super like. ``rater`` keeps a person's
    taste apart from any model's estimate; learning only ever reads ``human``. ``snapshot``
    freezes what produced the rated result (prompt, workflow, seed, controls, QC), so what the
    studio learns never depends on documents that change later.
    """

    __tablename__ = "ratings"
    __table_args__ = (Index("ix_ratings_target", "project_id", "target", "rater"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("rat"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    target: Mapped[str] = mapped_column(String(32))  # "video" or a shot id
    render_id: Mapped[str | None] = mapped_column(ForeignKey("renders.id"))
    asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"))
    rater: Mapped[str] = mapped_column(String(16), default="human")  # human | ai
    value: Mapped[int] = mapped_column(Integer)
    tags: Mapped[list[Any]] = mapped_column(default=list)
    note: Mapped[str | None] = mapped_column(Text)
    snapshot: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Image(Base):
    """One still picture the studio made or was given: generated, edited, inpainted,
    outpainted, upscaled, a character cutout or a storyboard still (docs/images.md).

    Files live under ``<data_dir>/images/<id>/``; ``rel_path`` is relative to that folder.
    ``parent_id`` points at the picture this one was made from, so edits form a chain.
    """

    __tablename__ = "images"
    __table_args__ = (Index("ix_images_listing", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("img"))
    # generate | edit | variation | inpaint | outpaint | upscale | upload | character | storyboard
    kind: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued | running | done | failed
    profile: Mapped[str | None] = mapped_column(String(32))
    workflow: Mapped[str | None] = mapped_column(String(64))
    prompt: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)   # semantic parameters sent
    request: Mapped[dict[str, Any]] = mapped_column(default=dict)  # what was asked (for redo)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("images.id"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    shot_id: Mapped[str | None] = mapped_column(String(32))
    job_id: Mapped[str | None] = mapped_column(String(64))
    rel_path: Mapped[str | None] = mapped_column(Text)
    source_rel_path: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    render_on: Mapped[str] = mapped_column(String(8), default="local")
    remote_id: Mapped[str | None] = mapped_column(String(64))
    duration_s: Mapped[float | None] = mapped_column(Float)
    error: Mapped[dict[str, Any] | None] = mapped_column()
    verdict: Mapped[int | None] = mapped_column(Integer)   # -2 .. 2, a person's judgement
    title: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_ready", "status", "run_after"),
        Index(
            "uq_jobs_active_dedupe",
            "dedupe_key",
            unique=True,
            postgresql_where=text("status IN ('QUEUED','RUNNING','RETRY_WAIT')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("job"))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(64))
    stage: Mapped[str | None] = mapped_column(String(32))
    agent_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="QUEUED")
    priority: Mapped[int] = mapped_column(Integer, default=100)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column()
    error: Mapped[dict[str, Any] | None] = mapped_column()
    dedupe_key: Mapped[str | None] = mapped_column(String(200))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    locked_by: Mapped[str | None] = mapped_column(String(100))
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @property
    def duration_s(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    from_state: Mapped[str | None] = mapped_column(String(32))
    to_state: Mapped[str | None] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(64))
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GpuLease(Base):
    __tablename__ = "gpu_leases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("gpu"))
    holder: Mapped[str] = mapped_column(String(100))
    resource_class: Mapped[str] = mapped_column(String(16))
    vram_gb: Mapped[float] = mapped_column(Float)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("apr"))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # publish | rights_ambiguity | cloud_gpu …
    status: Mapped[str] = mapped_column(String(16), default="pending")
    summary: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    requested_by: Mapped[str] = mapped_column(String(64))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_by: Mapped[str | None] = mapped_column(String(64))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)


class CostEntry(Base):
    __tablename__ = "cost_entries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))  # gpu_minutes | cloud_gpu_minutes | youtube_quota…
    amount: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16))
    usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("pub"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    dry_run: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(16))  # prepared | uploaded | failed
    request: Mapped[dict[str, Any]] = mapped_column()
    video_asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"))
    thumbnail_asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"))
    youtube_video_id: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[dict[str, Any] | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
