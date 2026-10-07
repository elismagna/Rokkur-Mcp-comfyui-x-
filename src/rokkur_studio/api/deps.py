from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from rokkur_studio.pipeline.context import StudioContext


def get_ctx(request: Request) -> StudioContext:
    return request.app.state.ctx  # type: ignore[no-any-return]


def get_session(request: Request) -> Iterator[Session]:
    ctx: StudioContext = request.app.state.ctx
    with ctx.db.transaction() as session:
        yield session
