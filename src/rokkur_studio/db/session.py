"""Engine and session factory. No module-level engine: callers own their Database object."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


class Database:
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        self.engine: Engine = create_engine(url, echo=echo, pool_pre_ping=True, future=True)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._factory()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """Session that commits on success and rolls back on any exception."""
        with self._factory() as session, session.begin():
            yield session

    def dispose(self) -> None:
        self.engine.dispose()
