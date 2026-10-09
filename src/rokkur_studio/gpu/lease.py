"""GPU lease manager: treats 8 GB of VRAM as a scarce, explicitly leased resource.

Leases live in Postgres so the API, every worker and anything else on the machine see
one consistent view. Acquisition is serialised with a transaction-scoped advisory lock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta

from sqlalchemy import func, select, text, update

from rokkur_studio.config import GpuSection
from rokkur_studio.db.models import GpuLease, utcnow
from rokkur_studio.db.session import Database

log = logging.getLogger(__name__)

_ADVISORY_KEY = 0x524F4B4B  # "ROKK"

GPU_CLASSES = ("GPU_LIGHT", "GPU_MEDIUM", "GPU_HEAVY")

Hook = Callable[[str], None]


class GpuUnavailable(TimeoutError):
    pass


class GpuLeaseManager:
    def __init__(self, db: Database, cfg: GpuSection, *,
                 before_heavy: list[Hook] | None = None,
                 after_heavy: list[Hook] | None = None) -> None:
        self.db, self.cfg = db, cfg
        self.before_heavy = before_heavy or []
        self.after_heavy = after_heavy or []
        self._batch = threading.local()

    def vram_for(self, resource_class: str) -> float:
        if resource_class not in GPU_CLASSES:
            raise ValueError(f"unknown resource class {resource_class!r}")
        return min(float(self.cfg.class_vram_gb.get(resource_class, self.cfg.vram_gb)),
                   self.cfg.vram_gb)

    def try_acquire(self, holder: str, resource_class: str) -> GpuLease | None:
        need = self.vram_for(resource_class)
        with self.db.transaction() as s:
            s.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _ADVISORY_KEY})
            now = utcnow()
            s.execute(update(GpuLease)
                      .where(GpuLease.released_at.is_(None), GpuLease.expires_at < now)
                      .values(released_at=now))
            used = s.scalar(select(func.coalesce(func.sum(GpuLease.vram_gb), 0.0))
                            .where(GpuLease.released_at.is_(None))) or 0.0
            heavy = s.scalar(select(func.count(GpuLease.id)).where(
                GpuLease.released_at.is_(None), GpuLease.resource_class == "GPU_HEAVY")) or 0
            if resource_class == "GPU_HEAVY" and heavy >= self.cfg.max_heavy_jobs:
                return None
            if used + need > self.cfg.vram_gb + 1e-9:
                return None
            lease = GpuLease(holder=holder, resource_class=resource_class, vram_gb=need,
                             expires_at=now + timedelta(seconds=self.cfg.lease_seconds))
            s.add(lease)
            s.flush()
            log.info("gpu lease acquired", extra={"data": {"holder": holder,
                     "class": resource_class, "vram_gb": need, "in_use_gb": used + need}})
            return lease

    def release(self, lease_id: str) -> None:
        with self.db.transaction() as s:
            s.execute(update(GpuLease).where(GpuLease.id == lease_id,
                                             GpuLease.released_at.is_(None))
                      .values(released_at=utcnow()))

    def active(self) -> list[GpuLease]:
        with self.db.session() as s:
            return list(s.scalars(select(GpuLease).where(
                GpuLease.released_at.is_(None), GpuLease.expires_at >= utcnow())))

    @contextmanager
    def lease(self, holder: str, resource_class: str, *, timeout_s: float = 600,
              poll_s: float = 2.0, should_abort: Callable[[], bool] | None = None
              ) -> Iterator[GpuLease]:
        """Block until the lease is granted; run VRAM hooks around heavy work."""
        deadline = time.monotonic() + timeout_s
        hooks = resource_class == "GPU_HEAVY" and not self._batching()
        if hooks:
            self._run_hooks(self.before_heavy, holder, "before_heavy")
        while (lease := self.try_acquire(holder, resource_class)) is None:
            if should_abort and should_abort():
                raise GpuUnavailable("aborted while waiting for GPU")
            if time.monotonic() >= deadline:
                raise GpuUnavailable(f"no {resource_class} capacity within {timeout_s}s")
            time.sleep(poll_s)
        try:
            yield lease
        finally:
            self.release(lease.id)
            if hooks:
                self._run_hooks(self.after_heavy, holder, "after_heavy")

    def _batching(self) -> bool:
        return bool(getattr(self._batch, "depth", 0))

    @contextmanager
    def heavy_batch(self, holder: str) -> Iterator[None]:
        """Run the heavy VRAM hooks once around a run of heavy leases, not around each one.

        Every shot of a render takes its own lease, but unloading ComfyUI's models after each
        shot made the next shot load the diffusion model and text encoder from disk again.
        Inside a batch the models stay loaded from shot to shot; Ollama is still unloaded
        first and ComfyUI is still freed at the end. Batches nest; only the outermost runs
        the hooks.
        """
        depth = getattr(self._batch, "depth", 0)
        if depth == 0:
            self._run_hooks(self.before_heavy, holder, "before_heavy")
        self._batch.depth = depth + 1
        try:
            yield
        finally:
            self._batch.depth = depth
            if depth == 0:
                self._run_hooks(self.after_heavy, holder, "after_heavy")

    @staticmethod
    def _run_hooks(hooks: list[Hook], holder: str, phase: str) -> None:
        for hook in hooks:
            try:
                hook(holder)
            except Exception:  # a failed unload must not block the render
                log.warning("gpu hook failed", exc_info=True,
                            extra={"data": {"phase": phase, "hook": getattr(hook, "__name__", "?")}})
