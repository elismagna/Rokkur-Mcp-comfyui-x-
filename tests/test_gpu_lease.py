from datetime import timedelta

import pytest

from rokkur_studio.config import GpuSection
from rokkur_studio.db.models import GpuLease, utcnow
from rokkur_studio.gpu.lease import GpuLeaseManager, GpuUnavailable


def mgr(db, **kw):
    return GpuLeaseManager(db, GpuSection(**kw))


def test_only_one_heavy_job_at_a_time(db):
    m = mgr(db)
    first = m.try_acquire("render-1", "GPU_HEAVY")
    assert first is not None
    assert m.try_acquire("render-2", "GPU_HEAVY") is None
    assert m.try_acquire("llm", "GPU_LIGHT") is None  # 8 GB already leased
    m.release(first.id)
    assert m.try_acquire("render-2", "GPU_HEAVY") is not None


def test_light_jobs_share_vram_budget(db):
    m = mgr(db)
    assert m.try_acquire("a", "GPU_LIGHT") and m.try_acquire("b", "GPU_MEDIUM")
    assert m.try_acquire("c", "GPU_LIGHT") is None  # 2 + 5 + 2 > 8


def test_future_gpu_upgrade_is_config_only(db):
    m = mgr(db, vram_gb=24, max_heavy_jobs=2, class_vram_gb={"GPU_LIGHT": 2, "GPU_MEDIUM": 6,
                                                             "GPU_HEAVY": 12})
    assert m.try_acquire("a", "GPU_HEAVY") and m.try_acquire("b", "GPU_HEAVY")


def test_expired_leases_are_reclaimed(db):
    m = mgr(db)
    lease = m.try_acquire("crashed", "GPU_HEAVY")
    with db.transaction() as s:
        s.get(GpuLease, lease.id).expires_at = utcnow() - timedelta(seconds=1)
    assert m.try_acquire("next", "GPU_HEAVY") is not None


def test_context_manager_runs_vram_hooks_and_tolerates_hook_failure(db):
    calls = []

    def unload(holder):
        calls.append(("unload_ollama", holder))
        raise RuntimeError("ollama down")  # must not block the render

    m = GpuLeaseManager(db, GpuSection(), before_heavy=[unload],
                        after_heavy=[lambda h: calls.append(("free_comfy", h))])
    with m.lease("job1", "GPU_HEAVY"):
        assert len(m.active()) == 1
    assert calls == [("unload_ollama", "job1"), ("free_comfy", "job1")]
    assert m.active() == []


def test_heavy_batch_runs_vram_hooks_once_around_many_leases(db):
    calls = []
    m = GpuLeaseManager(db, GpuSection(),
                        before_heavy=[lambda h: calls.append(("unload_ollama", h))],
                        after_heavy=[lambda h: calls.append(("free_comfy", h))])
    with m.heavy_batch("job1"):
        for _ in range(3):  # three shots: models stay loaded between them
            with m.lease("job1", "GPU_HEAVY"):
                assert len(m.active()) == 1
            assert m.active() == []
        # Nested: still only the outermost batch runs the hooks.
        with m.heavy_batch("job1"), m.lease("job1", "GPU_HEAVY"):
            pass
        assert calls == [("unload_ollama", "job1")]
    assert calls == [("unload_ollama", "job1"), ("free_comfy", "job1")]
    with m.lease("job2", "GPU_HEAVY"):  # outside a batch: hooks per lease again
        pass
    assert calls[-2:] == [("unload_ollama", "job2"), ("free_comfy", "job2")]


def test_lease_times_out(db):
    m = mgr(db)
    m.try_acquire("holder", "GPU_HEAVY")
    with pytest.raises(GpuUnavailable), m.lease("waiter", "GPU_HEAVY", timeout_s=0.2, poll_s=0.05):
        pass
