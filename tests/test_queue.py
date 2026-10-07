from datetime import timedelta

from rokkur_studio.config import JobsSection
from rokkur_studio.db.models import Event, Job, utcnow
from rokkur_studio.jobs import queue

CFG = JobsSection(backoff_base_s=0, backoff_max_s=0, lease_seconds=60)


def test_enqueue_dedupes_active_jobs(db):
    with db.transaction() as s:
        a = queue.enqueue(s, "analyze", dedupe_key="p1:stage")
        b = queue.enqueue(s, "render", dedupe_key="p1:stage")
    assert a is not None and b is None
    with db.transaction() as s:
        job = s.get(Job, a.id)
        queue.complete(s, job)
    with db.transaction() as s:
        assert queue.enqueue(s, "render", dedupe_key="p1:stage") is not None


def test_claim_is_exclusive_across_sessions(db):
    with db.transaction() as s:
        queue.enqueue(s, "a")
    s1, s2 = db.session(), db.session()
    try:
        j1 = queue.claim(s1, "w1", CFG)
        j2 = queue.claim(s2, "w2", CFG)  # row locked by s1 → skipped
        assert j1 is not None and j2 is None
    finally:
        s1.rollback(), s2.rollback(), s1.close(), s2.close()


def test_retry_then_dead_letter(db):
    with db.transaction() as s:
        job = queue.enqueue(s, "a", max_attempts=2)
    with db.transaction() as s:
        j = queue.claim(s, "w", CFG)
        assert queue.fail(s, j, {"code": "x"}, retryable=True, cfg=CFG) is True
        assert j.status == "RETRY_WAIT"
    with db.transaction() as s:
        j = queue.claim(s, "w", CFG)
        assert j is not None and j.id == job.id
        assert queue.fail(s, j, {"code": "x"}, retryable=True, cfg=CFG) is False
        assert j.status == "FAILED"
        types = [e.type for e in s.query(Event).order_by(Event.id)]
    assert types == ["JOB_RETRY_SCHEDULED", "JOB_FAILED"]


def test_permanent_error_skips_retries(db):
    with db.transaction() as s:
        queue.enqueue(s, "a", max_attempts=5)
    with db.transaction() as s:
        j = queue.claim(s, "w", CFG)
        assert queue.fail(s, j, {"code": "bad"}, retryable=False, cfg=CFG) is False
        assert j.status == "FAILED" and j.retry_count == 1


def test_expired_lease_is_reclaimed_and_counts_as_attempt(db):
    with db.transaction() as s:
        job = queue.enqueue(s, "a", max_attempts=3)
    with db.transaction() as s:
        j = queue.claim(s, "dead-worker", CFG)
        j.locked_until = utcnow() - timedelta(seconds=1)
    with db.transaction() as s:
        j = queue.claim(s, "w2", CFG)
        assert j.id == job.id and j.locked_by == "w2" and j.retry_count == 1


def test_backoff_is_exponential_and_capped():
    cfg = JobsSection(backoff_base_s=5, backoff_max_s=30)
    assert [queue.backoff_seconds(cfg, n) for n in (1, 2, 3, 4)] == [5, 10, 20, 30]


def test_heartbeat_extends_only_own_lease(db):
    with db.transaction() as s:
        queue.enqueue(s, "a")
    with db.transaction() as s:
        j = queue.claim(s, "w1", CFG)
        jid = j.id
    with db.transaction() as s:
        assert queue.heartbeat(s, jid, "w1", CFG)
        assert not queue.heartbeat(s, jid, "w2", CFG)


def test_cancel_project_jobs(db):
    from rokkur_studio.db.models import Project

    with db.transaction() as s:
        p = Project(name="x", status="RENDERING", render_profile="PREVIEW")
        s.add(p)
        s.flush()
        queue.enqueue(s, "render", project_id=p.id)
        assert queue.cancel_project_jobs(s, p.id) == 1
        assert queue.claim(s, "w", CFG) is None
