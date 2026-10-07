import pytest

from rokkur_studio.db.models import Event, Project
from rokkur_studio.domain.states import (
    TERMINAL,
    TRANSITIONS,
    InvalidTransition,
    ProjectStatus,
    can_transition,
)
from rokkur_studio.services.projects import (
    GateViolation,
    record_rights,
    resume,
    transition,
)

S = ProjectStatus


def test_every_state_has_a_transition_entry():
    assert set(TRANSITIONS) == set(ProjectStatus)


def test_minimum_states_present():
    names = {s.value for s in ProjectStatus}
    for required in ("DISCOVERED", "RIGHTS_REJECTED", "REPAIRING", "READY_TO_PUBLISH",
                     "MONITORING", "ARCHIVED", "FAILED", "CANCELLED"):
        assert required in names


def test_terminal_states_have_no_exits():
    for s in TERMINAL:
        assert TRANSITIONS[s] == frozenset()


def test_non_terminal_states_can_fail_and_cancel():
    for s in ProjectStatus:
        if s in TERMINAL or s is S.FAILED:
            continue
        assert can_transition(s, S.FAILED) and can_transition(s, S.CANCELLED)


@pytest.mark.parametrize("current,target", [
    (S.DISCOVERED, S.RENDERING),
    (S.RIGHTS_PENDING, S.DOWNLOADED_OR_INGESTED),
    (S.QUALITY_FAILED, S.READY_TO_PUBLISH),
    (S.RENDERING, S.PUBLISHED),
    (S.ARCHIVED, S.DISCOVERED),
])
def test_invalid_transitions_rejected(current, target):
    assert not can_transition(current, target)


def _project(session, status=S.DISCOVERED):
    p = Project(name="t", status=status.value, render_profile="PREVIEW")
    session.add(p)
    session.flush()
    return p


def test_transition_writes_audit_events(db):
    with db.transaction() as s:
        p = _project(s)
        transition(s, p, S.RIGHTS_PENDING, actor="tester", reason="go")
        events = s.query(Event).filter_by(project_id=p.id).all()
    assert [(e.type, e.from_state, e.to_state, e.actor) for e in events] == [
        ("STATE_CHANGED", "DISCOVERED", "RIGHTS_PENDING", "tester")]
    assert events[0].data["reason"] == "go"


def test_invalid_transition_raises_and_writes_nothing(db):
    with db.transaction() as s:
        p = _project(s)
        with pytest.raises(InvalidTransition):
            transition(s, p, S.RENDERING, actor="t")
        assert s.query(Event).count() == 0
        assert p.status == "DISCOVERED"


def test_rights_gate_blocks_ingestion_without_approval(db):
    with db.transaction() as s:
        p = _project(s, S.RIGHTS_PENDING)
        with pytest.raises(GateViolation):
            transition(s, p, S.RIGHTS_OK, actor="t")
        record_rights(s, p.id, category="USER_OWNED", status="approved", decided_by="t")
        transition(s, p, S.RIGHTS_OK, actor="t")
        assert p.status == "RIGHTS_OK"


def test_publication_gate_requires_rights_and_qc_pass(db):
    from rokkur_studio.services.projects import save_document

    with db.transaction() as s:
        p = _project(s, S.READY_TO_PUBLISH)
        with pytest.raises(GateViolation, match="rights"):
            transition(s, p, S.PUBLISHING, actor="t")
        record_rights(s, p.id, category="USER_OWNED", status="approved", decided_by="t")
        with pytest.raises(GateViolation, match="QC"):
            transition(s, p, S.PUBLISHING, actor="t")
        save_document(s, p.id, "qc_report", {"decision": "FAIL"}, created_by="qc")
        with pytest.raises(GateViolation, match="QC"):
            transition(s, p, S.PUBLISHING, actor="t")
        save_document(s, p.id, "qc_report", {"decision": "PASS"}, created_by="qc")
        transition(s, p, S.PUBLISHING, actor="t")


def test_failure_is_recoverable_via_resume(db):
    with db.transaction() as s:
        p = _project(s, S.RENDERING)
        transition(s, p, S.FAILED, actor="worker", reason="ComfyUI crashed")
        assert p.failed_from_state == "RENDERING"
        resume(s, p, actor="human")
        assert p.status == "WORKFLOW_READY"  # re-enters the state that schedules rendering
        assert p.failed_from_state is None
        with pytest.raises(InvalidTransition):
            resume(s, p, actor="human")
