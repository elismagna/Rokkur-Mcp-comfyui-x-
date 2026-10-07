"""Production state machine: the single table of legal project transitions."""

from __future__ import annotations

from enum import StrEnum


class ProjectStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    SCORED = "SCORED"
    RIGHTS_PENDING = "RIGHTS_PENDING"
    RIGHTS_OK = "RIGHTS_OK"
    RIGHTS_REJECTED = "RIGHTS_REJECTED"
    DOWNLOADED_OR_INGESTED = "DOWNLOADED_OR_INGESTED"
    ANALYZING = "ANALYZING"
    ANALYZED = "ANALYZED"
    CREATIVE_PLANNING = "CREATIVE_PLANNING"
    CREATIVE_READY = "CREATIVE_READY"
    WORKFLOW_COMPILING = "WORKFLOW_COMPILING"
    WORKFLOW_READY = "WORKFLOW_READY"
    RENDER_QUEUED = "RENDER_QUEUED"
    RENDERING = "RENDERING"
    QUALITY_CHECK = "QUALITY_CHECK"
    QUALITY_FAILED = "QUALITY_FAILED"
    REPAIRING = "REPAIRING"
    QUALITY_PASSED = "QUALITY_PASSED"
    EDITING = "EDITING"
    READY_TO_PUBLISH = "READY_TO_PUBLISH"
    PUBLISHING = "PUBLISHING"
    PUBLISHED = "PUBLISHED"
    MONITORING = "MONITORING"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


S = ProjectStatus

TERMINAL: frozenset[ProjectStatus] = frozenset({S.ARCHIVED, S.CANCELLED, S.RIGHTS_REJECTED})

# Forward edges of the happy path plus explicit loops. FAILED/CANCELLED are added below.
_EDGES: dict[ProjectStatus, set[ProjectStatus]] = {
    S.DISCOVERED: {S.SCORED, S.RIGHTS_PENDING},
    S.SCORED: {S.RIGHTS_PENDING, S.ARCHIVED},
    S.RIGHTS_PENDING: {S.RIGHTS_OK, S.RIGHTS_REJECTED},
    S.RIGHTS_OK: {S.DOWNLOADED_OR_INGESTED},
    S.RIGHTS_REJECTED: set(),
    S.DOWNLOADED_OR_INGESTED: {S.ANALYZING},
    S.ANALYZING: {S.ANALYZED},
    S.ANALYZED: {S.CREATIVE_PLANNING},
    S.CREATIVE_PLANNING: {S.CREATIVE_READY},
    S.CREATIVE_READY: {S.WORKFLOW_COMPILING, S.CREATIVE_PLANNING},
    S.WORKFLOW_COMPILING: {S.WORKFLOW_READY},
    S.WORKFLOW_READY: {S.RENDER_QUEUED, S.WORKFLOW_COMPILING},
    S.RENDER_QUEUED: {S.RENDERING},
    S.RENDERING: {S.QUALITY_CHECK, S.RENDER_QUEUED},
    S.QUALITY_CHECK: {S.QUALITY_PASSED, S.QUALITY_FAILED},
    S.QUALITY_FAILED: {S.REPAIRING},
    S.REPAIRING: {S.RENDER_QUEUED, S.QUALITY_CHECK},
    S.QUALITY_PASSED: {S.EDITING},
    S.EDITING: {S.READY_TO_PUBLISH},
    S.READY_TO_PUBLISH: {S.PUBLISHING, S.EDITING, S.ARCHIVED},
    S.PUBLISHING: {S.PUBLISHED, S.READY_TO_PUBLISH},
    S.PUBLISHED: {S.MONITORING, S.ARCHIVED},
    S.MONITORING: {S.ARCHIVED},
    S.ARCHIVED: set(),
    S.CANCELLED: set(),
    S.FAILED: {S.ARCHIVED},  # resume is handled separately (back to failed_from_state)
}

for _state, _targets in _EDGES.items():
    if _state not in TERMINAL and _state is not S.FAILED:
        _targets.update({S.FAILED, S.CANCELLED})
_EDGES[S.FAILED].add(S.CANCELLED)

TRANSITIONS: dict[ProjectStatus, frozenset[ProjectStatus]] = {
    k: frozenset(v) for k, v in _EDGES.items()
}

# States from which a FAILED project may be resumed: the state it failed in is re-entered.
RESUMABLE: frozenset[ProjectStatus] = frozenset(
    s for s in ProjectStatus if s not in TERMINAL and s is not S.FAILED
)


class InvalidTransition(ValueError):
    def __init__(self, current: ProjectStatus, target: ProjectStatus, reason: str = "") -> None:
        self.current, self.target = current, target
        msg = f"invalid transition {current} -> {target}"
        super().__init__(f"{msg}: {reason}" if reason else msg)


def can_transition(current: ProjectStatus, target: ProjectStatus) -> bool:
    return target in TRANSITIONS[current]


def assert_transition(current: ProjectStatus, target: ProjectStatus) -> None:
    if not can_transition(current, target):
        raise InvalidTransition(current, target)
