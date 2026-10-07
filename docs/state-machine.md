# Production state machine

Defined once in `src/rokkur_studio/domain/states.py`. All changes go through
`services.projects.transition()`, which rejects illegal edges, enforces gates and writes a
`STATE_CHANGED` event (plus a semantic event such as `RIGHTS_APPROVED`, `QC_FAILED`) in the
same transaction.

```
DISCOVERED ─► SCORED ─┐
     └────────────────┴► RIGHTS_PENDING ─► RIGHTS_OK ─► DOWNLOADED_OR_INGESTED ─► ANALYZING ─► ANALYZED
                              └► RIGHTS_REJECTED (terminal)
ANALYZED ─► CREATIVE_PLANNING ─► CREATIVE_READY ─► WORKFLOW_COMPILING ─► WORKFLOW_READY
WORKFLOW_READY ─► RENDER_QUEUED ─► RENDERING ─► QUALITY_CHECK ─┬► QUALITY_PASSED ─► EDITING ─► READY_TO_PUBLISH
                       ▲                                       └► QUALITY_FAILED ─► REPAIRING ─┘ (back to RENDER_QUEUED)
READY_TO_PUBLISH ─► PUBLISHING ─► PUBLISHED ─► MONITORING ─► ARCHIVED (terminal)
any non-terminal ─► FAILED (resumable) | CANCELLED (terminal)
```

## Gates

| Target | Requires |
|---|---|
| `RIGHTS_OK` | an approved rights decision on record |
| `DOWNLOADED_OR_INGESTED` | approved rights |
| `PUBLISHING` | approved rights and latest QC report `PASS` (and the project at `READY_TO_PUBLISH`) |

## Stage jobs

| State | Job |
|---|---|
| RIGHTS_PENDING | rights_check |
| RIGHTS_OK | ingest |
| DOWNLOADED_OR_INGESTED | analyze |
| ANALYZED | creative_plan |
| CREATIVE_READY | compile_workflow |
| WORKFLOW_READY / RENDER_QUEUED | render |
| QUALITY_CHECK | qc |
| QUALITY_FAILED | repair |
| QUALITY_PASSED | edit |

## Autonomy levels

| Level | Behaviour |
|---|---|
| 0 | Nothing advances automatically; use `POST /projects/{id}/advance` per stage |
| 1 | Runs rights, ingest, analysis and the creative brief, then waits |
| 2 (default) | Automatic production up to `READY_TO_PUBLISH`; publishing is manual |
| 3, 4 | Reserved for scheduled publishing / channel operation (Phases 6–10); behave like 2 today |

The level comes from the project's channel (`PATCH /channels/{id}`) or `studio.autonomy_level`.

## Failure and resume

`FAILED` stores `failed_from_state`. `POST /projects/{id}/resume` re-enters the state that
schedules the failed stage (e.g. a failure in `RENDERING` resumes at `WORKFLOW_READY`); shots
that already rendered are not rendered again.
