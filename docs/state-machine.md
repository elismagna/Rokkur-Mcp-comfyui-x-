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
| `QUALITY_PASSED` | latest QC report `PASS` |
| `PUBLISHING` | approved rights and latest QC report `PASS` (and the project at `READY_TO_PUBLISH`) |

## Repair limit

`repair` fails the project once `render.max_retries` rounds have not fixed QC, and opens a
`repair_budget` approval. Plain Resume is refused there, because it would stop again at once.
A human picks the way out (project page, Approvals, or the API):

- **More repair rounds** (`POST /projects/{id}/repair-more`, or approving the request): records
  `REPAIR_BUDGET_EXTENDED` with the rounds granted and resumes at `QUALITY_FAILED`.
- **Check quality again** (`POST /projects/{id}/recheck-quality`): resumes at
  `QUALITY_CHECK` and scores the current renders without rendering, for when QC itself changed.
- **Keep the renders** (`POST /projects/{id}/keep-renders`): saves a new QC report version
  marked `PASS` with an `override` block (who, when, which shots QC failed), records
  `QC_OVERRIDDEN` and moves `QUALITY_FAILED → QUALITY_PASSED`, the only edge that skips QC.

The same stop (and the same three ways out) applies when the repair planner has nothing it can
change for the project's workflow.

## Render budget

Before each render the cost guard checks the project's renders and GPU minutes against
`render.max_renders_per_project` and `costs.max_gpu_minutes_per_project`, plus any
`BUDGET_EXTENDED` grants. Over either, the render job fails with `budget_exceeded` and the
project goes to `FAILED`. Resume is refused while it is still over budget.
**Allow more renders** (`POST /projects/{id}/allow-more-renders`, default half the configured
render budget, GPU minutes in proportion) records `BUDGET_EXTENDED` and resumes rendering.

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
| 3, 4 | Like 2, and a finished video becomes an upload request on the Approvals page with a full plan (visibility, the next free release time, the default playlist). Nothing uploads until a person approves it (`docs/youtube.md`). Level 4 is reserved for channel operation (Phases 7–10) and behaves like 3 |

The level comes from the project's channel (`PATCH /channels/{id}`) or `studio.autonomy_level`.

## Failure and resume

`FAILED` stores `failed_from_state`. `POST /projects/{id}/resume` re-enters the state that
schedules the failed stage (e.g. a failure in `RENDERING` resumes at `WORKFLOW_READY`); shots
that already rendered are not rendered again.
