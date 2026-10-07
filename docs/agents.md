# Agents

Agents do reasoning only; job state lives in Postgres, never in chat history. Each role returns
a Pydantic schema (`agents/schemas.py`), and deterministic facts override model output
(e.g. the Creative Director cannot move shot boundaries measured by the Video Analyst).

## Providers (`agents.provider`)

| Provider | Status |
|---|---|
| `rule_based` (default) | Deterministic logic, no model, always available |
| `ollama` | `/api/chat` with the output schema as `format`; invalid JSON is fed back and retried `agents.max_output_retries` times, then the job fails recoverably |
| `rokkur_collective` | Refuses to run: the Collective's API has not been audited. Implement `RokkurCollectiveProvider.generate()` once its interface is known |

## Roles

| Role | Today |
|---|---|
| Video Analyst | Deterministic (`pipeline/analysis.py`): probe, scene cuts, shots, motion intensity. Pose/depth/segmentation/flow/landmarks listed as skipped until a workflow needs them |
| Creative Director | Rule-based or Ollama → `CreativeBrief` |
| Workflow Planner | Deterministic: manifest + per-shot semantic params from the render profile |
| QC Agent | Deterministic metrics (`pipeline/qc.py`); model-based metrics reported as not measured |
| Repair Planner | Rule-based mapping from QC recommendations to minimal parameter changes |
| Rights Agent | Deterministic gate; `RightsAssessment` schema ready for an advisory model |
| Scout, Trend Analyst | Schemas only (`TrendScore`); Phase 5 |
| Channel Manager | Drafts publication metadata; comments/analytics are Phases 7–8 |

`agents.max_per_project` is reserved for when roles run as concurrent agents.
