# Agents

Agents do reasoning only; job state lives in Postgres, never in chat history. Each role returns
a Pydantic schema (`agents/schemas.py`), and deterministic facts override model output
(e.g. the Creative Director cannot move shot boundaries measured by the Video Analyst).

## Providers (`agents.provider`)

| Provider | Status |
|---|---|
| `rule_based` | Deterministic logic, no model, always available |
| `ollama` (default in `config/studio.yaml`) | `/api/chat` with the output schema as `format` and `think: false`; invalid JSON is fed back and retried `agents.max_output_retries` times. If Ollama is down or still returns bad JSON, the role falls back to its rule-based logic and logs why, so a render never stalls on the model. `rokkur-studio agent-check` proves the configured model answers both roles. |
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
| Channel Manager | Rule-based or Ollama → `MetadataDraft` (title/description/tags). `services.publishing.apply_draft` re-adds `#shorts`, the AI disclosure and source attribution and drops any draft over YouTube's limits; comments/analytics are Phases 7–8 |

`agents.max_per_project` is reserved for when roles run as concurrent agents.
