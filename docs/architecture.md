# Architecture

A modular monolith (`src/rokkur_studio`) run as two process types from one image:
`rokkur-studio api` (FastAPI + dashboard) and `rokkur-studio worker` (job runner).
Postgres holds all state. ComfyUI and Ollama are external HTTP services.

```
Odysseus / you ──HTTP──► Studio API (FastAPI) ──► PostgreSQL ◄── Worker(s)
                         /ui dashboard            projects, jobs,     │
                                                  events, leases…     ├─► FFmpeg (subprocess)
                                                                      ├─► ComfyUI  (HTTP, GPU lease)
                         Asset store: data/projects/<id>/…  ◄─────────├─► Ollama   (HTTP)
                                                                      └─► YouTube  (Phase 6)
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | YAML config + `STUDIO_<SECTION>__<KEY>` env overrides; render profiles |
| `domain/states.py` | `ProjectStatus` and the single transition table |
| `domain/rights.py` | Rights categories and the deterministic eligibility gate |
| `db/models.py` | SQLAlchemy models: projects, sources, rights_decisions, assets, documents, renders, jobs, events, gpu_leases, approval_requests, cost_entries, publications, channels |
| `services/projects.py` | `transition()` (validates, enforces gates, audits), resume, documents |
| `services/commands.py` | Use cases shared by API/CLI/dashboard: create, start, cancel, resume, rights decisions, approvals |
| `jobs/queue.py` | Postgres queue: enqueue (dedupe), claim (SKIP LOCKED + expired-lease reclaim), heartbeat, retry with backoff, dead-letter |
| `jobs/worker.py` | Claims a job, runs its stage handler with a heartbeat, records the outcome, advances the project or fails it recoverably |
| `pipeline/driver.py` | Which job runs in which state; autonomy levels |
| `pipeline/stages.py` | Stage handlers: rights_check, ingest, analyze, creative_plan, compile_workflow, render, qc, repair, edit |
| `pipeline/analysis.py`, `pipeline/qc.py` | Deterministic video analysis and quality metrics |
| `pipeline/renderers.py` | `ComfyUIRenderer` (real) and `FFmpegPreviewRenderer` (non-AI stand-in) |
| `manifest/` | Versioned reconstruction manifest + builder + per-shot semantic params |
| `comfyui/` | HTTP client and template registry/compiler |
| `gpu/lease.py` | VRAM-budgeted leases with before/after hooks |
| `media/ffmpeg.py` | FFmpeg/ffprobe wrapper with logged commands and structured errors |
| `storage/` | `AssetStore` protocol and local filesystem implementation |
| `agents/` | Provider protocol (rule-based, Ollama, Collective placeholder), output schemas, roles |
| `services/publishing.py` | Metadata draft/validation and YouTube `videos.insert` dry run |
| `api/`, `dashboard/` | REST API and server-rendered dashboard |

## Job lifecycle

1. A state change (or `POST /projects/{id}/start|advance|resume`) calls `driver.advance()`,
   which enqueues the stage job for the current state with dedupe key `<project>:stage`, so a
   project never has two stage jobs active.
2. A worker claims it (`FOR UPDATE SKIP LOCKED`), sets a lease, and heartbeats while it runs.
3. The handler moves the project into its working state, does the work in short transactions,
   records documents/assets/renders/events, and moves to the next state.
4. On success the worker marks the job `SUCCEEDED` and enqueues the next stage (autonomy
   permitting). On `JobError` it retries with exponential backoff; on `PermanentJobError` or
   exhausted attempts it dead-letters the job and moves the project to `FAILED`, remembering
   `failed_from_state` so `POST /projects/{id}/resume` continues where it stopped.
5. A worker that dies leaves an expired lease; the next claim reclaims the job and counts
   the lost attempt.

Every job row carries `job_id, project_id, agent_id, stage, started_at, finished_at,
duration, status, error, retry_count`; every log line inside a job carries the same ids.

## Rendering, QC and repair

- The manifest splits the source into shots (scene cuts, capped at the profile's frame budget).
- `render` renders each shot that has no successful render, under a GPU lease for real GPU
  renderers, then splices the shots (`renders/assembled_vNN.mp4`).
- `qc` scores each shot against its source clip and passes or fails the project.
- `repair` asks the Repair Planner for parameter changes for the **failing shots only**,
  writes a new manifest version, supersedes those renders and sends the project back to
  `RENDER_QUEUED`. `render.max_retries` bounds the rounds; exhausting it fails the project
  and opens an approval request. A human then grants more rounds or keeps the renders
  (see `docs/state-machine.md`, Repair limit).
- QC structure compares edge maps, not brightness: a restyle may relight the whole scene and
  still keep the source layout, which is what the Canny-driven Wan workflow preserves.
- CUDA OOM never retries identical work: each OOM applies the next step of the profile's
  `degrade` ladder (clear cache → fewer frames → lower resolution → offload → lighter profile)
  and escalates when the ladder is exhausted.

## Cost controls

`render.max_renders_per_project`, `costs.max_gpu_minutes_per_project` (GPU minutes are
recorded per render in `cost_entries`), `render.max_retries` (repair rounds) and the OOM
ladder bound what one project can consume. Cloud GPU (`HYBRID_MAX`) has no implementation
yet and `costs.max_cloud_gpu_minutes` defaults to 0.
