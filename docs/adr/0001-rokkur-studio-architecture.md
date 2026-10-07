# ADR 0001: Rökkur Studio architecture

Status: Accepted (2026-10-07)

## Context

Rökkur Studio must run an AI-assisted video production pipeline on one Windows
workstation (RTX 3070, 8 GB VRAM, 32 GB RAM) alongside existing tools (Odysseus,
Rökkur Collective, Ollama, ComfyUI, Docker). Production runs take minutes to hours,
fail often (CUDA OOM, ComfyUI restarts, malformed agent output) and must never lose
state. Reasoning (LLM agents) and deterministic infrastructure (state, jobs, media) must
be kept apart. The existing components could not be inspected (see
`docs/current-state.md`), so integration is through their documented HTTP APIs only.

## Decision

1. **Modular monolith plus workers.** One Python package `rokkur_studio` with clear
   modules (`domain`, `db`, `jobs`, `gpu`, `storage`, `media`, `comfyui`, `manifest`,
   `pipeline`, `agents`, `youtube`, `api`). It runs as two process types from the same
   image: the API (`rokkur-studio api`) and one or more workers (`rokkur-studio worker`).
   No microservices until a real isolation need appears.
2. **PostgreSQL is the single source of truth** for projects, jobs, events, leases,
   assets, approvals and costs. SQLAlchemy 2 models, Alembic migrations.
3. **Durable job queue in Postgres** (`SELECT … FOR UPDATE SKIP LOCKED`), with
   leases (`locked_until`) so a crashed worker's jobs are reclaimed, exponential
   backoff retries, a per-job attempt budget and dead-lettering. No Redis, n8n or
   Temporal in phase 1: the queue needs nothing they add yet. Temporal is the planned
   upgrade if multi-step sagas outgrow this.
4. **Explicit production state machine.** `ProjectStatus` enum plus one transition
   table. Every transition goes through one function that rejects invalid moves,
   enforces gates (rights before ingest, rights + QC + ready before publish) and
   writes an audit event in the same transaction.
5. **GPU as a leased resource.** A Postgres-backed lease table with resource classes
   (`GPU_LIGHT`, `GPU_MEDIUM`, `GPU_HEAVY`) and a VRAM budget from config. By default
   one heavy job at a time. Before a heavy lease, hooks unload Ollama models
   (`keep_alive: 0`); after it, ComfyUI `/free` releases models.
6. **ComfyUI is a rendering service** used only through its HTTP API. Workflows are
   versioned API-format templates plus a semantic parameter map
   (`STYLE_PROMPT → node 6, input "text"`); application code never names node IDs.
7. **Versioned reconstruction manifest** (Pydantic, `version: 1`) is the only input
   to workflow compilation.
8. **Agents behind a provider protocol** with Pydantic output schemas. Malformed
   output is retried and then fails the job into a recoverable state. Providers:
   deterministic rule-based (default, no GPU), Ollama, Rökkur Collective (pending).
9. **Asset store interface** with a local filesystem implementation rooted at
   `data/projects/<project_id>/…`; S3/MinIO can be added behind the same interface.
10. **Configuration** in `config/studio.yaml` + `config/render_profiles.yaml`, overridable
    by `STUDIO_…` environment variables. Secrets only via env / Docker secrets.
11. **Rights first.** Every project carries a rights decision. Unknown rights block
    ingestion and publishing. No code downloads YouTube videos.
12. **Honest stand-ins.** Where a capability needs a model not yet integrated (identity
    similarity, prompt adherence), QC reports it as `not_measured` instead of inventing
    a score. The `ffmpeg_preview` renderer is a deterministic, non-AI stand-in used to
    exercise the pipeline end to end; it is labelled as such everywhere.

## Consequences

- Runs anywhere Docker + Postgres run; Windows-specific bits are limited to
  `host.docker.internal` URLs and a PowerShell wrapper.
- The queue's throughput ceiling (hundreds of jobs/second) is far above need.
- Rökkur Collective and Odysseus integration needs one follow-up once their interfaces
  are visible.
