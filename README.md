# Rökkur Studio

Local-first, multi-agent AI video production studio. It takes a permitted source video and a
creative brief, and runs it through **rights check → ingest → analysis → creative plan →
reconstruction manifest → ComfyUI workflow compilation → render → QC → shot-level repair →
encode → metadata/thumbnail → publish-ready project**, with every step persisted in Postgres,
audited, retryable and resumable.

Built for one Windows workstation (RTX 3070 8 GB, 32 GB RAM) next to the tools already there:
ComfyUI and Ollama keep running where they run; Studio talks to them over their HTTP APIs.

## What works today

| Area | State |
|---|---|
| Project API (FastAPI, OpenAPI at `/docs`) + dashboard at `/ui` | Working |
| Postgres models, Alembic migrations, audited state machine (26 states) | Working |
| Durable job queue (SKIP LOCKED, leases, backoff retries, dead-letter, resume) | Working |
| Rights gate (unknown rights block ingestion; human approval flow) | Working |
| FFmpeg service, scene detection, shot planning, motion analysis | Working |
| Versioned reconstruction manifest + per-shot semantic render params | Working |
| ComfyUI client + template compiler + GPU lease + OOM recovery ladder | Working against a fake ComfyUI in tests; **not yet run against a real ComfyUI** |
| `ffmpeg_preview` renderer (non-AI colour-grade stand-in) | Working, used by default |
| QC (temporal/motion/structure/artifacts/detail) + shot-level repair loop with budget | Working; identity/prompt adherence reported as *not measured* |
| Final Shorts encode, thumbnail, preview GIF, metadata draft | Working |
| YouTube publish | **Dry run only** (validated `videos.insert` request). OAuth/upload is Phase 6 |
| Ollama agent provider (JSON-schema output, malformed-output retry) | Working against a fake Ollama |
| Rökkur Collective / Odysseus integration | Not built: their interfaces could not be inspected (see `docs/current-state.md`) |
| Discovery, comments, analytics, learning, autonomy loop | Phases 5–10, not started |

## Quick start (Docker, Windows or Linux)

```powershell
copy .env.example .env          # set POSTGRES_PASSWORD
.\scripts\studio.ps1 up          # or: make up
.\scripts\studio.ps1 smoke-test  # synthetic video → … → dry-run publish, with one forced repair
```

Open http://127.0.0.1:8400/ui. The dashboard has:

- **Overview**: what is rendering, what needs you, finished videos, and whether ComfyUI,
  Ollama and YouTube are reachable.
- **New video**: pick a clip from the `media` folder (or upload one), describe the look,
  declare the rights, choose the quality, start.
- **Project page**: pipeline progress, the final video, the model-written brief, QC scores,
  and an editable title/description/tags with Dry run and Upload to YouTube buttons.
- **YouTube**, **Agents** (with a one-click model check), **Queue**, **Approvals**, **System**.

See `docs/setup-windows.md`.

Or render straight from the command line (Windows):

```powershell
.\scripts\studio.ps1 render myclip.mp4 --theme "1970s claymation" --rights USER_OWNED --evidence "I filmed it"
```

This runs the whole pipeline (rights gate, ComfyUI render, QC, dry-run publish) and prints
where the final video landed. Add `--profile PREVIEW` for a quick low-res pass.

To put a finished video on YouTube (private by default, after a one-time sign-in described
in `docs/youtube.md`):

```powershell
.\scripts\studio.ps1 publish <project id>
```

## Quick start (no Docker)

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # needs ffmpeg on PATH
export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/rokkur
rokkur-studio migrate
rokkur-studio api &        # http://127.0.0.1:8400
rokkur-studio worker &
rokkur-studio smoke-test --inject-fault
```

## Create a project

```bash
curl -X POST localhost:8400/projects -H 'content-type: application/json' -d '{
  "name": "Clay robot walk",
  "target_format": "youtube_short",
  "render_profile": "RTX3070_QUALITY",
  "source": {"platform": "local", "local_path": "/media/walk.mp4"},
  "rights": {"category": "USER_OWNED", "permission_evidence": "filmed by me"},
  "creative": {"theme": "1970s stop-motion sci-fi",
               "character_reference_path": "/media/character.png"},
  "autostart": true}'
```

## Commands

`make up | down | logs | test | migrate | studio | worker | comfy-check | youtube-auth |
smoke-test | audit | lint` — or the same names via `scripts\studio.ps1`.

## Documentation

- `docs/current-state.md` – Phase 0 audit and assumptions
- `docs/adr/0001-rokkur-studio-architecture.md` – architecture decision record
- `docs/milestones.md` – phase plan and status
- `docs/architecture.md`, `docs/state-machine.md`, `docs/comfyui.md`, `docs/agents.md`,
  `docs/rights.md`, `docs/youtube.md`, `docs/setup-windows.md`, `docs/setup-docker.md`,
  `docs/troubleshooting.md`
