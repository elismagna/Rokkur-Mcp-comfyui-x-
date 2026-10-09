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
| ComfyUI client + template compiler + GPU lease + OOM recovery ladder | Working; first real Wan 2.1 VACE render on the RTX 3070 on 2026-10-07 |
| `ffmpeg_preview` renderer (non-AI colour-grade stand-in) | Working, used by default |
| QC (temporal/motion/structure/artifacts/detail) + shot-level repair loop with budget | Working; identity/prompt adherence reported as *not measured* |
| Final Shorts encode, thumbnail, preview GIF, metadata draft | Working |
| YouTube publish (OAuth, resumable upload, thumbnail, quota ledger; private by default; scheduled public release, playlists, upload approvals at autonomy level 3) | Built and tested against a fake Google; waiting for the first real sign-in (`docs/youtube.md`) |
| Ollama agents (Creative Director, Channel Manager) | Working on the workstation's GPU (qwen3.5:9b, checked with `agent-check`) |
| AI director: source vision, cinematography, per-shot prompts, character tracker, Batch Prompt Schedule export (`docs/director.md`) | Both vision passes verified locally on qwen3.5:9b; observations still need review |
| Main subject: the app keeps the real subject or restyles it, from the prompt (`docs/subject.md`) | Built and tested here (CPU U²-Net masks, VACE keep workflow, composite); not yet run on the PC |
| Rökkur Collective / Odysseus integration | Not built: their interfaces could not be inspected (see `docs/current-state.md`) |
| Discovery, comments, analytics, learning, autonomy loop | Phases 5–10, not started |

## Quick start (Docker, Windows or Linux)

```powershell
copy .env.example .env          # set POSTGRES_PASSWORD
.\scripts\studio.ps1 up          # or: make up
.\scripts\studio.ps1 smoke-test  # synthetic video → … → dry-run publish, with one forced repair
```

Open http://127.0.0.1:8400/ui. The dashboard has:

The [2026-10-07 upgrade guide](docs/upgrade-2026-10-07.md) covers appearance references,
effective Wan controls, timing fixes, render comparisons and real RTX 3070 findings.

- **Overview**: what is rendering, what needs you, finished videos, and whether ComfyUI,
  Ollama and YouTube are reachable.
- **New video**: pick a clip from the `media` folder (or upload one), describe the look,
  declare the rights, choose the quality, start.
- **Project page**: pipeline progress, the final video, the model-written brief with each
  shot's frame, framing and prompt, the prompt schedule, QC scores, and an editable
  title/description/tags with Dry run and Upload to YouTube buttons: upload now, or private
  now and public at a set time, optionally into a playlist.
- **Director**: the global look, your characters, the allowed cinematography terms, and a
  form that shows how the prompt rules turn a description into a prompt.
- **YouTube**: sign-in, quota, release times and what is scheduled, your playlists.
- **Approvals**: rights questions, and at autonomy level 3 every upload the studio proposes.
- **Agents** (with a one-click model check), **Queue**, **System**.

See `docs/setup-windows.md`.

Or render straight from the command line (Windows):

```powershell
.\scripts\studio.ps1 render myclip.mp4 --theme "1970s claymation" --rights USER_OWNED --evidence "I filmed it"
```

This runs the whole pipeline (rights gate, ComfyUI render, QC, dry-run publish) and prints
where the final video landed. Add `--profile PREVIEW` for a quick low-res pass, and
`--character NEO` to keep a character from the Director page in every shot.

To put a finished video on YouTube (private by default, after a one-time sign-in described
in `docs/youtube.md`):

```powershell
.\scripts\studio.ps1 publish <project id>
.\scripts\studio.ps1 publish <project id> --at "2026-10-09 18:00" --playlist "Shorts"   # public at 18:00
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

`make up | down | logs | test | migrate | studio | worker | comfy-check | agent-check |
render | youtube-auth | publish | youtube-playlists | prompt-schedule | smoke-test | audit | lint` — or the same
names via `scripts\studio.ps1`.

## Documentation

- `docs/AI_HANDOFF.md` – shared handoff for AI assistants: state, decisions, verified results, open issues (start here)
- `docs/current-state.md` – Phase 0 audit and assumptions
- `docs/adr/0001-rokkur-studio-architecture.md` – architecture decision record
- `docs/milestones.md` – phase plan and status
- `docs/director.md` – how each shot's prompt is built: vocabulary, rules, characters, schedule
- `docs/subject.md` – keeping the real main subject or restyling it, and how the app decides
- `docs/research/` – online research behind the workflows (official templates, Civitai)
- `docs/architecture.md`, `docs/state-machine.md`, `docs/comfyui.md`, `docs/agents.md`,
  `docs/rights.md`, `docs/youtube.md`, `docs/setup-windows.md`, `docs/setup-docker.md`,
  `docs/troubleshooting.md`
