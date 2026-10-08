# AI handoff: Rökkur Studio

Shared notes for the AI assistants working on this repo (Claude and Codex). Read this first,
then inspect the files it points to before changing anything. Keep it short and current:
update **Current work** and the **Log** after meaningful work. Never put credentials here.

Last updated: 2026-10-08 by Claude (review of Codex's `5408dfe`). See the [local upgrade
findings](upgrade-2026-10-07.md) for implementation details and real render observations; the
latest Git commit is authoritative.

## How we share the work

- **Repo:** GitHub `elismagna/Rokkur-Mcp-comfyui-x-`, branch `main`. Elis's working copy is
  `C:\Users\Elis\rokkur studio` (Windows, RTX 3070 8 GB, 32 GB RAM).
- **Claude** works in a Linux cloud clone and pushes straight to `main`. It cannot see Elis's
  PC: no local files, Docker, ComfyUI, Ollama or GPU. It only sees what is pushed to GitHub and
  what Elis pastes into chat.
- **Codex** works in the local folder and can run the real stack.
- So: `git pull` before starting, commit and push when a piece of work is done, and never leave
  work uncommitted across a handoff. Put what you are doing in **Current work** so we don't
  both edit the same files.
- Local-only files that never go in git: `.env`, `secrets/` (YouTube OAuth client and token),
  `data/` (projects, renders, caches), `media/` (source clips). All are git-ignored.

## Project rules (from Elis's master instruction, still in force)

- Never commit OAuth refresh tokens, API tokens, Odysseus tokens, database credentials or keys.
- Don't expose the Docker socket, Postgres, ComfyUI or the API to the internet. Compose binds
  every port to `127.0.0.1`.
- Never bypass YouTube access controls or download restrictions. The studio never downloads
  from YouTube.
- Unknown rights block ingestion and publishing by default.
- Never publish incomplete or test renders. A dry run exists, and nothing uploads without a
  person's click (or approval at autonomy level 3+).
- Don't fake integrations. Unknown interfaces stay unimplemented and say so.
- Don't change working components until you understand them.

## What exists (detail lives in the linked docs)

A modular monolith, `src/rokkur_studio/`: FastAPI API plus dashboard at `/ui` (port 8400),
Postgres via SQLAlchemy/Alembic, and a Postgres job queue with workers. Pipeline: rights →
ingest → analysis → creative plan (AI director) → manifest → ComfyUI compile → render → QC →
shot-level repair → encode → metadata/thumbnail → publish.

| Area | Where | Docs |
|---|---|---|
| State machine (26 states), gates, resume | `domain/states.py`, `services/projects.py` | `docs/state-machine.md` |
| Job queue, worker, retries, leases | `jobs/` | `docs/architecture.md` |
| Stage handlers (render, QC, repair, edit…) | `pipeline/stages.py`, `pipeline/driver.py` | `docs/architecture.md` |
| QC metrics | `pipeline/qc.py` | `docs/architecture.md` |
| ComfyUI client, template compiler | `comfyui/`, `workflows/<name>/{workflow.json,params.yaml}` | `docs/comfyui.md` |
| Render profiles | `config/render_profiles.yaml` | `docs/comfyui.md` |
| Agents (Ollama + rule-based fallback) | `agents/` | `docs/agents.md` |
| AI director (vocabulary, DP pass, prompts, characters) | `director/` | `docs/director.md` |
| Rights gate | `domain/rights.py` | `docs/rights.md` |
| YouTube OAuth, upload, schedule, playlists, approvals | `youtube/`, `services/publishing.py` | `docs/youtube.md` |
| Use cases shared by API, dashboard and CLI | `services/commands.py` | |
| Dashboard | `dashboard/views.py`, `dashboard/templates/` | `README.md` |
| CLI | `cli.py`; Windows wrapper `scripts/studio.ps1`, Linux `Makefile` | `README.md` |
| Config | `config/studio.yaml`, overridden by `.env` (`STUDIO_<SECTION>__<KEY>`) | `.env.example` |

Other docs: `docs/current-state.md` (first audit of the PC), `docs/adr/0001-…` (architecture
decision), `docs/milestones.md` (phase status), `docs/setup-windows.md`,
`docs/troubleshooting.md`.

## Key decisions and why

- **Standalone repo; integrate over HTTP APIs only.** Odysseus and Rökkur Collective could
  not be inspected, so the studio calls nothing it can't see. `RokkurCollectiveProvider`
  refuses to run. (ADR 0001)
- **Postgres is the only source of truth** (projects, jobs, events, assets, approvals), and the
  queue uses `SKIP LOCKED`. No Redis or Temporal until there's a need.
- **Every status change goes through `transition()`**, which checks the edge, applies gates and
  writes an audit event. The exceptions are deliberate: `resume()` and the repair-limit actions.
- **Docker runs the studio only.** ComfyUI and Ollama stay on the Windows host, reached via
  `host.docker.internal`. The Docker socket is not mounted.
- **8 GB VRAM is the hard limit.** Ollama is unloaded before heavy jobs, ComfyUI is told to
  `/free` before agent calls, and only one heavy GPU job runs at a time (GPU lease). Without
  this, Ollama fell back to the CPU.
- **Render model: Wan 2.1 VACE 1.3B (fp16) with the umt5 fp8 text encoder, core nodes only.**
  It's the largest video-to-video model that fits 8 GB, and the PC has no custom node packs.
  The source clip's Canny edges drive VACE, and the prompt sets the look.
- **Wan constraints:** width and height are multiples of 16, 16 fps, frame count 4n+1 (max 81
  in RTX3070_QUALITY). One prompt per shot; the FizzNodes prompt schedule is only an export.
- **Agents return Pydantic schemas** using Ollama's `format`, and fall back to rules when
  Ollama is down. Measured facts override model output. The DP pass may only use a fixed
  cinematography vocabulary (`director/vocabulary.py`), at Elis's request.
- **QC uses deterministic metrics only.** Model-based metrics (identity, prompt adherence,
  style) are reported as `null` with a reason, never invented. Since `b69234d`, structure
  compares edge maps rather than brightness.
- **Publishing:** private by default. Public and scheduled uploads need `youtube.allow_public`,
  because `publishAt` makes a video public. `auto_publish` is not used.
- **Repair limit:** after `render.max_retries` rounds a person chooses more rounds, re-check, or
  keep. "Keep" writes a QC report version marked PASS with an `override` block.

## Installed on the PC (from audits and Elis's reports, 2026-10-07)

- Docker Desktop (WSL2). `studio.ps1 up` builds the image, migrates and starts db/api/worker.
- ComfyUI Desktop 0.39.1 (PyTorch 2.12.1+cu130), started with `--listen 0.0.0.0` so Docker can
  reach it. Core nodes only. Models are in
  `C:\Users\Elis\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models`:
  - `diffusion_models/wan2.1_vace_1.3B_fp16.safetensors`
  - `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  - `vae/wan_2.1_vae.safetensors`
  - Unused by the studio: `z_image_turbo_bf16`, `qwen_3_4b`.
  - `scripts/install-models.ps1` moves downloads into place and never deletes.
- Workflows in the repo:
  - `workflows/v2v_3070_quality`: Wan VACE, used by both profiles (PREVIEW at 320x576, 33
    frames, 8 steps; RTX3070_QUALITY at up to 576x1024, 81 frames, 20 steps).
  - `workflows/v2v_preview`: needs an SD 1.5 checkpoint that isn't installed, so it's unused.
- Ollama: the studio uses `qwen3.5:9b` (tools and vision). Also installed: `qwen3.5:4b`,
  `gemma4:12b`, `gemma3:12b`, `qwen2.5-coder:7b`, `deepseek-r1:8b`. `odysseus-vision:9b`,
  `satan-odysseus:9b` and `satan:latest` belong to Odysseus; don't use or change them.

## Run and test

```powershell
cd "C:\Users\Elis\rokkur studio"
git pull
.\scripts\studio.ps1 up            # build, migrate, start; dashboard http://127.0.0.1:8400/ui
.\scripts\studio.ps1 audit         # what the container can reach (writes data/audit.json)
.\scripts\studio.ps1 comfy-check   # templates vs the live ComfyUI /object_info and model files
.\scripts\studio.ps1 agent-check   # every agent role answers on the configured Ollama model
.\scripts\studio.ps1 comfy-render  # synthetic clip through real ComfyUI (PREVIEW)
.\scripts\studio.ps1 render myclip.mp4 --theme "..." --rights USER_OWNED --evidence "..."
.\scripts\studio.ps1 logs
```

Tests (`pytest`, plus `ruff check src tests` and `mypy src`) need FFmpeg and a Postgres
database in `TEST_DATABASE_URL`. **The test fixture drops and recreates every table in that
database, so never point it at the studio's `rokkur` database.** Use a separate `rokkur_test`.
The tests are integration tests: real FFmpeg, real Postgres, in-process worker, and fake
ComfyUI, Ollama and Google servers (`tests/fakes*.py`).

## Verified results

Tested directly by Codex on Elis's PC, 2026-10-07/08:
- Final suite: **199 passed**, with `ruff check src tests` and `mypy src` clean. Real FFmpeg
  and disposable Postgres; external APIs mocked in the suite. A `_test` database-name guard
  now prevents accidentally pointing tests at production.
- Real Qwen3.5:9b source-vision passes, plus two actual Wan renders on the RTX 3070. Exact
  settings, timings, visual limitations and QC results are in the upgrade findings.
- Deployed the rebuilt API and worker on 2026-10-08. Ten live dashboard pages returned 200;
  existing project players, HTTP range seeking, 390-pixel layout and browser scripts passed.
- `comfy-check`: installed Wan workflow v2 passes. The unused SD1.5 workflow lacks its
  checkpoint; hybrid workflow is absent. Unavailable profiles are disabled in the UI.
- Existing Ape project `proj_01a116f8ebdc_921a5f36` remains stopped at the **40/40 render
  budget**, after 8 repair rounds. It was not resumed or rerendered during deployment.
  Its previous outputs are reviewable; plain Resume cannot fix an exhausted render budget.
  Since Claude's review commit the project page offers **Allow 20 more renders and continue**
  (tested in the suite, not yet on the PC).

Tested on Elis's PC (Elis ran the commands and pasted the output):
- `up`, `audit` and migrations work. ComfyUI and Ollama are reachable from Docker.
- `comfy-render` produced a real Wan VACE render (after the multiple-of-16 fix).
- `agent-check`: Creative Director and Channel Manager answer on `qwen3.5:9b` on the GPU.
- First full `render` of Elis's own clip (RTX3070_QUALITY): all 6 shots rendered. QC then
  failed and the project stopped at the repair limit after 3 rounds. QC per shot (score,
  temporal, motion, structure): 001 2.99/2.79/4.44/0.0, 002 4.2/9.23/0.1/0.0,
  003 4.58/9.35/0.91/0.0, 004 6.17/6.59/7.9/1.31, 005 6.76/8.88/8.24/1.44 (pass),
  006 6.61/8.25/8.76/0.0 (pass). These scores used the old brightness-based structure metric.

Tested only in Claude's cloud container (fakes, not the real services):
- The full suite: 182 tests pass at `b69234d`, with ruff and mypy clean.
- YouTube upload, scheduling, playlists and approvals: tested against a fake Google only. No
  real sign-in has happened yet.
- Director passes: tested against a fake Ollama. They probably ran in the render above, but
  that isn't confirmed.
- The edge-based QC structure metric: tested on synthetic clips only.

Where outputs land on the PC: `data\projects\<project id>\`:
- `source\`, `analysis\analysis.json`, `manifests\manifest_vN.json`
- `work\clips\shot_NNN_<fps>fps.mp4` (the source cut per shot)
- `renders\shot_NNN\attempt_NN.mp4`, `renders\assembled_vNN.mp4`
- `qc\qc_vN.json`
- `final\final.mp4`, `final\preview.gif`, `thumbnails\thumbnail.jpg`

Other files: `data/director/asset_tracker.json` (characters and global look) and
`data/youtube/playlists.json`.

## Known issues (open)

1. **Real Wan fidelity remains imperfect.** Local reference-guided renders have filled surfaces,
   but can distort anatomy and motion. QC scored two real trials 5.29 and 5.19 (FAIL). Low-motion
   scoring is fixed and tested, but identity/anatomy/prompt fidelity are still unmeasured.
2. **Repairs now change seed and CONTROL_STRENGTH.** Unsupported style/identity/pose/depth
   changes are filtered and disclosed. Increasing guidance from 0.85 to 1.15 worsened the
   tested frog shot despite a better motion subscore. Treat adjustments as experiments.
   Automatic changes now stay within 0.7–1.0 (drift raises toward 1.0, flicker lowers by 0.1)
   and never push a value the user set further out, because QC rewards the higher strength
   while the anatomy it cannot see gets worse. DENOISE and OFFLOAD remain unavailable in this
   Wan graph and are reported as ignored.
3. **The vision pass and character anchors.** With a character anchor the DP pass keeps the
   story's subject (pose and expression) and takes only the observed background, so the
   source animal or actor is not written next to the anchor. Without an anchor the observation
   replaces the story subject, so a request like "turn the cat into a tiger" belongs in a
   character anchor, not the free-text prompt. Not yet checked on the real model.
4. One render job in the same project hit "ComfyUI unreachable … [Errno 101] Network is
   unreachable" before the run that got to QC. Not diagnosed; ComfyUI was probably not
   running at the time.
5. Rökkur Collective and Odysseus interfaces are still unknown, so nothing calls them.
6. `studio.ps1 test` on Windows has never been run. It runs `python -m pytest` on the host,
   which needs a venv with `.[dev]`, FFmpeg and a test Postgres.
7. Phases 5 (discovery) and 7–10 (community, analytics, learning, autonomy) are not started.

## Approaches that failed (don't repeat)

- Render sizes that aren't multiples of 16: the Wan KSampler fails (4200 vs 4320 tokens).
- `v2v_preview` on the PC: no SD 1.5 checkpoint, so PREVIEW was moved to the Wan template.
- `maxLength` in Ollama's `format` schema: HTTP 400 (grammar too large). Lengths are now
  enforced by Pydantic validation with retry.
- Building the image from three compose services at once: Docker Desktop fails, so only
  `migrate` builds.
- ComfyUI on its default `127.0.0.1` bind: unreachable from Docker. It needs `0.0.0.0`
  (Private network only in the firewall).
- QC structure as brightness correlation: a relit restyle with identical layout scored 0.
- Plain Resume at the repair limit: it re-entered repair and stopped again at once. It is now
  refused there.

## Current work

- **Claude (2026-10-08):** reviewed Codex's `5408dfe` and pushed the fixes listed in the Log.
  No edit in progress. Next for Elis: rebuild, then on the Ape project press **Allow 20 more
  renders and continue** (or Cancel it). `qc-shots.zip` is no longer needed.
- **Render quality experiments (proposed by Claude 2026-10-08, not run yet; Codex runs them on
  the PC because Claude cannot reach it).** Elis says ComfyUI `render_00054_` was the best so far
  and wants renders 61+ improved before adding much. Start from the 00054 graph (drag it into
  ComfyUI), keep its seed, change one thing per run, same shot each time:
  1. Resolution: 832×480 (Wan 1.3B's training size) vs our 576×320.
  2. Reference: none vs source first frame vs one image already in the target style.
  3. Control: Canny 0.2/0.5 (now) vs softer edges (thresholds 0.3/0.7, or a slight blur first),
     then depth via the `comfyui_controlnet_aux` add-on (Depth Anything V2 small). Hypothesis:
     edges cause the outline/clay look and broken fingers; depth keeps layout without lines.
  4. Steps 20 vs 30 at the best setting so far.
  5. Elis (renders 62/63): the room looks great (new tiles, towels) but the ape looks bad.
     Try keeping the real ape and restyling only the room: per-frame ape mask from SAM 2
     (segmentation add-on) into WanVaceToVideo's `control_masks` (white = regenerate the
     room, black = keep the source ape). Waiting on Elis's choice: keep the real ape, or
     restyle it too (depth guide + styled ape reference + precise subject prompt).
  Later, post only: an upscale model (core node, just a model file) and RIFE interpolation
  (`ComfyUI-Frame-Interpolation`) for 16→32 fps.
  **Report back through git so Claude can see the frames:** for each run commit
  `docs/validation/2026-10-08/<render>.png` (contact sheet:
  `ffmpeg -i render.mp4 -vf "fps=2,scale=320:-1,tile=4x3" -frames:v 1 render.png`) and
  `<render>.json` (settings: `ffprobe -v error -show_entries format_tags -of json render.mp4`),
  plus one line per run in the Log: what changed, QC score, what Elis thought.
- **Codex (2026-10-08):** reliability/UI upgrade is implemented, tested and deployed locally.
  This commit releases the previous file ownership; no further edit is in progress. The
  next useful work is real character/hand fidelity, repair comparison strategy, and a clear
  user flow for projects stopped at the total render budget. Do not promise that increasing
  source strength improves quality: the real comparison showed the opposite.

## Log (newest first)

- 2026-10-08 Claude: review of `5408dfe` (220 tests pass, ruff/mypy clean; dashboard script
  checked in Chromium). Added **Allow more renders** for projects stopped at the render budget
  (`BUDGET_EXTENDED`; Resume refused while still over). QC: frozen render of a low-motion source
  now fails (counts clearly moving pixels, so grain is not motion). Probe reads Matroska's
  stream `DURATION` (MKV/WebM with longer audio no longer pad the last shot and fail QC forever).
  Story merge also matches "1"/"shot_1" ids and falls back to position. Vision pass keeps the
  subject when there is a character anchor. No placeholder subject text in prompts. Automatic
  CONTROL_STRENGTH bounded to 0.7–1.0. ComfyUI timeouts and failed uploads are retried again
  (they had become permanent rejections). Padding no longer capped at 1 s (old short renders
  assemble at full length; normalized cache is now `_norm_v3`). An empty repair plan stops at
  the repair limit (keep / check again) instead of a dead end. A broken workflow file shows on
  the System page instead of a 500 on every page. Failed creates delete their uploads. Picking
  a render attempt no longer stops auto-refresh; Back after creating no longer leaves a
  disabled form; slider label restored with the draft.
- 2026-10-08 Codex: **199 tests passed**, lint/types clean; deployed and browser-checked the
  upgrade on port 8400. No media, credentials or production data are included in this commit.
- 2026-10-07 Codex: guided UI and attempt review; source-aware director passes; uploaded/source
  references wired into Wan; effective sampling/structure controls and repairs; valid frame
  lengths, CFR output and OOM timing; low-motion/frozen/edge-output QC; profile availability.
  Real local tests and remaining limitations are in [the upgrade guide](upgrade-2026-10-07.md).

- 2026-10-07 Claude `b69234d`: real stop reason on the project page; repair-limit actions
  (more rounds, check again, keep); edge-based QC structure.
- 2026-10-07 Claude `c285195`: scheduled release, playlists, upload approvals at autonomy 3.
- 2026-10-07 Claude `9b21e3a`: AI director passes.
- 2026-10-07 Claude: Phases 0–4 and 6 core (see `git log` for detail).
