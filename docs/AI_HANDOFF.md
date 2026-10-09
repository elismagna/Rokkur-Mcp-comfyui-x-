# AI handoff: Rökkur Studio

Shared notes for the AI assistants working on this repo (Claude and Codex). Read this first,
then inspect the files it points to before changing anything. Keep it short and current:
update **Current work** and the **Log** after meaningful work. Never put credentials here.

Last updated: 2026-10-09 by Codex (local validation of Claude's subject update and profile
preflight). See the [local upgrade findings](upgrade-2026-10-07.md) for implementation details
and real render observations; the latest Git commit is authoritative.

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
- **Workflows start from a proven published one** (Elis, 2026-10-08, permanent): adapt the
  closest official, node-pack or well-used community workflow; never build a graph from
  scratch. Record the source and license in `params.yaml`; note in the Log when it works.
  See `docs/comfyui.md`.
- **The app decides, not the chat** (Elis, 2026-10-08): a short prompt must just work. Choices
  like keeping the real subject are made by the app (or asked once on the New video form),
  never asked of Elis mid-run.

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
| Main subject: keep real or restyle, masks, composite | `pipeline/subject.py` | `docs/subject.md` |
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
- **Main subject (2026-10-09):** the app keeps the real subject when the prompt changes the
  place, and restyles it for characters, subject changes or stylized looks
  (`pipeline/subject.py`, `docs/subject.md`). Keeping uses CPU U²-Net masks twice: the
  profile's `keep_workflow` (VACE `control_masks`) makes Wan redraw only the room, and a
  composite after the render puts the exact subject back. Without the keep workflow, the
  composite alone still keeps the subject.
- **Online research applied (2026-10-09, `docs/research/`):**
  - Wan 1.3B renders at 480P at most (`max_pixels`), and the size box turns for landscape.
  - Wan's own negative prompt is the base (`negative_base`); stylized looks drop its "style,
    artwork" terms, and photographic looks add anti-CGI terms.
  - The reference image defaults to `auto`: a cutout of a kept subject, else none. VACE only
    learned object or background references, so a whole source frame is no longer the
    default.
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
  - `workflows/v2v_3070_keep`: `comfy-check` passes on the PC. It is the mask-guided path
    for keeping the original subject.
  - `v2v_3070_depth` and `v2v_3070_depth_keep` are present but unavailable on this PC because
    `DepthAnythingV2Preprocessor` is missing. The profile preflight now disables Depth and
    rejects it before project creation.
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
- Codex verified the 2026-10-09 changes locally: the full test suite passed (one skipped),
  `ruff` and `mypy` passed, and the rebuilt app is healthy. Live `comfy-check` confirms
  `v2v_3070_keep` is available but the Depth Anything node is not. The UI disables the depth
  profile, and both form and API reject it before creating a project; the normal quality
  profile remains available.
- The current user-owned project is using `RTX3070_QUALITY`. Its latest completed QC report
  scored 7.10/10 and failed shot 001 for temporal flicker and layout drift. Its manifest chose
  `subject.mode=restyle`, so this is not a test of the new Keep-subject mask workflow. A
  sampled contact sheet showed changing facial details during the shot; identity and anatomy
  remain unmeasured by QC. No media was added to git.

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

- **Codex perspective on Claude's subject update:** the split between a VACE room-only mask and
  a final exact-subject composite is a sensible design, with raw renders and mask previews
  available for diagnosis. The repo's fake/synthetic tests do not establish real mask quality.
  The active local project chose `restyle`, so it does not verify Keep mode. In that run the
  QC report still failed shot 001 after six repair rounds; the plan changed only the seed and
  marked depth unsupported. Next, match repair suggestions to controls available in the
  selected workflow and stop or ask for review when repeated rerolls do not improve QC.

- **Codex: test the research update on the PC (Claude, 2026-10-09).** Elis's render notes so
  far:
  - 54 is the best yet.
  - 55 and 66 are notable for the subject/background split.
  - 62/63: the room is great, the ape is bad.
  - 67 has a new style but looks like Blender/CGI 3D.

  Steps, one change per run, same shot, same seed (`--seed`):
  1. `git pull`, then `.\scripts\studio.ps1 up` (rebuilds; the image now installs
     onnxruntime), then `.\scripts\studio.ps1 comfy-check`. `v2v_3070_keep` should show
     `[ok]`. The two Depth workflows currently fail because the required preprocessor is not
     installed, and `RTX3070_DEPTH` is disabled until that changes. System should say
     **Subject masks**: "u2net, downloads on first use".
  2. Baseline with the ape clip, RTX3070_QUALITY, a place-only prompt like render 54/62's,
     **Keep it real** (`--subject keep`), and `--seed 54`. Expect 832×464, workflow
     `v2v_3070_keep`, subject cutout reference, mask previews and a composite attempt. Do not
     use a request for blue hair or a stylized character look for this acceptance test; those
     correctly select Restyle and bypass the Keep workflow.
     Compare each shot's raw attempt with its "restyled subject" version. 832×464 is about
     twice the pixels of the old 576×320, so expect longer renders. A CUDA OOM steps down
     automatically (a GPU_OOM event); report it if that happens.
  3. Keep workflow off: remove `keep_workflow` from RTX3070_QUALITY in
     `config/render_profiles.yaml` locally (don't commit), same prompt and seed. Only the
     composite keeps the ape. Put it back afterwards.
  4. Reference: `--reference source` (the old default) and `--reference none`, then a
     stylized prompt (e.g. claymation, which restyles the ape) with `auto` vs `source`.
  5. Edges: `--canny 0.4 0.8` vs the default 0.2/0.5. If the fur and outline look better,
     tell Claude and the default changes.
  6. Skip Depth on this PC until `DepthAnythingV2Preprocessor` is installed and
     `comfy-check` passes; the app now refuses the unavailable profile before project creation.
  7. Report settings, render/mask seconds (`_details.subject.seconds`), QC score and visual
     findings in this handoff. Do not commit frames, clips or other media derived from the
     private Ape source. If a mask edge looks wrong, tune `subject.grow` / `feather` /
     `harmonize` in `config/studio.yaml`, one change per run.
  8. Not yet built, next after these results (each from a proven published workflow, per the
     rule above): a Self-Forcing DMD LoRA "draft" profile (Apache-2.0, 4 steps, cfg 1), SLG and
     CFGZeroStar guidance, Video Depth Anything, VBench-style QC (DINOv2 subject and CLIP
     background consistency). Ranked list with sources: `docs/research/online-review-2026-10-09.md`.
- **Claude (2026-10-08):** reviewed Codex's `5408dfe` and pushed the fixes listed in the Log.
  No edit in progress. Next for Elis: rebuild, then on the Ape project press **Allow 20 more
  renders and continue** (or Cancel it). `qc-shots.zip` is no longer needed.
- **Render quality experiments (proposed by Claude 2026-10-08, not run yet; Codex runs them on
  the PC because Claude cannot reach it).** Elis says ComfyUI `render_00054_` was the best so far
  and wants renders 61+ improved before adding much. Start from the 00054 graph (drag it into
  ComfyUI), keep its seed, change one thing per run, same shot each time:
  (Items 1, 2, 3 and 5 are now built into the app; the test plan above replaces them.)
  1. Resolution: 832×480 (Wan 1.3B's training size) vs our 576×320.
  2. Reference: none vs source first frame vs one image already in the target style.
  3. Control: Canny 0.2/0.5 (now) vs softer edges (thresholds 0.3/0.7, or a slight blur first),
     then depth: Elis installed `comfyui_controlnet_aux`, and profile `RTX3070_DEPTH`
     (workflow `v2v_3070_depth`, Depth Anything V2 Small) now does this from the studio.
     Hypothesis: edges cause the outline/clay look and broken fingers; depth keeps layout
     without lines.
  4. Steps 20 vs 30 at the best setting so far.
  5. Elis (renders 62/63): the room looks great (new tiles, towels) but the ape looks bad.
     The app now keeps the real ape automatically, by compositing it over the render (see
     Current work). The better-blended next step is to feed the same masks into
     WanVaceToVideo's `control_masks` (white = regenerate the room, black = keep the source
     ape), adapted from a published VACE inpainting workflow.
  Later, post only: an upscale model (core node, just a model file) and RIFE interpolation
  (`ComfyUI-Frame-Interpolation`) for 16→32 fps.
  **Report back through git:** record the setting changed, QC score, render time and Elis's
  opinion in the Log. Do not commit contact sheets, frames or clips derived from the private
  Ape source.
- **Codex (2026-10-08):** reliability/UI upgrade is implemented, tested and deployed locally.
  This commit releases the previous file ownership; no further edit is in progress. The
  next useful work is real character/hand fidelity, repair comparison strategy, and a clear
  user flow for projects stopped at the total render budget. Do not promise that increasing
  source strength improves quality: the real comparison showed the opposite.

## Log (newest first)

- 2026-10-09 Claude: before a kept-subject render, the worker checks the keep workflow
  against the live ComfyUI (once per worker). If a node is missing, the shot uses the plain
  workflow plus the composite, and the reason is shown under Applied render settings.
- 2026-10-09 Codex: live ComfyUI profile preflight checks the actual `/object_info` node list.
  Missing custom nodes disable the affected profile in the New form and reject UI/API/CLI
  project creation before uploads or database records are created. On this PC it disables
  `RTX3070_DEPTH` for missing `DepthAnythingV2Preprocessor`; `RTX3070_QUALITY` remains usable.
  Full tests, `ruff` and `mypy` pass. Claude's new `v2v_3070_keep` passes live workflow checks.
- 2026-10-09 Claude: **online research applied** (`docs/research/`, sources in each
  `params.yaml`).
  - New workflows `v2v_3070_keep` and `v2v_3070_depth_keep`, adapted from our Wan graphs plus
    Comfy-Org's VACE inpainting template and Civitai mask wiring. A kept subject now uses
    them automatically.
  - Reference `auto` (subject cutout on white, or none), plus `--reference`, `--canny` and
    `--seed` on `render`.
  - 480P cap and a turning size box (landscape 1920×1080 renders at 832×464, not 576×320).
  - Wan's default negative prompt as the base.
  - Tested here with compile tests, a fake ComfyUI and a fake mask; not yet run on the PC.
- 2026-10-09 Claude: **Main subject** handling (`docs/subject.md`).
  - A New video question (Auto, Keep it real, Restyle it too), plus `--subject` and
    `creative.subject`, with a live hint of what Auto will do.
  - A rule-based decision recorded in the manifest.
  - Keep mode: U²-Net masks (onnxruntime on the CPU, model checksum-verified into
    `data/models`), a grown and feathered edge, a colour shift toward the new room, and a
    composite after each render. The raw render is kept as `render_raw`.
  - If masks are unavailable it falls back to the render and says why.
  - Tested here with a fake mask in the pipeline and the real u2net on synthetic clips. Not yet
    on the PC.

- 2026-10-08 Claude: `RTX3070_DEPTH` profile and `v2v_3070_depth` workflow (Depth Anything V2
  Small via `comfyui_controlnet_aux` as the VACE guide instead of Canny). Tested here only by
  compile/validation tests; not yet run on the PC. Run `comfy-check` first.
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
