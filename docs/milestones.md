# Milestone plan

Each milestone ends with runnable behaviour and passing tests.

| # | Milestone | Done when | State |
|---|---|---|---|
| 0 | Audit, ADR, plan | `docs/current-state.md`, ADR 0001, this plan | Done (remote audit blocked; `audit` command provided) |
| 1 | Core studio | `POST /projects` persists; state machine with audited transitions; Postgres job queue + worker; asset store; config; Docker Compose | Done (Compose file validated; image build not run in the build sandbox, which blocks Debian/PyPI TLS) |
| 2 | ComfyUI | Client (submit/monitor/cancel/history/outputs/free), template registry + compiler, GPU lease, OOM classification and degradation ladder; mocked render tracked end to end | Done (tested against a fake ComfyUI; real-instance run pending) |
| 3 | Video pipeline | FFmpeg service; ingest, probe, scene detection, shot plan, manifest; a real test video probed, split, rendered (preview renderer), QC'd, repaired, encoded | Done |
| 4 | Rökkur agents | Scout, Trend Analyst, Creative Director, Video Analyst, Workflow Planner, QC, Repair Planner on real providers with schemas | Creative Director + Channel Manager run on Ollama (qwen3.5:9b) with rule-based fallback; `agent-check` verifies. Scout/Trend Analyst wait for Phase 5; Video Analyst and QC stay deterministic by design. |
| 5 | YouTube discovery | Data API search/videos.list with quota ledger + cache, scoring, rights classification, dedupe | Not started |
| 6 | Publishing | OAuth installed-app flow, resumable upload, thumbnail, schedule, approval gate | Dry-run metadata + gates done; OAuth/upload not started |
| 7 | Community | Comment fetch, classification, reply policy, moderation queue | Not started |
| 8 | Analytics | Scheduled pulls, snapshots, dashboard, experiments | Not started (tables designed) |
| 9 | Learning | Skill candidates with verification lifecycle | Not started |
| 10 | Autonomy | Scheduled discovery → publish within gates | Not started |
