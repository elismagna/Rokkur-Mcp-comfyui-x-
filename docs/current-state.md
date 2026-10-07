# Current state (Phase 0 audit)

Audit date: 2026-10-07. Author: Claude (implementation agent).

## What was inspected

| Item | Result |
|---|---|
| GitHub repositories reachable from the build session | **None.** The account linked to this project exposes no repositories, so there was no existing Rökkur Studio, Odysseus or Rökkur Collective code to read. |
| Elis's Windows workstation (Odysseus, Rökkur Collective, ComfyUI, Docker, Ollama) | **Not reachable.** The account requires trusted devices, and the project's remote-device bridge does not support that yet, so no folder or shell on the workstation could be opened. |
| Build environment | Linux container with Python 3.12/3.13, PostgreSQL 16, FFmpeg 6.1, Docker 29. Used to build and test everything in this repository. |

## Consequences

Because none of the existing components could be inspected, Rökkur Studio is built as a
**new standalone repository** that integrates with the existing tools only through
documented, public interfaces:

| Component | How Studio talks to it | Status |
|---|---|---|
| ComfyUI | Its HTTP API (`POST /prompt`, `GET /history/{id}`, `GET /view`, `POST /upload/image`, `POST /interrupt`, `POST /queue`, `POST /free`, `GET /object_info`, `GET /system_stats`). The GUI is never automated. | Client implemented and tested against a fake server. Not yet run against Elis's real ComfyUI. |
| Ollama | Its HTTP API (`/api/chat` with a JSON-schema `format`, `/api/ps`, `keep_alive: 0` to unload). | Client implemented and tested against a fake server. |
| Rökkur Collective | **Unknown interface.** Exposed in Studio as an `AgentProvider` protocol; the Collective adapter refuses to run until its API is documented (see `docs/agents.md`). Nothing pretends to call it. | Pending audit on the workstation. |
| Odysseus | **Unknown interface.** Studio exposes a REST API with OpenAPI at `/docs` that Odysseus (or anything else) can call. | Pending audit. |
| Docker | Studio ships its own `docker-compose.yml` (Postgres + API + worker) and expects ComfyUI and Ollama to keep running where they already run (the Windows host), reached via `host.docker.internal`. | Compose file builds and runs in CI-like container. |
| Existing databases | None known. Studio owns its own Postgres database `rokkur`. | — |

## Assumptions (to verify on the workstation)

1. ComfyUI listens on `http://127.0.0.1:8188` on the host (Studio in Docker reaches it as `http://host.docker.internal:8188`).
2. Ollama listens on `http://127.0.0.1:11434` on the host.
3. Elis's existing video-to-video ComfyUI workflows use custom nodes (e.g. VideoHelperSuite). Studio does not guess those graphs: they are exported in **API format** from ComfyUI and registered as templates with a parameter map (see `docs/comfyui.md`). `make comfy-check` validates every template against the live `/object_info`.
4. Nothing in this repository modifies Odysseus, Rökkur Collective, ComfyUI or Ollama configuration.

## Workstation audit (2026-10-07, `data/audit.json` from Docker on Elis's PC)

| Check | Result |
|---|---|
| Studio stack | Image builds, Postgres 16.15 up, migrations applied, FFmpeg present in the image. Docker Desktop runs on WSL2. |
| Ollama | **Reachable** at `host.docker.internal:11434`. Nothing loaded at audit time. Installed: `qwen3.5:9b`, `qwen3.5:4b`, `odysseus-vision:9b` (from qwen3.5:9b), `satan-odysseus:9b` and `satan:latest` (qwen3.5 9B), `gemma4:12b`, `gemma3:12b`, `qwen2.5-coder:7b`, `deepseek-r1:8b`, all Q4_K_M. Studio's default agent model is now `qwen3.5:9b` (tools + vision, ~6.6 GB). The `odysseus-*` and `satan*` models look like Odysseus's own; Studio does not use or change them. |
| ComfyUI | **Not reachable** from the containers (`Network is unreachable` on `host.docker.internal:8188`). Either ComfyUI was not running, runs on another port, or listens on 127.0.0.1 only. Node classes and GPU stats are therefore still unknown. |
| Docker CLI inside the container | Not present, by design: the Docker socket is not mounted. |

## Next audit step

Make ComfyUI reachable from Docker (see `docs/troubleshooting.md`), then rerun
`scripts/studio.ps1 audit` and `scripts/studio.ps1 comfy-check`.
