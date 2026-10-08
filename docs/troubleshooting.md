# Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Project stuck in `RIGHTS_PENDING` | Rights are unknown or lack evidence. Check `/ui/approvals`; decide with `POST /projects/{id}/rights`. |
| `FAILED` with `no_source_file` | A platform source without a file. Studio never downloads; upload the permitted file or set `local_path`. |
| `FAILED` with `template_error` | The profile's workflow template is not installed or does not match. See `docs/comfyui.md`, run `comfy-check`. |
| `renderer_unavailable` retries | ComfyUI is not reachable at `comfyui.url`. From Docker it must be `http://host.docker.internal:8188`. |
| `oom_unrecoverable` | The OOM ladder ran out. Use `PREVIEW`, shorten shots (lower `max_frames`), or free VRAM (close other GPU apps). |
| `budget_exceeded` (`Stopped at this project's render budget`) | `render.max_renders_per_project` or `costs.max_gpu_minutes_per_project` reached; repair rounds count too. Look at the renders, then press **Allow N more renders and continue** on the project page (`POST /projects/{id}/allow-more-renders`, half the configured budget by default, with GPU minutes in proportion). Resume is refused while the project is still over budget; raising the limit in config also lets Resume work. |
| `Stopped after N repair rounds` (`repair budget exhausted`) | QC kept failing after `render.max_retries` rounds. Look at the QC table and the renders, then press **Try N more repairs**, **Check quality again** (after a QC change) or **Keep these renders** on the project page. Resume is refused here because it would stop again at once. |
| Jobs not running | No worker: `docker compose ps`, `make logs`. `/workers` shows workers holding leases and queue depth. |
| A crashed worker's job | Its lease expires (`jobs.lease_seconds`); another worker reclaims it automatically, counting one attempt. |
| GPU busy forever | A lease outlived a crash; it expires after `gpu.lease_seconds`. See `/gpu/leases`. |
| Audit says ComfyUI `Network is unreachable` or `Connection refused` | Studio runs in Docker and reaches ComfyUI via `host.docker.internal:8188`. Check `http://127.0.0.1:8188` opens on the PC. ComfyUI portable/manual binds to 127.0.0.1 by default, which Docker cannot reach: start it with `--listen 0.0.0.0 --port 8188` (ComfyUI Desktop: Settings > Server-Config > Host 0.0.0.0) and allow it only on **Private** networks in the Windows Firewall prompt. If ComfyUI uses another port, set `STUDIO_COMFYUI__URL` in `.env`. |

Logs are JSON lines on stdout with `job_id`, `project_id`, `stage`, `kind`, `agent_id`.
Every project's audit trail is at `GET /projects/{id}/events` and on its dashboard page.
