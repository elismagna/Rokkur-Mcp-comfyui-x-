# Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Project stuck in `RIGHTS_PENDING` | Rights are unknown or lack evidence. Check `/ui/approvals`; decide with `POST /projects/{id}/rights`. |
| `FAILED` with `no_source_file` | A platform source without a file. Studio never downloads; upload the permitted file or set `local_path`. |
| `FAILED` with `template_error` | The profile's workflow template is not installed or does not match. See `docs/comfyui.md`, run `comfy-check`. |
| `renderer_unavailable` retries | ComfyUI is not reachable at `comfyui.url`. From Docker it must be `http://host.docker.internal:8188`. |
| `oom_unrecoverable` | The OOM ladder ran out. Use `PREVIEW`, shorten shots (lower `max_frames`), or free VRAM (close other GPU apps). |
| `budget_exceeded` | `render.max_renders_per_project` or GPU minutes reached. Raise the limit in config, then `POST /projects/{id}/resume`. |
| `repair budget exhausted` | QC kept failing after `render.max_retries` rounds. Inspect the QC table, adjust the brief or profile, raise the limit, resume. |
| Jobs not running | No worker: `docker compose ps`, `make logs`. `/workers` shows workers holding leases and queue depth. |
| A crashed worker's job | Its lease expires (`jobs.lease_seconds`); another worker reclaims it automatically, counting one attempt. |
| GPU busy forever | A lease outlived a crash; it expires after `gpu.lease_seconds`. See `/gpu/leases`. |

Logs are JSON lines on stdout with `job_id`, `project_id`, `stage`, `kind`, `agent_id`.
Every project's audit trail is at `GET /projects/{id}/events` and on its dashboard page.
