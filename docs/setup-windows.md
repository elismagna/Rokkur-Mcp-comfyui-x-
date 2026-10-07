# Setup on the Windows workstation

1. Install Docker Desktop (WSL 2 backend) and Git. ComfyUI and Ollama stay as they are.
2. Clone or copy this repository, then in PowerShell:
   ```powershell
   copy .env.example .env      # set POSTGRES_PASSWORD; keep STUDIO_RENDER__RENDERER=ffmpeg_preview for now
   .\scripts\studio.ps1 up
   .\scripts\studio.ps1 audit        # writes data\audit.json: ComfyUI nodes, GPU, Ollama models
   .\scripts\studio.ps1 smoke-test   # full pipeline on a synthetic clip
   ```
3. Open http://127.0.0.1:8400/ui.
4. Check that Docker can reach ComfyUI: `.\scripts\studio.ps1 comfy-check`. ComfyUI listens
   on `127.0.0.1` by default; if the check cannot connect, start ComfyUI with `--listen` and
   allow port 8188 only from the Docker/WSL network in Windows Firewall. Never expose it to
   an untrusted network. The same applies to Ollama (`OLLAMA_HOST`).
5. Add your quality workflow (see `docs/comfyui.md`),
   then set `STUDIO_RENDER__RENDERER=comfyui` in `.env` and `.\scripts\studio.ps1 up`.
6. Put source videos and character images in `.\media` (or set `STUDIO_MEDIA_DIR`), and use
   `/media/<file>` paths in projects.

If PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
