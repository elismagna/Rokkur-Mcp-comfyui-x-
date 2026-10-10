# Local and cloud rendering

Every video renders either on **this PC** (your RTX 3070 through ComfyUI, free) or on a
**cloud server**: a ComfyUI you rent or run on another machine with a bigger GPU. You pick per
video on the New video page under **Where to render**. Everything else (rights, analysis, the
agents, masks, quality checks, the edit and publishing) still runs on your PC; only the
ComfyUI renders move.

Cloud mode is off until you set it up. Built and tested against a fake ComfyUI server
(`tests/test_cloud.py`). **Checked on a real cloud GPU on 2026-10-09:** a RunPod RTX 4090 (24 GB)
reached through an SSH tunnel. `comfy-check --cloud` reported the 4090, and `v2v_3070_quality` and
`v2v_3070_keep` passed. A first cloud render reached the workflow stage and exposed the
FUTURE_24GB size bug that is now fixed. A finished cloud render is not confirmed yet.

## What you need

1. **A ComfyUI server with a GPU.** Any provider that gives you a machine with ComfyUI works
   (for example RunPod, Vast.ai, or your own second PC). Rökkur only talks to ComfyUI's normal
   HTTP API, so nothing provider-specific is needed.
2. **The same models and custom nodes as your PC.** The cloud ComfyUI must have the files your
   workflows load (Wan 2.1 VACE 1.3B, umt5 text encoder, Wan VAE, and any node packs your
   workflows use). Check it with `.\scripts\studio.ps1 comfy-check --cloud`: it lists every
   workflow and any node the server is missing.
3. **A private way in.** Never leave a ComfyUI open to the internet: anyone who finds it can
   use your GPU and see your clips. Two safe options:
   - **SSH tunnel (recommended).** Most providers give SSH access. On your PC (not on the
     server) run this and keep the window open while rendering:
     ```
     ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 8189:127.0.0.1:8188 <user>@<server> -p <port> -i $env:USERPROFILE\.ssh\id_ed25519
     ```
     The studio then reaches the cloud ComfyUI at `http://host.docker.internal:8189`. The
     connection is encrypted and the server needs no public port. Docker Desktop on Windows
     reaches this 127.0.0.1-only tunnel, so it never has to be opened to your network.
     `ServerAliveInterval` keeps an idle tunnel from being dropped. `ExitOnForwardFailure`
     makes the tunnel stop with an error if port 8189 is already taken, instead of failing
     silently.
   - **An HTTPS address that checks a token**, such as a reverse proxy in front of ComfyUI.
     Rökkur sends the token as `Authorization: Bearer <token>` on every request (header and
     scheme are configurable). A provider's plain public proxy link usually has no login, so
     don't use one without a token check in front.

## Turn it on

Add to `.env` on your PC (never to `config/studio.yaml` or git), then restart the studio:

```
STUDIO_CLOUD__ENABLED=true
STUDIO_CLOUD__URL=http://host.docker.internal:8189
STUDIO_CLOUD__TOKEN=                    # only for a token-checking address
STUDIO_CLOUD__VRAM_GB=24                # the cloud GPU's memory
STUDIO_CLOUD__PRICE_PER_HOUR_USD=0      # what the provider charges an hour, for the estimate
STUDIO_COSTS__MAX_CLOUD_GPU_MINUTES=60  # per video
```

Optional: `STUDIO_CLOUD__DEFAULT=cloud` preselects Cloud for new videos;
`STUDIO_CLOUD__AUTH_HEADER` and `STUDIO_CLOUD__AUTH_SCHEME` change how the token is sent;
`STUDIO_COSTS__MAX_COST_PER_PROJECT_USD` stops a video when its estimated cloud cost reaches
that many dollars.

The System page then lists **Cloud ComfyUI** with its host, memory and price (never the token).

## What changes for a cloud video

- **Where to render** on the New video page offers *This PC* and *Cloud server*. The API takes
  `creative.render_on: "local" | "cloud"`. The choice is stored on the video; videos made
  before cloud mode render locally.
- Render profiles are checked against the server they render on. With a 24 GB cloud GPU,
  `FUTURE_24GB` becomes available for cloud videos (marked *cloud only* in the list).
  `HYBRID_MAX` still needs its `hybrid_quality` workflow, which does not exist yet.
- Your PC's GPU is **not locked** for a cloud render, so Ollama and the local ComfyUI are not
  unloaded for it.
- Each render's time (upload, queue, render and download) is recorded as *cloud GPU minutes*,
  with an estimated cost
  from `price_per_hour_usd`. The video page shows both. **This is an estimate:** a rented
  server bills for every hour it is switched on, rendering or not. Stop it when you are done.
- `costs.max_cloud_gpu_minutes` (default 60 per video) and the optional dollar cap stop a
  video at its budget, like the local GPU-minute budget. *Allow more renders* raises both.
- If the cloud server is off or unreachable, the render job waits and retries like a local
  ComfyUI outage. A cloud video never falls back to your PC on its own.

## Privacy

A cloud render uploads each shot's clip, masks and reference image to the cloud server and
downloads the result. Use cloud only for footage you are happy to send to that provider, and
delete it from the server afterwards (ComfyUI keeps inputs in its `input/` folder and results
in `output/`).

## Troubleshooting

- **`Server disconnected without sending a response`** from `comfy-check --cloud` means the studio
  reached the tunnel port but nothing answered behind it. Run
  `curl.exe -s http://127.0.0.1:8189/system_stats` on your PC:
  - If it prints nothing, the tunnel has stopped. Restart it.
  - To check the server side, log in and run `curl -s 127.0.0.1:8188/system_stats` there. If
    that also prints nothing, ComfyUI on the server has stopped.
- **`Cloud rendering is not set up`** means `.env` has no `STUDIO_CLOUD__ENABLED=true` and
  `STUDIO_CLOUD__URL` lines, or the studio was not restarted after adding them
  (`.\scripts\studio.ps1 up`).

## RunPod notes (from the first setup, 2026-10-09)

- **ComfyUI.** A plain PyTorch template has no ComfyUI. Install it into the persistent volume:
  ```
  cd /workspace && git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI && pip install -r requirements.txt
  ```
- **Starting ComfyUI.** Start it with:
  ```
  cd /workspace/ComfyUI && nohup python main.py --listen 127.0.0.1 --port 8188 > /workspace/comfyui.log 2>&1 &
  ```
  It does not start by itself when the pod restarts. Start it only once: a second copy fails
  on the `comfyui.db` lock, which is harmless.
- **Models.** Put the same files your PC uses under `/workspace/ComfyUI/models`, from the
  Comfy-Org `Wan_2.1_ComfyUI_repackaged` split files (VACE 1.3B fp16, umt5 xxl fp8, Wan VAE).
  The depth, draft and preview workflows also need the DepthAnythingV2 node pack, the
  self-forcing LoRA and the SD1.5 checkpoint. `comfy-check --cloud` lists anything missing.
- **SSH keys.** A key added in RunPod's settings only reaches pods started after you add it.
  For a pod that is already running, write it into `/root/.ssh/authorized_keys` from the Web
  Terminal with one `echo '<public key>' > /root/.ssh/authorized_keys` command.
  The SSH address and port are under the pod's Connect menu, and can change when the pod
  restarts.
- **Stopping the pod.** *Stop* keeps `/workspace` (models and ComfyUI). *Terminate* deletes it.
  A running pod bills by the hour whether it renders or not.
