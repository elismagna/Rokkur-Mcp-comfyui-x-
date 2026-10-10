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

## Start the cloud GPU with the studio (RunPod)

With a RunPod API key the studio switches your pod on and off for you: the desktop icon asks
whether to turn the cloud GPU on, starts the pod, starts ComfyUI on it and opens the tunnel.
Built and tested here against a fake RunPod API and a stand-in `ssh`; **not yet run on
Windows**.

1. **An API key.** In RunPod open *Settings → API Keys* and create a key. Give it the least
   access that works (read/write on Pods if RunPod offers that choice). Treat it like a
   password: it can start pods that bill your account.
2. **The pod's id**, shown on the pod in RunPod's *Pods* page.
3. Add to `.env` next to the lines above, then run `.\scripts\studio.ps1 up`:
   ```
   STUDIO_CLOUD__RUNPOD_API_KEY=<your key>
   STUDIO_CLOUD__RUNPOD_POD_ID=<the pod id>
   STUDIO_CLOUD__SSH_KEY=                  # empty: %USERPROFILE%\.ssh\id_ed25519
   ```
4. **The desktop icon:** `.\scripts\studio.ps1 shortcut` puts *Rökkur Studio* on your
   desktop with the studio's logo.

Double-clicking the icon starts Docker Desktop if needed, starts the studio, then asks
*Turn on the cloud GPU?* with the GPU and RunPod's hourly price. **Yes** starts the pod
(a minute or two), starts ComfyUI on it over SSH if it is not running, opens a hidden SSH tunnel
on port 8189 and waits until ComfyUI answers. **No** renders on this PC only. If the pod is
already on, the icon connects without asking. If anything fails, a message says why and the
studio opens anyway, rendering on this PC. The same steps from PowerShell:
`.\scripts\studio.ps1 launch` (`-Cloud` skips the question, `-NoCloud` skips the cloud),
`cloud-on`, and `cloud-pod status`.

**Stopping it.** A running pod bills by the hour, rendering or not.
- The rail shows a *Cloud GPU on* chip while the pod runs. The System page's **Cloud GPU**
  card has **Stop the cloud GPU** (refused while something renders on it).
- The worker stops the pod by itself after `cloud.auto_stop_idle_minutes` (default 30) without
  cloud work: no cloud job running or due, and an empty queue on the cloud ComfyUI. Set it to
  0 to turn this off.
- `.\scripts\studio.ps1 cloud-off` stops the pod and closes the tunnel.

The studio only ever *stops* the pod. It never terminates it, so `/workspace` (ComfyUI and the
models) stays. RunPod still charges a little for a stopped pod's disk.

**Notes**
- `cloud.ask_on_launch: false` (`STUDIO_CLOUD__ASK_ON_LAUNCH=false`) stops the question; the
  icon then connects to the cloud GPU only when it is already on.
- The icon uses the pod's public IP and SSH port from RunPod's API, which means the pod needs
  SSH over exposed TCP, the `ssh root@<ip> -p <port>` address. Your public key must be on the
  pod (see RunPod notes below).
- SSH runs without prompts, so a key with a passphrase must be loaded into the Windows
  ssh-agent first (`ssh-add`).
- The pod's host key is kept in `%USERPROFILE%\.ssh\rokkur_runpod_known_hosts`, separate from
  your normal `known_hosts`. When a restarted pod reuses an address, the old key for that
  address is forgotten first.
- ComfyUI on the pod is started from `cloud.remote_comfy_dir` (default `/workspace/ComfyUI`)
  on `cloud.remote_comfy_port` (8188), with its log in `/workspace/comfyui.log`.
- The tunnel closes when you sign out or restart Windows. Double-click the icon again to
  reconnect.

## What changes for a cloud video

- **Where to render** on the New video page offers *This PC* and *Cloud server*. The API takes
  `creative.render_on: "local" | "cloud"`. The choice is stored on the video; videos made
  before cloud mode render locally.
- Render profiles are checked against the server they render on. With a 24 GB cloud GPU,
  `FUTURE_24GB` and `CLOUD_14B` become available for cloud videos (marked *cloud only* in the
  list). `HYBRID_MAX` still needs its `hybrid_quality` workflow, which does not exist yet.
- Your PC's GPU is **not locked** for a cloud render, so Ollama and the local ComfyUI are not
  unloaded for it.
- Each render's time (upload, queue, render and download) is recorded as *cloud GPU minutes*,
  with an estimated cost
  from `price_per_hour_usd`. The video page shows both. **This is an estimate:** a rented
  server bills for every hour it is switched on, rendering or not. Stop it when you are done
  (with RunPod the studio can stop it for you, see above).
- `costs.max_cloud_gpu_minutes` (default 60 per video) and the optional dollar cap stop a
  video at its budget, like the local GPU-minute budget. *Allow more renders* raises both.
- If the cloud server is off or unreachable, the render job waits and retries like a local
  ComfyUI outage. A cloud video never falls back to your PC on its own.

## The Wan VACE 14B profile (CLOUD_14B)

`CLOUD_14B` renders with Wan 2.1 VACE **14B**, the larger sibling of the 1.3B model your PC
uses, at up to **720P** (1280x720 area; the 1.3B model is limited to 480P). It is made for a
24 GB cloud GPU and is never offered for this PC.

- **Workflows:** `workflows/v2v_cloud_14b` and `v2v_cloud_14b_keep` (for a kept subject). They
  follow Comfy-Org's official "Wan2.1 VACE 14B video to video" template (MIT) and are our
  1.3B graphs node for node except the model loader, so every studio option works the same.
  The template's CausVid speed-up LoRA is left out: its licence (CC-BY-NC) does not allow a
  channel that may earn money. Sampling uses the template's own non-LoRA settings: 20 steps,
  cfg 6, uni_pc/simple, shift 8, 16 fps, up to 81 frames.
- **Fitting 24 GB:** Comfy-Org publishes the 14B model only in fp16 (34.7 GB). The workflow
  loads it with ComfyUI's core `UNETLoader` set to `weight_dtype: fp8_e4m3fn`, which converts
  the weights to fp8 while loading, about 17 GB on the GPU. No custom nodes are needed.
- **Slow:** Comfy-Org notes about 40 minutes for one 81-frame shot at 720P on an RTX 4090
  without the speed-up LoRA, which is how this profile renders. Raise
  `STUDIO_COSTS__MAX_CLOUD_GPU_MINUTES` (default 60 per video) before a long video, or pick
  **Render size** 67% under advanced controls, which renders about 480P.
- **If it runs out of memory** the studio retries the shot smaller, then at fewer frames per
  second. It never switches to the 1.3B model mid-video, which would change the look between
  shots.

### Setting up the cloud server for it

The server needs these files (sizes from the Hugging Face API, 2026-10-10; all from
`Comfy-Org/Wan_2.1_ComfyUI_repackaged`, Apache-2.0):

| File | Folder under `ComfyUI/models` | Size |
|---|---|---|
| `wan2.1_vace_14B_fp16.safetensors` | `diffusion_models` | 34,675,323,640 bytes (32.3 GiB) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders` | 6,735,906,897 bytes |
| `wan_2.1_vae.safetensors` | `vae` | 253,815,318 bytes |

On a RunPod-style server where ComfyUI lives in `/workspace/ComfyUI`:

```
cd /workspace/ComfyUI/models/diffusion_models
wget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors
stat -c %s wan2.1_vace_14B_fp16.safetensors   # 34675323640 when complete
sha256sum wan2.1_vace_14B_fp16.safetensors    # f202a5c59b8a91ada1862c46a038214f1f7f216c61ec8350d25f69b919da4307
```

The text encoder and VAE download the same way into `text_encoders/` and `vae/` (replace the
folder and file name in the URL). Then run `.\scripts\studio.ps1 comfy-check --cloud`: it
reports `v2v_cloud_14b` as ok once the model is in place. On your PC, `comfy-check` lists the
14B workflows as "only for profiles this server cannot run" and does not fail on them.

Plan for (not yet measured on a real server):
- **Disk:** about 42 GB for these three files, so a volume of 60 GB or more with ComfyUI.
- **System RAM:** loading reads the 34.7 GB fp16 file before it is converted to fp8, so choose
  a server with at least 64 GB of RAM; with less, loading may swap or be killed.
- **GPU:** 24 GB. `STUDIO_CLOUD__VRAM_GB` (default 24) must say 24 or more, or the profile
  shows as unavailable for cloud videos too.

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
  It does not start by itself when the pod restarts; the desktop icon starts it for you (see
  above). Start it only once: a second copy fails on the `comfyui.db` lock, which is harmless.
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
