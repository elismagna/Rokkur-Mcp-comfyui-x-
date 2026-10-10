# Local and cloud rendering

Every video renders either on **this PC** (your RTX 3070 through ComfyUI, free) or on a
**cloud server**: a ComfyUI you rent or run on another machine with a bigger GPU. You pick per
video on the New video page under **Where to render**. Everything else (rights, analysis, the
agents, masks, quality checks, the edit and publishing) still runs on your PC; only the
ComfyUI renders move.

Cloud mode is off until you set it up. Built and tested here against a fake ComfyUI server
(`tests/test_cloud.py`); **not yet tried against a real cloud GPU**.

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
   - **SSH tunnel (recommended).** Most providers give SSH access. On your PC run
     `ssh -N -L 8189:127.0.0.1:8188 <user>@<server>` and keep it open while rendering. Then
     the studio reaches the cloud ComfyUI at `http://host.docker.internal:8189`, encrypted, and
     the server needs no public port.
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
  `FUTURE_24GB` and `CLOUD_14B` become available for cloud videos (marked *cloud only* in the
  list). `HYBRID_MAX` still needs its `hybrid_quality` workflow, which does not exist yet.
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
