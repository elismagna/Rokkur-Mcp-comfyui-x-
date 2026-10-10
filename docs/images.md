# Pictures: generate, change, repaint, extend, upscale

The studio makes and edits still pictures through the same ComfyUI it renders video with.
Dashboard: **Pictures** (`/ui/images`). API: `/images`. CLI: `rokkur-studio image`,
`rokkur-studio image-list`. Every picture is an `Image` row with its files under
`data/images/<id>/`; a picture made from another one points at it (`parent_id`), so edits form a
chain you can walk back.

## Operations

| Operation | What happens | Profile and workflow |
|---|---|---|
| Create | text to image, up to `images.max_batch` pictures per request | `KLEIN_4B` → `img_klein_t2i`; `ZIMAGE_TURBO` → `img_zimage_t2i` |
| Change | an instruction ("make it night") applied to a picture; the output keeps its size | `KLEIN_4B` → `img_klein_edit` |
| Variations | the edit model with a fixed variation instruction and a new seed | `KLEIN_4B` → `img_klein_edit` |
| Repaint | the area you paint is regenerated from the prompt, the rest is kept pixel for pixel | `KLEIN_4B` → `img_klein_inpaint` |
| Extend | the picture is padded on the sides you choose and the border is generated | `KLEIN_4B` → `img_klein_outpaint` |
| Upscale | RealESRGAN x4, brought down to 2× when asked | `UPSCALE` → `img_upscale` |

Sizes are rounded to multiples of 16 and capped at the profile's `max_pixels` (1 MP by
default); pictures you give the studio are scaled down to that area first and never scaled up.
The distilled models take no negative prompt and want guidance 1. A fixed seed makes a request
reproducible; a batch uses the seed for the first picture and counts up when the GPU had to
render them one at a time.

Profiles live in `config/image_profiles.yaml`; defaults for the page in `config/studio.yaml`
under `images:`. Each profile maps an operation to a template in `workflows/`, so a new model
is a new profile and a new template, not new code.

## Models to install

All weights go into ComfyUI's models folder (`scripts/install-models.ps1` moves them from
Downloads):

| File | Folder | From | Licence |
|---|---|---|---|
| `flux-2-klein-4b-fp8.safetensors` | `diffusion_models` | huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8 | Apache-2.0 |
| `qwen_3_4b.safetensors` | `text_encoders` | already on the PC (Z-Image) | Apache-2.0 |
| `flux2-vae.safetensors` | `vae` | huggingface.co/Comfy-Org/flux2-dev (split_files/vae) | Apache-2.0 |
| `ae.safetensors` | `vae` | huggingface.co/Comfy-Org/z_image_turbo (split_files/vae); only for `ZIMAGE_TURBO` | Apache-2.0 |
| `RealESRGAN_x4plus.pth` | `upscale_models` | github.com/xinntao/Real-ESRGAN releases v0.1.0 | BSD-3-Clause |

FLUX.2 [klein] 4B is the choice for the 8 GB RTX 3070: the fp8 weights need about 6 GB, it
renders in 4 steps, and one model does generation, editing, repainting and extension.
Z-Image-Turbo (6B, already downloaded on the PC except its VAE) gives a second look for
text-to-image; Comfy-Org says it fits 16 GB, so on 8 GB ComfyUI offloads weights to RAM and it is
slow. Then run `scripts\studio.ps1 comfy-check`: every `img_*` template is validated against the
live `/object_info`, including the model file names.

## How a request runs

`services/images.py:request_images` validates the request, prepares the input files (scaling,
the alpha mask for a repaint, the padding for an extension), creates one row per picture and
queues one `image` job. `pipeline/images.py:image_job` takes a GPU lease of the profile's class
(so a picture never runs beside a video render; Ollama is unloaded first), uploads the input,
compiles the template and polls ComfyUI exactly as video shots do. A CUDA OOM walks the
profile's `degrade` ladder (`clear_cache`, `single_image`, `reduce_resolution`), never
resubmitting identical work; after the ladder the pictures fail with that reason. Cloud mode
works the same way through the cloud server and records `cloud_gpu_minutes`.

The repaint mask comes from the page's brush as a PNG (white = repaint). The studio turns it into
the source's alpha channel (`FFmpeg.alpha_from_mask`), which is how ComfyUI's own inpaint example
passes a mask, and `GrowMask` widens it by the blend edge you set. **Select the subject** asks the
same CPU subject model the video pipeline uses (`pipeline/subject.py`) for the main subject;
**Select the background** inverts it.

## Hand-offs to video

- **Use as a video reference** opens New video with the picture as the appearance reference
  (`character_reference_path`), so Wan is shown the look you approved.
- **Use as thumbnail** copies the picture into a finished video's thumbnails; the next upload
  uses it.
- Pictures with a verdict are kept out of nothing: the verdict is for you, the taste profile
  learns only from video ratings.

## Not done, and honest limits

- No real GPU has rendered these templates yet. The graphs are adapted node for node from the
  Comfy-Org klein and Z-Image templates and ComfyUI's inpaint/outpaint/upscale examples, and
  the studio's compiler, the fake ComfyUI tests and `comfy-check` cover the wiring; the first
  real render on the PC is the acceptance test (`docs/AI_HANDOFF.md`).
- Repainting with klein uses the sampler's noise mask around the edit model's reference; the
  edge quality on real pictures is unverified. If it disappoints, FLUX.1 Fill (a true inpaint
  model, 12B, too big for 8 GB without offloading) is the published alternative.
- There is no text rendering or typography model, no multi-reference composition, no image
  ControlNet in the studio yet (the Z-Image Fun Union ControlNet template exists upstream and
  would be the next adaptation).
