# Create: pictures and clips from a prompt

The **Create** page (`/ui/create`) brings the studio layout of
[Open Higgsfield AI](https://github.com/Anil-matcha/Open-Higgsfield-AI) into Rökkur. That
project is by Anil Matcha, MIT licence, now renamed "Open Generative AI". It runs on Rökkur's
own ComfyUI, on this PC or on the cloud GPU. Videos still come from the New video page. Create
is for quick takes you can look at, rate, animate and turn into a video project.

## What we took from Open Higgsfield, and what we did not

| Open Higgsfield | Rökkur Create |
|---|---|
| Image Studio: text to picture | **Image** tab: Z-Image Turbo, 6 shapes, 1 to 4 takes at once |
| Video Studio: text to video, start frame to video | **Video** tab: Wan 2.1 VACE 1.3B, text to clip or start picture to clip, 2/3/5 s |
| Cinema Studio: camera, lens, focal length, aperture | **Cinema** tab: those four plus shot size, angle, lighting (the director's vocabulary) and 13 camera moves |
| Generation history, upload history | **Your takes** grid with filter, like/dislike, Again, Animate, Save, Delete; recent pictures are offered as start pictures |
| 200+ cloud models through Muapi.ai (paid API key) | Not added. Everything runs on your ComfyUI; a paid engine would be opt-in only |
| Local video via a separate Wan2GP install | Not needed: ComfyUI already runs Wan |
| Lip sync, audio, marketing, agents, "no filters" | Not added |

Rökkur keeps everything the front end lacks: the rights gate, the director, QC, ratings,
Night School and private-by-default YouTube publishing. A finished clip's **Make a video**
button opens New video with the clip as the source and the evidence filled in, so the
full pipeline can restyle it.

## How a take runs

1. The form (or `POST /create/takes`) calls `services/create.py`. It writes the Cinema choices
   into the prompt as plain phrases: framing first, then your words, then the look. The
   page shows that text under **The model reads**.
2. Each take is a `create_take` job (`pipeline/create.py`). On this PC it waits for the GPU
   lease, so it never runs next to a video's shots on the 8 GB card. On the cloud server it
   skips the lease and is logged as `cloud_gpu_minutes`.
3. The result and a thumbnail go to `data/create/<take id>/`. Takes are not projects: they
   never count against a video's render budget.

With the `ffmpeg_preview` renderer, a take is a test pattern at the right size. This lets
you try the page without a GPU.

## Workflows

All three follow the rule that a workflow starts from a proven published one. Each
`params.yaml` lists its source and what changed.

| Kind | Workflow | Source (Comfy-Org workflow_templates, MIT) | Models |
|---|---|---|---|
| Picture | `t2i_zimage_turbo` | `image_z_image_turbo.json`, subgraph flattened, same values (8 steps, cfg 1, res_multistep, shift 3) | `z_image_turbo_bf16`, `qwen_3_4b`, `ae.safetensors` |
| Clip from text | `t2v_3070_vace` | `video_wan_vace_14B_t2v.json` with its own bypassed 1.3B loaders switched on and no CausVid LoRA (20 steps, cfg 6) | VACE 1.3B, umt5 fp8, Wan VAE (already used by videos) |
| Clip from a picture | `i2v_3070_vace` | `video_wan_vace_flf2v.json` image-batch subgraph with only a start frame; the core `ImageBatch` node replaces the newer batch and math nodes | same as above |

The PC already has Z-Image Turbo and qwen_3_4b. `ae.safetensors` (the FLUX VAE) may be
missing; `comfy-check` lists Create workflows as `(Create page)` and names any missing file.
It's on [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) under
`split_files/vae/`.

## Settings

In `config/studio.yaml` under `create:`, each overridable as `STUDIO_CREATE__<KEY>`:

- `enabled`
- `image_workflow`, `video_workflow`, `animate_workflow`
- `video_fps` (16)
- `max_frames` (81, which is 5 s)
- `video_steps` (20)
- `image_steps` (8)
- `negative_prompt` (empty uses Wan's own negative from the default profile)

## API

- `GET /create/options`: shapes, lengths and Cinema choices.
- `GET /create/takes`: recent takes.
- `GET /create/takes/{id}`: one take.
- `POST /create/takes`: make takes. The body is `{"mode": "image" | "video", "prompt", "aspect", "seconds", "seed", "count", "camera": {...}, "start_take", "render_on"}`. A video with `start_take` (a finished picture) animates it.

## Known limits

- Clips are 480P and at most 5 s, which is what VACE 1.3B is made for. On the 4090 the 14B
  model could do 720P, but Create doesn't have a 14B workflow yet.
- Animating a picture with VACE 1.3B keeps the first frame. How well the rest follows it has
  not been tested on the PC yet.
- None of the three workflows has run on the real ComfyUI yet. They are tested against the fake
  ComfyUI, and their graphs are checked link by link.
