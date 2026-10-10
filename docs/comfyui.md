# ComfyUI integration

Studio uses ComfyUI only through its HTTP API (`comfyui/client.py`): `POST /upload/image`
(input clip), `POST /prompt`, `GET /history/{id}` (polled), `GET /queue`, `POST /interrupt`
/ `POST /queue {"delete"}` (cancel), `GET /view` (outputs), `POST /free` (unload models),
`GET /object_info` and `GET /system_stats` (checks). The GUI is never automated.

## Templates

A template is a directory in `workflows/`:

```
workflows/<name>/workflow.json   # exported with "Save (API Format)"
workflows/<name>/params.yaml     # semantic parameter → node/input map
```

```yaml
name: v2v_3070_quality
version: 1
resource_class: GPU_HEAVY
parameters:
  STYLE_PROMPT: {node: "6", input: text, type: str, required: true}
  INPUT_VIDEO:  {node: "10", input: video, type: str, required: true}
  SEED:         {node: "3", input: seed, type: int, default: 0}
  DENOISE:      {node: "3", input: denoise, type: float, min: 0, max: 1}
  POSE_STRENGTH: {targets: [{node: "21", input: strength}], type: float}
```

The compiler substitutes values, enforces types/ranges/required parameters, never mutates
the template, and reports parameters the template does not accept (`ignored`) instead of
dropping them silently. Application code only uses semantic names: `STYLE_PROMPT,
NEGATIVE_PROMPT, REFERENCE_IMAGE, INPUT_VIDEO, SEED, WIDTH, HEIGHT, FPS, FRAME_COUNT, STEPS,
CFG, DENOISE, POSE_STRENGTH, DEPTH_STRENGTH, IDENTITY_STRENGTH, STYLE_STRENGTH, OFFLOAD,
CHECKPOINT, OUTPUT_PREFIX, MASK_VIDEO, CANNY_LOW, CANNY_HIGH`.

### Shipped

- `v2v_preview`: per-frame img2img over the shot clip using ComfyUI **core** nodes only
  (`LoadVideo`, `GetVideoComponents`, `ImageScale`, `VAEEncode`, `KSampler`, `VAEDecode`,
  `CreateVideo`, `SaveVideo`; node and input names checked against ComfyUI's
  `comfy_extras/nodes_video.py`). Needs a 2025+ ComfyUI and an SD1.5 checkpoint
  (`CHECKPOINT`, default `v1-5-pruned-emaonly.safetensors`). It flickers by design; it proves
  the render path.
- `v2v_3070_quality` (profiles `PREVIEW`, `RTX3070_QUALITY`, `FUTURE_24GB`): Wan 2.1 VACE 1.3B,
  core nodes only. The source clip's Canny edges guide VACE; an optional reference image
  (node 20) sets the look.
- `v2v_3070_depth` (profile `RTX3070_DEPTH`): the same graph with a depth map from Depth
  Anything V2 (`DepthAnythingV2Preprocessor`, `comfyui_controlnet_aux` add-on) as the guide
  instead of edges. It defaults to the Small depth model, the only size under Apache-2.0
  (Base/Large/Giant are non-commercial). Experiment for fur, hands and the outline look.
- `v2v_3070_keep` and `v2v_3070_depth_keep` (each profile's `keep_workflow`): the two graphs
  above plus a subject mask (`MASK_VIDEO`, core nodes `ImageToMask`, `ThresholdMask`,
  `InvertMask`, `ImageCompositeMasked`). Wan regenerates the room and keeps the real subject.
  They run only when a project keeps its subject and its masks exist (`docs/subject.md`).
  Adapted from Comfy-Org's VACE inpainting template (MIT) and the mask wiring of Civitai
  workflows 1605242, 1470557 and 1680850; sources are in their `params.yaml`, research notes
  in `docs/research/`.

- `img_klein_t2i`, `img_klein_edit`, `img_klein_inpaint`, `img_klein_outpaint`, `img_zimage_t2i`
  and `img_upscale`: still pictures (`docs/images.md`), adapted from the Comfy-Org FLUX.2
  [klein] and Z-Image-Turbo templates and ComfyUI's inpaint, outpaint and upscale examples.
  Their semantic names: `PROMPT, SOURCE_IMAGE, WIDTH, HEIGHT, BATCH, STEPS, CFG, SEED,
  MEGAPIXELS, MASK_GROW, PAD_LEFT/TOP/RIGHT/BOTTOM, FEATHER, SCALE_BY, UPSCALE_MODEL`.

`CANNY_LOW` / `CANNY_HIGH` set the Canny thresholds (default 0.2 / 0.5). The CLI takes
`--canny LOW HIGH` and the API `creative.canny_low` / `canny_high`, for A/B runs against
0.4 / 0.8, the node defaults kept in Comfy-Org's VACE v2v template.

Profiles cap Wan 1.3B at 480P (`max_pixels: 399360`, 480×832), the size it was trained on, and
the size box turns with the source, so a landscape clip renders at 832×464 instead of
576×320. `negative_base` puts Wan's own default negative prompt in front of ours
(`manifest/builder.py:negative_prompt`).

### Adding or changing a workflow: start from a proven one

Elis's standing rule (2026-10-08): never build a workflow graph from scratch. Start from the
closest proven, published workflow, then adapt it to our parameters. Good sources:

- the official ComfyUI and Comfy-Org examples, or docs.comfy.org;
- a node pack's own `example_workflows`;
- a well-used community workflow, for example on Civitai.

Write the source URL and its license in the comments of `params.yaml`. Commit a copy of the
original only when its license allows redistribution. When an adapted workflow works on the
PC, note that in the handoff log. Leave the rule aside only when adapting would break the
pipeline, and say why.

### To add from your installation

`HYBRID_MAX` expects `hybrid_quality`, which depends on your models and custom nodes, so
Studio does not guess it. Export the workflow in API format into
`workflows/hybrid_quality/workflow.json`, write `params.yaml`, then run
`make comfy-check`, which validates every template against your live `/object_info`
(installed node classes, input names and model filenames). Until then, that profile is
refused when a project is created.

## Renderer selection

`render.renderer: ffmpeg_preview` (default) renders with a deterministic FFmpeg colour grade,
no GPU, so the whole pipeline is testable. Switch to `comfyui` once `comfy-check` passes.

## VRAM (8 GB)

Each render takes a GPU lease of the profile's `resource_class`. Heavy leases are exclusive
(`gpu.max_heavy_jobs: 1`). Before a heavy lease, loaded Ollama models are unloaded
(`keep_alive: 0`); after it, `POST /free` asks ComfyUI to unload models. A CUDA OOM
(`torch.OutOfMemoryError` / "CUDA out of memory" in the execution error) triggers the
profile's `degrade` ladder; identical work is never resubmitted.
