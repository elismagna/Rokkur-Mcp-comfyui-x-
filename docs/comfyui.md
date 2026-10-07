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
CHECKPOINT, OUTPUT_PREFIX`.

### Shipped

- `v2v_preview`: per-frame img2img over the shot clip using ComfyUI **core** nodes only
  (`LoadVideo`, `GetVideoComponents`, `ImageScale`, `VAEEncode`, `KSampler`, `VAEDecode`,
  `CreateVideo`, `SaveVideo`; node and input names checked against ComfyUI's
  `comfy_extras/nodes_video.py`). Needs a 2025+ ComfyUI and an SD1.5 checkpoint
  (`CHECKPOINT`, default `v1-5-pruned-emaonly.safetensors`). It flickers by design; it proves
  the render path.

### To add from your installation

`RTX3070_QUALITY` and `FUTURE_24GB` expect `v2v_3070_quality`; `HYBRID_MAX` expects
`hybrid_quality`. These depend on your models and custom nodes, so Studio does not guess
them. Export your best 8 GB video-to-video workflow in API format into
`workflows/v2v_3070_quality/workflow.json`, write `params.yaml`, then run
`make comfy-check`, which validates every template against your live `/object_info`
(installed node classes, input names and model filenames). Until then, projects using those
profiles fail fast at `compile_workflow` with a clear message when `render.renderer=comfyui`.

## Renderer selection

`render.renderer: ffmpeg_preview` (default) renders with a deterministic FFmpeg colour grade,
no GPU, so the whole pipeline is testable. Switch to `comfyui` once `comfy-check` passes.

## VRAM (8 GB)

Each render takes a GPU lease of the profile's `resource_class`. Heavy leases are exclusive
(`gpu.max_heavy_jobs: 1`). Before a heavy lease, loaded Ollama models are unloaded
(`keep_alive: 0`); after it, `POST /free` asks ComfyUI to unload models. A CUDA OOM
(`torch.OutOfMemoryError` / "CUDA out of memory" in the execution error) triggers the
profile's `degrade` ladder; identical work is never resubmitted.
