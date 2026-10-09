# Rökkur Studio retrospective: current choices vs proven online sources (2026-10-09)

> Copied into the repo on 2026-10-09 from Claude's project notes, so Codex can read it. Paths under `knowledge/` refer to Claude's side: the downloaded workflow JSONs are not committed, because their licences differ. The Comfy-Org templates are at github.com/Comfy-Org/workflow_templates (MIT). What was applied is in `docs/AI_HANDOFF.md`.

Research only, by the Civitai-workflows thread. The repo is unchanged; the "Rökkur Studio build" thread
implements. Repo state compared: `main` at f38e830 (workflows v2v_3070_quality v2, v2v_3070_depth v1,
v2v_preview; config/render_profiles.yaml; director/prompts.py; pipeline/qc.py; pipeline/renderers.py;
manifest/builder.py).

Legend: **[V]** = verified in a downloaded file, source code or HF/GitHub licence file. **[C]** = claimed by a
page or author, not tested by us. Licences: OK = commercial use allowed, **NC** = non-commercial (avoid for
YouTube), ? = unknown.

Companion files:
- Civitai findings: `knowledge/civitai-wan-workflows.md`, JSONs in `knowledge/civitai-workflows/`.
- Official Comfy-Org templates (MIT licence **[V]**), copied unchanged to `knowledge/official-workflows/`:
  `comfy-org_video_wan_vace_14B_v2v.json`, `comfy-org_video_wan_vace_inpainting.json`,
  `comfy-org_video_wan_vace_14B_ref2v.json`, `comfy-org_utility_video_segment_sam3.json`.
  Source: github.com/Comfy-Org/workflow_templates/tree/main/templates.

---

## Top recommendations (ranked by expected gain for effort)

| # | Change | Gain | Effort | Licence | Source |
|---|---|---|---|---|---|
| 1 | Stop using the source's first frame as the VACE reference image; use none, or a subject cutout on white | High (likely cause of half-real "CGI" look) | Low | OK | Comfy-Org ref2v template note **[V]** |
| 2 | Subject-keeping mode built from the official **VACE inpainting template** (SAM3 text mask -> GrowMask -> control_masks), inverted to keep the ape | High (ape stays real; background restyles) | Medium | SAM License (commercial OK) **[V]**; template MIT | `comfy-org_video_wan_vace_inpainting.json` **[V]** |
| 3 | Cap 1.3B renders at 480P area (e.g. 480x832); profiles now allow 576x1024 | Medium-high (stability) | Low | — | Wan2.1 README **[V]** |
| 4 | Canny thresholds 0.4/0.8 (official) instead of 0.2/0.5 | Medium (fewer fur-noise edges -> less outline/clay look) | Trivial | — | Comfy-Org v2v template **[V]** |
| 5 | Wan's official negative prompt as the base, minus the style words for stylised renders | Medium | Low | — | Wan2.1 `shared_config.py` **[V]** |
| 6 | Director prompts: Wan-style 80-100 word prose with simple motion verbs; drop the action strip-out and SD-style weights for Wan | Medium | Medium | — | Wan2.1 `prompt_extend.py` **[V]** |
| 7 | Speed: Self-Forcing DMD 1.3B LoRA, 4 steps, cfg 1 as a "draft" profile (QC repair loops get ~5x cheaper) | High speed, unknown quality | Low | Apache-2.0 **[V]** | Kijai/WanVideo_comfy + Civitai 1719791 **[V]** |
| 8 | QC: add DINOv2-small subject consistency + CLIP background consistency + RAFT/Farneback warp error (VBench method) | Medium (catches melting subjects that edge-QC misses) | Medium | Apache/MIT/BSD **[V]** | VBench (Apache-2.0) **[V]** |
| 9 | Depth control: switch to **Video Depth Anything Small** (temporally consistent) if depth flicker shows | Medium | Medium (custom node) | Apache-2.0 **[V]** | DepthAnything/Video-Depth-Anything **[V]** |
| 10 | Long shots: core `WanContextWindowsManual` (81 frames, overlap 30) instead of hard 81-frame limits/splits | Medium (longer shots without seams) | Low-medium | — | ComfyUI `nodes_context_windows.py` **[V]** |

Details for each follow.

### 1. Reference image: VACE wants an object on a plain background, not a style or scene frame
- **Ours [V]:** `pipeline/renderers.py` lines ~104-108: when no character reference is given and
  `_REFERENCE_MODE == "source"`, the worker grabs the **source clip's first frame** (`ffmpeg.thumbnail(at=0)`)
  and feeds it as `reference_image`.
- **Official [V]** (template note in `comfy-org_video_wan_vace_14B_ref2v.json`): "VACE does not use data with
  style reference for training. Currently, it only has the functions of object or background reference.
  Therefore, at 'Load reference image', you should upload a image with a solid-colored background or a
  background image. ... WanVaceToVideo only supports a single reference_image."
- **Why it matters:** the full real-world first frame tells VACE to reproduce the *real* ape **and** the
  real background as objects. That fights the prompt's style. The result is a mix of photo-real subject
  and restyled surroundings, which reads as CGI. This fits render 62/63 (bad ape, great background) and
  67 (Blender look), but that link is inferred, not tested.
- **Change:** default `_REFERENCE_MODE` to `none` for restyles. For identity, use a **subject cutout on
  white**: rembg u2net (MIT **[V]** via danielgatis/rembg) or BiRefNet (MIT **[V]**). Civitai workflows
  1470557/1605242 do exactly this with `ImageRemoveBackground+` -> `ImageRemoveAlpha #FFFFFF` **[V]**.
  A restyled keyframe of the subject (cut out) is the best identity anchor for "restyle the ape" mode.
- **A/B:** same seed/clip, reference = first frame vs none vs cutout.

### 2. Subject keeping: adapt the official VACE inpainting template
**Template [V]** (`comfy-org_video_wan_vace_inpainting.json`, all **core** nodes, needs a recent ComfyUI:
node versions 0.19-0.21 in the file):
`LoadVideo -> GetVideoComponents -> ImageFromBatch(0, 81) -> ResizeImageMaskNode (dims floor(x/16)*16)`
-> subgraph **SAM3_Detect** (CheckpointLoaderSimple `sam3.1_multiplex_fp16.safetensors` 1.63 GB, text prompt
e.g. "cat", threshold 0.5, refine_iterations 2) -> **GrowMask(expand 20, tapered)** -> `control_masks`.
The same mask goes through InvertMask -> MaskToImage -> **ImageCompositeMasked** onto the frames -> `control_video`,
so the masked region is painted black/filled. Then ModelSamplingSD3 **shift 5**, KSampler 4 steps cfg 1
uni_pc/simple with the CausVid LoRA on (or 20 steps cfg 6 off).
- **Semantics [V]** (`comfy_extras/nodes_wan.py`): mask 1 = regenerate, 0 = keep. As published, the template
  regenerates the detected object. **Keep-the-ape mode:** InvertMask the SAM3 mask before `control_masks`,
  and in `control_video` keep the original pixels inside the ape and put **depth** (not black) in the
  background. **Restyle-the-ape mode:** use the template as is, but fill the ape area with its depth map, so
  silhouette, fur outline and hand pose survive.
- **Auto decision (Elis wanted one question at most):** SAM3 takes a **text prompt** ("ape"), so the app can
  build masks from the brief's subject noun with no clicks. The SAM3 video template
  (`comfy-org_utility_video_segment_sam3.json`) notes a 32-token limit and `name:N` syntax ("eye:2") **[V]**.
- **Licences:** SAM License (Meta) grants use, reproduction and derivatives royalty-free, with trade-control and
  military exclusions, and no non-commercial clause **[V, HF Comfy-Org/sam3.1 LICENSE]**. SAM 2.1
  Apache-2.0 **[V]**. BiRefNet MIT **[V]**. **RMBG-2.0: "bria-rmbg-2.0" licence, gated, non-commercial
  without a BRIA agreement [V licence name; NC terms per BRIA C]: avoid.** RMBG-1.4 is also a BRIA licence
  **[V name]**. ComfyUI-RMBG node pack is GPL-3.0 **[V]** (fine to run, just don't vendor it). MatAnyone
  (video matting) is S-Lab **NC [V]**: avoid.
- **Fallbacks if SAM3_Detect is missing on Desktop 0.39.1:** kijai/ComfyUI-segment-anything-2 (Apache-2.0
  **[V]**, needs click points or a detector) or BiRefNet per frame (no clicks, "main subject" only; it
  can flicker, so GrowMask + temporal blur).
- VRAM: SAM3.1 fp16 1.63 GB file **[V]**; it runs before Wan and is unloaded, so it should fit 8 GB **[C]**.

### 3. Resolution: 1.3B is a 480P model
- **Ours [V]:** RTX3070_QUALITY/DEPTH `max_width 576, max_height 1024`; `builder.fit_within` uses those
  limits, so a 1080x1920 source renders at 576x1024. The workflow default is 480x832.
- **Official [V]** (Wan2.1 README): "VACE-1.3B ... Supports 480P"; "The 1.3B model is capable of generating
  videos at 720P resolution. However, due to limited training at this resolution, the results are generally
  less stable ... we recommend using 480P". The Comfy-Org template table says VACE-1.3B 480P yes, 720P no
  **[V]**.
- **Change:** cap the pixel area at 480x832 (399,360 px) for 1.3B profiles (portrait 480x832, landscape
  832x480, square ~624x624), still multiples of 16. Upscale afterwards if needed.

### 4. Canny thresholds
- **Ours [V]:** `Canny low 0.2 / high 0.5` (node 13, v2v_3070_quality).
- **Official [V]:** `Canny 0.4 / 0.8` in `video_wan_vace_14B_v2v` (Comfy-Org). The template note says the
  preprocessor can be swapped for any controlnet_aux one.
- Lower thresholds keep many weak edges. Fur and foliage turn into dense line noise, which VACE reproduces as
  outlines (fits the "clay/outline" suspicion in the depth workflow's own comment). Try 0.4/0.8 first.
  Depth is still the better guide for fur.

### 5. Negative prompt
- **Ours [V]:** workflow default `"blurry, deformed, watermark, text"`. Tracker default (docs/director.md)
  `"blurry, deformed, drawing, cartoon, illustration, distorted hands"`.
- **Official Wan default [V]** (`wan/configs/shared_config.py`):
  `色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走`
  (bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, still,
  overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn
  hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy
  background, three legs, many people in the background, walking backwards). Comfy-Org templates use it
  without the first term **[V]**. The model was tuned with Chinese negatives, so keep them in Chinese **[C]**.
- **Change:** base negative = the Wan default. For **stylised** renders, drop `风格，作品，画作，画面`
  (style/artwork/painting/picture), which push toward photo-real. This is the same idea as our existing
  `merge_negatives` "theme asks for it" rule; add the Chinese terms to that mapping. Add anti-CGI words for
  photoreal looks: "3D render, CGI, plastic, waxy skin" **[C, community practice]**.
- Note: negatives only work with cfg > 1 (not in the 4-step distilled mode, rec. 7).

### 6. Prompt shape for Wan (director/prompts.py, docs/director.md)
**Wan's own prompt-extension system prompt [V]** (`wan/utils/prompt_extend.py`, LM_EN_SYS_PROMPT) asks for:
style first ("Japanese-style fresh film photography, ..."); then the subject with appearance, expression and
posture; spatial relationships and shot scale at the end ("Medium shot half-body portrait", "Close-up,
low-angle view"); "**Emphasize motion information and different camera movements**"; "add natural actions of
the target using **simple and direct verbs**"; "around **80-100 words**"; full sentences. The README says
prompt extension "can effectively enrich the details ... we recommend enabling" it **[V]**.

Differences in ours **[V]**:
- **Action strip-out** turns `is walking` into `in mid-stride` and removes verbs. That suits SD image models,
  but Wan is a video model whose guide asks for motion verbs. Keep verbs for Wan (simple present: "the ape
  walks slowly, turns its head"). Keep the strip-out only for image backends.
- **Rule of Nouns / tag lists:** Wan's examples are comma-joined descriptive sentences, style first, framing
  last. Ours puts weighted framing first. Reorder for Wan: style/medium -> subject + action -> environment ->
  lighting/atmosphere -> shot size/angle/camera movement.
- **Weights `(close-up shot:1.3)`:** ComfyUI does parse weights for umt5. `WanT5Tokenizer` subclasses
  `SDTokenizer`, and `ClipTokenWeightEncoder` interpolates against an empty encoding **[V source]**. But
  Wan was never trained with weighted prompts, and the effect on T5 is reported as weak or unpredictable
  **[C]**. Recommend weight 1.0 (no brackets) for Wan profiles. The code already supports that via
  `director.framing_weight`/`angle_weight`.
- **Prompt extension:** our Creative Director already runs on qwen3.5:9b. Feeding it Wan's LM_EN_SYS_PROMPT
  verbatim (with our vocabulary as constraints) is the "proven base" version of our compiler.
- Length: there is no 77-token limit (umt5, ComfyUI max_length unbounded **[V]**), so 80-100 words is fine.

### 7. 8 GB speed and quality
| Option | Settings | Licence | Status |
|---|---|---|---|
| Self-Forcing DMD 1.3B LoRA (`Kijai/WanVideo_comfy/LoRAs/Wan2_1_self_forcing_1_3B/Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16`) | LoraLoaderModelOnly, 4 steps, cfg 1, lcm/simple, shift 8 (Civitai 1719791 JSON) **[V]** | Apache-2.0 (gdhe17/Self-Forcing, guandeh17/Self-Forcing) **[V]** | Best fit |
| Self-Forcing SID v2 1.3B LoRA, rCM 1.3B LoRA (same Kijai repo **[V files]**) | similar few-step | SID Apache **[V]**; rCM ? (HF 401) | A/B later |
| CausVid 1.3B LoRA (`Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32`) | Official template: strength 0.7, 2-4 steps, cfg 1 **[V]**; "may shake and become blurry", try 0.3-0.7 **[V template note]** | **NC**: tianweiy/CausVid and lightx2v CausVid are CC-BY-NC-4.0 **[V]** | Avoid for YouTube even though the Comfy template uses it |
| lightx2v step/cfg distill | 14B only (lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill, Apache-2.0 **[V]**) | OK | Only with 14B |
| TeaCache | Kijai wrapper thresh 0.1 **[V Civitai 1470557]**; native `WanVideoTeaCacheKJ` in KJNodes **[V]**; core `EasyCache`/`LazyCache` **[V]** | Apache-2.0 **[V]** | Small win at 20 steps; pointless at 4 |
| VACE 14B GGUF (QuantStack/Wan2.1_14B_VACE-GGUF, Apache-2.0 **[V]**) | Q4-Q5 + offload + CausVid/lightx2v | OK (except CausVid) | Too slow for 8 GB as default **[C]**; batch-class only |
| SageAttention | `PathchSageAttentionKJ` in KJNodes **[V]** | Apache-2.0 **[V]** | Needs a Triton/sageattention install on Windows; medium effort |
| Steps | Wan native default **50**; README for 1.3B: guide scale 6, shift 8-12 **[V]**; Comfy template 20/cfg 6 **[V]** | — | Ours 20/6/8 matches the template. Try 30 for finals |

Speed reference **[V]**: Wan README: T2V-1.3B "requires only 8.19 GB VRAM ... 5-second 480P on an RTX 4090 in
about 4 minutes" (50 steps, no optimisation). docs.comfy.org: VACE-14B on a 4090, 640x640x49 about 7 min.
I found **no verified RTX 3070 timings**: Reddit returned 403 to this environment. Measure ours from
the render logs instead.

### 8. Automatic QC (pipeline/qc.py)
**Ours [V]:** deterministic edge-structure NCC vs source, motion correlation, moving share, sharpness,
black/frame-mismatch checks. Identity, prompt adherence, style and deformation are reported as `null`.

Proven method: **VBench** (Apache-2.0 **[V]**, github.com/Vchitect/VBench) dimensions, each runnable on 8 GB or CPU:
| Metric | How | Model / licence | Catches |
|---|---|---|---|
| Subject consistency | cosine sim of DINO features, each frame vs first and vs previous | DINOv2-small, Apache-2.0 **[V]** (VBench uses DINO ViT-B/16 [C]) | melting/identity drift of the ape |
| Background consistency | same with CLIP image features | LAION CLIP ViT-B/32, MIT **[V]** (open_clip code MIT-style **[V]**) | background popping |
| Temporal flicker | mean abs diff between consecutive frames on static areas | none (numpy) | flicker |
| Warping error | optical flow source t->t+1, warp render frame, masked L1 | RAFT, BSD-3 **[V]** (torchvision raft_small) or OpenCV Farneback on CPU | temporal tearing |
| Aesthetic | LAION improved aesthetic predictor (MLP on CLIP ViT-L/14) | Apache-2.0 **[V]** | CGI/ugly frames (weak signal) |
| Subject-region checks | compute the above inside the SAM3 mask only | — | "ape bad, background great" exactly |

Avoid: **pyiqa / IQA-PyTorch: PolyForm Noncommercial [V]**. **DOVER: S-Lab NC [V]**. MUSIQ via pyiqa
inherits that problem. An identity check for the ape works with DINOv2 cosine between the reference cutout and
masked render crops; faces need nothing extra. A deformation check could use a VLM (qwen-vl on Ollama) asking
"are the hands/face deformed?"; that one is a heuristic, so keep it advisory.

### 9. Depth control
- **Ours [V]:** `DepthAnythingV2Preprocessor depth_anything_v2_vits.pth res 512`. Small is Apache-2.0 and
  others are CC-BY-NC, as our params.yaml already says (matches HF **[V]**).
- Per-frame depth flickers. **Video Depth Anything Small** (Apache-2.0 code and Small model **[V]**) is
  temporally consistent. A ComfyUI node exists from the community (ComfyUI-Video-Depth-Anything) **[C]**.
  Use it if depth renders show shimmering surfaces. Civitai 1719791 runs DepthAnythingV2 vits at res 512 too
  **[V]**, so our setting matches the proven 1.3B workflow.

### 10. Long shots
- **Ours [V]:** `max_frames 81`, and frames above that are limited per shot.
- **Core [V]:** `WanContextWindowsManual` ("Wan Context Windows", length 81 = 4n+1, overlap default 30,
  uniform schedule, pyramid fuse, optional FreeNoise). Kijai README: 1025 frames with window 81 / overlap 16 on
  1.3B "used under 5GB VRAM" (5090, 10 min) **[C]**. Comfy inpainting template note: VACE is trained on about
  81 frames, so for longer clips chunk at 81 and keep prompt and seed consistent **[V]**.

---

## Node-by-node diff: v2v_3070_quality vs Comfy-Org `video_wan_vace_14B_v2v` (1.3B path)

| Item | Ours | Official template | Note |
|---|---|---|---|
| Model | wan2.1_vace_1.3B_fp16 | 14B active; **1.3B fp16 loader present but bypassed** | same file |
| Text encoder | umt5_xxl_fp8_e4m3fn_scaled | fp16 active; fp8 in the 1.3B path | same |
| LoRA | none | 1.3B path: CausVid 1.3B **0.7** (LoraLoader); 14B: CausVid 0.3 | CausVid NC: replace with Self-Forcing |
| ModelSamplingSD3 shift | 8 | 8 | same (inpainting template uses 5) |
| KSampler | 20 steps, cfg 6, uni_pc, simple | 4 steps, cfg 1, uni_pc, simple (LoRA); note: "Default steps 20, cfg 6.0" | same when no LoRA |
| Control preprocess | Canny 0.2/0.5 | **Canny 0.4/0.8** | see rec. 4 |
| WanVaceToVideo | 480x832, 81, batch 1, strength 1.0 | 720x720, 81, 1, strength 1.0 | resolution: rec. 3 |
| Reference image | source first frame by default | dedicated reference image (object/background) | rec. 1 |
| Negative | "blurry, deformed, watermark, text" | Wan Chinese default (minus 色调艳丽) | rec. 5 |
| Positive | compiled tag list | ~60-word descriptive prose | rec. 6 |
| TrimVideoLatent | trim from node 14 output | same | same |
| CreateVideo fps | 16 | 16 | same |
| Video input | LoadVideo -> GetVideoComponents -> ImageScale(center) | LoadVideo -> GetVideoComponents | ours adds a crop; fine |

v2v_3070_depth: identical except DepthAnythingV2 vits 512, which matches Civitai 1719791 **[V]**.
v2v_preview: SD 1.5 per-frame img2img (euler/normal, 8 steps, denoise 0.55). There is no proven published
equivalent worth keeping; the PREVIEW profile already uses the Wan workflow. Recommend making PREVIEW = Wan 1.3B +
Self-Forcing LoRA at 4 steps (rec. 7) and retiring v2v_preview.

Our `DENOISE` is computed and reported as ignored for VACE, which is correct: WanVaceToVideo starts from an
empty latent and the KSampler denoise is 1.0 in both templates **[V]**.

---

## Verified vs claimed (summary)
Verified in files or source: the official templates' settings and notes; WanVaceToVideo mask semantics;
Wan default negative and prompt-extension rules; the 480P recommendation; SAM3_Detect, GrowMask,
ContextWindows, EasyCache, CFGZeroStar and SkipLayerGuidanceDiT being core nodes on ComfyUI master;
KJNodes node names; HF/GitHub licences listed above; our repo's values.

Claimed only: that the first-frame reference causes the CGI look (inference); T5 weight effects; TeaCache
and SageAttention gains; SAM3 fitting in 8 GB alongside the pipeline; 3070 speeds (none found); VBench
model choices; Video-Depth-Anything ComfyUI node quality.

Unknown and needing a check on the PC: whether ComfyUI Desktop 0.39.1's bundled core has `SAM3_Detect`,
`WanContextWindowsManual`, `EasyCache`, `ResizeImageMaskNode`, `ComfyMathExpression`, `ComfySwitchNode`.
The `comfy-check` command can query /object_info for these.

## Couldn't access
- Reddit (r/StableDiffusion, r/comfyui): HTTP 403 from this environment.
- GitHub API for repos outside the project scope (directory listings), so Kijai's `example_workflows`
  folder could not be listed. Raw files with known paths worked.
- Civitai downloads needing login, and HF nvidia/rCM (401): see civitai-wan-workflows.md.
