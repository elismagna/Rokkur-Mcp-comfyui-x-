# Civitai Wan VACE workflows for Rökkur Studio (researched 2026-10-09)

> Copied into the repo on 2026-10-09 from Claude's project notes, so Codex can read it. Paths under `knowledge/` refer to Claude's side: the downloaded workflow JSONs are not committed, because their licences differ. The Comfy-Org templates are at github.com/Comfy-Org/workflow_templates (MIT). What was applied is in `docs/AI_HANDOFF.md`.

Scope: Wan 2.1 VACE video-to-video on an RTX 3070 (8 GB). Our current graph: VACE 1.3B fp16 + umt5 fp8,
Canny or DepthAnythingV2 -> WanVaceToVideo (control_video + reference_image, strength 1.0) -> KSampler
uni_pc/simple 20 steps cfg 6, ModelSamplingSD3 shift 8, ~480x832, 81 frames, 16 fps. Problem: the subject
(ape face, fur, hands) melts or looks CGI while the background restyles well.

Legend: **[verified]** = I read it in the downloaded workflow JSON, the ComfyUI source, or a Hugging Face
API/model card. **[claim]** = only stated on a Civitai page, not tested by us.

Copied JSONs (in `knowledge/civitai-workflows/`, all from Civitai pages marked "derivatives allowed"):

| File | Source | Engine |
|---|---|---|
| `vace13b-native-depth-pose-canny.json` | civitai.com/models/1719791 | native core + 1.3B |
| `vace13b-native-mask-composite.json` | civitai.com/models/1719791 | native core + 1.3B |
| `vace-native-sam2-mask-reference.json` | civitai.com/models/1605242 | native core, 14B GGUF |
| `vace13b-sam2-subject-replace-wrapper.json` | civitai.com/models/1470557 | Kijai WanVideoWrapper, 1.3B |
| `vace-masking-extension-pftq-wrapper.json` | civitai.com/models/1536883 | Kijai WanVideoWrapper |

The prompt text in `vace-native-sam2-mask-reference.json` was blanked (the author's example prompt was NSFW);
the graph is unchanged.

---

## Top recommendations for our pipeline (ranked by gain for effort on 8 GB)

### 1. Keep the subject with VACE `control_masks` plus a composite control video (biggest gain)
**What to change:** build a per-frame subject mask (SAM 2 or BiRefNet/RMBG), grow + blur it a little, then feed
WanVaceToVideo with:
- `control_masks` = **inverted** subject mask (white = background = regenerate, black = subject = keep).
- `control_video` = per-frame composite: original RGB pixels inside the subject, our depth (or Canny) map in
  the background.

**Why:** the ComfyUI source decides what is kept **[verified, comfy_extras/nodes_wan.py]**:
`inactive = control_video*(1-mask)+0.5`, `reactive = control_video*mask+0.5`. Mask 1 (white) = regenerate,
mask 0 = keep as context. With mask 0 over the ape, VACE gets the real fur/face pixels as fixed context and
blends the new background around them in the same pass. That gives better edges and lighting than a
post-hoc U2-Net composite. Missing mask frames are padded with 1.0 (regenerate), so the mask must cover all
81 frames.

**Proven graph to start from:** `vace-native-sam2-mask-reference.json` (native nodes, the closest to ours)
**[verified]**: VHS_LoadVideo -> ImageResize+ -> Sam2Segmentation (PointsEditor clicks,
sam2.1_hiera_large, "video" mode) -> GrowMaskWithBlur (expand 25, blur on) -> ImageCompositeMasked (paste a
flat colour panel #a2ad9c over the masked area) -> AIO_Preprocessor Canny -> `control_video`. The same mask
goes through MaskToImage -> ImageToMask(red) -> `control_masks`. As published, that graph masks the subject,
which **replaces** it. For our "keep the ape" mode, put InvertMask before both uses, and put depth instead
of the grey panel in the regenerated area. The author of 1470557 suggests exactly that ("could be enhanced
by incorporating depth/openpose... instead of just a gray background") **[claim]**.

**Two modes the app can choose automatically:**
- *Keep subject, restyle world:* mask = inverted subject; the subject area holds original pixels.
- *Restyle subject, keep world:* mask = subject; the subject area holds **depth of the subject** (not flat
  grey) so fur silhouette and hand pose survive. This is the 1470557 layout with depth swapped in.

**Nodes needed:** segment-anything-2 (kijai), comfyui-kjnodes (GrowMaskWithBlur, PointsEditor),
comfyui_essentials (ImageResize+, ImageCompositeMasked is core), comfyui_controlnet_aux (installed).
SAM 2 needs click points, so for automation use a text-prompted segmenter instead: Florence2 or
GroundingDINO -> SAM2, or BiRefNet/RMBG for "main subject" masks (no clicks). The 1680850 AIO workflow
describes "V2V background change = subject switch + invert mask" with SAM **[claim; download needs login]**.

**VRAM:** SAM2 large in fp16 runs before the sampler and frees VRAM. On 8 GB use `sam2.1_hiera_small`/`base`
if it OOMs **[claim]**. Masks at 480x832 are cheap.

### 2. Self-Forcing DMD 1.3B LoRA: 4 steps, cfg 1 (speed, so more QC repairs fit)
**What to change:** add `LoraLoaderModelOnly` (between UNETLoader and ModelSamplingSD3) with
`Kijai/WanVideo_comfy/LoRAs/Wan2_1_self_forcing_1_3B/Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors`
**[verified file exists]**, then sample at 4 steps, cfg 1.0, sampler `lcm`, scheduler `simple`,
shift 8. Those exact sampler settings are in `vace13b-native-depth-pose-canny.json` **[verified]**. That graph
uses the pre-merged model `Wan2.1-T2V-1.3B-Self-Forcing-DMD-VACE-FP16` with WanVaceToVideo + DepthAnythingV2
vits, the same node as ours. The author says it "works on 6 GB VRAM" **[claim]**.

**Licence:** Self-Forcing weights are Apache-2.0 on gdhe17/Self-Forcing **[verified HF card]**. The
merged `lym00/Wan2.1_T2V_1.3B_SelfForcing_VACE` repo is labelled **CC-BY-NC-SA-4.0 (non-commercial)**
**[verified]**, so for YouTube use the Kijai LoRA on our own VACE 1.3B, not the lym00 merge. Kijai's repo
has no licence tag; the LoRA is extracted from the Apache weights.

**Expected effect:** about 5x fewer steps. At cfg 1, the negative prompt is ignored. Quality vs our 20-step
cfg 6 run is untested; A/B it on render 54's clip.

Other 1.3B accelerators seen in Kijai's repo **[verified files]**: `Wan2_1_self_forcing_sid_v2_1_3B_lora`,
`Wan_2_1_T2V_1_3B_480p_rCM_lora` (rCM licence page needs login: unknown),
`Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32` (**CausVid is CC-BY-NC-4.0: non-commercial, avoid**).

### 3. Reference image = subject cut out on white
**What to change:** before WanVaceToVideo `reference_image`, run the reference through background removal and
put it on a plain white background. Both 1470557 and 1605242 do this **[verified]**:
`LoadImage -> ImageRemoveBackground+ (RemBGSession+ "u2net: general purpose") -> LayerUtility:
ImageRemoveAlpha (fill #FFFFFF)`. A busy reference background leaks into the scene. A clean ape cutout (a
good frame from the source clip) anchors face and fur identity across frames.
**Nodes:** comfyui_essentials (rembg nodes), comfyui_layerstyle (or plain core compositing on white).

### 4. Lower shift and cfg when the goal is to *retain* source detail
pftq (1536883) **[claim, page text]**: "Keep CFG 2-3 and Shift=1 to retain as much detail from the existing
footage as possible", and higher values "introduced artifacts". Their workflow sampler is 50 steps,
cfg 2.0, shift 1.0 **[verified in JSON]**. For the "keep subject" mode, try cfg 3 / shift 3 against our cfg 6 /
shift 8 before reaching for LoRAs. The `ColorMatch` node (comfyui-kjnodes, method `mkl`) fixes colour drift
**[claim]**. ColorMatch exists in KJNodes **[verified]**.

### 5. Guidance helpers that are core nodes (no new packs)
- **CFGZeroStar** (core): used before KSampler in 1805031 **[verified]**. It is said to reduce early-step
  overshoot, which shows up as plastic/oversaturated looks **[claim]**.
- **SkipLayerGuidanceDiT** (core): set to layers "9,10", scale 3, 0.01-0.8 in 1805031, muted by default
  **[verified]**. SLG is a common Wan anatomy/hands fix **[claim]**; KJNodes also has
  `SkipLayerGuidanceWanVideo` **[verified node exists]**. Only works with cfg > 1, so not with the
  Self-Forcing 4-step mode.
- **EasyCache / LazyCache** (core, `comfy_extras/nodes_easycache.py` on ComfyUI master **[verified]**): a
  TeaCache-style step skip without a custom pack. Check that Desktop 0.39.1 includes it. Wrapper workflows use
  WanVideoTeaCache thresh 0.1 **[verified in 1470557]**; native equivalent in KJNodes:
  `WanVideoTeaCacheKJ` **[verified node exists]**.
- **WanVideoEnhanceAVideoKJ** (KJNodes, native): 1470557 runs the wrapper version at weight 3
  **[verified]**. Enhance-A-Video is said to improve temporal consistency and detail **[claim]**.
- **NAG** (`WanVideoNAG` in KJNodes **[verified exists]**): gives a working negative prompt at cfg 1, so it
  pairs with the Self-Forcing LoRA **[claim]**.

### 6. Fur, faces, hands, anti-CGI LoRAs: nothing usable for 1.3B on Civitai
Civitai searches for 1.3B LoRAs with realism/detail/fur/animal/film/cinematic/ape returned **zero** results.
The 1.3B LoRAs that exist are styles (Ghibli, pixel art, Arcane) **[verified search results]**. The known
anti-CGI LoRA `Wan_FusionX_FaceNaturalizer` (civitai.com/models/1755105) is **14B only** and carries
Image+RentCivit commercial rights only. `Wan 2.1 1.3B tools` (1934938) has a DiffSynth `highresfix` 1.3B LoRA
("avoids image corruption and grayscale" at high res **[claim]**), commercial rights RentCivit only.
For hands and fur the cheaper levers are: keep them in the masked "keep" region (rec. 1), use depth rather than
Canny inside the subject (Canny draws fur as noise edges), add SLG, and use negatives like "3d render, CGI,
plastic, smooth skin, poorly drawn hands" (1605242 negative: "poorly drawn hands/faces, deformed limbs"
**[verified]**).

### 7. Not worth it on 8 GB now
VACE 14B (GGUF Q4-Q8 + CausVid 14B LoRA at 6 steps) is what most Civitai VACE workflows use. 1605242 loads
`Wan2.1-VACE-14B-Q8_0.gguf` **[verified]**. With 8 GB it needs heavy offload and is slow, a batch-class
option at best. The 1680850 author says "1.3B VACE GGUF fails to give good result" **[claim]**, so
quantising 1.3B is not a VRAM win either; stay on 1.3B fp16.

---

## Per-workflow notes

**1719791 "Vace control depth/openpose/canny/splines/extend"** (JustSomeGuy). Commercial: Image, RentCivit,
Rent, Sell, SellMerge; derivatives allowed. Base: Wan 1.3B. **[verified in JSON]** native core
UNETLoader `Wan2.1-T2V-1.3B-Self-Forcing-DMD-VACE-FP16` -> ModelSamplingSD3 shift 8 -> KSampler 4 steps
cfg 1 lcm/simple; WanVaceToVideo strength 1.0, 41-45 frames, ~448-600 px; DepthAnythingV2 `vits` res 512;
CLIPLoaderGGUF umt5 Q6_K. Packs: inspire-pack (prompt caching; swap for CLIPTextEncode), impact-pack,
ComfyUI-GGUF, WanVideoWrapper (only WanVideoImageResizeToClosest / VACEStartToEndFrame), KJNodes,
controlnet_aux, VHS, was-ns. "last_example" shows mask building with ImageToMask -> InvertMask ->
ReplaceImagesInBatch into `control_masks`. VRAM: "works on 6gb" **[claim]**. Model licence: lym00 merge is
CC-BY-NC-SA (see rec. 2).

**1605242 "VACE-14B-GGUF AIO ControlNet and Mask Segment"** (mellinjohan297). **Commercial use: none
allowed on Civitai** (allowCommercialUse empty). This flag covers the asset; we copy only the graph pattern
for internal reference. Derivatives allowed. **[verified]** native WanVaceToVideo 720x720x81, KSampler 6
steps cfg 1 uni_pc/normal, shift 8, CausVid 14B LoRA 0.25 (rgthree Power Lora Loader), SageAttention patch,
UnetLoaderGGUF Q8, SAM2 mask -> GrowMaskWithBlur(25, blur) -> grey composite + `control_masks`
(muted by default), rembg reference on white. Packs: ComfyUI-GGUF, comfyui-multigpu, segment-anything-2,
KJNodes, rgthree, essentials, controlnet_aux, comfyroll, layerstyle, reactor (face swap; we skip). Model: 14B
(not for 8 GB without offload).

**1470557 "VACE Subject Replace"** (theartofficialtrainer). Commercial: full; derivatives allowed.
**[verified]** Kijai WanVideoWrapper: `Wan2_1-T2V-1_3B_bf16` + `Wan2_1_VACE_1_3B_preview_bf16`, model quant
fp8_e4m3fn, T5 fp8 offloaded, sageattn; WanVideoSampler 30 steps cfg 3 shift 5 unipc; TeaCache 0.1;
Enhance-A-Video 3.0; VACEEncode 832x480x85 strength 1. Mask: SAM2.1 large video mode (PointsEditor clicks) ->
GrowMaskWithBlur(10) -> grey #a2ad9c fill -> input_frames + input_masks. Reference: u2net cutout on white,
padded. Also contains an SDXL Juggernaut + depth ControlNet branch to make a first-frame reference. Packs:
WanVideoWrapper, KJNodes, VHS, essentials, comfyroll, segment-anything-2, controlnet_aux, layerstyle. 1.3B,
so it is 8 GB-friendly **[claim]**. Wrapper nodes differ from our native graph, so use it for the masking
pattern only.

**1536883 "Wan VACE Masking & Extension"** (pftq). Commercial: full; derivatives allowed. **[verified]** Wrapper,
14B in the JSON (page lists 1.3B fp16/fp8 alternatives), sampler 50 steps cfg 2 shift 1, CausVid LoRA and
BlockSwap 15 present but muted, TeaCache muted. **[claim]** mask convention: grey #7F7F7F in the source video,
mask white where grey = regenerate (matches the ComfyUI source). The page also says "Wan 2.1 1.3B seems best for
LoRAs if you are trying to do something drastically different". It suggests ColorMatch MKL against drift.

**1598288 "VACE + CausVid LoRA"** (theartofficialtrainer). Commercial: full. Pose2Vid only. **[verified]** The native
JSON's UNETLoader points at `FramePackI2V_HY_bf16` (a wrong model left in), with KSampler 4 steps cfg 8. Not
copied. The wrapper version uses CausVid 14B at 0.4, 4 steps cfg 1 shift 3. CausVid is non-commercial
(see rec. 2).

**1805031 "Wan2.1 Vace T2I Inpainting gguf 8g"** (SamLiu). Single-image (1 frame) inpaint/outpaint with
VACE 14B Q4_0 GGUF + lightx2v I2V 14B LoRA, 8-12 steps euler/bong_tangent, CFGZeroStar, SLG muted,
INPAINT_MaskedFill neutral. The LoRA stack includes NSFW LoRAs. Not copied; only the guidance-node settings
are noted above. Useful later for fixing a single keyframe (e.g. a hand) at 8 GB **[claim]**.

**1674121 "Simple Self-Forcing Wan1.3B+Vace"** (davcha). Commercial: full. Core nodes + VHS only **[claim]**.
**Its text says mask 1 = keep original. That is backwards per the ComfyUI source** (mask 1 = regenerate).
Download needs a Civitai login.

**1680850 "WAN2.1-VACE-14B 1.3B GGUF 6 steps AIO"** (kukalikuk). Commercial: Image, RentCivit, Rent (no
Sell). Switch guide **[claim]**: "V2V subject change = Image1+VidRef+control+SAM ON; V2V background change =
same + invert mask". 1.3B mode uses the lym00 Self-Forcing VACE merge at 6 steps. Recommends the
4x-ClearRealityV1 upscaler. Download needs login.

**1583890 / 1603600 (T8star)** "Custom Background Strong Consistency Dual Mask" and "VACE+14B+SAM2 Split Body
Reference". 14B; download needs login. Their titles match our "keep character, new background" goal; worth a
look if Elis downloads them manually.

---

## Couldn't access
- Civitai downloads that need a logged-in account (HTTP 401, "creator requires you to be logged in"):
  1674121, 1680850, 1583890, 1603600. Elis can download these in a browser and drop them into
  `knowledge/civitai-workflows/`. A Civitai API key in the environment would also work; none was used.
- Not downloaded on purpose (multi-GB archives with bundled videos/models): 1631379 (15 GB), 1604714 (1.1 GB).
- Hugging Face `nvidia/rCM` and `worstcase/rCM` returned 401, so the rCM LoRA licence is unknown.
- `QuantStack/Wan2.1_T2V_1.3B_VACE-GGUF` does not exist (401/not found).
- Nothing was executed or installed. All files were read as JSON only.
