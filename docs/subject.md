# The main subject: keep it real or restyle it

Wan 2.1 VACE 1.3B restyles rooms well but melts animals and people. On renders 62/63 of the ape
clip, the new tiles and towels looked great and the ape looked bad. So the studio decides, per
project, whether the real main subject goes back over the render.

## Asking once, before anything renders

The New video form has one question, **Main subject**:

- **Let the studio decide** (the default).
- **Keep it real, restyle everything around it.**
- **Restyle it too.**

While you type, the hint under the field shows what the studio will do and why. The CLI takes
`--subject auto|keep|restyle`, and the API takes `creative.subject`. Nothing pauses the
pipeline to ask later.

With **Let the studio decide**, `pipeline/subject.py:decide_subject` applies these rules in
order:

1. If you picked a character (saved character, description or appearance reference), the
   subject is **restyled** into it.
2. If the prompt asks to keep the subject, it is **kept**. Examples: "keep the ape real", "only
   the background", "don't change the ape".
3. If the prompt changes the subject, it is **restyled**. Examples: "turn the ape into a robot",
   "wearing a tuxedo", "replace the ape with a tiger". Changing the place is not a subject
   change: "turn the bathroom into a jungle" keeps the subject.
4. If the look is stylized, the subject is **restyled**, because a real ape in a cartoon room
   looks pasted in. Examples: anime, claymation, Pixar, 3D render, watercolor, comic, Ghibli.
5. Otherwise the subject is **kept**: the prompt changes the place and the look, which is
   what Wan does well.

The decision and its reason are stored in the manifest (`subject`). They show on the project
page under Creative brief, marked "(planned)" until the manifest exists. Manifests from
before this feature have no `subject` and restyled everything.

Known gap: an adjective-only change of the subject ("make the ape golden") is not detected
as a subject change, so it keeps the subject. Pick **Restyle it too** for that.

## How the subject is kept

The subject is kept twice: Wan draws the new room around the real subject, and then the
subject's exact pixels go back over the render.

1. **Mask.** Before each shot renders, U²-Net, a salient-object model, finds the main subject
   in every frame of the source shot. This runs on the CPU, outside the GPU lease.
   - It uses rembg's ONNX export of the Apache-2.0 weights, run with onnxruntime.
   - The source is cropped to the render's aspect, the way ComfyUI's ImageScale crops.
   - The masks are averaged over neighbouring frames so the edge does not shimmer.
   - Masks are cached per shot in `work/masks/`, so repair rounds and the composite reuse
     them. A grey mask preview video sits next to the cache.
2. **Wan keeps the subject.** The profile's `keep_workflow` runs instead of its usual workflow
   (`v2v_3070_keep`, or `v2v_3070_depth_keep` for RTX3070_DEPTH). It gets the masks as a
   video, `MASK_VIDEO`, and:
   - inverts them into WanVaceToVideo's `control_masks`, so Wan regenerates the room (white)
     and keeps the subject (black);
   - builds the control video from the source pixels on the subject and the usual Canny
     edges (or depth) everywhere else, so the room still follows the source layout.

   The wiring comes from Comfy-Org's VACE inpainting template and the Civitai workflows
   listed in `workflows/v2v_3070_keep/params.yaml`, per the workflow rule in
   `docs/comfyui.md`.
3. **Reference.** With the reference set to **Let the studio decide** (`reference_mode:
   auto`), Wan also gets the subject cut out on white as its reference image. VACE was trained
   on object or background references, not on scenes (Comfy-Org ref2v template note). The old
   default, the whole first source frame, probably pulled the real room and light into the
   restyle; that is inferred from renders 62, 63 and 67, not tested yet.
   A restyled subject gets no reference in `auto`. `cutout`, `source` and `none` force a
   choice; an uploaded character reference always wins.
4. **Edge.** The mask is grown by `subject.grow` (1.2% of the short side) to cover Wan's halo
   around the subject. It is then feathered by `subject.feather`.
5. **Colour.** The subject's colours move part of the way toward the new room
   (`subject.harmonize`, 0.5). The offset comes from a ring just outside the subject, and it
   is one value per shot, so it does not flicker.
6. **Composite.** After the render, the result is written to
   `renders/shot_NNN/attempt_NN_subject.mp4` and becomes the shot's render, so QC, repair and
   assembly use it.
   - The untouched render stays as a `render_raw` asset.
   - The attempt picker on the project page lists it as "restyled subject".

Steps 4 to 6 run on the CPU after each render. If the keep workflow is missing or fails to
load, the shot renders with the profile's usual workflow and only the composite keeps the
subject. The project page lists the workflow each attempt used under Applied render settings.

The full frame is restyled instead, with the reason shown under the shot preview, when:

- the median mask covers less than `subject.min_coverage` (1%), meaning no clear subject;
- it covers more than `subject.max_coverage` (75%), meaning the subject is the whole frame;
- masks cannot be made, for example when onnxruntime is missing or the model cannot be
  downloaded. The render still completes.

## The model file

`u2net.onnx` is 176 MB. The worker downloads it on the first keep render into
`data/models/`, from rembg's GitHub release, and keeps it only if its SHA-256 matches. The
System page shows whether masks are ready.

For an offline machine, put the file there by hand and set `subject.download: false`. The
finer-edged `isnet-general-use` (179 MB, about 3x slower on the CPU) is the other choice for
`subject.model`.

Measured in the cloud container on 4 CPU threads: about 0.35 s per frame for u2net. That is
about 25–35 s for an 81-frame shot, including the composite.

## Limits and next steps

- A salient-object model picks the most prominent thing. That can include objects the
  subject holds, or something else when the subject is small. Check the mask preview video
  when a result looks odd.
- If Wan moved the subject's outline by more than the growth margin, bits of Wan's own
  subject can show around the edge.
- The keep workflows have only been checked here by compile tests and a fake ComfyUI. Their
  first real renders on the PC decide whether they stay the default for kept subjects.
- The mask is binary inside Wan (ThresholdMask at 0.5), as in the Comfy-Org template. If Wan
  leaves a seam at the subject's edge, try growing the mask a little before it goes to Wan.
