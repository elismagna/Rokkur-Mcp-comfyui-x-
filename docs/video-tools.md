# Video tools: extending a clip or a finished video

**Extend** continues a video past its last frame with Wan 2.1 VACE 1.3B, the studio's render
model. It exists in two places:

- **Before a video is made**: New video → *Extend this clip first*. A clip from the media
  folder gets a longer copy next to it (`<name>_extended_<id>.mp4`); pick the copy as the
  source afterwards. Needs the clip's rights line, like any source.
- **After the render**: the project page → *Extend the video* on a finished video. The longer
  cut becomes the latest final video (`final_extended_vN.mp4`, asset kind `final` with
  `extended: true`); the previous cut stays in Renders. The quality check is not run again on
  the continuation; the event log records the extension.

API: `POST /projects/{id}/extend` (`seconds`, `prompt`, `seed`, `steps`, `reference_image_path`,
`render_on`). The continuation's prompt defaults to the brief's scene prompt.

## How it works (`pipeline/extend.py`, workflow `v2v_3070_extend`)

VACE takes a *control video* and a *control mask* per frame. For an extension the control
video is the clip's last 9 frames followed by white frames, and the mask is black over those 9
(keep) and white over the rest (generate). The model paints a continuation that starts from
the real frames; the studio then trims the 9 overlap frames off the result, matches the
original's size and frame rate, joins the two and copies the original's audio back. The
appearance reference is the clip's last frame unless you pick a picture from the library.

The graph is adapted from Comfy-Org's *Wan2.1 VACE first-last frame* template (control video
`[frame, white…, frame]`, masks `[black, white…, black]`, MIT) and our `v2v_3070_keep`
(loaders, sampler and mask wiring); the Canny node is left out because the control frames are
real pixels. Length follows the render profile: at most `max_frames` of VACE output, in 4n+1
steps, so a 3 s extension at 16 fps renders 57 frames (9 kept + 48 new). Longer extensions
are capped at 8 s per request; run it again on the result for more.

## Honest status

Tested with the fake ComfyUI (which echoes the control video, so the tests check the
plumbing: frame counts, masks, trimming, joining, audio, the new final and the events). Not
yet rendered on a real GPU; VACE's own documentation lists temporal extension as a supported
task, and the first real run on the PC is the acceptance test. If the seam at the overlap is
visible, the next knobs are a longer overlap (`OVERLAP_FRAMES`) and a higher
`CONTROL_STRENGTH`.
