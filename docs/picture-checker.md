# Picture checker and stability metrics

QC used to judge a render from 64×64 grey frames. That is enough to catch black frames, frozen
motion and layout drift, but not the texture boiling Elis saw on the first cloud render ("good
but stability is needed"). This change adds two things:

- **Stronger stability metrics** in `pipeline/qc.py`: deterministic, measured on every shot.
- **The picture checker** in `pipeline/vision.py`: a contact sheet you can look at, and an
  optional AI review of that sheet by the vision model already installed.

Neither changes the `overall` score or the pass threshold, which were calibrated on real
renders. Stability can fail a shot only when a floor is set (`min_stability`, off by default).

## The metrics

All scores are 0–10, 10 best. A score that could not be computed is `null`, and the shot's
`not_measured` says why. The report's `not_measured` lists only what no shot measured.

| Key | What it measures | How |
|---|---|---|
| `stability` | Do surfaces hold still while the scene moves? Low = boiling textures, shimmer. | For each pair of neighbouring frames, optical flow (OpenCV Farneback) is estimated on the **source** pair. Render frame *t* is warped with it and compared with render frame *t+1*; the source gets the same treatment as a baseline for occlusions and lighting. The render's excess residual (worst 10% of pairs dropped) maps to the score. |
| `flicker` | Whole-frame brightness pulsing beyond the source's. | Per-frame mean brightness minus a 5-frame moving average, RMS, render minus source. Grey frames only, so luma flicker, not colour. |
| `temporal_consistency`, `motion`, `structure`, `detail`, `artifact_score` | Unchanged. | See `docs/architecture.md`. |

Why the source's flow: real motion is then not punished (the render may move exactly as the
source does), and flicker cannot hide inside a flow estimated on the flickering render.
Pixels that fail a forward-backward flow check (occlusions) are left out of both residuals.
Whole-frame brightness is removed before the stability residual, so a brightness pulse shows
up as `flicker`, not as `stability`. The render is brought to the source's contrast first
(gain bounded to 0.5–2), so a softer or punchier restyle is not punished or rewarded for it.

Stability is measured on larger grey frames when the QC stage passes them (`source_detail`,
`render_detail`, 192 px wide, see `qc.detail_size`); otherwise on the 64×64 frames, which see
less. Its method and frame size are recorded in `stability_method`.

**Calibration.** The mappings (`STABILITY_SCALE = 6`, `FLICKER_SCALE = 4` grey levels) are
reasoned, not fitted to real renders yet. On synthetic tests H.264 compression alone (even at
CRF 35) stays above 9.8, and 3-pixel boiling of ±8 grey levels scores about 4 while the old
metrics still pass the shot. Check the numbers on a few real renders before setting a floor.

### New recommendations

QC only ever appends to the existing recommendations, on failing shots:

| Recommendation | When |
|---|---|
| `STABILIZE` | `stability` < 7, or below the `min_stability` floor (the shot then fails with "picture not steady (stability 4.2, floor 6.0)") |
| `CALM_EDGES` | edge-like output, or stability low while `structure` ≥ 6: the layout holds but the guide drew texture edges |
| `DEFLICKER` | `flicker` < 7 and the render's brightness pulsing is at least 1.5× the source's |
| `MORE_DETAIL` | `detail` < 5 |

The AI review adds `FOLLOW_PROMPT`, `FIX_ANATOMY` and `STABILIZE` (below).

## Reading the contact sheet

`vision.contact_sheet(source_rgb, render_rgb)` returns one image (a PNG via
`vision.png_bytes`, no Pillow needed). Frames are sampled evenly through the shot, first and
last included, and **time runs left to right**:

1. **Top row:** the source.
2. **Second row:** the render at the same moments.
3. **Third row:** the change heatmap over a darkened copy of the render. Black means the render
   changes the way the source does. Red, then yellow, then white means it changes more than the
   source between neighbouring frames: boiling, shimmer, flicker. Each cell averages the frame
   pairs of its stretch of the shot, so an unsteady stretch is not missed between samples.
   Full white is 24 grey levels of excess change.

There is no text on the sheet; the layout is fixed instead. Without OpenCV the heatmap falls
back to plain frame differences, which also light up the moving edges of a relit render.

## The AI picture review (advisory)

`vision.PictureChecker(provider).review(...)` shows the sheet to the agent provider and returns
a `PictureReview`: a one- or two-sentence description of what the render actually shows, and
0–10 scores for prompt adherence, style consistency, subject identity, anatomy and steadiness,
plus issues from a fixed list (`melting_subject`, `extra_limbs`, `face_distortion`,
`hand_distortion`, `identity_drift`, `style_drift`, `prompt_ignored`, `source_look_leaks`,
`texture_boiling`, `flicker`, `blurry`, `black_or_broken_frames`) and a short note.

- It uses the vision model the studio already runs: Ollama `qwen3.5:9b` (about 6.6 GB, reads
  images). Nothing new is downloaded. A separate `director.vision_model` works too.
- It is **a model's opinion, not a measurement.** `vision.apply_review` copies it into the QC
  shot as `identity`, `prompt_adherence`, `style_consistency` and `hand_body_deformation`
  (10 = no deformation), with a `picture_review` block, and **never changes PASS/FAIL**. On a
  failing shot it appends repair hints: `FOLLOW_PROMPT` (prompt adherence ≤ 4, or the prompt
  was ignored / the source look leaks through), `FIX_ANATOMY` (anatomy ≤ 4 or melting, limb,
  face or hand issues) and `STABILIZE` (steadiness ≤ 4, or boiling or flicker).
- When the provider cannot see images (the rule-based provider, a text-only model), is down,
  or keeps answering with invalid JSON, there is **no review**: the scores stay `null`, the
  shot gets `picture_review_skipped` with the reason, and nothing is invented. After one
  "unavailable" the checker stops asking for the rest of the check.

## Licences

- **OpenCV** (`opencv-python-headless`): Apache-2.0 since OpenCV 4.5; the wheel's bundled
  FFmpeg libraries are LGPL. Both allow commercial use, including monetized YouTube videos.
  The headless wheel needs no extra system libraries in the Docker image.
- The review model is Elis's existing Qwen3.5 install; the PNG encoder is our own code.
