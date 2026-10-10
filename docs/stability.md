# Temporal stability

QC on Elis's cloud render said "good, but stability is needed": surfaces that should hold still
shimmer and boil. This page explains why Wan 2.1 VACE 1.3B does that, the three things the studio
now does about it, and how repairs tune settings instead of only rerolling the seed.

Code: `pipeline/stabilize.py`, `pipeline/renderers.py` (control smoothing), `RepairPlanner` in
`agents/roles.py`. Tests: `tests/test_stabilize.py`, `tests/test_repair_tuning.py`. Everything
below was measured on synthetic clips with real FFmpeg only; nothing here has run on a real Wan
render yet.

## Why Wan VACE 1.3B shimmers

- **The Canny guide flickers.** ComfyUI's Canny node runs on every source frame by itself. Grain
  and compression noise near the thresholds switch weak edges on and off from frame to frame, and
  VACE redraws whatever the guide shows, so texture boils. Our thresholds (0.2/0.5) keep far more
  weak edges (fur, foliage, noise) than the official 0.4/0.8 template
  (`docs/research/online-review-2026-10-09.md`, section 4).
- **1.3B above 480P.** The Wan2.1 README says 1.3B results at 720P "are generally less stable" and
  recommends 480P. Profiles already cap the area at 480×832 (`max_pixels`).
- **High CFG.** Workflows that aim to keep source detail run CFG 2-3 and report artifacts above
  that (`docs/research/civitai-wan-workflows.md`, section 4); ours is 6. A claim from community
  pages, not measured here, so auto-tuning lowers CFG only as a bounded, recorded experiment.

## 1. A calmer guide: `smooth_control` (0-1, default 0)

Before the source clip is uploaded as VACE's control video, the ComfyUI renderer runs it through a
temporal-only `hqdn3d` (luma strength 20 × amount, chroma 15 × amount). hqdn3d weighs each pixel's
change between frames, so grain settles while real motion, a large change, passes almost untouched.
The filter keeps every frame's timing, and the render details record
`control_smoothing: {amount, filter}`. Local and cloud renders run the same code.

- Only the control clip is smoothed. The subject mask, the reference image and the kept-subject
  composite still use the original clip.
- On a still, grainy synthetic scene, OpenCV Canny at about 0.2/0.5 toggled 16% / 53% / 68% fewer
  edge pixels between frames at 0.3 / 0.6 / 0.9 (heavy grain); with light grain 0.3 already cut it
  by 59%.
- Gotcha: hqdn3d replaces a spatial strength of exactly 0 with its default (4), which blurs every
  frame. The filters use 0.01, which matched the unfiltered clip exactly in a test.
- The preview renderer has no control video and ignores the setting.

## 2. A steadier render: `stabilize` (`auto` | `off` | `light` | `strong`, default `auto`)

Applied to the finished render, **before** the kept-subject composite, so the real subject's pixels
stay crisp.

| Level | FFmpeg filter | Does |
|---|---|---|
| `light` | `deflicker=mode=am:size=5` | evens out frame brightness over 5 frames (luma only) |
| `strong` | `deflicker=mode=am:size=9`, then `hqdn3d` temporal 10/8, spatial 0.01 | also calms boiling texture and colour shimmer on still surfaces |
| `auto` | tries `light` | |
| `off` | none | your choice; auto-tuning never turns it back on |

`strong` falls back to `light` when strong is not measurably steadier. Each candidate is kept only
when `pick_steadier` agrees, scoring raw and steadied clips against the source with QC's own
`score_shot`:

- steadiness rises by at least 0.3 (motion-compensated `stability` when QC measures it for both
  clips, else `temporal_consistency`);
- `detail` falls by at most 1.0;
- `overall` does not fall.

Otherwise the raw render stands, and the reason is recorded. Frame count and fps never change (QC
compares frame counts with the source); a changed count rejects the candidate.

Measured on synthetic clips: alternating brightness flicker fell by more than half at every
level; `mode=am` and `mode=pm` both cut it by about 80%, but `pm` brightened the whole clip, so
`am` is used. On a still frame with fresh grain each frame, `strong` cut frame-to-frame change by
about 70%, while `light` changed it no more than re-encoding does. On a clean still picture
`strong` changed nothing beyond re-encoding noise: no spatial blur.

Limits: deflicker evens out luma only. A real fast lighting change (a lamp switching on) is
spread over a few frames. Temporal denoise can soften thin, fast motion; `pick_steadier` guards
detail and overall score, not every pixel.

## 3. Repairs that tune themselves

`RepairPlanner.plan(..., auto_tune=True)` is the default. Each failing shot still gets a new seed;
on top of that every recommendation, or a low score when QC has no recommendation for it yet,
moves one setting one bounded step. `RepairAction.tuning` records each rule that fired and why,
e.g. `STABILIZE: smooth_control 0.0 -> 0.3 (stability 5.4 < 7; a time-smoothed source draws a
steadier Canny/depth guide)`.

| Trigger | Change per round | Bound |
|---|---|---|
| `STABILIZE` / `DEFLICKER`, or temporal consistency, stability or flicker below 7 | `stabilize` auto/light → strong; `smooth_control` +0.3; `cfg` −0.5 only when steadiness is the worst score and nobody asked for `FOLLOW_PROMPT` | strong; 0.9; 4.0 |
| `CALM_EDGES` | `canny_low` +0.1, `canny_high` +0.15 (0.2/0.5 → 0.3/0.65 → official 0.4/0.8) | 0.5 / 0.9 |
| Layout drift: `ADD_DEPTH_CONTROL` or structure below 5 | raised Canny thresholds step back down by the same amounts; `control_strength` as before | 0.15 / 0.4; 0.7-1.0 |
| `FOLLOW_PROMPT` or prompt adherence below 5 | `cfg` +1 | 8.0 |
| `FIX_ANATOMY` / `MORE_DETAIL`, or hand/body score below 5 or detail below 4 | `steps` +6 (once, even for both) | 32 |

Rules the planner keeps:

- A value you set outside a bound is never pushed further out, and a stabilizer you turned off stays
  off. `CALM_EDGES` and layout drift together leave the thresholds alone (they pull opposite ways).
- Workflow inputs (`cfg`→`CFG`, `steps`→`STEPS`, `canny_low`/`canny_high`→`CANNY_LOW`/`CANNY_HIGH`,
  `shift`→`SHIFT`) change only when the workflow has them; the rest are listed as unavailable.
  `stabilize` and `smooth_control` are applied by the studio, so every workflow accepts them. A
  workflow with no repair controls still stops for a person, as before, unless the studio's own
  stabilizer can help.
- Never the same combination twice: when the shot's `tried` list (earlier attempts' settings, see
  `RepairPlanner.settings_from_params`) already holds the planned combination, it adds the first
  untried safe step: stronger stabilizer, more smoothing, more steps, lower CFG. If all were tried,
  only the seed changes, and the action says so.
- `shift` is not tuned automatically: nothing measured says which way helps our renders (the
  template uses 8; detail-keeping workflows use 1-3). It is a manual setting.
- `auto_tune=False` gives exactly the planner from before.

Scores are read as QC writes them: 0-10, higher is better. A score QC did not measure is simply
absent and triggers nothing.
