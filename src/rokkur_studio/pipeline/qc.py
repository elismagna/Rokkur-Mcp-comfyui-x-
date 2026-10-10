"""Quality control: measurable, deterministic metrics only.

Scores that need a vision model (identity, prompt adherence, style consistency, hand/body
deformation) are ``null`` with a reason unless the AI picture review (``pipeline/vision.py``)
copied its advisory opinion in; they are never invented here. Motion-compensated stability
needs OpenCV and is ``null`` with a reason when it cannot be imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rokkur_studio.pipeline.analysis import motion_series

NOT_MEASURED = {
    "identity": "needs the AI picture review (a vision model) or a face/identity embedding model",
    "prompt_adherence": "needs the AI picture review (a vision model)",
    "style_consistency": "needs the AI picture review (a vision model) or a style embedding model",
    "hand_body_deformation": "needs the AI picture review (a vision model) or a pose/keypoint model",
}
# Picture-level scores computed here; a shot lists the ones it could not compute.
MEASURED = ("stability", "flicker")

# Grey-level scales of the 0-10 mappings below (10 * exp(-excess / scale)). Not yet
# calibrated on real renders, which is why ``min_stability`` defaults to off.
STABILITY_SCALE = 6.0   # mean excess warp residual, same scale as the temporal metric
FLICKER_SCALE = 4.0     # RMS excess of high-passed frame brightness


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b))
    if n < 3 or a[:n].std() < 1e-6 or b[:n].std() < 1e-6:
        return None
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-6 else 0.0


def _edge_maps(frames: np.ndarray) -> np.ndarray:
    """Gradient magnitude per frame, box-blurred 3x3 so a one-pixel shift still overlaps.

    Tone-blind: a restyle that relights or recolours the scene keeps its edges, and the Wan
    VACE workflow is driven by the source's Canny edges.
    """
    f = frames.astype(np.float32)
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gx[:, :, 1:-1] = f[:, :, 2:] - f[:, :, :-2]
    gy[:, 1:-1, :] = f[:, 2:, :] - f[:, :-2, :]
    mag = np.hypot(gx, gy)
    pad = np.pad(mag, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = mag.shape[1:]
    blurred = np.zeros_like(mag)
    for i in range(3):
        for j in range(3):
            blurred += pad[:, i:i + h, j:j + w]
    return blurred / 9.0


def _moving_share(frames: np.ndarray) -> float:
    """Share of pixels that clearly change between frames: motion, not codec noise or grain."""
    if len(frames) < 2:
        return 0.0
    return float((np.abs(np.diff(frames.astype(np.int16), axis=0)) > 12).mean())


def _sharpness(frames: np.ndarray) -> float:
    f = frames.astype(np.float32)
    lap = (f[:, 1:-1, 1:-1] * 4 - f[:, :-2, 1:-1] - f[:, 2:, 1:-1] - f[:, 1:-1, :-2]
           - f[:, 1:-1, 2:])
    return float(lap.var()) if lap.size else 0.0


# -- motion-compensated stability --------------------------------------------------------
def _cv2() -> Any:
    """OpenCV, or None when it is not installed (or a system library it needs is missing)."""
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def detail_size(width: int, height: int, target_width: int = 192) -> tuple[int, int]:
    """Frame size for the stability metric: ``target_width`` wide (never upscaled), aspect
    kept, both sides even so ffmpeg's scaler accepts them."""
    w = max(2, min(target_width, width) // 2 * 2)
    return w, max(2, round(w * height / max(width, 1) / 2) * 2)


def _contrast_scale(src: np.ndarray, out: np.ndarray) -> float:
    """Gain that brings the render to the source's contrast, bounded so a flat render is not
    amplified into noise. A restyle may legitimately be softer or punchier than the source."""
    return float(np.clip(float(src.std()) / max(float(out.std()), 1.0), 0.5, 2.0))


def _flow(cv2: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense Farneback flow: ``a(p) ~ b(p + flow(p))``."""
    return np.asarray(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0))


def _warp(cv2: Any, img: np.ndarray, flow: np.ndarray,
          grid: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """``img`` resampled at ``p + flow(p)``: frame a brought onto frame b's pixels."""
    return np.asarray(cv2.remap(img.astype(np.float32), grid[0] + flow[..., 0],
                                grid[1] + flow[..., 1], cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE))


def _pair_residuals(cv2: Any, src_a: np.ndarray, src_b: np.ndarray, out_a: np.ndarray,
                    out_b: np.ndarray, gain: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """|warp(a) - b| for the source and the render, both warped with the SOURCE's flow, and the
    mask of pixels where that flow is trustworthy.

    The source's flow means real motion is not punished, and flicker cannot hide inside a flow
    estimated on the flickering render. Whole-frame brightness is removed first: that is what
    the ``flicker`` metric measures. Pixels that fail the forward-backward check (occlusions,
    flow failures) are masked out of both residuals.
    """
    h, w = src_a.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid = (gx, gy)
    back = _flow(cv2, src_b, src_a)   # where each pixel of b was in a
    fwd = _flow(cv2, src_a, src_b)
    round_trip = np.stack([_warp(cv2, fwd[..., k], back, grid) for k in range(2)], axis=-1)
    err = ((back + round_trip) ** 2).sum(-1)
    valid = err < 0.01 * ((back ** 2).sum(-1) + (round_trip ** 2).sum(-1)) + 0.5
    if valid.mean() < 0.2:  # flow failed almost everywhere: compare all pixels instead
        valid = np.ones_like(valid)

    def residual(a: np.ndarray, b: np.ndarray, g: float) -> np.ndarray:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        return np.abs(_warp(cv2, a - a.mean(), back, grid) - (b - b.mean())) * g

    return residual(src_a, src_b, 1.0), residual(out_a, out_b, gain), valid


def excess_change(source_a: np.ndarray, source_b: np.ndarray, render_a: np.ndarray,
                  render_b: np.ndarray) -> np.ndarray:
    """Per-pixel change of the render between two neighbouring grey frames beyond what the
    source does there (float32 grey levels, >= 0): where the picture boils or shimmers.

    Motion-compensated with the source's optical flow when OpenCV is available, else plain
    frame differences (which also light up moving edges of a relit render)."""
    cv2 = _cv2()
    if cv2 is None:
        ds = np.abs(source_b.astype(np.float32) - source_a.astype(np.float32))
        dr = np.abs(render_b.astype(np.float32) - render_a.astype(np.float32))
        return np.maximum(0.0, dr - ds)
    rs, ro, valid = _pair_residuals(cv2, source_a, source_b, render_a, render_b)
    return np.where(valid, np.maximum(0.0, ro - rs), 0.0).astype(np.float32)


def _stability(cv2: Any, src: np.ndarray, out: np.ndarray) -> float:
    """0-10 from the robust mean of the render's excess warp residual over the source's."""
    gain = _contrast_scale(src, out)
    excess = []
    for t in range(len(src) - 1):
        rs, ro, valid = _pair_residuals(cv2, src[t], src[t + 1], out[t], out[t + 1], gain)
        excess.append(max(0.0, float(ro[valid].mean()) - float(rs[valid].mean())))
    # Drop the worst 10% of frame pairs: single spikes are counted as failed frames already,
    # and stability is about persistent boiling or shimmer.
    kept = np.sort(excess)[:max(1, len(excess) - len(excess) // 10)]
    return 10.0 * float(np.exp(-float(kept.mean()) / STABILITY_SCALE))


def _flicker(src: np.ndarray, out: np.ndarray) -> tuple[float, float, float]:
    """(score 0-10, render RMS, source RMS) of whole-frame brightness changes faster than a
    5-frame moving average; real lighting changes in the source are the baseline."""

    def high_pass(frames: np.ndarray) -> np.ndarray:
        m = frames.astype(np.float32).mean(axis=(1, 2))
        smooth = np.convolve(np.pad(m, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
        return np.asarray(m - smooth)

    rms_src = float(np.sqrt(np.mean(high_pass(src) ** 2)))
    rms_out = float(np.sqrt(np.mean(high_pass(out) ** 2))) * _contrast_scale(src, out)
    return 10.0 * float(np.exp(-max(0.0, rms_out - rms_src) / FLICKER_SCALE)), rms_out, rms_src


def _detail_frames(source_detail: np.ndarray | None, render_detail: np.ndarray | None,
                   cv2: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """The larger frames for the stability metric, same count and size, or None."""
    if source_detail is None or render_detail is None:
        return None
    n = min(len(source_detail), len(render_detail))
    src, out = source_detail[:n], render_detail[:n]
    if out.shape[1:] != src.shape[1:]:
        h, w = src.shape[1:]
        out = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in out]) \
            if n else out.reshape(0, h, w)
    return src, out


def score_shot(source: np.ndarray, render: np.ndarray, *, threshold: float,
               shot_id: str, frame_offset: int = 0, source_detail: np.ndarray | None = None,
               render_detail: np.ndarray | None = None,
               min_stability: float = 0.0) -> dict[str, Any]:
    """Score one shot from grey uint8 frames of the source and the render (64x64).

    ``source_detail``/``render_detail`` (grey, larger, e.g. ``detail_size``) sharpen the
    stability metric; without them it uses the 64x64 frames. ``min_stability`` > 0 fails a
    shot whose measured stability is below it; 0 leaves the decision to the existing checks.
    """
    if (source_detail is None) != (render_detail is None):
        raise ValueError("source_detail and render_detail must be given together")
    issues: list[str] = []
    recs: list[str] = []
    n = min(len(source), len(render))
    if n == 0:
        return {"shot_id": shot_id, "decision": "FAIL", "overall": 0.0,
                "issues": ["render has no frames"], "recommendations": ["RERENDER_SHOT"],
                "failed_frames": [], "stability": None, "flicker": None,
                "not_measured": {k: "render has no frames" for k in MEASURED}}
    frame_mismatch = abs(len(source) - len(render)) > max(2, 0.1 * len(source))
    if frame_mismatch:
        issues.append(f"frame count {len(render)} differs from source {len(source)}")
        recs.append("RERENDER_SHOT")
    src, out = source[:n], render[:n]
    ds, dr = motion_series(src), motion_series(out)

    # Temporal consistency: excess frame-to-frame change over what the source has.
    excess = np.maximum(0.0, dr - ds) if len(dr) else np.zeros(0)
    excess_change_mean = float(excess.mean()) if len(excess) else 0.0
    temporal = 10.0 * float(np.exp(-excess_change_mean / 6.0))
    spike_floor = max(12.0, 3.0 * float(np.median(dr))) if len(dr) else 12.0
    failed = [int(i + 1 + frame_offset) for i in np.where(excess > spike_floor)[0]]

    # Motion preservation: does output motion follow source motion over time?
    c = _corr(ds, dr)
    source_motion = float(ds.mean()) if len(ds) else 0.0
    output_motion = float(dr.mean()) if len(dr) else 0.0
    # Without motion variation to correlate, reward an output that is equally still.
    low_motion = source_motion < 1.5
    if low_motion:
        # Near-static source correlations are dominated by codec noise, not motion.
        motion = 10.0 * float(np.exp(-max(0.0, output_motion - source_motion) / 3.0))
        # Small but real source motion (a figure crossing a still scene) that the render
        # lost: it froze. Counted in clearly changing pixels, so grain is not motion.
        src_moving, out_moving = _moving_share(src), _moving_share(out)
        if src_moving > 0.002 and out_moving < 0.25 * src_moving:
            motion = min(motion, 10.0 * out_moving / src_moving)
    else:
        ratio = min(source_motion, output_motion) / max(source_motion, output_motion, 1e-6)
        motion = 10.0 * ratio if c is None else 10.0 * max(0.0, c)

    # Structure preservation: per-frame correlation of edge maps with the source. Brightness
    # is not compared: a restyle may relight the scene and keep the layout exactly.
    structure = 10.0 * max(0.0, float(np.mean([
        _ncc(a, b) for a, b in zip(_edge_maps(src), _edge_maps(out), strict=True)])))

    black = [int(i + frame_offset) for i, f in enumerate(out) if f.mean() < 8]
    failed = sorted(set(failed) | set(black))
    artifact = round(min(10.0, 10.0 * len(failed) / n * 3), 2)

    src_sharp, out_sharp = _sharpness(src), _sharpness(out)
    detail = 10.0 * min(1.0, out_sharp / src_sharp) if src_sharp > 1e-6 else 10.0
    # A sudden loss of filled surfaces into bright contours is not extra detail.
    src_dark = float((src < 16).mean())
    out_dark = float((out < 16).mean())
    edge_like = out_dark > 0.65 and out_dark - src_dark > 0.3 and out_sharp > src_sharp * 1.3
    if edge_like:
        issues.append("possible edge-map output: most filled surfaces became black; visual review needed")
        recs.append("RERENDER_SHOT")
        detail = min(detail, 3.0)

    # Stability and flicker are reported next to, not inside, ``overall``: its formula and the
    # pass threshold were calibrated on real renders.
    overall = (0.3 * temporal + 0.3 * motion + 0.25 * structure + 0.15 * detail) - 0.3 * artifact
    overall = round(max(0.0, min(10.0, overall)), 2)

    if temporal < 7:
        issues.append(f"temporal flicker (score {temporal:.1f})")
        recs += ["CHANGE_SEED", "REDUCE_STYLE_STRENGTH"]
    if motion < 6:
        issues.append(f"motion not preserved (score {motion:.1f})")
        recs.append("ADD_POSE_CONTROL")
    if structure < 5:
        issues.append(f"layout drift from source (score {structure:.1f})")
        recs += ["REDUCE_STYLE_STRENGTH", "ADD_DEPTH_CONTROL"]
    if black:
        issues.append(f"{len(black)} black/corrupt frames")
        recs.append("RERENDER_SHOT")
    elif failed:
        issues.append(f"{len(failed)} unstable frames")
        recs.append("REPAIR_FRAMES" if len(failed) <= 3 else "RERENDER_SHOT")

    # -- picture steadiness: motion-compensated stability and brightness flicker -----------
    not_measured: dict[str, str] = {}
    extra: list[str] = []  # recommendations added after the existing ones, never instead
    cv2 = _cv2()
    stability: float | None = None
    stability_method: str | None = None
    if cv2 is None:
        not_measured["stability"] = ("needs OpenCV (pip install opencv-python-headless) for "
                                     "optical flow")
    else:
        pair = _detail_frames(source_detail, render_detail, cv2) or (src, out)
        if len(pair[0]) < 2:
            not_measured["stability"] = "fewer than 2 frames"
        else:
            stability = round(_stability(cv2, pair[0], pair[1]), 2)
            h, w = pair[0].shape[1:]
            stability_method = f"source optical flow warp residual at {w}x{h}"
    unsteady = stability is not None and min_stability > 0 and stability < min_stability
    if stability is not None and (unsteady or stability < 7):
        issues.append(f"picture not steady (stability {stability:.1f}, floor {min_stability:.1f})"
                      if unsteady else f"textures not steady (stability {stability:.1f})")
        extra.append("STABILIZE")
        if structure >= 6:  # layout holds, surfaces boil: the guide drew texture edges
            extra.append("CALM_EDGES")
    if edge_like:
        extra.append("CALM_EDGES")

    flicker: float | None = None
    if n < 3:
        not_measured["flicker"] = "fewer than 3 frames"
    else:
        flicker_score, rms_out, rms_src = _flicker(src, out)
        flicker = round(flicker_score, 2)
        if flicker < 7 and rms_out >= 1.5 * rms_src:
            issues.append(f"brightness flicker beyond the source (flicker {flicker:.1f})")
            extra.append("DEFLICKER")
    if detail < 5:
        extra.append("MORE_DETAIL")

    passed = (overall >= threshold and not black and len(failed) <= 3 and not edge_like
              and not frame_mismatch and not unsteady)
    return {
        "shot_id": shot_id,
        "decision": "PASS" if passed else "FAIL",
        "overall": overall,
        "temporal_consistency": round(temporal, 2),
        "motion": round(motion, 2),
        "motion_method": "low-motion difference" if low_motion else "motion correlation",
        "visual_review_required": edge_like,
        "structure": round(structure, 2),
        "detail": round(detail, 2),
        "artifact_score": artifact,
        "stability": stability,
        "stability_method": stability_method,
        "flicker": flicker,
        "not_measured": not_measured,
        "failed_frames": failed,
        "issues": issues,
        "recommendations": ["PASS"] if passed else list(dict.fromkeys(
            (recs or ["CHANGE_SEED"]) + extra)),
    }


def summarize(shots: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [s[key] for s in shots if s.get(key) is not None]
        return round(float(np.mean(values)), 2) if values else None

    # Only what no shot measured is "not measured"; a shot's own reason wins over the default.
    not_measured = {k: reason for k, reason in NOT_MEASURED.items() if avg(k) is None}
    for key in MEASURED:
        if avg(key) is None:
            not_measured[key] = next((s["not_measured"][key] for s in shots
                                      if key in (s.get("not_measured") or {})),
                                     "not measured for these shots")
    passed = all(s["decision"] == "PASS" for s in shots)
    return {
        "decision": "PASS" if passed else "FAIL",
        "threshold": threshold,
        "overall": avg("overall"),
        "motion": avg("motion"),
        "temporal_consistency": avg("temporal_consistency"),
        "background": avg("structure"),
        "detail": avg("detail"),
        "artifact_score": avg("artifact_score"),
        "stability": avg("stability"),
        "flicker": avg("flicker"),
        **{k: avg(k) for k in NOT_MEASURED},
        "not_measured": not_measured,
        "picture_reviewed": [s["shot_id"] for s in shots if s.get("picture_review")],
        "failed_frames": sorted(f for s in shots for f in s["failed_frames"]),
        "failed_shots": [s["shot_id"] for s in shots if s["decision"] != "PASS"],
        "shots": shots,
    }
