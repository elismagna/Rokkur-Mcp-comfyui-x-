"""Quality control: measurable, deterministic metrics only.

Metrics needing a vision model (identity similarity, prompt adherence, style consistency,
hand/body deformation) are reported as ``null`` with a reason rather than invented.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rokkur_studio.pipeline.analysis import motion_series

NOT_MEASURED = {
    "identity": "needs a face/identity embedding model (Phase 4)",
    "prompt_adherence": "needs a vision-language model (Phase 4)",
    "style_consistency": "needs a style embedding model (Phase 4)",
    "hand_body_deformation": "needs a pose/keypoint model (Phase 4)",
}


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


def _sharpness(frames: np.ndarray) -> float:
    f = frames.astype(np.float32)
    lap = (f[:, 1:-1, 1:-1] * 4 - f[:, :-2, 1:-1] - f[:, 2:, 1:-1] - f[:, 1:-1, :-2]
           - f[:, 1:-1, 2:])
    return float(lap.var()) if lap.size else 0.0


def score_shot(source: np.ndarray, render: np.ndarray, *, threshold: float,
               shot_id: str, frame_offset: int = 0) -> dict[str, Any]:
    issues: list[str] = []
    recs: list[str] = []
    n = min(len(source), len(render))
    if n == 0:
        return {"shot_id": shot_id, "decision": "FAIL", "overall": 0.0,
                "issues": ["render has no frames"], "recommendations": ["RERENDER_SHOT"],
                "failed_frames": []}
    frame_mismatch = abs(len(source) - len(render)) > max(2, 0.1 * len(source))
    if frame_mismatch:
        issues.append(f"frame count {len(render)} differs from source {len(source)}")
        recs.append("RERENDER_SHOT")
    src, out = source[:n], render[:n]
    ds, dr = motion_series(src), motion_series(out)

    # Temporal consistency: excess frame-to-frame change over what the source has.
    excess = np.maximum(0.0, dr - ds) if len(dr) else np.zeros(0)
    flicker = float(excess.mean()) if len(excess) else 0.0
    temporal = 10.0 * float(np.exp(-flicker / 6.0))
    spike_floor = max(12.0, 3.0 * float(np.median(dr))) if len(dr) else 12.0
    failed = [int(i + 1 + frame_offset) for i in np.where(excess > spike_floor)[0]]

    # Motion preservation: does output motion follow source motion over time?
    c = _corr(ds, dr)
    # Without motion variation to correlate, reward an output that is equally still.
    low_motion = float(ds.mean() if len(ds) else 0) < 1.5
    if low_motion:
        # Near-static source correlations are dominated by codec noise, not motion.
        motion = 10.0 * float(np.exp(-max(0.0, float(dr.mean() if len(dr) else 0)
                                          - float(ds.mean() if len(ds) else 0)) / 3.0))
    else:
        source_motion = float(ds.mean())
        output_motion = float(dr.mean() if len(dr) else 0)
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

    passed = overall >= threshold and not black and len(failed) <= 3 and not edge_like and not frame_mismatch
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
        "failed_frames": failed,
        "issues": issues,
        "recommendations": ["PASS"] if passed else list(dict.fromkeys(recs or ["CHANGE_SEED"])),
    }


def summarize(shots: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [s[key] for s in shots if s.get(key) is not None]
        return round(float(np.mean(values)), 2) if values else None

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
        **{k: None for k in NOT_MEASURED},
        "not_measured": NOT_MEASURED,
        "failed_frames": sorted(f for s in shots for f in s["failed_frames"]),
        "failed_shots": [s["shot_id"] for s in shots if s["decision"] != "PASS"],
        "shots": shots,
    }
