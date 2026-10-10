"""Temporal stabilization: calmer motion guides before a render, steadier renders after it.

Wan VACE 1.3B shimmers mostly because its Canny guide flickers (edges drawn from grain switch
on and off between frames) and because it renders above its 480P training size or at a high
CFG (docs/stability.md). Two FFmpeg-only tools help without a GPU:

- ``control_smoothing_filter``: temporal-only denoise of the source clip before it becomes the
  Canny/depth guide, so the guide's edges hold still where the scene does.
- ``stabilize_filter`` / ``stabilize_render``: brightness deflicker, and at ``strong`` also a
  temporal-only denoise, applied to the finished render. ``pick_steadier`` keeps the result only
  when QC measures it as meaningfully steadier without losing detail.

Every filter keeps the frame count and frame rate, because QC compares frame counts with the
source.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from rokkur_studio.media.ffmpeg import FFmpeg, FFmpegError
from rokkur_studio.pipeline.qc import score_shot

log = logging.getLogger(__name__)

LEVELS = ("auto", "off", "light", "strong")
AUTO_LEVEL = "light"  # what "auto" tries; pick_steadier decides whether it stays

# hqdn3d replaces a spatial strength of exactly 0 with its default (4), which blurs every frame.
# 0.01 makes its spatial weights vanish: on a still test clip the output matched the input
# exactly, while "0" lost sharpness. Only the temporal terms do any work below.
_NO_SPATIAL = 0.01

# "am" (arithmetic mean) over "pm": on a synthetic alternating flicker both cut frame-to-frame
# brightness change by about 80%, but pm lifted the whole clip toward its brightest frames.
_FILTERS = {
    "off": "",
    "light": "deflicker=mode=am:size=5",
    "strong": ("deflicker=mode=am:size=9,"
               f"hqdn3d=luma_spatial={_NO_SPATIAL}:chroma_spatial={_NO_SPATIAL}:"
               "luma_tmp=10:chroma_tmp=8"),
}

# Keep a steadied clip only for a real gain, at little cost in detail (QC points, 0-10).
MIN_GAIN = 0.3
MAX_DETAIL_DROP = 1.0

# Temporal strength of the control smoothing at amount 1.0. On a still synthetic scene with
# heavy grain, 0.3 / 0.6 / 0.9 cut OpenCV Canny edge flicker by about 15% / 50% / 70%; with
# light grain 0.3 already cut it by about 60%.
_SMOOTH_LUMA, _SMOOTH_CHROMA = 20.0, 15.0


def resolve_level(value: object) -> str:
    """The stabilizer level a setting means: ``off``, ``light`` or ``strong``.

    ``auto`` (also an unset value) is the level tried automatically, :data:`AUTO_LEVEL`.
    Raises ``ValueError`` for anything outside :data:`LEVELS`.
    """
    if value is None or value == "":
        return AUTO_LEVEL
    if isinstance(value, bool):
        return AUTO_LEVEL if value else "off"
    level = value.strip().lower() if isinstance(value, str) else None
    if level not in LEVELS:
        raise ValueError(f"stabilize must be one of {', '.join(LEVELS)}, not {value!r}")
    return AUTO_LEVEL if level == "auto" else level


def candidate_levels(value: object) -> list[str]:
    """Levels to try for a setting, strongest first. ``strong`` falls back to ``light`` when the
    strong pass is not measurably steadier (it can cost detail that light keeps)."""
    return {"off": [], "light": ["light"], "strong": ["strong", "light"]}[resolve_level(value)]


def stabilize_filter(level: str) -> str:
    """FFmpeg ``-vf`` chain for a level (``auto`` resolves first). Empty for ``off``."""
    return _FILTERS[resolve_level(level)]


def stabilize_render(ffmpeg: FFmpeg, render: Path, out: Path, *, level: str, fps: float) -> Path:
    """Write a steadied copy of ``render`` to ``out`` at the same fps and frame count.

    Returns ``render`` itself, untouched, when the level is ``off``.
    """
    vf = stabilize_filter(level)
    if not vf:
        return render
    return ffmpeg.filter_video(render, out, vf, fps=fps)


def _steadiness_key(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Motion-compensated stability when QC measured it for both clips, else the frame-difference
    temporal consistency score."""
    if before.get("stability") is not None and after.get("stability") is not None:
        return "stability"
    return "temporal_consistency"


def _numbers(result: dict[str, Any], key: str) -> dict[str, Any]:
    return {"steadiness": result.get(key), "detail": result.get("detail"),
            "structure": result.get("structure"), "overall": result.get("overall")}


def pick_steadier(source_gray: np.ndarray, raw_gray: np.ndarray, steady_gray: np.ndarray, *,
                  threshold: float) -> tuple[bool, dict[str, Any]]:
    """Should the steadied clip replace the raw render? Both are scored against the source with
    QC's own ``score_shot``. Kept only when steadiness rises by at least :data:`MIN_GAIN`, detail
    falls by at most :data:`MAX_DETAIL_DROP` and the overall score does not fall.

    Returns the decision and the before/after numbers for the render record.
    """
    before = score_shot(source_gray, raw_gray, threshold=threshold, shot_id="raw")
    after = score_shot(source_gray, steady_gray, threshold=threshold, shot_id="steadied")
    key = _steadiness_key(before, after)
    record: dict[str, Any] = {"metric": key, "before": _numbers(before, key),
                              "after": _numbers(after, key)}
    values = (before.get(key), after.get(key), before.get("detail"), after.get("detail"))
    if any(v is None for v in values):
        return False, {**record, "kept": False, "reason": "steadiness could not be measured"}
    gain = float(after[key]) - float(before[key])
    detail_drop = float(before["detail"]) - float(after["detail"])
    if gain < MIN_GAIN:
        reason = f"{key.replace('_', ' ')} changed {gain:+.2f}, less than the +{MIN_GAIN} needed"
    elif detail_drop > MAX_DETAIL_DROP:
        reason = f"detail fell {detail_drop:.2f}, more than {MAX_DETAIL_DROP}"
    elif float(after["overall"]) < float(before["overall"]):
        reason = f"overall fell {float(before['overall']) - float(after['overall']):.2f}"
    else:
        return True, {**record, "kept": True,
                      "reason": f"{key.replace('_', ' ')} rose {gain:+.2f}"}
    return False, {**record, "kept": False, "reason": reason}


def stabilize_shot(ffmpeg: FFmpeg, *, source: Path, render: Path, value: object, fps: float,
                   qc_fps: float, threshold: float) -> tuple[Path, dict[str, Any]]:
    """Try the levels a shot's ``stabilize`` setting allows on a finished render and keep the first
    one QC measures as steadier (``pick_steadier``).

    ``source`` and ``qc_fps`` should be what the QC stage reads (the shot's source clip at the
    manifest fps), so this decision agrees with the QC that follows. Returns the clip to use
    (``render`` when nothing helped) and a record for the render's details. Stabilizer trouble
    never fails a render: the raw render stands and the reason is recorded.
    """
    requested = "auto" if value is None or value == "" else str(value)
    try:
        levels = candidate_levels(value)
    except ValueError as exc:
        return render, {"requested": requested, "kept": None, "reason": str(exc)}
    if not levels:
        return render, {"requested": requested, "kept": None, "reason": "stabilizer is off"}
    tried: list[dict[str, Any]] = []
    try:
        src = ffmpeg.read_gray_frames(source, 64, 64, fps=qc_fps)
        raw = ffmpeg.read_gray_frames(render, 64, 64, fps=qc_fps)
        raw_frames = ffmpeg.probe(render).frame_count
        for level in levels:
            out = render.with_name(f"{render.stem}_steady_{level}.mp4")
            stabilize_render(ffmpeg, render, out, level=level, fps=fps)
            frames = ffmpeg.probe(out).frame_count
            if frames != raw_frames:  # QC would fail the shot for a frame count mismatch
                tried.append({"level": level, "kept": False,
                              "reason": f"frame count changed ({raw_frames} -> {frames})"})
                out.unlink(missing_ok=True)
                continue
            keep, numbers = pick_steadier(src, raw, ffmpeg.read_gray_frames(out, 64, 64,
                                                                            fps=qc_fps),
                                          threshold=threshold)
            tried.append({"level": level, **numbers})
            if keep:
                return out, {"requested": requested, "kept": level, "tried": tried}
            out.unlink(missing_ok=True)
    except FFmpegError as exc:
        log.warning("stabilizer failed", extra={"data": {"render": str(render),
                                                         "error": exc.summary}})
        return render, {"requested": requested, "kept": None, "tried": tried,
                        "reason": f"stabilizer failed: {exc.summary}"}
    except OSError as exc:  # disk full: the raw render stands
        return render, {"requested": requested, "kept": None, "tried": tried,
                        "reason": f"stabilizer failed: {exc}"}
    return render, {"requested": requested, "kept": None, "tried": tried,
                    "reason": "no level was measurably steadier"}


def control_smoothing_filter(amount: float) -> str | None:
    """Temporal-only ``hqdn3d`` for the source clip before it becomes the Canny/depth guide.

    ``amount`` runs 0-1 (values outside are clamped); 0 means no smoothing and gives None.
    hqdn3d weighs each pixel's change between frames, so grain and shimmer settle while real
    motion, a large change, passes almost untouched.
    """
    value = float(amount)
    if math.isnan(value):
        raise ValueError("smooth_control must be a number between 0 and 1")
    value = min(1.0, max(0.0, value))
    if value <= 0:
        return None
    return (f"hqdn3d=luma_spatial={_NO_SPATIAL}:chroma_spatial={_NO_SPATIAL}:"
            f"luma_tmp={round(_SMOOTH_LUMA * value, 2):g}:"
            f"chroma_tmp={round(_SMOOTH_CHROMA * value, 2):g}")
