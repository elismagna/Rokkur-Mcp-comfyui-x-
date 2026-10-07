"""Deterministic video analysis: probe, scene boundaries, shot list, motion intensity.

Only signals the current render path consumes are computed. Pose, depth, segmentation,
optical flow and face landmarks are listed as ``skipped`` until a workflow needs them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rokkur_studio.media.ffmpeg import FFmpeg

MIN_SHOT_S = 0.5

SKIPPED_SIGNALS = {
    "pose": "computed inside ComfyUI control preprocessors when a template enables pose",
    "depth": "computed inside ComfyUI control preprocessors when a template enables depth",
    "segmentation": "not used by installed templates",
    "optical_flow": "not used by installed templates",
    "facial_landmarks": "not used by installed templates",
}


def split_shots(duration: float, cuts: list[float], max_shot_s: float) -> list[tuple[float, float]]:
    """Scene cuts → shots, merging slivers shorter than MIN_SHOT_S and splitting long shots."""
    bounds = [0.0] + [c for c in cuts if MIN_SHOT_S <= c <= duration - MIN_SHOT_S] + [duration]
    merged: list[float] = [bounds[0]]
    for b in bounds[1:-1]:
        if b - merged[-1] >= MIN_SHOT_S:
            merged.append(b)
    merged.append(duration)
    shots: list[tuple[float, float]] = []
    for start, end in zip(merged, merged[1:], strict=False):
        pieces = max(1, int(np.ceil((end - start) / max_shot_s - 1e-9)))
        step = (end - start) / pieces
        shots += [(round(start + i * step, 3), round(start + (i + 1) * step, 3))
                  for i in range(pieces)]
    return shots


def motion_series(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive grey frames (0..255)."""
    if len(frames) < 2:
        return np.zeros(0)
    f = frames.astype(np.float32)
    return np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))


def classify_motion(intensity: float) -> str:
    if intensity < 0.05:
        return "static"
    if intensity < 0.25:
        return "gentle"
    if intensity < 0.6:
        return "moderate"
    return "high"


def analyze_video(ffmpeg: FFmpeg, path: Path, *, max_shot_s: float,
                  scene_threshold: float = 0.3, sample_fps: float = 12) -> dict[str, Any]:
    info = ffmpeg.probe(path)
    cuts = ffmpeg.detect_scenes(path, scene_threshold)
    frames = ffmpeg.read_gray_frames(path, 64, 64, fps=sample_fps)
    motion = motion_series(frames)
    shots = []
    for i, (start, end) in enumerate(split_shots(info.duration, cuts, max_shot_s), 1):
        a, b = int(start * sample_fps), max(int(start * sample_fps) + 1, int(end * sample_fps) - 1)
        # Exclude the cut frame itself so a hard cut does not read as motion.
        seg = motion[a:b - 1] if b - 1 > a else motion[a:b]
        raw = float(np.median(seg)) if len(seg) else 0.0
        intensity = round(min(1.0, raw / 30.0), 3)
        shots.append({
            "shot_id": f"shot_{i:03d}", "start": start, "end": end,
            "duration": round(end - start, 3), "motion_intensity": intensity,
            "motion_type": classify_motion(intensity), "camera": "unknown",
        })
    return {
        "probe": info.to_dict(),
        "duration": info.duration,
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "has_audio": info.has_audio,
        "scene_cuts": cuts,
        "scene_threshold": scene_threshold,
        "shots": shots,
        "signals_computed": ["probe", "scene_boundaries", "shot_duration", "motion_intensity"],
        "signals_skipped": SKIPPED_SIGNALS,
    }
