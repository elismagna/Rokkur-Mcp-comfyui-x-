"""Find the characters in a clip before prompting: cutouts of the main subject on white.

The CPU subject model (``pipeline/subject.py``) marks the salient subject in a handful of
frames spread over the clip. Each distinct subject becomes a picture in the library (kind
``character``) that New video can hand to Wan as the appearance reference, and that the
Director's character list can name. The model sees one salient subject per frame, so two
people in one frame come out as the more prominent one; the page says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rokkur_studio.config import SubjectSection
from rokkur_studio.media.ffmpeg import FFmpeg, FFmpegError
from rokkur_studio.pipeline.subject import SubjectMasker, subject_alpha, unusable

MAX_CHARACTERS = 4
SAMPLE_FRAMES = 8
SAME_SUBJECT = 0.985   # cosine similarity of colour signatures above which two cutouts are one subject


def sample_times(ffmpeg: FFmpeg, video: Path, samples: int = SAMPLE_FRAMES) -> list[float]:
    """Moments to look at: the middle of each scene, topped up with evenly spread times."""
    info = ffmpeg.probe(video)
    duration = max(info.duration, 0.1)
    try:
        cuts = [c for c in ffmpeg.detect_scenes(video) if 0 < c < duration]
    except FFmpegError:
        cuts = []
    edges = [0.0, *sorted(cuts), duration]
    times = [(a + b) / 2 for a, b in zip(edges, edges[1:], strict=False)]
    spread = [duration * (i + 0.5) / samples for i in range(samples)]
    for t in spread:
        if len(times) >= samples:
            break
        if all(abs(t - u) > duration / (2 * samples) for u in times):
            times.append(t)
    return sorted(min(max(t, 0.0), max(duration - 0.05, 0.0)) for t in times[:samples])


def signature(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """A small colour fingerprint of the subject: a 4x4x4 RGB histogram of its pixels."""
    weights = alpha.ravel()
    if weights.sum() < 1:
        return np.zeros(64, np.float32)
    bins = (frame.reshape(-1, 3) // 64).astype(np.int64)
    index = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    hist = np.bincount(index, weights=weights, minlength=64)[:64].astype(np.float32)
    return hist / max(float(np.linalg.norm(hist)), 1e-6)


def crop_box(alpha: np.ndarray, margin: float = 0.12) -> tuple[int, int, int, int]:
    """``(top, bottom, left, right)`` around the subject, with a margin, square-ish."""
    ys, xs = np.where(alpha > 0.5)
    h, w = alpha.shape
    if len(ys) == 0:
        return 0, h, 0, w
    top, bottom, left, right = int(np.min(ys)), int(np.max(ys)) + 1, int(np.min(xs)), int(np.max(xs)) + 1
    pad = int(margin * max(bottom - top, right - left))
    return (max(0, top - pad), min(h, bottom + pad), max(0, left - pad), min(w, right + pad))


def find_characters(ffmpeg: FFmpeg, masker: SubjectMasker, video: Path, out_dir: Path, *,
                    settings: SubjectSection, width: int = 768,
                    max_characters: int = MAX_CHARACTERS) -> list[dict[str, Any]]:
    """Cutouts of the distinct subjects seen in ``video``, written under ``out_dir``.

    Returns, best first: ``path`` (the cutout on white, VACE's reference format), ``crop``
    (the cutout cropped to the subject, for the page), ``time`` of the frame it came from,
    ``share`` of that frame it covers and ``seen`` (frames it appeared in).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    found: list[dict[str, Any]] = []
    for index, t in enumerate(sample_times(ffmpeg, video)):
        frame_png = out_dir / f"frame_{index:02d}.png"
        try:
            ffmpeg.thumbnail(video, frame_png, at=t, width=width)
            small = ffmpeg.read_rgb_frames(frame_png, masker.size, masker.size)
            full = ffmpeg.read_rgb_frames(frame_png, *ffmpeg.image_size(frame_png))[0]
        except FFmpegError:
            continue
        if len(small) == 0:
            continue
        prob = masker.predict(small[:1])[0]
        share = float((prob > 0.5).mean())
        if unusable(share, settings):
            continue
        h, w = full.shape[:2]
        alpha = subject_alpha(prob, h, w, grow_px=0, feather_px=2)
        sig = signature(full, alpha)
        same = next((c for c in found if float(np.dot(c["_sig"], sig)) >= SAME_SUBJECT), None)
        if same is not None:
            same["seen"] += 1
            if share > same["share"]:  # keep the clearest view of this subject
                same.update(_write_cutout(ffmpeg, full, alpha, out_dir, int(same["index"]), t,
                                          share))
            continue
        position = len(found)
        entry: dict[str, Any] = {"index": position, "seen": 1, "_sig": sig}
        entry.update(_write_cutout(ffmpeg, full, alpha, out_dir, position, t, share))
        found.append(entry)
    found.sort(key=lambda c: (-c["seen"], -c["share"]))
    for c in found:
        del c["_sig"]
    return found[:max_characters]


def _write_cutout(ffmpeg: FFmpeg, frame: np.ndarray, alpha: np.ndarray, out_dir: Path,
                  index: int, t: float, share: float) -> dict[str, Any]:
    a = alpha[..., None]
    cutout = np.round(a * frame + (1 - a) * 255).astype(np.uint8)
    path = out_dir / f"character_{index + 1}.png"
    ffmpeg.write_image(cutout, path)
    top, bottom, left, right = crop_box(alpha)
    crop = out_dir / f"character_{index + 1}_crop.png"
    ffmpeg.write_image(cutout[top:bottom, left:right], crop)
    return {"path": str(path), "crop": str(crop), "time": round(float(t), 2),
            "share": round(share, 3), "box": [int(top), int(bottom), int(left), int(right)]}


def describe(name: str, clip_name: str, share: float) -> str:
    """The Director's description for a character found in a clip (edit it on the page)."""
    size = "fills much of the frame" if share > 0.35 else "a smaller figure in the frame"
    return (f"{name}, the main subject of {clip_name}; appearance exactly as in the reference "
            f"picture, {size}")
