"""The picture checker: what a render actually looks like (docs/picture-checker.md).

* ``contact_sheet`` puts frames sampled through a shot side by side: the source, the render,
  and a heatmap of where the render changes more than the source (boiling, shimmer, flicker).
* ``png_bytes`` encodes it without Pillow, for the dashboard and for a vision model.
* ``PictureChecker`` shows the sheet to a vision model (Elis's local Ollama ``qwen3.5:9b``)
  and returns its ``PictureReview``. That is a model's opinion, not a measurement:
  ``apply_review`` copies it into the QC shot as advisory scores and never flips PASS/FAIL,
  and when no model can see the image there is no review, never an invented one.
"""

from __future__ import annotations

import logging
import struct
import zlib
from typing import Any, Literal, get_args

import numpy as np
from pydantic import BaseModel, Field, field_validator

from rokkur_studio.agents.providers import AgentOutputError, AgentProvider, AgentUnavailable
from rokkur_studio.pipeline.qc import excess_change

log = logging.getLogger(__name__)

SEPARATOR = 200          # grey of the thin lines between cells
HEAT_FULL_SCALE = 24.0   # excess change (grey levels) drawn at full heat
PAIRS_PER_CELL = 6       # frame pairs averaged into one heatmap cell, spread over its span

PictureIssue = Literal[
    "melting_subject", "extra_limbs", "face_distortion", "hand_distortion", "identity_drift",
    "style_drift", "prompt_ignored", "source_look_leaks", "texture_boiling", "flicker",
    "blurry", "black_or_broken_frames",
]
_ANATOMY_ISSUES = {"melting_subject", "extra_limbs", "face_distortion", "hand_distortion"}
_PROMPT_ISSUES = {"prompt_ignored", "source_look_leaks"}
_STEADINESS_ISSUES = {"texture_boiling", "flicker"}


# -- contact sheet --------------------------------------------------------------------------
def frame_size(width: int, height: int, longest: int = 256) -> tuple[int, int]:
    """Cell size for a contact sheet: the longest side ``longest`` (never upscaled), aspect
    kept, both sides even for ffmpeg's scaler."""
    scale = min(1.0, longest / max(width, height, 1))
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def _rgb(frames: np.ndarray) -> np.ndarray:
    """``(n, h, w, 3)`` uint8 from RGB or grey ``(n, h, w)`` frames."""
    f = np.asarray(frames)
    if f.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {f.dtype}")
    if f.ndim == 3:
        return np.repeat(f[..., None], 3, axis=-1)
    if f.ndim == 4 and f.shape[-1] == 3:
        return f
    raise ValueError(f"frames must be (n, h, w) or (n, h, w, 3), got {f.shape}")


def _resize(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize, enough to line a mismatched render up with its source."""
    ys = (np.arange(h) * frames.shape[1] / h).astype(int)
    xs = (np.arange(w) * frames.shape[2] / w).astype(int)
    return frames[:, ys][:, :, xs]


def _grey(frames: np.ndarray) -> np.ndarray:
    luma = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.clip(luma + 0.5, 0, 255).astype(np.uint8)


def _hot(v: np.ndarray) -> np.ndarray:
    """Black -> red -> yellow -> white for 0..1."""
    return np.stack([np.clip(3 * v, 0, 1), np.clip(3 * v - 1, 0, 1), np.clip(3 * v - 2, 0, 1)],
                    axis=-1) * 255.0


def _heat_cell(src: np.ndarray, out: np.ndarray, pairs: list[int], base: np.ndarray) -> np.ndarray:
    """Mean excess change over ``pairs`` (frame t to t+1), over a dimmed render frame so the
    hot spots can be located on the picture."""
    excess = np.mean([excess_change(src[t], src[t + 1], out[t], out[t + 1]) for t in pairs], axis=0)
    v = np.clip(excess / HEAT_FULL_SCALE, 0.0, 1.0)
    alpha = np.clip(3 * v, 0.0, 1.0)[..., None]
    dim = np.repeat(base.astype(np.float32)[..., None] * 0.35, 3, axis=-1)
    return np.clip(dim * (1 - alpha) + _hot(v) * alpha + 0.5, 0, 255).astype(np.uint8)


def sample_indices(n: int, columns: int) -> list[int]:
    """``columns`` frame indices spread evenly over ``n`` frames, first and last included."""
    if n <= 0:
        return []
    return [int(i) for i in np.unique(np.linspace(0, n - 1, max(1, min(columns, n))).round())]


def contact_sheet(source_rgb: np.ndarray, render_rgb: np.ndarray, *, columns: int = 4,
                  heat: bool = True, gap: int = 2) -> np.ndarray:
    """An ``(H, W, 3)`` uint8 sheet: row 1 source frames, row 2 the render at the same moments,
    row 3 (``heat``) where the render changes more than the source between neighbouring
    frames. Time runs left to right; each heat cell averages the frame pairs of its stretch
    of the shot. No text is drawn (no Pillow); the layout is fixed instead.
    """
    src, out = _rgb(source_rgb), _rgb(render_rgb)
    n = min(len(src), len(out))
    if n == 0:
        raise ValueError("contact sheet needs at least one source and one render frame")
    src, out = src[:n], out[:n]
    h, w = src.shape[1:3]
    if out.shape[1:3] != (h, w):
        out = _resize(out, h, w)
    idx = sample_indices(n, columns)
    rows = [[src[i] for i in idx], [out[i] for i in idx]]
    if heat and n >= 2:
        gs, go = _grey(src), _grey(out)
        bounds = [0] + [(a + b) // 2 for a, b in zip(idx, idx[1:], strict=False)] + [n - 1]
        cells = []
        for k, i in enumerate(idx):
            lo, hi = min(bounds[k], n - 2), min(max(bounds[k] + 1, bounds[k + 1]), n - 1)
            pairs = sorted({int(t) for t in np.linspace(lo, hi - 1, min(PAIRS_PER_CELL, hi - lo))
                            .round()}) or [lo]
            cells.append(_heat_cell(gs, go, pairs, go[i]))
        rows.append(cells)
    sheet = np.full((len(rows) * h + (len(rows) - 1) * gap, len(idx) * w + (len(idx) - 1) * gap, 3),
                    SEPARATOR, dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            y, x = r * (h + gap), c * (w + gap)
            sheet[y:y + h, x:x + w] = cell
    return sheet


def png_bytes(image: np.ndarray) -> bytes:
    """Encode a uint8 grey ``(h, w)``, RGB ``(h, w, 3)`` or RGBA ``(h, w, 4)`` image as PNG.

    Pure Python (zlib + struct) so no imaging library is needed. Every row uses PNG's "Up"
    filter, which compresses photographic frames far better than none at no real cost.
    """
    img = np.ascontiguousarray(image)
    if img.dtype != np.uint8:
        raise ValueError(f"PNG needs uint8 pixels, got {img.dtype}")
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4) or 0 in img.shape:
        raise ValueError(f"PNG needs an (h, w), (h, w, 3) or (h, w, 4) image, got {image.shape}")
    h, w, ch = img.shape
    rows = img.reshape(h, w * ch)
    up = rows.copy()
    up[1:] = rows[1:] - rows[:-1]  # uint8 arithmetic wraps modulo 256, as the filter wants
    raw = np.concatenate([np.full((h, 1), 2, dtype=np.uint8), up], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", w, h, 8, {1: 0, 3: 2, 4: 6}[ch], 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


# -- AI picture review ----------------------------------------------------------------------
class PictureReview(BaseModel):
    """A vision model's honest look at one shot's contact sheet. Scores 0-10, 10 best."""

    description: str = Field(min_length=1, max_length=400)   # what the render actually shows
    prompt_adherence: int = Field(ge=0, le=10)
    style_consistency: int = Field(ge=0, le=10)
    subject_identity: int = Field(ge=0, le=10)
    anatomy: int = Field(ge=0, le=10)       # 10: bodies, faces and hands intact
    steadiness: int = Field(ge=0, le=10)    # 10: surfaces hold still between frames
    issues: list[PictureIssue] = Field(max_length=len(get_args(PictureIssue)))
    notes: str = Field(max_length=300)

    @field_validator("issues")
    @classmethod
    def _unique(cls, v: list[PictureIssue]) -> list[PictureIssue]:
        return list(dict.fromkeys(v))


class PictureChecker:
    """Shows a shot's contact sheet to a vision-capable agent provider for a ``PictureReview``.

    Returns None, with the reason on ``skipped`` and in the log, when the provider cannot see
    images, is down, or keeps answering with invalid JSON: QC then simply has no AI review.
    """

    role = "picture_checker"
    instructions = (
        "You check one shot of a restyled video. The attached image is a contact sheet of "
        "frames sampled evenly through the shot, time running left to right. Top row: the "
        "source video. Second row: the render, the same moments restyled. Third row, when "
        "sheet.rows lists it: a change heatmap over a darkened copy of the render, black where "
        "the render changes like the source and red to yellow to white where it changes more "
        "than the source between neighbouring frames (texture boiling, shimmer, flicker). "
        "Judge the render row against the theme and prompt; use the source row to see what "
        "should be kept: subject, pose, layout and motion. In description say in one or two "
        "plain sentences what the render actually shows, not what the prompt asked for. "
        "Score honestly from 0 to 10, 10 best: prompt_adherence (the render shows the "
        "prompt's look and setting, not the source's look), style_consistency (one style "
        "across all render frames), subject_identity (the main subject stays the same "
        "character across frames and fits the prompt), anatomy (bodies, faces and hands "
        "intact; melting, extra or missing limbs score low), steadiness (surfaces and "
        "textures hold still; use the heatmap when present). A mediocre render gets a "
        "mediocre score: do not round up. In issues list only problems you can see, using "
        "allowed_issues; an empty list is fine. If the frames are too small to judge a "
        "detail, do not list an issue for it. notes: one short sentence for the person "
        "fixing the shot."
    )

    def __init__(self, provider: AgentProvider, *, max_calls: int = 24) -> None:
        self.provider, self.max_calls = provider, max_calls
        self.calls = 0
        self.skipped: str | None = None  # why the last review() returned None
        self._down = False

    def _skip(self, shot_id: str, reason: str) -> None:
        self.skipped = reason
        log.warning("picture checker: no review", extra={"data": {
            "shot": shot_id, "provider": self.provider.name, "reason": reason}})

    def review(self, *, shot_id: str, theme: str, prompt: str, sheet_png: bytes,
               columns: int = 4, heat: bool = True) -> PictureReview | None:
        """The model's review of one sheet, or None (see ``skipped``). ``columns``/``heat``
        describe the sheet as ``contact_sheet`` drew it."""
        self.skipped = None
        if self._down:
            self._skip(shot_id, "the vision model was unavailable earlier in this check")
            return None
        if self.calls >= self.max_calls:
            self._skip(shot_id, f"review limit of {self.max_calls} shots reached")
            return None
        if not self.provider.supports_images():
            self._skip(shot_id, f"agent provider {self.provider.name!r} cannot see images")
            return None
        payload: dict[str, Any] = {
            "shot_id": shot_id, "theme": theme[:500], "prompt": prompt[:1500],
            "sheet": {"columns": columns, "time": "left to right",
                      "rows": ["source", "render"] + (["change heatmap"] if heat else [])},
            "scores": "integers 0-10, 10 best",
            "allowed_issues": list(get_args(PictureIssue)),
        }
        self.calls += 1
        try:
            return self.provider.generate(self.role, self.instructions, payload, PictureReview,
                                          images=[sheet_png])
        except AgentUnavailable as exc:
            self._down = True  # do not wait on a dead model for every remaining shot
            self._skip(shot_id, f"vision model unavailable: {exc}")
        except AgentOutputError as exc:
            self._skip(shot_id, f"vision model gave no valid review: {exc}")
        return None


def apply_review(shot_result: dict[str, Any], review: PictureReview, *,
                 by: str | None = None) -> dict[str, Any]:
    """A copy of a QC shot with the review's scores filled in (identity, prompt_adherence,
    style_consistency, hand_body_deformation; 10 best) and a ``picture_review`` block.

    Advisory: the decision is never changed. A failing shot gets matching repair hints
    appended (FOLLOW_PROMPT, FIX_ANATOMY, STABILIZE); existing ones are kept.
    """
    out = dict(shot_result)
    out.update(identity=review.subject_identity, prompt_adherence=review.prompt_adherence,
               style_consistency=review.style_consistency, hand_body_deformation=review.anatomy)
    out["picture_review"] = {"description": review.description, "issues": list(review.issues),
                             "notes": review.notes, "steadiness": review.steadiness,
                             **({"by": by} if by else {})}
    if out.get("decision") != "PASS":
        issues = set(review.issues)
        recs = list(out.get("recommendations") or [])
        if review.prompt_adherence <= 4 or issues & _PROMPT_ISSUES:
            recs.append("FOLLOW_PROMPT")
        if review.anatomy <= 4 or issues & _ANATOMY_ISSUES:
            recs.append("FIX_ANATOMY")
        if review.steadiness <= 4 or issues & _STEADINESS_ISSUES:
            recs.append("STABILIZE")
        out["recommendations"] = list(dict.fromkeys(recs))
    return out


def check_shot(checker: PictureChecker | None, shot_result: dict[str, Any],
               source_rgb: np.ndarray, render_rgb: np.ndarray, *, theme: str, prompt: str,
               columns: int = 4, heat: bool = True) -> tuple[dict[str, Any], bytes]:
    """Contact sheet for one scored shot and, when ``checker`` gets a review, the shot with it
    applied (else ``picture_review_skipped`` says why). Returns (shot, sheet PNG)."""
    sheet = png_bytes(contact_sheet(source_rgb, render_rgb, columns=columns, heat=heat))
    if checker is None:
        return shot_result, sheet
    n = min(len(source_rgb), len(render_rgb))  # tell the model the layout actually drawn
    review = checker.review(shot_id=str(shot_result.get("shot_id", "")), theme=theme,
                            prompt=prompt, sheet_png=sheet,
                            columns=len(sample_indices(n, columns)), heat=heat and n >= 2)
    if review is None:
        return {**shot_result, "picture_review_skipped": checker.skipped}, sheet
    return apply_review(shot_result, review, by=checker.provider.name), sheet
