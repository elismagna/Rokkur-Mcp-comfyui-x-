"""The main subject: decide whether the real one stays, and lay it back over the render.

Wan 2.1 VACE 1.3B restyles rooms well but melts animals and people (renders 62/63 of the ape
clip, 2026-10-08). When the prompt changes the place rather than the subject, Studio keeps the
real subject: a salient-object model (U²-Net, Apache-2.0, run on the CPU with onnxruntime) finds
it in every frame of the source shot, and the source pixels go back over the render with a soft,
colour-matched edge. The raw render is kept next to the result for comparison.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import numpy as np

from rokkur_studio.config import SubjectSection
from rokkur_studio.media.ffmpeg import FFmpeg

log = logging.getLogger(__name__)

SubjectMode = Literal["keep", "restyle"]
SubjectChoice = Literal["auto", "keep", "restyle"]

# -- deciding -----------------------------------------------------------------------------
# Words for the place, the look and the whole frame: "turn the bathroom into a spa" restyles
# the place, not the subject.
_PLACE = (r"(?:room|bathroom|bedroom|kitchen|house|home|interior|background|backdrop|scene|"
          r"place|setting|environment|world|walls?|floor|ceiling|sky|city|street|forest|"
          r"landscape|location|space|area|surroundings|lighting|light|colou?rs?|palette|"
          r"weather|season|time|era|mood|atmosphere|vibe|tone|feel|tiles?|furniture|camera|"
          r"angle|everything|it|this|video|clip|shot|footage|look|style)")
# A word that can name the subject: not the place, not an article (so the regex cannot step
# over "the" to reach "the room").
_SUBJECT_WORD = rf"(?!(?:{_PLACE}|the|a|an|my|its|sure|up|over)\b)\w+"
_KEEP = re.compile(
    r"\bkeep (?:the |him |her |it |them |this )?(?:\w+ )?(?:real|realistic|unchanged|untouched|"
    r"original|as (?:it|he|she|they) (?:is|are))\b"
    rf"|\b(?:only|just) (?:change |restyle |transform )?(?:the )?{_PLACE}\b"
    r"|\b(?:background|backdrop|room|environment|setting|surroundings) only\b"
    rf"|\b(?:don'?t|do not|never) (?:change|touch|restyle|alter|transform) (?:the )?{_SUBJECT_WORD}",
    re.I)
_TRANSFORM = re.compile(
    rf"\b(?:turn|transform|change|convert|make|replace)\w* (?:the |him |her |them )?"
    rf"{_SUBJECT_WORD}(?: \w+){{0,3}}? (?:into|to|with|looks? like)\b"
    rf"|\breimagin\w* (?:the |him |her |them )?{_SUBJECT_WORD} as\b"
    r"|\b(?:dressed (?:up )?as|wearing|in an? \w*\s?costume)\b", re.I)
_STYLIZED = re.compile(
    r"\b(anime|cartoon\w*|toon|cel[- ]shad\w*|clay\w*|plasticine|stop[- ]motion|pixar|disney|"
    r"dreamworks|ghibli|3d(?: render\w*| animat\w*| cartoon| look| style)|cgi|blender|"
    r"low[- ]poly|voxel|pixel art|8[- ]bit|16[- ]bit|lego|watercolou?r|oil paint\w*|painting|"
    r"painted|gouache|sketch\w*|pencil (?:drawing|sketch)|charcoal|ink (?:drawing|wash)|"
    r"comics?|comic book|manga|illustrat\w*|origami|paper[- ]?cut\w*|papercraft|felted|"
    r"animation|animated|video game|unreal engine)\b", re.I)


def is_stylized(text: str) -> bool:
    """Whether a look asks for a drawn, animated or otherwise non-photographic style."""
    return bool(_STYLIZED.search(text))


@dataclass(frozen=True)
class SubjectDecision:
    mode: SubjectMode
    reason: str
    decided_by: Literal["you", "auto"]

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "reason": self.reason, "decided_by": self.decided_by}


def decide_subject(creative: Mapping[str, Any]) -> SubjectDecision:
    """Keep the real subject or restyle it, from the project's creative input.

    The person's own choice wins. Otherwise: a chosen character, a prompt that changes the
    subject, or a stylized look (a real ape in a cartoon room looks pasted in) restyle it; a
    prompt that only changes the place keeps it, since that is where Wan 1.3B is strong.
    """
    choice = str(creative.get("subject") or "auto")
    if choice == "keep":
        return SubjectDecision("keep", "You chose to keep the real subject.", "you")
    if choice == "restyle":
        return SubjectDecision("restyle", "You chose to restyle the subject.", "you")
    if any(creative.get(k) for k in ("character_key", "character_description",
                                     "character_reference_path", "character_reference_asset")):
        return SubjectDecision("restyle", "You picked a character, so the subject becomes it.",
                               "auto")
    text = " ".join(str(creative.get(k) or "") for k in ("theme", "style", "prompt"))
    text = re.sub(r"\s+", " ", text)
    if m := _KEEP.search(text):
        return SubjectDecision("keep", f'Your prompt keeps the subject ("{m.group(0).strip()}").', "auto")
    if m := _TRANSFORM.search(text):
        return SubjectDecision("restyle",
                               f'Your prompt changes the subject ("{m.group(0).strip()}").', "auto")
    if m := _STYLIZED.search(text):
        return SubjectDecision(
            "restyle", f'The look is stylized ("{m.group(0)}"), and a real subject would look '
            "pasted in.", "auto")
    return SubjectDecision("keep", "Your prompt changes the place and the look, not the subject.",
                           "auto")


# -- the mask model -----------------------------------------------------------------------
class MaskerUnavailable(RuntimeError):
    """The subject mask model cannot run here (onnxruntime missing, model not downloadable)."""


@dataclass(frozen=True)
class MaskModel:
    url: str
    sha256: str
    size: int  # square input side
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


# The ONNX exports rembg publishes (MIT); the weights are Apache-2.0 (U²-Net, DIS/IS-Net).
_RELEASES = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"
MODELS: dict[str, MaskModel] = {
    "u2net": MaskModel(f"{_RELEASES}/u2net.onnx",
                       "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491", 320,
                       (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "isnet-general-use": MaskModel(
        f"{_RELEASES}/isnet-general-use.onnx",
        "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a", 1024,
        (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
}


class SubjectMasker(Protocol):
    name: str
    size: int

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """``(n, size, size, 3)`` uint8 RGB → ``(n, size, size)`` float32 subject probability."""
        ...


def fetch_model(spec: MaskModel, path: Path) -> Path:
    """Download a model into ``path``; the file only appears once its checksum matches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.part")
    digest = hashlib.sha256()
    log.info("downloading subject mask model", extra={"data": {"url": spec.url}})
    try:
        with httpx.stream("GET", spec.url, follow_redirects=True,
                          timeout=httpx.Timeout(30, read=120)) as response:
            response.raise_for_status()
            with part.open("wb") as fh:
                for chunk in response.iter_bytes(1 << 20):
                    digest.update(chunk)
                    fh.write(chunk)
    except (httpx.HTTPError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise MaskerUnavailable(f"could not download the subject mask model ({exc}); put "
                                f"{Path(spec.url).name} from {spec.url} into {path.parent}") from exc
    if digest.hexdigest() != spec.sha256:
        part.unlink(missing_ok=True)
        raise MaskerUnavailable(f"the downloaded {path.name} does not match its checksum")
    part.replace(path)
    return path


class OnnxSubjectMasker:
    """Salient-object masks on the CPU. The session opens on first use, once per instance."""

    def __init__(self, settings: SubjectSection, data_dir: Path) -> None:
        self.name: str = settings.model
        self.spec = MODELS[settings.model]
        self.size = self.spec.size
        self.path = Path(settings.model_dir or Path(data_dir) / "models") / f"{self.name}.onnx"
        self.download, self.threads = settings.download, settings.threads
        self._session: Any = None
        self._error: str | None = None

    def problem(self) -> str | None:
        """Why masks cannot be made here, without downloading anything (for system checks)."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return "onnxruntime is not installed"
        if not self.path.is_file() and not self.download:
            return f"{self.path} is missing and subject.download is off"
        return None

    def _open(self) -> Any:
        if self._error:
            raise MaskerUnavailable(self._error)
        if self._session is not None:
            return self._session
        try:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise MaskerUnavailable("onnxruntime is not installed; rebuild the studio image "
                                        "or pip install onnxruntime") from exc
            if not self.path.is_file():
                if not self.download:
                    raise MaskerUnavailable(f"{self.path} is missing and subject.download is off")
                fetch_model(self.spec, self.path)
            options = ort.SessionOptions()
            if self.threads:
                options.intra_op_num_threads = self.threads
            try:
                self._session = ort.InferenceSession(str(self.path), options,
                                                     providers=["CPUExecutionProvider"])
            except Exception as exc:  # onnxruntime raises its own error types
                raise MaskerUnavailable(f"could not load {self.path.name}: {exc}") from exc
        except MaskerUnavailable as exc:
            self._error = str(exc)  # one attempt per render job, not one per shot
            raise
        return self._session

    def predict(self, frames: np.ndarray) -> np.ndarray:
        session = self._open()
        name = session.get_inputs()[0].name
        mean = np.array(self.spec.mean, np.float32)
        std = np.array(self.spec.std, np.float32)
        out = np.empty((len(frames), self.size, self.size), np.float32)
        for i, frame in enumerate(frames):
            x = frame.astype(np.float32)
            x /= max(float(x.max()), 1.0)  # as rembg prepares its input
            x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
            try:
                out[i] = session.run(None, {name: x})[0][0, 0]
            except Exception as exc:
                raise MaskerUnavailable(f"the subject mask model failed: {exc}") from exc
        return np.clip(out, 0.0, 1.0)


# -- masks and compositing (pure numpy) ---------------------------------------------------
def smooth_in_time(probs: np.ndarray, radius: int = 1) -> np.ndarray:
    """Average each mask with its neighbours so the edge does not shimmer from frame to frame."""
    n = len(probs)
    if radius <= 0 or n < 2:
        return probs
    padded = np.pad(probs, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(probs, dtype=np.float32)
    for k in range(2 * radius + 1):
        out += padded[k:k + n]
    return out / (2 * radius + 1)


def resize(a: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of one ``(h, w)`` float image."""
    h, w = a.shape
    if (h, w) == (height, width):
        return a.astype(np.float32)
    ys = np.clip((np.arange(height) + 0.5) * h / height - 0.5, 0, h - 1)
    xs = np.clip((np.arange(width) + 0.5) * w / width - 0.5, 0, w - 1)
    y0, x0 = ys.astype(np.intp), xs.astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    a = a.astype(np.float32)
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bottom = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def _window(a: np.ndarray, start: int, n: int, axis: int) -> np.ndarray:
    return a[start:start + n] if axis == 0 else a[:, start:start + n]


def grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """Max filter over a square of side ``2 * radius + 1``."""
    if radius <= 0:
        return mask
    out = mask
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = _window(padded, 0, n, axis).copy()
        for k in range(1, 2 * radius + 1):
            np.maximum(acc, _window(padded, k, n, axis), out=acc)
        out = acc
    return out


def _box(mask: np.ndarray, radius: int, axis: int) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius + 1, radius)
    c = np.cumsum(np.pad(mask, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = mask.shape[axis]
    return ((_window(c, 2 * radius + 1, n, axis) - _window(c, 0, n, axis))
            / (2 * radius + 1)).astype(np.float32)


def feather(mask: np.ndarray, width: int) -> np.ndarray:
    """Soften the edge over about ``width`` pixels each side (two box passes per axis)."""
    radius = max(1, width // 2) if width > 0 else 0
    for _ in range(2 if radius else 0):
        mask = _box(_box(mask, radius, 0), radius, 1)
    return mask


def subject_alpha(prob: np.ndarray, height: int, width: int, *, grow_px: int,
                  feather_px: int) -> np.ndarray:
    """Blend weight of the source at render size: 1 on the subject, 0 away from it."""
    m = np.clip((resize(prob, height, width) - 0.3) / 0.4, 0.0, 1.0)
    return np.clip(feather(grow(m, grow_px), feather_px), 0.0, 1.0)


def colour_shift(source: np.ndarray, rendered: np.ndarray, alphas: list[np.ndarray],
                 frames: list[int], *, strength: float, ring_px: int) -> np.ndarray:
    """Per-channel offset that moves the subject's colours toward the new room around it.

    Compares the source and the render in a ring just outside the subject, over a few frames,
    so the offset is one value for the whole shot and cannot flicker.
    """
    if strength <= 0:
        return np.zeros(3, np.float32)
    src_sum = np.zeros(3)
    out_sum = np.zeros(3)
    count = 0
    for alpha, i in zip(alphas, frames, strict=True):
        ring = (grow((alpha > 0.5).astype(np.float32), ring_px) > 0.5) & (alpha < 0.05)
        k = int(ring.sum())
        if k:
            src_sum += source[i][ring].sum(axis=0)
            out_sum += rendered[i][ring].sum(axis=0)
            count += k
    if count < 100:
        return np.zeros(3, np.float32)
    shift = strength * (out_sum - src_sum) / count
    return np.clip(shift, -40, 40).astype(np.float32)


def shot_masks(ffmpeg: FFmpeg, masker: SubjectMasker, *, clip: Path, fps: float, frames: int,
               aspect: tuple[int, int], cache: Path) -> np.ndarray:
    """Subject probability for each of the shot's first ``frames`` frames, ``(frames, s, s)``.

    The source is centre-cropped to ``aspect`` (the render's shape) first. The result is kept
    in ``<cache>_<model>.npy``, so the VACE mask, the cutout reference, the composite and later
    repair rounds all reuse one pass of the model.
    """
    masks_file = cache.with_name(f"{cache.name}_{masker.name}.npy")
    if masks_file.is_file():
        loaded = np.load(masks_file)
        if (loaded.ndim == 3 and loaded.shape[0] >= frames
                and loaded.shape[1:] == (masker.size, masker.size)):
            return loaded[:frames].astype(np.float32) / 255
    small = ffmpeg.read_rgb_frames(clip, masker.size, masker.size, fps=fps, frames=frames,
                                   aspect=aspect)
    probs = smooth_in_time(masker.predict(small))
    masks_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(masks_file, np.round(probs * 255).astype(np.uint8))
    return probs


def subject_share(probs: np.ndarray) -> float:
    """Median share of the frame the subject covers."""
    return float(np.median((probs > 0.5).mean(axis=(1, 2))))


def unusable(share: float, settings: SubjectSection) -> str | None:
    """Why masks with this coverage should not be used, or None."""
    if share < settings.min_coverage:
        return "no clear subject in this shot"
    if share > settings.max_coverage:
        return "the subject fills most of the frame"
    return None


def vace_mask_video(ffmpeg: FFmpeg, probs: np.ndarray, out: Path, *, width: int, height: int,
                    fps: float) -> Path:
    """The subject in white on black at render size, for WanVaceToVideo's ``control_masks``
    (the keep workflow inverts it, so the room is regenerated and the subject is kept)."""
    frames = np.stack([np.round(np.clip((resize(p, height, width) - 0.3) / 0.4, 0, 1) * 255)
                       .astype(np.uint8) for p in probs])
    return ffmpeg.write_frames(frames, out, fps)


def cutout_reference(ffmpeg: FFmpeg, probs: np.ndarray, *, clip: Path, fps: float, width: int,
                     height: int, out: Path) -> Path:
    """The subject from the source on plain white, as VACE expects a reference image.

    VACE was trained on object or background references, not on scenes (Comfy-Org's ref2v
    template note), so a whole source frame drags the real room and lighting into the restyle.
    Uses the first frame unless the subject is barely in it, then the frame showing most of it.
    """
    shares = (probs > 0.5).mean(axis=(1, 2))
    i = 0 if shares[0] >= 0.5 * float(np.median(shares)) else int(np.argmax(shares))
    frame = ffmpeg.read_rgb_frames(clip, width, height, fps=fps, frames=i + 1,
                                   aspect=(width, height))[i]
    a = subject_alpha(probs[i], height, width, grow_px=0, feather_px=2)[..., None]
    return ffmpeg.write_image(np.round(a * frame + (1 - a) * 255).astype(np.uint8), out)


def keep_subject(ffmpeg: FFmpeg, masker: SubjectMasker, settings: SubjectSection, *,
                 clip: Path, render: Path, fps: float, out: Path, cache: Path) -> dict[str, Any]:
    """Lay the source shot's main subject over its render, written to ``out``.

    ``cache`` names the shot's mask files (see ``shot_masks``); a mask preview video is written
    next to them. Returns what was done; ``kept`` is False when the shot has no clear subject or
    the subject fills the frame, and ``out`` is then not written.
    """
    started = time.monotonic()
    info = ffmpeg.probe(render)
    width, height = info.width, info.height
    rendered = ffmpeg.read_rgb_frames(render, width, height)
    n = len(rendered)
    if n == 0:
        return {"kept": False, "reason": "the render has no frames"}
    aspect = (width, height)
    probs = shot_masks(ffmpeg, masker, clip=clip, fps=fps, frames=n, aspect=aspect, cache=cache)
    share = subject_share(probs)
    result: dict[str, Any] = {"model": masker.name, "coverage": round(share, 3)}
    if reason := unusable(share, settings):
        return {**result, "kept": False, "reason": reason}

    source = ffmpeg.read_rgb_frames(clip, width, height, fps=fps, frames=n, aspect=aspect)
    short = min(width, height)
    grow_px = round(settings.grow * short)
    feather_px = round(settings.feather * short)

    def alpha(i: int) -> np.ndarray:
        return subject_alpha(probs[i], height, width, grow_px=grow_px, feather_px=feather_px)

    sample = sorted({round(i) for i in np.linspace(0, n - 1, min(n, 12))})
    shift = colour_shift(source, rendered, [alpha(i) for i in sample], sample,
                         strength=settings.harmonize, ring_px=max(4, 3 * grow_px))
    composite = np.empty_like(rendered)
    mask_frames = np.empty((n, height, width), np.uint8)
    for i in range(n):
        a = alpha(i)
        mask_frames[i] = np.round(a * 255)
        subject = np.clip(source[i].astype(np.float32) + shift, 0, 255)
        composite[i] = np.round(a[..., None] * subject + (1 - a[..., None]) * rendered[i])
    ffmpeg.write_frames(composite, out, fps)
    preview = cache.with_name(f"{cache.name}_{masker.name}_preview.mp4")
    if not preview.is_file():
        ffmpeg.write_frames(mask_frames, preview, fps)
    return {**result, "kept": True, "colour_shift": [round(float(v), 1) for v in shift],
            "mask_video": str(preview), "seconds": round(time.monotonic() - started, 1)}
