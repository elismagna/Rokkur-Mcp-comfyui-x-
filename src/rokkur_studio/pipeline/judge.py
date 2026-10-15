"""The quality judge: learns what botched and well-made AI images look like (docs/judge.md).

CLIP ViT-B/16 (an ONNX export, on the CPU) turns each frame into 512 numbers that describe what
it looks like; a small linear head on top, trained here, turns those into scores. Training data:

- RichHF-18K (Google Research): about 18,000 AI-generated images (Pick-a-Pic), each rated by
  people for artifacts (plausibility: melted hands, extra limbs, distorted faces) and overall
  quality, 0 to 1. ``references fetch`` downloads them, keeps only the CLIP features, and
  deletes the images.
- Your own examples: images in ``<data_dir>/references/good`` and ``.../bad``, or frames you mark
  on the Night School page. They count ``judge.user_weight`` times as much as a RichHF image,
  because they look like this studio's renders and RichHF's stills do not.

The judge scores a render's frames 0 to 10 (higher: fewer artifacts, better made) and reports its
worst frame. It is advisory in QC; practice counts it as a third of the quality score.
Accuracy is measured on RichHF's own test split, which training never sees.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from rokkur_studio.config import Settings
from rokkur_studio.pipeline.subject import MaskModel, fetch_model

log = logging.getLogger(__name__)

# Xenova's ONNX export of openai/clip-vit-base-patch16 (MIT): input pixel_values, output the
# projected 512-d image_embeds.
CLIP = MaskModel(
    "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main/onnx/vision_model.onnx",
    "b5170b47c0ecc667c9af5b42ed460c341377706d1466d177798c65fe49ea1e6b", 224,
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

RICHHF = "https://huggingface.co/datasets/Exploration/richhf_18k_with_images/resolve/main/data"
RICHHF_FILES = {"test": ["test-00000-of-00001.parquet"],
                "validation": ["validation-00000-of-00001.parquet"],
                "train": [f"train-0000{i}-of-00006.parquet" for i in range(6)]}
TARGETS = ("plausibility", "overall")  # what the head predicts, 0-1, higher is better
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class JudgeUnavailable(RuntimeError):
    """The judge cannot run: no CLIP model, onnxruntime or pyarrow missing, nothing trained."""


def references_dir(settings: Settings) -> Path:
    return Path(settings.studio.data_dir) / "references"


def head_path(settings: Settings) -> Path:
    return Path(settings.studio.data_dir) / "models" / "quality_judge.npz"


# -- CLIP features --------------------------------------------------------------------------
class ClipEncoder:
    """CLIP image features on the CPU; the session opens on first use."""

    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.studio.data_dir) / "models" / "clip-vit-b16-vision.onnx"
        self.download = settings.judge.download
        self._session: Any = None

    def _open(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise JudgeUnavailable("onnxruntime is not installed") from exc
            if not self.path.is_file():
                if not self.download:
                    raise JudgeUnavailable(f"{self.path} is missing and judge.download is off")
                fetch_model(CLIP, self.path, what="CLIP model for the quality judge",
                            error=JudgeUnavailable)
            self._session = ort.InferenceSession(str(self.path),
                                                 providers=["CPUExecutionProvider"])
        return self._session

    @staticmethod
    def prepare(rgb: np.ndarray) -> np.ndarray:
        """One RGB uint8 image to CLIP's input: shortest side 224, centre crop, normalised."""
        import cv2

        h, w = rgb.shape[:2]
        k = CLIP.size / min(h, w)
        if (h, w) != (CLIP.size, CLIP.size):
            rgb = cv2.resize(rgb, (max(CLIP.size, round(w * k)), max(CLIP.size, round(h * k))),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        y, x = (rgb.shape[0] - CLIP.size) // 2, (rgb.shape[1] - CLIP.size) // 2
        crop = rgb[y:y + CLIP.size, x:x + CLIP.size].astype(np.float32) / 255.0
        mean, std = np.array(CLIP.mean, np.float32), np.array(CLIP.std, np.float32)
        return ((crop - mean) / std).transpose(2, 0, 1)

    def embed(self, images: Iterable[np.ndarray], batch: int = 32) -> np.ndarray:
        """``(n, 512)`` unit-length features for RGB uint8 images of any size."""
        session = self._open()
        out: list[np.ndarray] = []
        chunk: list[np.ndarray] = []
        for img in images:
            chunk.append(self.prepare(img))
            if len(chunk) == batch:
                out.append(session.run(None, {"pixel_values": np.stack(chunk)})[0])
                chunk = []
        if chunk:
            out.append(session.run(None, {"pixel_values": np.stack(chunk)})[0])
        if not out:
            return np.zeros((0, 512), np.float32)
        e = np.concatenate(out).astype(np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)


def decode(data: bytes) -> np.ndarray | None:
    """Encoded image bytes to RGB uint8, or None when they are not an image."""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -- RichHF-18K -----------------------------------------------------------------------------
def fetch_richhf(settings: Settings, encoder: ClipEncoder, *, splits: Iterable[str],
                 limit: int | None = None, progress: Callable[[str], None] = log.info
                 ) -> dict[str, int]:
    """Download RichHF-18K parquet files one at a time, keep their CLIP features and scores in
    ``references/richhf/<file>.npz`` and delete the images. A finished file is skipped, so an
    interrupted fetch picks up where it stopped. ``limit``: at most this many images per file."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise JudgeUnavailable("pyarrow is not installed (pip install pyarrow)") from exc
    folder = references_dir(settings) / "richhf"
    folder.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in splits:
        for name in RICHHF_FILES[split]:
            out = folder / f"{Path(name).stem}.npz"
            if out.is_file():
                counts[name] = int(np.load(out)["y"].shape[0])
                progress(f"{name}: already fetched ({counts[name]} images)")
                continue
            part = folder / f"{name}.part"
            progress(f"{name}: downloading")
            try:
                with httpx.stream("GET", f"{RICHHF}/{name}", follow_redirects=True,
                                  timeout=httpx.Timeout(30, read=300)) as response:
                    response.raise_for_status()
                    with part.open("wb") as fh:
                        for block in response.iter_bytes(1 << 20):
                            fh.write(block)
                started = time.monotonic()
                feats: list[np.ndarray] = []
                ys: list[np.ndarray] = []
                pf = pq.ParquetFile(part)
                for group in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(group, columns=[
                        "image", "artifact_score", "overall_score"])
                    rows = zip(table.column("image").to_pylist(),
                               table.column("artifact_score").to_pylist(),
                               table.column("overall_score").to_pylist(), strict=True)
                    images, scores = [], []
                    for image, artifact, overall in rows:
                        rgb = decode(image["bytes"]) if image and image.get("bytes") else None
                        if rgb is not None and artifact is not None and overall is not None:
                            images.append(rgb)
                            scores.append((artifact, overall))
                    if limit is not None:
                        room = limit - int(sum(len(y) for y in ys))
                        images, scores = images[:room], scores[:room]
                    if images:
                        feats.append(encoder.embed(images))
                        ys.append(np.array(scores, np.float32))
                    done = int(sum(len(y) for y in ys))
                    progress(f"{name}: {done} images, {time.monotonic() - started:.0f} s")
                    if limit is not None and done >= limit:
                        break
            except (httpx.HTTPError, OSError) as exc:
                raise JudgeUnavailable(f"could not fetch {name}: {exc}") from exc
            finally:
                part.unlink(missing_ok=True)
            x = np.concatenate(feats).astype(np.float16) if feats else np.zeros((0, 512))
            y = np.concatenate(ys) if ys else np.zeros((0, 2), np.float32)
            np.savez(out, x=x, y=y)
            counts[name] = len(y)
    return counts


# -- your own examples ----------------------------------------------------------------------
def user_examples(settings: Settings, encoder: ClipEncoder | None
                  ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Features and targets for the images in ``references/good`` (1) and ``bad`` (0). Features
    are cached by file content, so only new images are run through CLIP. Without an encoder
    only cached images count."""
    root = references_dir(settings)
    cache_path = root / "user_cache.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.is_file():
        data = np.load(cache_path)
        cache = dict(zip(data["keys"].tolist(), data["x"], strict=True))
    xs, ys, counts = [], [], {"good": 0, "bad": 0}
    fresh: dict[str, np.ndarray] = {}
    for label, value in (("good", 1.0), ("bad", 0.0)):
        folder = root / label
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_TYPES:
                continue
            raw = path.read_bytes()
            key = hashlib.sha256(raw).hexdigest()
            feat = cache.get(key)
            if feat is None and encoder is not None:
                rgb = decode(raw)
                if rgb is None:
                    continue
                feat = encoder.embed([rgb])[0]
                fresh[key] = feat
            if feat is None:
                continue
            xs.append(feat.astype(np.float32))
            ys.append((value, value))
            counts[label] += 1
    if fresh:
        cache.update(fresh)
        root.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(list(cache)), x=np.stack(list(cache.values())))
    if not xs:
        return np.zeros((0, 512), np.float32), np.zeros((0, 2), np.float32), counts
    return np.stack(xs), np.array(ys, np.float32), counts


def mark_frames(settings: Settings, frames: np.ndarray, *, label: str, name: str,
                count: int = 4) -> list[Path]:
    """Save ``count`` evenly spaced frames of a render as good or bad examples."""
    import cv2

    if label not in ("good", "bad"):
        raise ValueError("label must be good or bad")
    folder = references_dir(settings) / label
    folder.mkdir(parents=True, exist_ok=True)
    picks = np.linspace(0, len(frames) - 1, num=min(count, len(frames))).round().astype(int)
    paths = []
    for i in dict.fromkeys(picks.tolist()):
        path = folder / f"{name}_f{i:03d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


# -- training -------------------------------------------------------------------------------
def _ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    xb = np.c_[x, np.ones(len(x), np.float32)]
    xw = xb * w[:, None]
    reg = lam * np.eye(xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0  # never shrink the intercept
    return np.linalg.solve(xb.T.astype(np.float64) @ xw + reg, xw.T.astype(np.float64) @ y)


def _pairwise(pred: np.ndarray, truth: np.ndarray, max_pairs: int = 200_000) -> float:
    """How often the judge orders two images the way people did (0.5 is chance)."""
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(truth), max_pairs)
    j = rng.integers(0, len(truth), max_pairs)
    keep = truth[i] != truth[j]
    return float(np.mean((pred[i] > pred[j])[keep] == (truth[i] > truth[j])[keep]))


def _richhf(settings: Settings) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    folder = references_dir(settings) / "richhf"
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, names in RICHHF_FILES.items():
        parts = [np.load(folder / f"{Path(n).stem}.npz") for n in names
                 if (folder / f"{Path(n).stem}.npz").is_file()]
        if parts:
            out[split] = (np.concatenate([p["x"] for p in parts]).astype(np.float32),
                          np.concatenate([p["y"] for p in parts]).astype(np.float32))
    return out


def train(settings: Settings, encoder: ClipEncoder | None) -> dict[str, Any]:
    """Fit the judge on RichHF (train + validation) and your examples; measure it on RichHF's
    test split. Saves ``models/quality_judge.npz`` and returns what it measured."""
    cfg = settings.judge
    data = _richhf(settings)
    ux, uy, counts = user_examples(settings, encoder)
    xs = [d[0] for k, d in data.items() if k != "test"] + [ux]
    ys = [d[1] for k, d in data.items() if k != "test"] + [uy]
    public = sum(len(d[1]) for k, d in data.items() if k != "test")
    x, y = np.concatenate(xs), np.concatenate(ys)
    if len(y) < 20 or (public == 0 and min(counts.values()) < 5):
        raise JudgeUnavailable("Too few examples to train on: run `references fetch` or add at "
                               "least 5 good and 5 bad images")
    w = np.ones(len(y), np.float32)
    w[public:] = cfg.user_weight
    head = _ridge(x, y, w, cfg.ridge)
    report: dict[str, Any] = {"trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                              "public_images": public, "your_good": counts["good"],
                              "your_bad": counts["bad"]}
    if "test" in data:
        tx, ty = data["test"]
        pred = np.c_[tx, np.ones(len(tx))] @ head
        report["test_images"] = len(ty)
        for k, name in enumerate(TARGETS):
            report[f"{name}_pearson"] = round(float(np.corrcoef(pred[:, k], ty[:, k])[0, 1]), 3)
            report[f"{name}_pairwise"] = round(_pairwise(pred[:, k], ty[:, k]), 3)
    if len(uy):
        pred = np.c_[ux, np.ones(len(ux))] @ head
        report["your_examples_right"] = round(float(np.mean(
            (pred.mean(axis=1) >= 0.5) == (uy[:, 0] >= 0.5))), 3)
    path = head_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, head=head.astype(np.float32), report=json.dumps(report))
    return report


# -- scoring --------------------------------------------------------------------------------
@dataclass
class QualityJudge:
    encoder: ClipEncoder
    head: np.ndarray            # (513, 2)
    report: dict[str, Any]

    @classmethod
    def load(cls, settings: Settings) -> QualityJudge | None:
        """The trained judge, or None when none is trained or it is switched off."""
        path = head_path(settings)
        if not settings.judge.enabled or not path.is_file():
            return None
        data = np.load(path)
        return cls(ClipEncoder(settings), data["head"], json.loads(str(data["report"])))

    def score_frames(self, frames: np.ndarray) -> dict[str, Any]:
        """0-10 scores for a render's frames: the mean, the worst frame and which one."""
        feats = self.encoder.embed(list(frames))
        pred = np.clip(np.c_[feats, np.ones(len(feats))] @ self.head, 0.0, 1.0)
        per_frame = pred.mean(axis=1) * 10
        worst = int(np.argmin(per_frame))
        return {"judge": round(float(per_frame.mean()), 2),
                "judge_worst": round(float(per_frame[worst]), 2), "judge_worst_frame": worst,
                "judge_plausibility": round(float(pred[:, 0].mean() * 10), 2),
                "judge_overall": round(float(pred[:, 1].mean() * 10), 2)}


def judge_for(ctx_extras: dict[str, Any], settings: Settings) -> QualityJudge | None:
    """The judge for this process, reloaded when it was retrained (``extras["quality_judge"]``
    replaces it in tests)."""
    if "quality_judge" in ctx_extras:
        return ctx_extras["quality_judge"]  # type: ignore[no-any-return]
    path = head_path(settings)
    stamp = path.stat().st_mtime if path.is_file() else None
    cached = ctx_extras.get("_quality_judge_loaded")
    if cached is None or cached[0] != stamp:
        cached = (stamp, QualityJudge.load(settings))
        ctx_extras["_quality_judge_loaded"] = cached
    return cached[1]  # type: ignore[no-any-return]


def score_render(judge: QualityJudge | None, read_frames: Callable[[int], np.ndarray],
                 frames: int = 8) -> dict[str, Any]:
    """The judge's scores for one render, or ``{}`` when there is no judge or it failed.
    ``read_frames(n)`` returns n evenly spaced square RGB frames of the render."""
    if judge is None:
        return {}
    try:
        return judge.score_frames(read_frames(frames))
    except JudgeUnavailable as exc:
        log.warning("quality judge unavailable: %s", exc)
        return {"judge_skipped": str(exc)}
