"""The quality judge: examples, training, RichHF fetching and scoring (no real CLIP model)."""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from typing import Any

import cv2
import numpy as np
import pytest

from rokkur_studio.pipeline import judge
from rokkur_studio.services import practice
from rokkur_studio.services.projects import latest_document
from tests.test_dashboard import client_for
from tests.test_pipeline import create, run, status


class FakeEncoder:
    """Stands in for CLIP: brightness and noisiness, padded to 512 numbers."""

    def __init__(self) -> None:
        self.seen = 0

    def embed(self, images: Any, batch: int = 32) -> np.ndarray:
        out = []
        for img in images:
            self.seen += 1
            g = np.asarray(img, np.float32).mean(axis=2) / 255
            noise = float(np.abs(np.diff(g, axis=1)).mean())
            f = np.zeros(512, np.float32)
            f[0], f[1], f[2] = g.mean(), noise, 0.5
            out.append(f / np.linalg.norm(f))
        return np.array(out, np.float32).reshape(-1, 512)


def _clean(seed: int) -> np.ndarray:
    return np.full((64, 64, 3), 150 + seed % 40, np.uint8)


def _botched(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)


def _write_examples(settings, n: int = 8) -> None:
    root = judge.references_dir(settings)
    for label, make in (("good", _clean), ("bad", _botched)):
        (root / label).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(root / label / f"{i}.png"), make(i))


def test_judge_learns_your_examples_and_scores_frames(settings):
    _write_examples(settings)
    enc = FakeEncoder()
    with pytest.raises(judge.JudgeUnavailable):
        judge.train(settings, enc)  # 16 images: too few without public data
    _write_examples(settings, n=12)
    report = judge.train(settings, enc)
    assert report["your_good"] == 12 and report["your_bad"] == 12
    assert report["your_examples_right"] == 1.0
    seen = enc.seen
    judge.train(settings, enc)
    assert enc.seen == seen  # features are cached by file content
    loaded = judge.QualityJudge.load(settings)
    assert loaded is not None
    loaded.encoder = enc  # type: ignore[assignment]
    good = loaded.score_frames(np.stack([_clean(i) for i in range(4)]))
    bad = loaded.score_frames(np.stack([_botched(i) for i in range(4)]))
    assert good["judge"] > 7 > 3 > bad["judge"]
    settings.judge.enabled = False
    assert judge.QualityJudge.load(settings) is None


def test_fetch_keeps_features_and_scores_and_deletes_the_images(settings, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [(_clean(i), 0.9, 0.8) for i in range(6)] + [(_botched(i), 0.1, 0.2)
                                                         for i in range(6)]
    table = pa.table({
        "image": [{"bytes": cv2.imencode(".png", img)[1].tobytes(), "path": None}
                  for img, _, _ in rows],
        "artifact_score": [a for _, a, _ in rows], "overall_score": [o for _, _, o in rows]})
    buf = io.BytesIO()
    pq.write_table(table, buf, row_group_size=5)
    urls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self, size: int):  # noqa: ANN202
            yield buf.getvalue()

    @contextmanager
    def stream(method: str, url: str, **kw: Any):  # noqa: ANN202
        urls.append(url)
        yield Response()

    monkeypatch.setattr(judge.httpx, "stream", stream)
    enc = FakeEncoder()
    counts = judge.fetch_richhf(settings, enc, splits=["test", "validation"], progress=print)
    assert counts == {"test-00000-of-00001.parquet": 12,
                      "validation-00000-of-00001.parquet": 12}
    folder = judge.references_dir(settings) / "richhf"
    assert not list(folder.glob("*.part"))  # images are not kept
    judge.fetch_richhf(settings, enc, splits=["test"], progress=print)
    assert len(urls) == 2  # a finished file is not fetched again
    judge.fetch_richhf(settings, enc, splits=["train"], limit=3, progress=print)
    assert int(np.load(folder / "train-00000-of-00006.npz")["y"].shape[0]) == 3
    report = judge.train(settings, enc)
    assert report["test_images"] == 12 and report["plausibility_pairwise"] > 0.9


class FixedJudge:
    report = {"trained_at": "now", "public_images": 100, "your_good": 1, "your_bad": 1}

    def score_frames(self, frames: np.ndarray) -> dict[str, Any]:
        assert frames.shape[1:] == (224, 224, 3) and len(frames) <= 8
        return {"judge": 4.0, "judge_worst": 2.0, "judge_worst_frame": 1,
                "judge_plausibility": 4.0, "judge_overall": 4.0}


def test_judge_scores_count_in_practice_and_advise_qc(ctx, sample_video):
    ctx.extras["quality_judge"] = FixedJudge()
    ctx.settings.practice.variants = 1
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"  # advisory: a low judge score fails nothing
    with ctx.db.session() as s:
        qc = latest_document(s, pid, "qc_report").data  # type: ignore[union-attr]
    assert all(shot["judge"] == 4.0 and shot["judge_worst"] == 2.0 for shot in qc["shots"])
    from tests.test_practice import _queue

    _queue(ctx, pid)
    run(ctx)
    with ctx.db.session() as s:
        report = latest_document(s, pid, practice.REPORT).data  # type: ignore[union-attr]
    v = report["variants"][0]
    assert v["judge"] == 4.0 and v["reviewed"]  # the judge alone makes it count
    assert v["score"] == round((v["qc"] + 4.0) / 2, 2)
    page = client_for(ctx).get("/ui/practice").text
    assert "Quality judge" in page and "Botched" in page


def test_marking_a_render_saves_frames_as_examples(ctx, sample_video):
    from tests.test_practice import _queue

    ctx.settings.practice.variants = 1
    pid = create(ctx, sample_video)
    run(ctx)
    _queue(ctx, pid)
    run(ctx)
    with ctx.db.session() as s:
        asset = latest_document(s, pid, practice.REPORT).data["variants"][0]["asset_id"]  # type: ignore[union-attr]
    r = client_for(ctx).post("/ui/references/mark", data={
        "project_id": pid, "asset_id": asset, "label": "bad"}, follow_redirects=False)
    assert r.status_code == 303 and "botched" in r.headers["location"]
    saved = list((judge.references_dir(ctx.settings) / "bad").glob("*.png"))
    assert 1 <= len(saved) <= 4
    assert cv2.imread(str(saved[0])).shape == (448, 448, 3)


def test_quality_score_with_and_without_a_judge():
    assert practice.quality_score({"overall": 6.0})["reviewed"] is False
    judged = practice.quality_score({"overall": 6.0, "judge": 8.0})
    assert judged["score"] == 7.0 and judged["reviewed"] and judged["ai"] is None
    assert json.dumps(judged)
