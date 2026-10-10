"""Motion-compensated stability, brightness flicker and the new QC recommendations."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from rokkur_studio.pipeline import qc

pytest.importorskip("cv2")


def moving_scene(n: int = 24, size: int = 128, speed: int = 2, seed: int = 0) -> np.ndarray:
    """A smooth texture panning sideways with a bright disc crossing it: real motion that a
    steady render must not be punished for."""
    rng = np.random.default_rng(seed)
    cells = rng.normal(0, 1, (size // 4 + 2, (size + n * speed) // 4 + 2))
    big = np.kron(cells, np.ones((4, 4)))
    pad = np.pad(big, 2, mode="edge")
    smooth = sum(pad[i:i + big.shape[0], j:j + big.shape[1]] for i in range(5) for j in range(5)) / 25
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min()) * 160 + 40
    yy, xx = np.mgrid[0:size, 0:size]
    frames = []
    for t in range(n):
        f = smooth[:size, t * speed:t * speed + size].copy()
        f[(yy - size // 2) ** 2 + (xx - 20 - 4 * t) ** 2 < 14 ** 2] = 235
        frames.append(f)
    return np.clip(np.array(frames), 0, 255).astype(np.uint8)


def small(frames: np.ndarray) -> np.ndarray:
    """2x2 block mean: the 64x64 frames QC always gets (ffmpeg's area scaler does the same)."""
    n, h, w = frames.shape
    return frames.reshape(n, h // 2, 2, w // 2, 2).mean(axis=(2, 4)).round().astype(np.uint8)


def boil(frames: np.ndarray, sigma: float, seed: int = 5) -> np.ndarray:
    """Texture that re-draws itself every frame in 3-pixel patches, as a diffusion render's
    surfaces do when they shimmer."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, frames.shape)[:, ::3, ::3]
    noise = np.kron(noise, np.ones((1, 3, 3)))[:, :frames.shape[1], :frames.shape[2]]
    return np.clip(frames + noise * sigma, 0, 255).astype(np.uint8)


def score(src: np.ndarray, out: np.ndarray, **kw: Any) -> dict[str, Any]:
    return qc.score_shot(small(src), small(out), threshold=6.5, shot_id="s",
                         source_detail=src, render_detail=out, **kw)


def test_a_steady_render_of_moving_footage_scores_high_stability():
    src = moving_scene()
    for out in (src, (255 - src).astype(np.uint8)):  # identical, and relit with the same motion
        r = score(src, out)
        assert r["stability"] >= 9.5 and r["flicker"] >= 9.5
        assert r["decision"] == "PASS" and not any("steady" in i for i in r["issues"])
        assert r["not_measured"] == {} and "128x128" in r["stability_method"]


def test_boiling_textures_score_clearly_lower_even_where_the_old_metrics_pass():
    src = moving_scene()
    steady, boiling = score(src, src), score(src, boil(src, 6))
    assert boiling["stability"] < 6 and steady["stability"] - boiling["stability"] > 3
    # The calibrated overall score and decision do not change: this is what "good but
    # stability is needed" looks like.
    assert boiling["decision"] == "PASS"
    assert any("textures not steady" in i for i in boiling["issues"])
    # The 64x64 fallback sees it too, only less sharply.
    coarse = qc.score_shot(small(src), small(boil(src, 6)), threshold=6.5, shot_id="s")
    assert coarse["stability"] < steady["stability"] - 2
    assert "64x64" in coarse["stability_method"]


def test_min_stability_fails_an_unsteady_shot_and_recommends_stabilize():
    src = moving_scene()
    r = score(src, boil(src, 6), min_stability=6.0)
    assert r["decision"] == "FAIL"
    assert any(i.startswith("picture not steady (stability ") and i.endswith(", floor 6.0)")
               for i in r["issues"])
    assert "STABILIZE" in r["recommendations"]
    assert "CALM_EDGES" in r["recommendations"]  # layout holds while the surfaces boil
    # New hints are appended after the existing ones.
    assert r["recommendations"][-2:] == ["STABILIZE", "CALM_EDGES"]
    assert r["recommendations"][0] not in {"STABILIZE", "CALM_EDGES"}
    assert score(src, src, min_stability=6.0)["decision"] == "PASS"


def test_new_metrics_do_not_change_the_calibrated_score():
    src = moving_scene()
    for out in (src, boil(src, 6), boil(src, 12)):
        with_detail = score(src, out)
        without = qc.score_shot(small(src), small(out), threshold=6.5, shot_id="s")
        keys = ("overall", "temporal_consistency", "motion", "structure", "detail", "decision",
                "failed_frames")
        assert {k: with_detail[k] for k in keys} == {k: without[k] for k in keys}


def test_heavy_boiling_keeps_the_existing_recommendations_and_appends_new_ones():
    src = moving_scene()
    r = score(src, boil(src, 12))
    assert r["decision"] == "FAIL"
    recs = r["recommendations"]
    assert recs[:2] == ["CHANGE_SEED", "REDUCE_STYLE_STRENGTH"]
    assert recs.index("STABILIZE") > recs.index("REDUCE_STYLE_STRENGTH")


def test_brightness_flicker_is_flicker_not_texture_instability():
    src = moving_scene()
    pulse = (np.arange(len(src)) % 2 * 30)[:, None, None]
    flickering = np.clip(src.astype(int) + pulse, 0, 255).astype(np.uint8)
    r = score(src, flickering)
    assert r["flicker"] < 5 and r["stability"] >= 9  # whole-frame brightness is not boiling
    assert r["decision"] == "FAIL" and "DEFLICKER" in r["recommendations"]
    assert "CHANGE_SEED" in r["recommendations"]  # the temporal metric still speaks too
    assert any("brightness flicker beyond the source" in i for i in r["issues"])
    # The same flashes in the source (real lighting) are the baseline, not a defect.
    same = score(flickering, flickering)
    assert same["flicker"] >= 9.5 and "DEFLICKER" not in same["recommendations"]


def test_without_opencv_stability_is_null_with_the_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # import cv2 now raises ImportError
    src = moving_scene()
    r = score(src, boil(src, 6), min_stability=6.0)
    assert r["stability"] is None and r["stability_method"] is None
    assert "OpenCV" in r["not_measured"]["stability"]
    assert r["decision"] == "PASS"  # an unmeasured metric cannot fail a shot
    assert r["flicker"] is not None  # needs no OpenCV
    s = qc.summarize([r], 6.5)
    assert s["stability"] is None and "OpenCV" in s["not_measured"]["stability"]
    # The heatmap helper falls back to plain frame differences.
    a, b = small(src)[0], small(src)[1]
    m = qc.excess_change(a, b, a, b)
    assert m.shape == a.shape and float(m.max()) == 0.0


def test_new_recommendations_for_blur_and_edge_output():
    src = moving_scene()
    s64 = small(src).astype(np.float32)
    pad = np.pad(s64, ((0, 0), (3, 3), (3, 3)), mode="edge")
    blurred = sum(pad[:, i:i + 64, j:j + 64] for i in range(7) for j in range(7)) / 49
    # A truncated render fails on frame count; a soft one also gets MORE_DETAIL.
    r = qc.score_shot(small(src), blurred.astype(np.uint8)[:10], threshold=6.5, shot_id="s")
    assert r["detail"] < 5 and r["decision"] == "FAIL"
    assert r["recommendations"][0] == "RERENDER_SHOT" and "MORE_DETAIL" in r["recommendations"]

    a = np.full((10, 64, 64), 160, dtype=np.uint8)
    a[:, 16:48, 16:48] = 110
    b = np.zeros_like(a)
    b[:, 16:48, 16] = b[:, 16:48, 47] = 255
    b[:, 16, 16:48] = b[:, 47, 16:48] = 255
    edges = qc.score_shot(a, b, threshold=6.5, shot_id="edges")
    assert edges["visual_review_required"] and "CALM_EDGES" in edges["recommendations"]


def test_detail_frames_must_be_given_together_and_may_differ_in_size():
    src = moving_scene()
    with pytest.raises(ValueError):
        qc.score_shot(small(src), small(src), threshold=6.5, shot_id="s", source_detail=src)
    # A render decoded at another size is resized to the source's before comparing.
    wide = np.repeat(src, 2, axis=2)
    r = qc.score_shot(small(src), small(src), threshold=6.5, shot_id="s",
                      source_detail=src, render_detail=wide)
    assert r["stability"] >= 9


def test_too_few_frames_are_reported_not_scored():
    one = moving_scene(n=1)
    r = score(one, one)
    assert r["stability"] is None and r["flicker"] is None
    assert set(r["not_measured"]) == {"stability", "flicker"}
    empty = qc.score_shot(small(one), small(one)[:0], threshold=6.5, shot_id="s")
    assert empty["decision"] == "FAIL" and empty["stability"] is None


def test_through_h264_the_codec_is_not_instability_but_boiling_is(ffmpeg, tmp_path):
    """Encoded and read back the way the QC stage does: 64x64 frames plus detail frames."""
    src = moving_scene(n=33, size=256)
    ffmpeg.write_frames(src, tmp_path / "source.mp4", 16)
    results = {}
    for name, frames in (("steady", (255 - src).astype(np.uint8)), ("boiling", boil(src, 8))):
        ffmpeg.write_frames(frames, tmp_path / f"{name}.mp4", 16)
        dw, dh = qc.detail_size(256, 256)
        read = [ffmpeg.read_gray_frames(tmp_path / f"{clip}.mp4", w, h, fps=16)
                for clip in ("source", name) for w, h in ((64, 64), (dw, dh))]
        results[name] = qc.score_shot(read[0], read[2], threshold=6.5, shot_id=name,
                                      source_detail=read[1], render_detail=read[3])
    assert results["steady"]["stability"] >= 9.5 and results["steady"]["flicker"] >= 9.5
    assert results["boiling"]["stability"] < 6 and results["boiling"]["decision"] == "PASS"


def test_detail_size_keeps_aspect_and_even_sides():
    assert qc.detail_size(1920, 1080) == (192, 108)
    assert qc.detail_size(1080, 1920) == (192, 342)
    assert qc.detail_size(100, 50) == (100, 50)  # never upscaled


def test_summary_averages_measured_and_reviewed_scores_and_lists_only_the_unmeasured():
    base = {"decision": "PASS", "overall": 9, "failed_frames": [], "motion": 9,
            "temporal_consistency": 9, "structure": 9, "detail": 9, "artifact_score": 0}
    shots = [
        {**base, "shot_id": "a", "stability": 8.0, "flicker": 10.0, "not_measured": {},
         "identity": 6, "prompt_adherence": 4, "style_consistency": 8,
         "hand_body_deformation": 7, "picture_review": {"description": "a fox"}},
        {**base, "shot_id": "b", "stability": 6.0, "flicker": 9.0, "not_measured": {}},
    ]
    s = qc.summarize(shots, 6.5)
    assert s["stability"] == 7.0 and s["flicker"] == 9.5
    assert s["identity"] == 6 and s["prompt_adherence"] == 4 and s["hand_body_deformation"] == 7
    assert s["not_measured"] == {} and s["picture_reviewed"] == ["a"]

    unreviewed = qc.summarize([{**base, "shot_id": "c", "stability": None, "flicker": 9.0,
                                "not_measured": {"stability": "needs OpenCV"}}], 6.5)
    assert unreviewed["identity"] is None
    assert set(unreviewed["not_measured"]) == {*qc.NOT_MEASURED, "stability"}
    assert unreviewed["not_measured"]["stability"] == "needs OpenCV"
