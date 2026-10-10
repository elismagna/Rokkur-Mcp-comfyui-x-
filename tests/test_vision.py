"""The picture checker: contact sheet, PNG encoder, AI picture review (fake providers only)."""

from __future__ import annotations

import base64
import json
import struct
import sys
import zlib
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from rokkur_studio.agents.providers import (
    AgentOutputError,
    AgentUnavailable,
    OllamaProvider,
    RuleBasedProvider,
)
from rokkur_studio.pipeline import qc, vision
from rokkur_studio.pipeline.vision import PictureChecker, PictureReview

REVIEW = {"description": "A red fox in a snowy pine forest, painted in watercolour.",
          "prompt_adherence": 8, "style_consistency": 7, "subject_identity": 9, "anatomy": 6,
          "steadiness": 5, "issues": ["texture_boiling"], "notes": "Calm the snow texture."}


def clip(n: int = 9, h: int = 32, w: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (h, w + n, 3)).astype(np.uint8)
    return np.stack([base[:, t:t + w] for t in range(n)])  # panning RGB frames


def decode_png(data: bytes) -> np.ndarray:
    """Minimal decoder for what png_bytes writes (8-bit, filters None/Up), with CRC checks."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(tag + body)
        chunks.append((tag, body))
        pos += 12 + length
    assert [t for t, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    w, h, depth, colour, *_ = struct.unpack(">IIBBBBB", chunks[0][1])
    assert depth == 8
    ch = {0: 1, 2: 3, 6: 4}[colour]
    raw = np.frombuffer(zlib.decompress(chunks[1][1]), dtype=np.uint8).reshape(h, 1 + w * ch)
    rows = np.zeros((h, w * ch), dtype=np.uint8)
    for y in range(h):
        assert raw[y, 0] in (0, 2)
        rows[y] = raw[y, 1:] + (rows[y - 1] if raw[y, 0] == 2 and y else 0)
    return rows.reshape(h, w, ch).squeeze()


# -- contact sheet ------------------------------------------------------------------------
def test_contact_sheet_layout_source_render_and_heat_rows():
    src, out = clip(), clip(seed=1)
    sheet = vision.contact_sheet(src, out, columns=4)
    assert sheet.dtype == np.uint8 and sheet.shape == (3 * 32 + 2 * 2, 4 * 48 + 3 * 2, 3)
    idx = vision.sample_indices(9, 4)
    assert idx == [0, 3, 5, 8]
    assert np.array_equal(sheet[:32, :48], src[0])                # row 1: source, first frame
    assert np.array_equal(sheet[34:66, 150:198], out[8])          # row 2: render, last frame
    assert (sheet[32:34] == vision.SEPARATOR).all()               # thin separator
    assert vision.contact_sheet(src, out, heat=False).shape == (66, 198, 3)
    # Grey frames, fewer frames than columns, and a render decoded at another size all work.
    grey = src[..., 0]
    assert vision.contact_sheet(grey[:2], np.repeat(grey[:2], 2, axis=2)).shape == (100, 98, 3)
    assert vision.contact_sheet(grey[:1], grey[:1]).shape == (66, 48, 3)  # no pair, no heat
    with pytest.raises(ValueError):
        vision.contact_sheet(src[:0], out)


def test_heat_row_lights_up_where_the_render_boils():
    pytest.importorskip("cv2")
    src = clip(n=12)
    rng = np.random.default_rng(3)
    boiling = np.clip(src + rng.normal(0, 14, src.shape), 0, 255).astype(np.uint8)
    steady = vision.contact_sheet(src, src)[68:].astype(float)
    hot = vision.contact_sheet(src, boiling)[68:].astype(float)
    assert hot[..., 0].mean() > 2 * steady[..., 0].mean() + 20


def test_heat_row_without_opencv_uses_plain_differences(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert vision.contact_sheet(clip(), clip(seed=2)).shape == (100, 198, 3)


def test_frame_size_keeps_the_longest_side_and_even_dimensions():
    assert vision.frame_size(1920, 1080) == (256, 144)
    assert vision.frame_size(1080, 1920) == (144, 256)
    assert vision.frame_size(200, 100) == (200, 100)


# -- PNG ----------------------------------------------------------------------------------
def test_png_bytes_round_trip_for_rgb_grey_and_rgba():
    rgb = clip()[0]
    data = vision.png_bytes(rgb)
    w, h, depth, colour = struct.unpack(">IIBB", data[16:26])
    assert data[12:16] == b"IHDR" and (w, h, depth, colour) == (48, 32, 8, 2)
    assert np.array_equal(decode_png(data), rgb)
    grey = rgb[..., 1]
    assert np.array_equal(decode_png(vision.png_bytes(grey)), grey)
    rgba = np.concatenate([rgb, np.full((32, 48, 1), 128, np.uint8)], axis=-1)
    assert np.array_equal(decode_png(vision.png_bytes(rgba)), rgba)
    with pytest.raises(ValueError):
        vision.png_bytes(rgb.astype(np.float32))
    with pytest.raises(ValueError):
        vision.png_bytes(np.zeros((0, 4, 3), np.uint8))


def test_png_bytes_decode_with_opencv():
    cv2 = pytest.importorskip("cv2")
    sheet = vision.contact_sheet(clip(), clip(seed=1))
    decoded = cv2.imdecode(np.frombuffer(vision.png_bytes(sheet), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded[..., ::-1], sheet)  # OpenCV decodes to BGR


# -- picture review -----------------------------------------------------------------------
class SeeingProvider:
    name = "fake_vision"

    def __init__(self, reply: dict[str, Any] | None = None,
                 error: Exception | None = None) -> None:
        self.reply, self.error = reply or REVIEW, error
        self.calls: list[dict[str, Any]] = []

    def supports_images(self) -> bool:
        return True

    def generate(self, role, instructions, payload, schema, images=None):
        self.calls.append({"role": role, "instructions": instructions, "payload": payload,
                           "schema": schema, "images": images})
        if self.error is not None:
            raise self.error
        return schema.model_validate(self.reply)


class BlindProvider(SeeingProvider):
    name = "fake_text"

    def supports_images(self) -> bool:
        return False


def test_picture_checker_sends_the_sheet_and_returns_the_review():
    provider = SeeingProvider()
    png = vision.png_bytes(vision.contact_sheet(clip(), clip(seed=1)))
    review = PictureChecker(provider).review(shot_id="shot_001", theme="watercolour",
                                             prompt="a red fox in snow", sheet_png=png)
    assert isinstance(review, PictureReview) and review.steadiness == 5
    call = provider.calls[0]
    assert call["images"] == [png] and call["schema"] is PictureReview
    assert call["role"] == "picture_checker"
    assert call["payload"]["sheet"]["rows"] == ["source", "render", "change heatmap"]
    assert call["payload"]["prompt"] == "a red fox in snow"
    assert "texture_boiling" in call["payload"]["allowed_issues"]
    for words in ("Top row", "Second row", "Third row", "left to right", "honestly"):
        assert words in call["instructions"]


def test_picture_checker_gives_no_review_when_the_model_cannot_see():
    for provider in (BlindProvider(), RuleBasedProvider()):
        checker = PictureChecker(provider)
        assert checker.review(shot_id="s", theme="t", prompt="p", sheet_png=b"png") is None
        assert "cannot see images" in (checker.skipped or "")
    blind = BlindProvider()
    PictureChecker(blind).review(shot_id="s", theme="t", prompt="p", sheet_png=b"png")
    assert blind.calls == []  # nothing was asked, nothing invented


def test_picture_checker_gives_no_review_when_the_model_fails():
    down = SeeingProvider(error=AgentUnavailable("connection refused"))
    checker = PictureChecker(down)
    assert checker.review(shot_id="a", theme="t", prompt="p", sheet_png=b"png") is None
    assert "unavailable" in (checker.skipped or "")
    assert checker.review(shot_id="b", theme="t", prompt="p", sheet_png=b"png") is None
    assert len(down.calls) == 1  # a dead model is not asked again for every shot

    bad = SeeingProvider(error=AgentOutputError("picture_checker", "invalid output"))
    checker = PictureChecker(bad)
    assert checker.review(shot_id="a", theme="t", prompt="p", sheet_png=b"png") is None
    assert checker.review(shot_id="b", theme="t", prompt="p", sheet_png=b"png") is None
    assert len(bad.calls) == 2 and "no valid review" in (checker.skipped or "")

    limited = PictureChecker(SeeingProvider(), max_calls=1)
    assert limited.review(shot_id="a", theme="t", prompt="p", sheet_png=b"png") is not None
    assert limited.review(shot_id="b", theme="t", prompt="p", sheet_png=b"png") is None


def test_picture_review_schema_is_bounded():
    assert PictureReview.model_validate({**REVIEW, "issues": ["flicker", "flicker"]}).issues \
        == ["flicker"]
    for bad in ({"anatomy": 11}, {"steadiness": -1}, {"issues": ["looks_weird"]},
                {"description": ""}, {"description": "x" * 801}, {"notes": "x" * 601}):
        with pytest.raises(ValidationError):
            PictureReview.model_validate({**REVIEW, **bad})
    with pytest.raises(ValidationError):  # every field must be answered, issues included
        PictureReview.model_validate({k: v for k, v in REVIEW.items() if k != "issues"})


def test_picture_checker_through_ollama_sends_the_png_as_an_image():
    calls: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision"]})
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(REVIEW)}})

    provider = OllamaProvider("http://ollama", "qwen3.5:9b",
                              transport=httpx.MockTransport(handle))
    png = vision.png_bytes(vision.contact_sheet(clip(), clip(seed=1)))
    review = PictureChecker(provider).review(shot_id="s", theme="t", prompt="p", sheet_png=png)
    assert review is not None and review.subject_identity == 9
    user = calls[0]["messages"][1]
    assert base64.b64decode(user["images"][0]) == png
    assert "maxLength" not in json.dumps(calls[0]["format"])  # grammar-safe schema


# -- applying the review --------------------------------------------------------------------
def shot(decision: str) -> dict[str, Any]:
    return {"shot_id": "shot_001", "decision": decision, "overall": 7.0, "failed_frames": [],
            "issues": [], "recommendations": ["PASS"] if decision == "PASS" else ["CHANGE_SEED"]}


def test_apply_review_is_advisory_and_never_flips_the_decision():
    poor = PictureReview.model_validate({**REVIEW, "prompt_adherence": 2, "anatomy": 3,
                                         "steadiness": 2, "issues": ["melting_subject"]})
    passed = shot("PASS")
    out = vision.apply_review(passed, poor, by="ollama")
    assert out["decision"] == "PASS" and out["recommendations"] == ["PASS"]
    assert passed.get("identity") is None and "picture_review" not in passed  # not mutated
    assert (out["identity"], out["prompt_adherence"], out["style_consistency"],
            out["hand_body_deformation"]) == (9, 2, 7, 3)
    assert out["picture_review"] == {"description": REVIEW["description"],
                                     "issues": ["melting_subject"], "notes": REVIEW["notes"],
                                     "steadiness": 2, "by": "ollama"}

    good = PictureReview.model_validate({**REVIEW, "steadiness": 9, "issues": []})
    failed = vision.apply_review(shot("FAIL"), good)
    assert failed["decision"] == "FAIL" and failed["recommendations"] == ["CHANGE_SEED"]
    assert "by" not in failed["picture_review"]

    worst = vision.apply_review(shot("FAIL"), poor)
    assert worst["decision"] == "FAIL"
    assert worst["recommendations"] == ["CHANGE_SEED", "FOLLOW_PROMPT", "FIX_ANATOMY",
                                        "STABILIZE"]
    leaks = PictureReview.model_validate({**REVIEW, "issues": ["source_look_leaks", "flicker",
                                                               "hand_distortion"]})
    assert vision.apply_review(shot("FAIL"), leaks)["recommendations"][1:] == [
        "FOLLOW_PROMPT", "FIX_ANATOMY", "STABILIZE"]


def test_reviewed_scores_reach_the_qc_summary():
    reviewed = vision.apply_review(shot("PASS"), PictureReview.model_validate(REVIEW))
    s = qc.summarize([reviewed, {**shot("PASS"), "shot_id": "shot_002"}], 6.5)
    assert s["identity"] == 9 and s["hand_body_deformation"] == 6
    assert "identity" not in s["not_measured"] and s["picture_reviewed"] == ["shot_001"]


def test_check_shot_builds_the_sheet_and_folds_in_the_review():
    src, out = clip(), clip(seed=1)
    result, png = vision.check_shot(PictureChecker(SeeingProvider()), shot("FAIL"), src, out,
                                    theme="watercolour", prompt="a fox")
    assert png.startswith(b"\x89PNG") and result["picture_review"]["by"] == "fake_vision"
    assert result["recommendations"] == ["CHANGE_SEED", "STABILIZE"]  # steadiness 5, boiling

    skipped, _ = vision.check_shot(PictureChecker(BlindProvider()), shot("FAIL"), src, out,
                                   theme="t", prompt="p")
    assert "cannot see images" in skipped["picture_review_skipped"]
    assert "picture_review" not in skipped
    unchanged, _ = vision.check_shot(None, shot("PASS"), src, out, theme="t", prompt="p")
    assert unchanged == shot("PASS")

    few = SeeingProvider()
    vision.check_shot(PictureChecker(few), shot("PASS"), src[:2], out[:2], theme="t",
                      prompt="p")
    assert few.calls[0]["payload"]["sheet"]["columns"] == 2
