"""Temporal stabilization: render deflicker/denoise, the keep-or-reject check and the calmer
control guide. Real FFmpeg on synthetic clips; ComfyUI is faked."""
from pathlib import Path

import numpy as np
import pytest

from rokkur_studio.comfyui.client import ComfyClient
from rokkur_studio.comfyui.compiler import TemplateRegistry
from rokkur_studio.media.ffmpeg import FFmpeg, FFmpegError
from rokkur_studio.pipeline import stabilize
from rokkur_studio.pipeline.renderers import ComfyUIRenderer, RenderRejected
from rokkur_studio.pipeline.stabilize import (
    candidate_levels,
    control_smoothing_filter,
    pick_steadier,
    resolve_level,
    stabilize_filter,
    stabilize_render,
    stabilize_shot,
)
from tests.fakes import FakeComfyUI

ROOT = Path(__file__).resolve().parents[1]
FPS = 16


def _options(vf: str, name: str) -> dict[str, str]:
    part = next(p for p in vf.split(",") if p.startswith(name + "="))
    return dict(kv.split("=") for kv in part[len(name) + 1:].split(":"))


def _brightness_flicker(ffmpeg: FFmpeg, path: Path) -> float:
    """Mean change of a frame's average brightness from one frame to the next."""
    means = ffmpeg.read_gray_frames(path, 64, 64).astype(np.float32).mean(axis=(1, 2))
    return float(np.abs(np.diff(means)).mean())


def _pixel_change(ffmpeg: FFmpeg, path: Path) -> float:
    frames = ffmpeg.read_gray_frames(path, 160, 288).astype(np.float32)
    return float(np.abs(np.diff(frames, axis=0)).mean())


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory, ffmpeg: FFmpeg,
          sample_video: Path) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("stabilize")
    base = f"fps={FPS},scale=160:288"
    still = f"{base},trim=end_frame=1,loop=loop=31:size=1,setpts=N/{FPS}/TB"
    return {
        "clean": ffmpeg.filter_video(sample_video, d / "clean.mp4", base, fps=FPS),
        # What QC calls temporal flicker: every other frame brighter.
        "flicker": ffmpeg.filter_video(sample_video, d / "flicker.mp4",
                                       f"{base},eq=enable='mod(n,2)':brightness=0.1", fps=FPS),
        "still": ffmpeg.filter_video(sample_video, d / "still.mp4", still, fps=FPS),
        # A surface that should hold still but boils: the same frame with fresh grain each time.
        "boil": ffmpeg.filter_video(sample_video, d / "boil.mp4",
                                    f"{still},noise=alls=8:allf=t", fps=FPS),
    }


# -- levels and filters -------------------------------------------------------------------
@pytest.mark.parametrize(("value", "level"), [
    ("auto", "light"), (None, "light"), ("", "light"), ("OFF", "off"), (" Strong ", "strong"),
    ("light", "light"), (True, "light"), (False, "off"),
])
def test_resolve_level_maps_auto_to_the_level_tried_automatically(value, level):
    assert resolve_level(value) == level


@pytest.mark.parametrize("value", ["medium", 2, 0.5, "auto light"])
def test_resolve_level_rejects_unknown_settings(value):
    with pytest.raises(ValueError, match="stabilize must be one of"):
        resolve_level(value)


def test_candidate_levels_fall_back_from_strong_to_light():
    assert candidate_levels("auto") == candidate_levels("light") == ["light"]
    assert candidate_levels("strong") == ["strong", "light"]
    assert candidate_levels("off") == []


def test_stabilize_filters_deflicker_and_denoise_only_in_time():
    assert stabilize_filter("off") == ""
    assert stabilize_filter("auto") == stabilize_filter("light")
    light, strong = stabilize_filter("light"), stabilize_filter("strong")
    assert light.startswith("deflicker=") and "hqdn3d" not in light
    assert int(_options(strong, "deflicker")["size"]) > int(_options(light, "deflicker")["size"])
    hq = {k: float(v) for k, v in _options(strong, "hqdn3d").items()}
    # Spatial 0 would be replaced by hqdn3d's default and blur every frame; a tiny value is off.
    assert 0 < hq["luma_spatial"] <= 0.01 and 0 < hq["chroma_spatial"] <= 0.01
    assert hq["luma_tmp"] > 0 and hq["chroma_tmp"] > 0


# -- stabilize_render ---------------------------------------------------------------------
@pytest.mark.parametrize("level", ["light", "strong", "auto"])
def test_stabilize_render_keeps_frames_and_fps_and_calms_flicker(ffmpeg, clips, tmp_path, level):
    raw = clips["flicker"]
    out = stabilize_render(ffmpeg, raw, tmp_path / f"{level}.mp4", level=level, fps=FPS)
    before, after = ffmpeg.probe(raw), ffmpeg.probe(out)
    assert after.frame_count == before.frame_count and after.fps == before.fps == FPS
    assert after.duration == pytest.approx(before.duration, abs=0.001)
    assert _brightness_flicker(ffmpeg, out) < 0.5 * _brightness_flicker(ffmpeg, raw)


def test_stabilize_render_off_returns_the_render_untouched(ffmpeg, clips, tmp_path):
    out = tmp_path / "off.mp4"
    assert stabilize_render(ffmpeg, clips["flicker"], out, level="off", fps=FPS) == clips["flicker"]
    assert not out.exists()


def test_strong_calms_boiling_texture_that_light_leaves(ffmpeg, clips, tmp_path):
    boil = clips["boil"]
    light = stabilize_render(ffmpeg, boil, tmp_path / "light.mp4", level="light", fps=FPS)
    strong = stabilize_render(ffmpeg, boil, tmp_path / "strong.mp4", level="strong", fps=FPS)
    light_change = _pixel_change(ffmpeg, light)
    # Deflicker alone keeps the grain (re-encoding smooths a little of it).
    assert light_change > 0.8 * _pixel_change(ffmpeg, boil)
    assert _pixel_change(ffmpeg, strong) < 0.5 * light_change
    assert ffmpeg.probe(strong).frame_count == ffmpeg.probe(boil).frame_count


def test_strong_leaves_a_still_picture_as_sharp_as_it_was(ffmpeg, clips, tmp_path):
    out = stabilize_render(ffmpeg, clips["still"], tmp_path / "s.mp4", level="strong", fps=FPS)
    a = ffmpeg.read_gray_frames(clips["still"], 160, 288).astype(int)
    b = ffmpeg.read_gray_frames(out, 160, 288).astype(int)
    assert np.abs(a - b).mean() < 0.5  # no spatial blur: only re-encoding noise


# -- pick_steadier ------------------------------------------------------------------------
def _scene(n: int = 24) -> np.ndarray:
    rng = np.random.default_rng(3)
    texture = rng.integers(40, 200, (64, 64)).astype(np.float32)
    frames = []
    for t in range(n):
        f = np.roll(texture, t, axis=1).copy()
        f[20:34, 5 + t:19 + t] = 250
        frames.append(f)
    return np.clip(np.array(frames), 0, 255).astype(np.uint8)


def _flickering(frames: np.ndarray, amount: int = 30) -> np.ndarray:
    lift = (np.arange(len(frames)) % 2 * amount)[:, None, None]
    return np.clip(frames.astype(int) + lift, 0, 255).astype(np.uint8)


def _blurred(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    pad = np.pad(f, ((0, 0), (2, 2), (2, 2)), mode="edge")
    out = sum(pad[:, i:i + 64, j:j + 64] for i in range(5) for j in range(5)) / 25
    return out.astype(np.uint8)


def test_pick_steadier_keeps_a_clip_that_is_steadier_at_the_same_detail():
    source = _scene()
    keep, record = pick_steadier(source, _flickering(source), source, threshold=6.5)
    assert keep and record["kept"] and record["metric"] == "temporal_consistency"
    assert record["after"]["steadiness"] - record["before"]["steadiness"] >= stabilize.MIN_GAIN
    assert set(record["before"]) == {"steadiness", "detail", "structure", "overall"}


def test_pick_steadier_rejects_no_gain_and_lost_detail():
    source = _scene()
    keep, record = pick_steadier(source, source, source, threshold=6.5)
    assert not keep and "less than" in record["reason"]
    keep, record = pick_steadier(source, _flickering(source), _blurred(source), threshold=6.5)
    assert not keep and "detail fell" in record["reason"]
    assert record["after"]["steadiness"] > record["before"]["steadiness"]


def _fake_scores(monkeypatch, before: dict, after: dict) -> None:
    def score(source, render, *, threshold, shot_id, frame_offset=0):
        return dict(before if shot_id == "raw" else after)
    monkeypatch.setattr(stabilize, "score_shot", score)


def test_pick_steadier_prefers_motion_compensated_stability(monkeypatch):
    frames = _scene(4)
    _fake_scores(monkeypatch,
                 {"temporal_consistency": 9.0, "stability": 5.0, "detail": 8.0, "overall": 7.0},
                 {"temporal_consistency": 9.0, "stability": 6.0, "detail": 7.5, "overall": 7.2})
    keep, record = pick_steadier(frames, frames, frames, threshold=6.5)
    assert keep and record["metric"] == "stability"
    assert (record["before"]["steadiness"], record["after"]["steadiness"]) == (5.0, 6.0)


def test_pick_steadier_rejects_a_lower_overall_or_unmeasured_stability(monkeypatch):
    frames = _scene(4)
    _fake_scores(monkeypatch,
                 {"temporal_consistency": 5.0, "stability": None, "detail": 8.0, "overall": 7.0},
                 {"temporal_consistency": 6.0, "stability": None, "detail": 8.0, "overall": 6.9})
    keep, record = pick_steadier(frames, frames, frames, threshold=6.5)
    assert not keep and record["metric"] == "temporal_consistency"
    assert "overall fell" in record["reason"]
    _fake_scores(monkeypatch, {"overall": 0.0}, {"overall": 0.0})  # a render with no frames
    keep, record = pick_steadier(frames, frames, frames, threshold=6.5)
    assert not keep and record["reason"] == "steadiness could not be measured"


# -- stabilize_shot: what the render stage calls --------------------------------------------
def test_stabilize_shot_keeps_the_steadied_render_when_qc_measures_it_steadier(ffmpeg, clips,
                                                                               tmp_path):
    raw = tmp_path / "attempt_01.mp4"
    raw.write_bytes(clips["flicker"].read_bytes())
    path, info = stabilize_shot(ffmpeg, source=clips["clean"], render=raw, value="auto",
                                fps=FPS, qc_fps=FPS, threshold=6.5)
    assert path == tmp_path / "attempt_01_steady_light.mp4" and path.exists()
    assert info["requested"] == "auto" and info["kept"] == "light"
    assert info["tried"][0]["after"]["steadiness"] > info["tried"][0]["before"]["steadiness"]
    assert ffmpeg.probe(path).frame_count == ffmpeg.probe(raw).frame_count


def test_stabilize_shot_tries_strong_first_then_keeps_the_raw_render(ffmpeg, clips, tmp_path):
    raw = tmp_path / "attempt_02.mp4"
    raw.write_bytes(clips["clean"].read_bytes())  # already as steady as its source
    path, info = stabilize_shot(ffmpeg, source=clips["clean"], render=raw, value="strong",
                                fps=FPS, qc_fps=FPS, threshold=6.5)
    assert path == raw and info["kept"] is None
    assert [t["level"] for t in info["tried"]] == ["strong", "light"]
    assert info["reason"] == "no level was measurably steadier"
    assert not list(tmp_path.glob("*_steady_*.mp4"))  # rejected copies are not left behind


def test_stabilize_shot_off_unknown_and_failures_leave_the_raw_render(ffmpeg, clips, tmp_path,
                                                                     monkeypatch):
    raw = clips["flicker"]
    kwargs = {"source": clips["clean"], "render": raw, "fps": FPS, "qc_fps": FPS,
              "threshold": 6.5}
    assert stabilize_shot(ffmpeg, value="off", **kwargs) == (
        raw, {"requested": "off", "kept": None, "reason": "stabilizer is off"})
    path, info = stabilize_shot(ffmpeg, value="wobbly", **kwargs)
    assert path == raw and "stabilize must be one of" in info["reason"]

    def broken(*args, **kw):
        raise FFmpegError(["ffmpeg"], 1, "No such filter: 'deflicker'")
    monkeypatch.setattr(stabilize, "stabilize_render", broken)
    path, info = stabilize_shot(ffmpeg, value="auto", **kwargs)
    assert path == raw and info["reason"] == "stabilizer failed: No such filter: 'deflicker'"


# -- calmer control guide -----------------------------------------------------------------
def test_control_smoothing_filter_is_temporal_only_and_scales_with_the_amount():
    assert control_smoothing_filter(0) is None
    assert control_smoothing_filter(-0.4) is None
    half = {k: float(v) for k, v in _options(control_smoothing_filter(0.5) or "", "hqdn3d").items()}
    full = {k: float(v) for k, v in _options(control_smoothing_filter(1.0) or "", "hqdn3d").items()}
    assert half["luma_spatial"] <= 0.01 and half["chroma_spatial"] <= 0.01
    assert 0 < half["luma_tmp"] < full["luma_tmp"] and 0 < half["chroma_tmp"] < full["chroma_tmp"]
    assert control_smoothing_filter(3) == control_smoothing_filter(1.0)  # clamped to 0-1
    assert control_smoothing_filter("0.5") == control_smoothing_filter(0.5)  # type: ignore[arg-type]
    for bad in ("lots", float("nan")):
        with pytest.raises(ValueError):
            control_smoothing_filter(bad)  # type: ignore[arg-type]


def test_control_smoothing_steadies_a_grainy_still_clip(ffmpeg, clips, tmp_path):
    vf = control_smoothing_filter(0.6)
    assert vf
    out = ffmpeg.filter_video(clips["boil"], tmp_path / "smooth.mp4", vf, fps=FPS)
    assert _pixel_change(ffmpeg, out) < 0.6 * _pixel_change(ffmpeg, clips["boil"])
    assert ffmpeg.probe(out).frame_count == ffmpeg.probe(clips["boil"]).frame_count


def _render(ffmpeg: FFmpeg, clip: Path, out: Path, **extra) -> tuple[dict, Path, FakeComfyUI]:
    fake = FakeComfyUI()
    client = ComfyClient("http://comfy", transport=fake.transport())
    params = {"STYLE_PROMPT": "spa", "WIDTH": 160, "HEIGHT": 288, "FPS": FPS,
              "FRAME_COUNT": 29, "_OUTPUT_FRAMES": 27, "_REFERENCE_MODE": "none", **extra}
    outcome = ComfyUIRenderer(client, TemplateRegistry(ROOT / "workflows"), ffmpeg,
                              timeout_s=10, poll_s=0).render_shot(
        clip=clip, params=params, workflow="v2v_3070_quality", out=out)
    workflow = next(iter(fake.prompts.values()))
    control = out.with_name(f"{out.stem}_uploaded_control.mp4")
    control.write_bytes(fake.uploads[workflow["10"]["inputs"]["file"]])
    return outcome.details, control, fake


def test_renderer_smooths_the_control_clip_in_time_and_records_it(ffmpeg, clips, tmp_path):
    plain, plain_control, _ = _render(ffmpeg, clips["boil"], tmp_path / "plain.mp4")
    smooth, smooth_control, _ = _render(ffmpeg, clips["boil"], tmp_path / "smooth.mp4",
                                        _SMOOTH_CONTROL=0.6)
    assert plain["control_smoothing"] is None
    assert smooth["control_smoothing"] == {"amount": 0.6, "filter": control_smoothing_filter(0.6)}
    # Same timing as before (FRAME_COUNT frames at the render fps), steadier pixels.
    for control in (plain_control, smooth_control):
        info = ffmpeg.probe(control)
        assert info.frame_count == 29 and info.fps == FPS
    assert _pixel_change(ffmpeg, smooth_control) < 0.6 * _pixel_change(ffmpeg, plain_control)


def test_renderer_ignores_zero_smoothing_and_rejects_nonsense(ffmpeg, clips, tmp_path):
    details, _, _ = _render(ffmpeg, clips["clean"], tmp_path / "zero.mp4", _SMOOTH_CONTROL=0)
    assert details["control_smoothing"] is None
    with pytest.raises(RenderRejected, match="smooth_control"):
        _render(ffmpeg, clips["clean"], tmp_path / "bad.mp4", _SMOOTH_CONTROL="lots")
