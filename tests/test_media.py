import numpy as np
import pytest

from rokkur_studio.media.ffmpeg import FFmpegError
from rokkur_studio.pipeline import qc
from rokkur_studio.pipeline.analysis import analyze_video, split_shots


def test_probe_and_scene_detection(ffmpeg, sample_video):
    info = ffmpeg.probe(sample_video)
    assert (info.width, info.height, info.fps) == (360, 640, 24.0)
    assert info.has_audio and abs(info.duration - 4) < 0.1
    assert ffmpeg.detect_scenes(sample_video) == [2.0]


def test_probe_uses_the_video_length_when_matroska_audio_runs_longer(ffmpeg, tmp_path):
    out = tmp_path / "long_audio.mkv"
    ffmpeg._run(ffmpeg._ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24:duration=3",
                           "-f", "lavfi", "-i", "sine=duration=3.6",
                           "-c:v", "libx264", "-c:a", "aac", str(out)))
    info = ffmpeg.probe(out)
    assert abs(float(info.raw["format"]["duration"]) - 3.6) < 0.1  # the container's length
    assert info.duration == 3.0 and info.frame_count == 72


def test_cut_splice_attach_encode(ffmpeg, sample_video, tmp_path):
    a = ffmpeg.cut(sample_video, tmp_path / "a.mp4", start=0, end=2, fps=12, width=288, height=512)
    b = ffmpeg.cut(sample_video, tmp_path / "b.mp4", start=2, end=4, fps=12, width=288, height=512)
    spliced = ffmpeg.concat([a, b], tmp_path / "ab.mp4")
    assert ffmpeg.probe(spliced).frame_count == 48
    with_audio = ffmpeg.attach_audio(spliced, sample_video, tmp_path / "av.mp4")
    final = ffmpeg.encode_final(with_audio, tmp_path / "final.mp4", width=1080, height=1920, fps=24)
    info = ffmpeg.probe(final)
    assert (info.width, info.height, info.video_codec, info.has_audio) == (1080, 1920, "h264", True)
    assert ffmpeg.thumbnail(final, tmp_path / "t.jpg", at=1).stat().st_size > 0
    assert ffmpeg.preview_gif(final, tmp_path / "p.gif", seconds=1).stat().st_size > 0
    frames = ffmpeg.extract_frames(spliced, tmp_path / "frames", fps=2)
    assert len(frames) == 8
    assert any("concat" in " ".join(h["cmd"]) for h in ffmpeg.history)  # commands are logged


def test_errors_are_structured(ffmpeg, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(FFmpegError) as exc:
        ffmpeg.probe(bad)
    d = exc.value.to_dict()
    assert d["returncode"] != 0 and d["cmd"][0].endswith("ffprobe") and d["stderr_tail"]


def test_split_shots_merges_slivers_and_splits_long_shots():
    assert split_shots(10, [0.2, 5.0, 5.3], max_shot_s=10) == [(0, 5.0), (5.0, 10)]
    assert split_shots(9, [], max_shot_s=4) == [(0, 3), (3, 6), (6, 9)]


def test_analysis_finds_shots_and_documents_skipped_signals(ffmpeg, sample_video):
    a = analyze_video(ffmpeg, sample_video, max_shot_s=4)
    assert [s["shot_id"] for s in a["shots"]] == ["shot_001", "shot_002"]
    assert a["shots"][0]["end"] == 2.0
    assert "pose" in a["signals_skipped"]


def test_qc_passes_identical_and_flags_black_frames(ffmpeg, sample_video):
    frames = ffmpeg.read_gray_frames(sample_video, 64, 64, fps=12)
    good = qc.score_shot(frames, frames, threshold=6.5, shot_id="s")
    assert good["decision"] == "PASS" and good["overall"] > 9
    broken = frames.copy()
    broken[10:16] = 0
    bad = qc.score_shot(frames, broken, threshold=6.5, shot_id="s", frame_offset=100)
    assert bad["decision"] == "FAIL"
    assert set(range(110, 116)) <= set(bad["failed_frames"])
    assert "RERENDER_SHOT" in bad["recommendations"]


def test_qc_flags_flicker():
    rng = np.random.default_rng(0)
    base = np.tile(rng.integers(0, 255, (1, 64, 64), dtype=np.uint8), (24, 1, 1))
    flicker = base.copy()
    flicker[::2] = np.clip(flicker[::2].astype(int) + 90, 0, 255).astype(np.uint8)
    r = qc.score_shot(base, flicker, threshold=6.5, shot_id="s")
    assert r["decision"] == "FAIL" and "CHANGE_SEED" in r["recommendations"]


def test_qc_summary_reports_unmeasured_metrics_as_null():
    s = qc.summarize([{"shot_id": "a", "decision": "PASS", "overall": 9, "failed_frames": [],
                       "motion": 9, "temporal_consistency": 9, "structure": 9, "detail": 9,
                       "artifact_score": 0}], 6.5)
    assert s["decision"] == "PASS" and s["identity"] is None and "identity" in s["not_measured"]


def test_qc_structure_ignores_relighting_but_catches_a_different_layout(ffmpeg, sample_video):
    frames = ffmpeg.read_gray_frames(sample_video, 64, 64, fps=12)
    relit = 255 - frames  # same edges, every tone changed: what a strong restyle can do
    assert qc.score_shot(frames, relit, threshold=6.5, shot_id="s")["structure"] > 9
    moved = frames.transpose(0, 2, 1)  # bars turned sideways: the layout no longer matches
    r = qc.score_shot(frames, moved, threshold=6.5, shot_id="s")
    assert r["structure"] < 5 and any("layout drift" in i for i in r["issues"])
