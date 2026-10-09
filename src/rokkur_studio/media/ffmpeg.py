"""FFmpeg/ffprobe wrapper. Every command is logged; failures raise structured diagnostics."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _tag_seconds(tags: dict[str, Any]) -> float | None:
    """Seconds from a Matroska ``DURATION`` tag such as ``00:00:03.000000000``."""
    value = next((v for k, v in tags.items() if k.upper() == "DURATION"), None)
    try:
        h, m, s = str(value).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (TypeError, ValueError):
        return None


class FFmpegError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd, self.returncode = cmd, returncode
        self.stderr_tail = stderr[-4000:]
        super().__init__(f"{Path(cmd[0]).name} exited {returncode}: {self.summary}")

    @property
    def summary(self) -> str:
        lines = [ln for ln in self.stderr_tail.splitlines() if ln.strip()]
        return lines[-1] if lines else "no stderr"

    def to_dict(self) -> dict[str, Any]:
        return {"cmd": self.cmd, "returncode": self.returncode, "stderr_tail": self.stderr_tail}


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    video_codec: str | None
    has_audio: bool
    audio_codec: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


@dataclass
class FFmpeg:
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    timeout_s: float = 3600
    history: list[dict[str, Any]] = field(default_factory=list)

    def available(self) -> bool:
        return bool(shutil.which(self.ffmpeg_bin) and shutil.which(self.ffprobe_bin))

    # -- plumbing -------------------------------------------------------------------------
    def _run(self, cmd: list[str], *, capture_stdout: bool = False,
             stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        started = time.monotonic()
        log.info("ffmpeg command", extra={"data": {"cmd": cmd}})
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout_s, check=False,
                                  input=stdin)
        except FileNotFoundError as exc:
            raise FFmpegError(cmd, 127, f"executable not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(cmd, -1, f"timed out after {self.timeout_s}s") from exc
        elapsed = time.monotonic() - started
        self.history.append({"cmd": cmd, "returncode": proc.returncode, "seconds": round(elapsed, 3)})
        if proc.returncode != 0:
            raise FFmpegError(cmd, proc.returncode, proc.stderr.decode("utf-8", "replace"))
        return proc

    def _ff(self, *args: str) -> list[str]:
        return [self.ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *args]

    # -- inspection -----------------------------------------------------------------------
    def probe(self, path: Path) -> MediaInfo:
        cmd = [self.ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format",
               "-show_streams", str(path)]
        data = json.loads(self._run(cmd).stdout or b"{}")
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if video is None:
            raise FFmpegError(cmd, 0, f"no video stream in {path}")
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        fps = float(Fraction(rate)) if rate not in ("0/0", "") else 0.0
        # Matroska/WebM keep the stream's own length in a tag; the container duration can be
        # the longer audio's, which would make the last shot run past the end of the video.
        duration = float(video.get("duration") or _tag_seconds(video.get("tags", {}))
                         or data.get("format", {}).get("duration") or 0)
        frames = video.get("nb_frames")
        return MediaInfo(
            path=str(path),
            duration=duration,
            width=int(video["width"]),
            height=int(video["height"]),
            fps=round(fps, 3),
            frame_count=int(frames) if frames and str(frames).isdigit() else round(duration * fps),
            video_codec=video.get("codec_name"),
            has_audio=audio is not None,
            audio_codec=audio.get("codec_name") if audio else None,
            raw=data,
        )

    def detect_scenes(self, path: Path, threshold: float = 0.3) -> list[float]:
        """Timestamps (s) where FFmpeg's scene score exceeds ``threshold``."""
        cmd = [self.ffmpeg_bin, "-hide_banner", "-nostdin", "-i", str(path), "-an",
               "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
        proc = self._run(cmd)
        text = proc.stderr.decode("utf-8", "replace")
        return sorted({round(float(m), 3) for m in re.findall(r"pts_time:([0-9.]+)", text)})

    def read_gray_frames(self, path: Path, width: int = 64, height: int = 64,
                         fps: float | None = None) -> np.ndarray:
        """Decode to an ``(n, height, width)`` uint8 array of downscaled grey frames."""
        vf = f"scale={width}:{height}:flags=area,format=gray"
        if fps:
            vf = f"fps={fps}," + vf
        cmd = [self.ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
               "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        raw = self._run(cmd).stdout
        frame = width * height
        n = len(raw) // frame
        return np.frombuffer(raw[: n * frame], dtype=np.uint8).reshape(n, height, width)

    def read_rgb_frames(self, path: Path, width: int, height: int, *, fps: float | None = None,
                        frames: int | None = None,
                        aspect: tuple[int, int] | None = None) -> np.ndarray:
        """Decode to an ``(n, height, width, 3)`` uint8 RGB array.

        ``aspect`` (w, h) centre-crops to that shape first, as ComfyUI's ImageScale with
        crop=center does, before scaling to ``width`` x ``height``. ``frames`` pads (repeating
        the last frame) or trims to exactly that many frames.
        """
        vf = [f"fps={fps}"] if fps else []
        if aspect:
            aw, ah = aspect
            vf.append(f"crop='min(iw,ih*{aw}/{ah})':'min(ih,iw*{ah}/{aw})'")
        vf.append(f"scale={width}:{height}")
        if frames:
            vf += ["tpad=stop=-1:stop_mode=clone", f"trim=end_frame={frames}"]
        cmd = [self.ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
               "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = self._run(cmd).stdout
        frame = width * height * 3
        n = len(raw) // frame
        return np.frombuffer(raw[: n * frame], dtype=np.uint8).reshape(n, height, width, 3)

    # -- transformation -------------------------------------------------------------------
    def write_frames(self, frames: np.ndarray, out: Path, fps: float) -> Path:
        """Encode ``(n, h, w, 3)`` RGB or ``(n, h, w)`` grey uint8 frames to H.264."""
        n, h, w = frames.shape[:3]
        pix = "rgb24" if frames.ndim == 4 else "gray"
        self._run(self._ff("-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", str(fps),
                           "-i", "-", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(out)),
                  stdin=np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
        return out

    def write_image(self, frame: np.ndarray, out: Path) -> Path:
        """Write one ``(h, w, 3)`` uint8 RGB frame as an image (format from the suffix)."""
        h, w = frame.shape[:2]
        self._run(self._ff("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
                           "-frames:v", "1", str(out)),
                  stdin=np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        return out

    def extract_frames(self, path: Path, out_dir: Path, *, fps: float | None = None,
                       pattern: str = "frame_%05d.png") -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        args = ["-i", str(path)]
        if fps:
            args += ["-vf", f"fps={fps}"]
        self._run(self._ff(*args, str(out_dir / pattern)))
        return sorted(out_dir.glob(pattern.replace("%05d", "*")))

    def extract_audio(self, path: Path, out: Path) -> Path:
        self._run(self._ff("-i", str(path), "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le",
                           str(out)))
        return out

    def cut(self, path: Path, out: Path, *, start: float, end: float, fps: float | None = None,
            width: int | None = None, height: int | None = None, keep_audio: bool = False) -> Path:
        """Frame-accurate re-encoded segment, optionally normalised to fps/size."""
        filters = []
        if fps:
            filters.append(f"fps={fps}")
        if width and height:
            filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
            filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
        args = ["-ss", f"{start:.3f}", "-i", str(path), "-t", f"{max(0.04, end - start):.3f}"]
        if filters:
            args += ["-vf", ",".join(filters)]
        if fps:
            args += ["-r", str(fps)]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
        args += ["-c:a", "aac"] if keep_audio else ["-an"]
        self._run(self._ff(*args, str(out)))
        return out

    def filter_video(self, path: Path, out: Path, vf: str, *, fps: float | None = None) -> Path:
        # Supply the encoder rate as well as the fps filter: otherwise a trimmed final
        # frame can have zero duration in MP4 on newer FFmpeg versions.
        rate = ["-r", str(fps)] if fps else []
        self._run(self._ff("-i", str(path), "-vf", vf, *rate, "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)))
        return out

    def frames_to_video(self, frames_glob_pattern: str, out: Path, fps: float) -> Path:
        """Combine an image sequence (e.g. ``dir/frame_%05d.png``) into H.264."""
        self._run(self._ff("-framerate", str(fps), "-i", frames_glob_pattern, "-r", str(fps),
                           "-c:v", "libx264",
                           "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)))
        return out

    def concat(self, clips: list[Path], out: Path) -> Path:
        """Splice clips (same codec/size/fps) via the concat demuxer, re-encoding for safety."""
        if not clips:
            raise ValueError("concat needs at least one clip")
        listing = out.with_suffix(".concat.txt")
        listing.write_text(
            "".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8"
        )
        try:
            self._run(self._ff("-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264",
                               "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-an",
                               str(out)))
        finally:
            listing.unlink(missing_ok=True)
        return out

    def attach_audio(self, video: Path, audio_source: Path, out: Path, *,
                     normalize: bool = True) -> Path:
        args = ["-i", str(video), "-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        if normalize:
            args += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
        self._run(self._ff(*args, str(out)))
        return out

    def encode_final(self, video: Path, out: Path, *, width: int, height: int, fps: float,
                     codec: str = "h264", crf: int = 18) -> Path:
        """Platform delivery encode: fit+pad to target, yuv420p, faststart, AAC audio if present."""
        vcodec = {"h264": "libx264", "h265": "libx265"}[codec]
        vf = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
        args = ["-i", str(video), "-vf", vf, "-c:v", vcodec, "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k"]
        if codec == "h265":
            args += ["-tag:v", "hvc1"]
        self._run(self._ff(*args, str(out)))
        return out

    def thumbnail(self, video: Path, out: Path, *, at: float, width: int = 1280) -> Path:
        self._run(self._ff("-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
                           "-vf", f"scale={width}:-2", str(out)))
        return out

    def preview_gif(self, video: Path, out: Path, *, width: int = 320, fps: int = 10,
                    seconds: float = 4) -> Path:
        vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen[p];"
              f"[b][p]paletteuse")
        self._run(self._ff("-t", str(seconds), "-i", str(video), "-vf", vf, str(out)))
        return out

    def make_test_video(self, out: Path, *, seconds: float = 4, width: int = 360,
                        height: int = 640, fps: int = 24, with_audio: bool = True,
                        scene_cut: bool = True) -> Path:
        """Synthetic, rights-free fixture (testsrc2 + moving box + tone)."""
        half = seconds / 2
        if scene_cut:
            vf_src = (f"testsrc2=size={width}x{height}:rate={fps}:duration={half}[a];"
                      f"mandelbrot=size={width}x{height}:rate={fps}[m];"
                      f"[m]trim=duration={half},setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[v]")
        else:
            vf_src = f"testsrc2=size={width}x{height}:rate={fps}:duration={seconds}[v]"
        args = ["-filter_complex", vf_src, "-map", "[v]"]
        if with_audio:
            args = ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", *args,
                    "-map", "0:a", "-c:a", "aac"]
        # The concat filter has a microsecond timebase. Explicit CFR prevents codec /
        # FFmpeg-version-dependent rounding from changing the fixture's frame rate.
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), "-t", str(seconds)]
        self._run(self._ff(*args, str(out)))
        return out
