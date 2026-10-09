"""Shot renderers. ``comfyui`` is the real path; ``ffmpeg_preview`` is a deterministic,
non-AI stand-in that exercises the full pipeline without a GPU."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rokkur_studio.comfyui.client import (
    ComfyClient,
    ComfyError,
    ComfyExecutionError,
    ComfyUnavailable,
    ComfyValidationError,
)
from rokkur_studio.comfyui.compiler import TemplateRegistry, compile_workflow
from rokkur_studio.jobs.errors import JobCancelled
from rokkur_studio.media.ffmpeg import FFmpeg

log = logging.getLogger(__name__)


class RenderOOM(RuntimeError):
    """CUDA out of memory: must be degraded, never retried identically."""


class RenderUnavailable(RuntimeError):
    """Renderer backend unreachable: retry later."""


class RenderRejected(RuntimeError):
    """Inputs invalid for the backend: retrying identically cannot help."""


@dataclass
class RenderOutcome:
    path: Path
    seconds: float
    remote_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class Renderer(Protocol):
    name: str

    def render_shot(self, *, clip: Path, params: dict[str, Any], workflow: str, out: Path,
                    on_progress: Callable[[dict[str, Any]], None] | None = None
                    ) -> RenderOutcome: ...


class FFmpegPreviewRenderer:
    """Colour-grade stand-in. Supports ``_FAULT`` (black|flicker) to exercise QC/repair."""

    name = "ffmpeg_preview"

    def __init__(self, ffmpeg: FFmpeg) -> None:
        self.ffmpeg = ffmpeg

    def render_shot(self, *, clip: Path, params: dict[str, Any], workflow: str, out: Path,
                    on_progress: Callable[[dict[str, Any]], None] | None = None
                    ) -> RenderOutcome:
        started = time.monotonic()
        s = float(params.get("STYLE_STRENGTH", 0.7))
        vf = [f"fps={params['FPS']}",
              f"scale={params['WIDTH']}:{params['HEIGHT']}",
              f"eq=saturation={1 + 0.8 * s:.3f}:contrast={1 + 0.3 * s:.3f}",
              f"hue=h={25 * s:.1f}",
              "unsharp=5:5:0.6"]
        fault = params.get("_FAULT")
        if fault == "black":
            vf.append("drawbox=enable='between(n,3,8)':x=0:y=0:w=iw:h=ih:color=black:t=fill")
        elif fault == "flicker":
            vf.append("eq=enable='mod(n,2)':brightness=0.35")
        self.ffmpeg.filter_video(clip, out, ",".join(vf), fps=params["FPS"])
        return RenderOutcome(out, time.monotonic() - started, None,
                             {"renderer": self.name, "fault": fault, "vf": vf})


class ComfyUIRenderer:
    name = "comfyui"

    def __init__(self, client: ComfyClient, registry: TemplateRegistry, ffmpeg: FFmpeg, *,
                 timeout_s: float, poll_s: float,
                 should_cancel: Callable[[], bool] | None = None) -> None:
        self.client, self.registry, self.ffmpeg = client, registry, ffmpeg
        self.timeout_s, self.poll_s, self.should_cancel = timeout_s, poll_s, should_cancel

    def render_shot(self, *, clip: Path, params: dict[str, Any], workflow: str, out: Path,
                    on_progress: Callable[[dict[str, Any]], None] | None = None
                    ) -> RenderOutcome:
        started = time.monotonic()
        template = self.registry.get(workflow)
        work = out.parent / f"{out.stem}_comfy"
        work.mkdir(parents=True, exist_ok=True)
        values = {k: v for k, v in params.items() if not k.startswith("_")}
        reference = values.get("REFERENCE_IMAGE")
        reference_kind = str(params.get("_REFERENCE_KIND") or "uploaded") if reference else "none"
        if reference and "REFERENCE_IMAGE" not in template.spec.parameters:
            raise RenderRejected("This workflow cannot use a character reference image")
        mask = values.pop("MASK_VIDEO", None)
        if mask and "MASK_VIDEO" not in template.spec.parameters:
            raise RenderRejected(f"Workflow {workflow} cannot use a subject mask")
        if ("REFERENCE_IMAGE" in template.spec.parameters and not reference
                and params.get("_REFERENCE_MODE", "source") == "source"):
            reference = str(self.ffmpeg.thumbnail(clip, work / "reference.png", at=0,
                                                  width=int(params["WIDTH"])))
            reference_kind = "source first frame"
        prepared = clip
        if any(n["class_type"] == "WanVaceToVideo" for n in template.workflow.values()):
            prepared = self._frames(clip, work / "control.mp4", params)
        try:
            folder = f"rokkur/{self.client.client_id}/{out.stem}"
            uploaded = self.client.upload_input(prepared, subfolder=folder)
            if mask:
                # Same frame timing as the control video, so mask frame i covers source frame i.
                mask_path = Path(mask)
                if not mask_path.is_file():
                    raise RenderRejected(f"Subject mask is not readable by the worker: {mask}")
                values["MASK_VIDEO"] = self.client.upload_input(
                    self._frames(mask_path, work / "subject_mask.mp4", params), subfolder=folder)
            if reference:
                ref_path = Path(reference)
                if not ref_path.is_file():
                    raise RenderRejected(f"Reference image is not readable by the worker: {reference}")
                values["REFERENCE_IMAGE"] = self.client.upload_input(ref_path, subfolder=folder)
            compiled = compile_workflow(template, {
                **values,
                "INPUT_VIDEO": uploaded,
            })
            prompt_id = self.client.submit(compiled.workflow)
            result = self.client.wait(prompt_id, timeout_s=self.timeout_s, poll_s=self.poll_s,
                                      on_progress=on_progress, should_cancel=self.should_cancel)
        except ComfyExecutionError as exc:
            if exc.is_oom:
                raise RenderOOM(str(exc)) from exc
            raise RenderRejected(str(exc)) from exc
        except ComfyValidationError as exc:
            raise RenderRejected(f"{exc}: {exc.details}") from exc
        except ComfyUnavailable as exc:
            raise RenderUnavailable(str(exc)) from exc
        except ComfyError as exc:
            if exc.code == "cancelled":
                raise JobCancelled() from exc
            raise RenderUnavailable(str(exc)) from exc  # a timeout or failed upload: retry
        videos = [o for o in result.outputs if Path(o.filename).suffix.lower()
                  in (".mp4", ".webm", ".mkv", ".mov", ".gif")]
        images = [o for o in result.outputs if Path(o.filename).suffix.lower()
                  in (".png", ".jpg", ".jpeg", ".webp")]
        if videos:
            raw = self.client.download(videos[-1], work / Path(videos[-1].filename).name)
        elif images:
            for i, img in enumerate(sorted(images, key=lambda o: o.filename), 1):
                self.client.download(img, work / f"frame_{i:05d}.png")
            raw = self.ffmpeg.frames_to_video(str(work / "frame_%05d.png"),
                                               work / "frames.mp4", params["FPS"])
        else:
            raise RenderRejected(f"prompt {result.prompt_id} produced no image/video output")
        wanted = int(params.get("_OUTPUT_FRAMES", params["FRAME_COUNT"]))
        self.ffmpeg.filter_video(raw, out, f"fps={params['FPS']},"
            f"scale={params['WIDTH']}:{params['HEIGHT']},"
            f"tpad=stop=-1:stop_mode=clone,trim=end_frame={wanted},setpts=PTS-STARTPTS",
            fps=params["FPS"])
        return RenderOutcome(out, time.monotonic() - started, result.prompt_id, {
            "renderer": self.name, "workflow": compiled.template,
            "workflow_version": compiled.template_version, "applied": compiled.applied,
            "ignored_params": sorted(compiled.ignored),
            "reference": reference_kind,
            "subject_mask": bool(mask),
            "output_frames": self.ffmpeg.probe(out).frame_count,
            "execution_seconds": result.execution_seconds})

    def _frames(self, video: Path, out: Path, params: dict[str, Any]) -> Path:
        """``video`` at the render's fps and exactly FRAME_COUNT frames (the last one held)."""
        return self.ffmpeg.filter_video(video, out,
            f"fps={params['FPS']},tpad=stop=-1:stop_mode=clone,"
            f"trim=end_frame={params['FRAME_COUNT']},setpts=PTS-STARTPTS", fps=params["FPS"])
