"""Build a reconstruction manifest and per-shot semantic render parameters."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from rokkur_studio.agents.schemas import CreativeBrief
from rokkur_studio.config import RenderProfile
from rokkur_studio.manifest.schema import (
    CameraSpec,
    ControlsSpec,
    IdentitySpec,
    MotionSpec,
    ReconstructionManifest,
    ShotSpec,
    StyleSpec,
    VideoSpec,
)
from rokkur_studio.pipeline.subject import is_stylized


def fit_within(width: int, height: int, max_w: int, max_h: int, multiple: int = 16, *,
               max_pixels: int | None = None) -> tuple[int, int]:
    """Scale (w, h) to fit the (max_w, max_h) box keeping aspect; round down to ``multiple``.

    The box turns with the source, so a landscape clip in a 576x1024 box may be 1024 wide (it
    used to come out 576x320). ``max_pixels`` also caps the area: Wan 1.3B is trained at
    480x832 and is "less stable" above it (Wan 2.1 README).

    16 because video diffusion models (Wan, LTX, Hunyuan) patchify the 8x latent in 2x2 tiles:
    a 568-pixel side gives mismatched token counts and the sampler fails.
    """
    if (width > height) != (max_w > max_h):
        max_w, max_h = max_h, max_w
    scale = min(1.0, max_w / width, max_h / height)
    if max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (width * height)))
    w = max(multiple, int(width * scale) // multiple * multiple)
    h = max(multiple, int(height * scale) // multiple * multiple)
    return w, h


def shot_seed(project_id: str, shot_id: str) -> int:
    return int(hashlib.sha256(f"{project_id}:{shot_id}".encode()).hexdigest()[:8], 16)


def build_manifest(*, project_id: str, source_asset: str, analysis: dict[str, Any],
                   brief: CreativeBrief, profile: RenderProfile, target_format: str,
                   reference_image: str | None = None) -> ReconstructionManifest:
    width, height = fit_within(analysis["width"], analysis["height"], profile.max_width,
                               profile.max_height, max_pixels=profile.max_pixels)
    fps = float(min(analysis["fps"] or profile.fps, profile.fps))
    motion_by_id = {s["shot_id"]: s for s in analysis["shots"]}
    shots = []
    for plan in brief.shot_plan:
        a = motion_by_id.get(plan.shot_id, {})
        shots.append(ShotSpec(
            shot_id=plan.shot_id,
            start=plan.start,
            end=plan.end,
            motion=MotionSpec(type=a.get("motion_type", plan.motion_type),
                              intensity=a.get("motion_intensity", 0.0)),
            camera=CameraSpec(type=plan.camera),
            controls=ControlsSpec(**{k: bool(v) for k, v in profile.controls.items()
                                     if k in ControlsSpec.model_fields}),
            seed=shot_seed(project_id, plan.shot_id),
            prompt=plan.prompt,
        ))
    return ReconstructionManifest(
        project_id=project_id,
        source_asset=source_asset,
        video=VideoSpec(fps=fps, width=width, height=height, duration=analysis["duration"]),
        style=StyleSpec(theme=brief.theme, prompt=brief.prompt,
                        negative_prompt=brief.negative_prompt, strength=brief.style_strength),
        identity=IdentitySpec(reference_image=reference_image, strength=brief.identity_strength),
        shots=shots,
        render_profile=profile.name,
        target_format=target_format,
    )


# Wan's default negative (wan/configs/shared_config.py) carries "style, artwork, painting,
# picture", which push toward photographs; a stylized look must not be steered away from itself.
_STYLE_TERMS = ("风格", "作品", "画作", "画面")
# Community practice against the plastic CGI look (Civitai research, 2026-10-09).
_ANTI_CGI = "3d render, cgi, plastic, waxy skin"


def negative_prompt(base: str, manifest: ReconstructionManifest) -> str:
    """The shot's negative prompt: the profile's base (Wan's own default), then the director's."""
    ours = manifest.style.negative_prompt
    if not base:
        return ours
    stylized = is_stylized(f"{manifest.style.theme} {manifest.style.prompt}")
    terms = [t.strip() for t in re.split(r"[，,]", base)
             if t.strip() and not (stylized and t.strip() in _STYLE_TERMS)]
    parts = ["，".join(terms), "" if stylized else _ANTI_CGI, ours]
    return ", ".join(part for part in parts if part)


def shot_params(manifest: ReconstructionManifest, shot: ShotSpec,
                profile: RenderProfile) -> dict[str, Any]:
    """Semantic parameters for the workflow compiler (never node IDs)."""
    o = shot.overrides
    style = float(o.get("style_strength", manifest.style.strength))
    identity = float(o.get("identity_strength", manifest.identity.strength))
    scale = float(o.get("resolution_scale", 1.0))
    fps = float(o.get("fps", manifest.video.fps))
    box_w, box_h = fit_within(manifest.video.width, manifest.video.height, profile.max_width,
                              profile.max_height, max_pixels=profile.max_pixels)
    width, height = fit_within(int(manifest.video.width * scale),
                               int(manifest.video.height * scale), box_w, box_h)
    # A degraded profile must preserve the whole shot, not silently truncate its tail.
    limit = (profile.max_frames - 1) // profile.frame_multiple * profile.frame_multiple + 1
    fps = min(fps, float(profile.fps), limit / shot.duration)
    output_frames = max(1, round(shot.duration * fps))
    frames = min(limit, math.ceil((output_frames - 1) / profile.frame_multiple)
                 * profile.frame_multiple + 1)
    prompt = shot.prompt or manifest.style.prompt
    extra = str(o.get("prompt_extra", "")).strip()
    if extra:  # a direction added while the video renders (commands.adjust_remaining_shots)
        prompt = f"{prompt}, {extra}" if prompt else extra
    params: dict[str, Any] = {
        "STYLE_PROMPT": prompt,
        "NEGATIVE_PROMPT": negative_prompt(profile.negative_base, manifest),
        "SEED": int(o.get("seed", shot.seed)),
        "WIDTH": width,
        "HEIGHT": height,
        "FPS": fps,
        "FRAME_COUNT": frames,
        "STEPS": int(o.get("steps", profile.steps)),
        "CFG": float(o.get("cfg", 6.0)),
        "CONTROL_STRENGTH": float(o.get("control_strength", 1.0)),
        "_OUTPUT_FRAMES": output_frames,
        "_REFERENCE_MODE": manifest.identity.reference_mode,
        "DENOISE": round(min(0.95, max(0.2, profile.denoise * style / 0.7)), 3),
        "STYLE_STRENGTH": style,
        "IDENTITY_STRENGTH": identity,
        "POSE_STRENGTH": 1.0 if o.get("pose", shot.controls.pose) else 0.0,
        "DEPTH_STRENGTH": 1.0 if o.get("depth", shot.controls.depth) else 0.0,
        "OFFLOAD": bool(o.get("offload", profile.offload)),
    }
    for key in ("canny_low", "canny_high"):
        if o.get(key) is not None:
            params[key.upper()] = float(o[key])
    if manifest.identity.reference_image:
        params["REFERENCE_IMAGE"] = manifest.identity.reference_image
    return params
