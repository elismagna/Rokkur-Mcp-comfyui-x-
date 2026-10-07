"""Build a reconstruction manifest and per-shot semantic render parameters."""

from __future__ import annotations

import hashlib
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


def fit_within(width: int, height: int, max_w: int, max_h: int, multiple: int = 8) -> tuple[int, int]:
    """Scale (w, h) to fit (max_w, max_h) keeping aspect; round down to ``multiple``."""
    scale = min(1.0, max_w / width, max_h / height)
    w = max(multiple, int(width * scale) // multiple * multiple)
    h = max(multiple, int(height * scale) // multiple * multiple)
    return w, h


def shot_seed(project_id: str, shot_id: str) -> int:
    return int(hashlib.sha256(f"{project_id}:{shot_id}".encode()).hexdigest()[:8], 16)


def build_manifest(*, project_id: str, source_asset: str, analysis: dict[str, Any],
                   brief: CreativeBrief, profile: RenderProfile, target_format: str,
                   reference_image: str | None = None) -> ReconstructionManifest:
    width, height = fit_within(analysis["width"], analysis["height"], profile.max_width,
                               profile.max_height)
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


def shot_params(manifest: ReconstructionManifest, shot: ShotSpec,
                profile: RenderProfile) -> dict[str, Any]:
    """Semantic parameters for the workflow compiler (never node IDs)."""
    o = shot.overrides
    style = float(o.get("style_strength", manifest.style.strength))
    identity = float(o.get("identity_strength", manifest.identity.strength))
    scale = float(o.get("resolution_scale", 1.0))
    fps = float(o.get("fps", manifest.video.fps))
    width, height = fit_within(int(manifest.video.width * scale),
                               int(manifest.video.height * scale),
                               manifest.video.width, manifest.video.height)
    frames = min(profile.max_frames, max(1, round(shot.duration * fps)))
    params: dict[str, Any] = {
        "STYLE_PROMPT": manifest.style.prompt,
        "NEGATIVE_PROMPT": manifest.style.negative_prompt,
        "SEED": int(o.get("seed", shot.seed)),
        "WIDTH": width,
        "HEIGHT": height,
        "FPS": fps,
        "FRAME_COUNT": frames,
        "STEPS": profile.steps,
        "DENOISE": round(min(0.95, max(0.2, profile.denoise * style / 0.7)), 3),
        "STYLE_STRENGTH": style,
        "IDENTITY_STRENGTH": identity,
        "POSE_STRENGTH": 1.0 if o.get("pose", shot.controls.pose) else 0.0,
        "DEPTH_STRENGTH": 1.0 if o.get("depth", shot.controls.depth) else 0.0,
        "OFFLOAD": bool(o.get("offload", profile.offload)),
    }
    if manifest.identity.reference_image:
        params["REFERENCE_IMAGE"] = manifest.identity.reference_image
    return params
