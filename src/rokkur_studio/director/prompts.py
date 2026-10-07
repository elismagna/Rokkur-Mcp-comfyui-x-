"""Prompt compiler: brief + framing + asset tracker -> one diffusion prompt per shot, and the
Batch Prompt Schedule export.

Prompt order (Rule of Nouns, left to right):

    <prefix>, <weighted shot size>, <weighted angle>, <movement>, <subject block>,
    <lighting>, <background>, <theme>, <style>, <extra detail>, <global style modifiers>

The subject block is the character anchor followed by the shot's physical state, never split
by other terms. Only the two framing terms carry weights.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rokkur_studio.agents.schemas import CreativeBrief, DirectorNotes, ShotPlan
from rokkur_studio.director.assets import AssetTracker
from rokkur_studio.director.rules import (
    clean_phrase,
    dedupe_terms,
    narrative_leftovers,
    strip_actions,
    tidy,
    weighted,
)
from rokkur_studio.director.vocabulary import PHRASES


@dataclass(frozen=True)
class Weights:
    framing: float = 1.3
    angle: float = 1.25


def subject_block(anchor: str | None, state: str) -> str:
    return ", ".join(p for p in (clean_phrase(anchor), strip_actions(state)) if p)


def shot_prompt(shot: ShotPlan, *, prefix: str, anchor: str | None, style_terms: Sequence[str],
                weights: Weights) -> str:
    framing = [
        weighted(PHRASES[shot.shot_size], weights.framing) if shot.shot_size else "",
        weighted(PHRASES[shot.camera_angle], weights.angle) if shot.camera_angle else "",
        PHRASES[shot.camera_movement] if shot.camera_movement else "",
    ]
    parts = [clean_phrase(prefix), *framing, subject_block(anchor, shot.subject),
             PHRASES[shot.lighting] if shot.lighting else "", strip_actions(shot.background),
             *(clean_phrase(t) for t in style_terms)]
    return dedupe_terms(tidy(", ".join(p for p in parts if p)))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z-]+", text.lower()))


def merge_negatives(parts: Sequence[str | None], *, wanted: str = "") -> tuple[str, list[str]]:
    """Merged negative prompt, minus any term the theme explicitly asks for.

    A global negative such as "cartoon" would fight a "cartoon noir" theme, so terms whose
    words all appear in ``wanted`` are dropped and reported.
    """
    merged = dedupe_terms(tidy(", ".join(clean_phrase(p) for p in parts if p)))
    want = _words(wanted)
    kept: list[str] = []
    dropped: list[str] = []
    for term in (t for t in merged.split(", ") if t):
        words = _words(term)
        (dropped if words and words <= want else kept).append(term)
    return ", ".join(kept), dropped


def look(tracker: AssetTracker, *, theme: str, style: str = "", extra: str = "",
         global_look: bool = True) -> tuple[str, list[str]]:
    """(prefix, style terms) for a video: its theme and style, then the global look."""
    terms = [theme, style, extra]
    if global_look:
        terms.append(tracker.style_modifiers)
    return (tracker.prompt_prefix if global_look else ""), terms


def compile_brief(brief: CreativeBrief, *, creative_input: dict[str, Any],
                  tracker: AssetTracker, anchor: str | None, character_key: str | None,
                  weights: Weights, notes: DirectorNotes | None = None) -> CreativeBrief:
    """Fill every shot's ``prompt`` and the merged negative prompt (pure; returns a copy)."""
    notes = (notes or DirectorNotes()).model_copy(deep=True)
    global_look = bool(creative_input.get("use_global_look", True))
    prefix, style_terms = look(tracker, theme=brief.theme, style=brief.style,
                               extra=creative_input.get("prompt") or "", global_look=global_look)
    out = brief.model_copy(deep=True)
    if anchor:
        out.character = clean_phrase(anchor)
    for shot in out.shot_plan:
        shot.prompt = shot_prompt(shot, prefix=prefix, anchor=anchor, style_terms=style_terms,
                                  weights=weights)
        left = narrative_leftovers(shot.prompt)
        if left:
            notes.warnings.append(f"{shot.shot_id}: story phrasing left in prompt: "
                                  f"{', '.join(left)}")
    wanted = " ".join([brief.theme, brief.style, creative_input.get("prompt") or ""])
    out.negative_prompt, dropped = merge_negatives(
        [tracker.negative_prompt if global_look else None, brief.negative_prompt], wanted=wanted)
    if dropped:
        notes.warnings.append("left out of the negative prompt because the theme asks for it: "
                              + ", ".join(dropped))
    notes.global_look = global_look
    notes.character_key = character_key
    notes.weights = {"framing": weights.framing, "angle": weights.angle}
    out.director = notes
    return out


def preview(tracker: AssetTracker, *, theme: str = "", subject: str = "", background: str = "",
            shot_size: str | None = None, camera_angle: str | None = None,
            camera_movement: str | None = None, lighting: str | None = None,
            character_key: str | None = None, global_look: bool = True,
            weights: Weights | None = None) -> dict[str, Any]:
    """Compile one prompt from hand-picked parts (the Director page's try-it form)."""
    shot = ShotPlan.model_validate({
        "shot_id": "preview", "start": 0, "end": 1, "subject": subject,
        "background": background, "shot_size": shot_size or None,
        "camera_angle": camera_angle or None, "camera_movement": camera_movement or None,
        "lighting": lighting or None})
    anchor = tracker.character(character_key)
    prefix, terms = look(tracker, theme=theme, global_look=global_look)
    negative, dropped = merge_negatives([tracker.negative_prompt if global_look else None],
                                        wanted=theme)
    return {"prompt": shot_prompt(shot, prefix=prefix, anchor=anchor, style_terms=terms,
                                  weights=weights or Weights()),
            "negative": negative, "subject_state": strip_actions(subject),
            "dropped_negatives": dropped,
            "character": anchor, "unknown_character": bool(character_key and not anchor)}


# -- Batch Prompt Schedule export ---------------------------------------------------------
@dataclass
class Schedule:
    text: str
    fps: int
    interval: int
    max_frames: int
    negative: str
    keyframes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "fps": self.fps, "interval": self.interval,
                "max_frames": self.max_frames, "negative": self.negative,
                "keyframes": self.keyframes}


def build_schedule(shots: Sequence[ShotPlan], negative: str, *, duration: float,
                   fps: int = 24, interval: int = 24, inline_negative: bool = True) -> Schedule:
    """Keyframed prompts for FizzNodes' Batch Prompt Schedule.

    A keyframe every ``interval`` frames (frame 24 = one second at 24 fps), plus one on each
    side of every cut: the node blends prompts between keyframes, so without them a cut would
    fade across up to a second. The text is the node's format: ``"frame": "prompt",`` lines,
    with ``--neg`` separating the negative prompt.
    """
    if not shots:
        raise ValueError("no shots to schedule")
    total = max(1, round(duration * fps))
    bounds = [(s, round(s.start * fps), round(s.end * fps)) for s in shots]
    frames = set(range(0, total, max(1, interval)))
    for _, start, _ in bounds[1:]:
        if 0 < start < total:
            frames.update({start - 1, start})

    def shot_at(frame: int) -> ShotPlan:
        for shot, start, end in bounds:
            if start <= frame < end:
                return shot
        return bounds[-1][0] if frame >= bounds[-1][1] else bounds[0][0]

    lines, keyframes = [], []
    for frame in sorted(frames):
        shot = shot_at(frame)
        text = shot.prompt
        if inline_negative and negative:
            text = f"{text} --neg {negative}"
        lines.append(f'"{frame}": {json.dumps(text, ensure_ascii=False)}')
        keyframes.append({"frame": frame, "shot_id": shot.shot_id})
    return Schedule(text=",\n".join(lines), fps=fps, interval=interval, max_frames=total,
                    negative=negative, keyframes=keyframes)
