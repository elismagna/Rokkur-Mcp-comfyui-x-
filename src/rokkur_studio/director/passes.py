"""The director loop: story pass -> cinematography pass -> prompt compiler."""

from __future__ import annotations

from typing import Any

from rokkur_studio.agents.providers import AgentProvider
from rokkur_studio.agents.roles import CreativeDirector, DirectorOfPhotography
from rokkur_studio.agents.schemas import CreativeBrief, DirectorNotes
from rokkur_studio.config import DirectorSection
from rokkur_studio.director.assets import AssetTracker, resolve_character
from rokkur_studio.director.prompts import Weights, compile_brief


def direct(provider: AgentProvider, creative_input: dict[str, Any], analysis: dict[str, Any],
           target_format: str, *, tracker: AssetTracker, settings: DirectorSection,
           dp_provider: AgentProvider | None = None,
           keyframes: dict[str, bytes] | None = None) -> tuple[CreativeBrief, str]:
    """Run every pass and return the finished brief and who wrote its story.

    Pass 1 (Creative Director) writes the brief and each shot's subject as physical states.
    Pass 2 (Director of Photography) picks framing and lighting from the allowed vocabulary,
    looking at each shot's middle frame when the model can read images. Pass 3 is
    deterministic: the action strip-out, the Rule of Nouns order, the framing weights, the
    character anchor and the global look. Each model pass falls back to rules on its own.
    """
    key, anchor, warning = resolve_character(creative_input, tracker)
    story_images = (keyframes if settings.vision == "auto" and keyframes
                    and provider.supports_images() else None)
    brief, story_by = CreativeDirector(provider).run(creative_input, analysis, target_format,
                                                     character=anchor, keyframes=story_images)
    notes = DirectorNotes(story_by=story_by, character_key=key,
                          warnings=[warning] if warning else [])
    if not settings.enabled:
        if creative_input.get("negative_prompt"):
            brief.negative_prompt = creative_input["negative_prompt"]
        brief.director = notes
        return brief, story_by
    dp = dp_provider or provider
    vision = settings.vision == "auto" and bool(keyframes) and dp.supports_images()
    framed, framing_by = DirectorOfPhotography(
        dp, vision=vision, max_calls=settings.max_framing_calls).run(brief, keyframes)
    notes.framing_by = framing_by
    notes.vision = any((s.framing_by or "").endswith("+vision") for s in framed.shot_plan)
    if not notes.vision and not story_images:
        notes.warnings.append("Source images were not read by a vision model; check the shot descriptions.")
    weights = Weights(framing=settings.framing_weight, angle=settings.angle_weight)
    return compile_brief(framed, creative_input=creative_input, tracker=tracker, anchor=anchor,
                         character_key=key, weights=weights, notes=notes), story_by
