"""Agent roles. Each produces a validated schema; deterministic fallbacks keep the pipeline
runnable without a model, and LLM output is constrained by deterministic facts."""

from __future__ import annotations

import logging
from typing import Any

from rokkur_studio.agents.providers import (
    AgentOutputError,
    AgentProvider,
    AgentUnavailable,
    RuleBasedProvider,
)
from rokkur_studio.agents.schemas import (
    CreativeBrief,
    MetadataDraft,
    RepairAction,
    RepairPlan,
    ShotFraming,
    ShotPlan,
    ShotStory,
    StoryBrief,
)
from rokkur_studio.director.vocabulary import (
    CAMERA_ANGLES,
    CAMERA_MOVEMENTS,
    DEFAULT_ANGLE,
    DEFAULT_SHOT_SIZE,
    LIGHTING_STYLES,
    MOVEMENT_FOR_MOTION,
    SHOT_SIZES,
    lighting_for,
)

log = logging.getLogger(__name__)

_ASPECT = {"youtube_short": "9:16", "youtube_video": "16:9"}


class CreativeDirector:
    """Story pass: turns the user's creative input + video analysis into a production brief,
    with each shot's subject written as concrete physical states."""

    role = "creative_director"
    instructions = (
        "Turn the creative input and the source analysis into a production brief. Keep the "
        "shot boundaries from the analysis; describe style, theme, prompt strategy, identity "
        "and background requirements. For every shot in shot_plan give: intent, one sentence "
        "on what the shot should look like in the new style; subject, the subject as concrete "
        "visible states with nouns first (pose, expression, wardrobe details), never actions "
        "or story: write 'in mid-stride, head turned sharply, focused expression', not 'is "
        "walking', 'suddenly runs' or 'thinking about home'; background, the setting as "
        "nouns. Use commas between details, never full stops."
    )
    anchor_note = (
        " The main character is fixed by character_anchor and is added to every shot "
        "automatically: do not restate their clothing or appearance in subject, give only "
        "their pose and expression."
    )

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    def run(self, creative_input: dict[str, Any], analysis: dict[str, Any],
            target_format: str, *, character: str | None = None) -> tuple[CreativeBrief, str]:
        shots = [ShotPlan(shot_id=s["shot_id"], start=s["start"], end=s["end"],
                          motion_type=s.get("motion_type", "unknown"),
                          camera=s.get("camera", "unknown")) for s in analysis["shots"]]
        if not isinstance(self.provider, RuleBasedProvider):
            try:
                payload: dict[str, Any] = {"creative_input": creative_input,
                                           "analysis": analysis, "target_format": target_format}
                if character:
                    payload["character_anchor"] = character
                story = self.provider.generate(
                    self.role, self.instructions + (self.anchor_note if character else ""),
                    payload, StoryBrief)
                # Facts win over the model: shot timing comes from the analysis.
                planned = story.shot_plan + [ShotStory(**s.model_dump(include=set(
                    ShotStory.model_fields))) for s in shots[len(story.shot_plan):]]
                merged = [s.model_copy(update={"intent": b.intent, "subject": b.subject,
                                               "background": b.background})
                          for s, b in zip(shots, planned, strict=False)]
                brief = CreativeBrief(**story.model_dump(exclude={"shot_plan"}),
                                      shot_plan=merged)
                brief.target_duration = analysis["duration"]
                if character:
                    brief.character = character
                return brief, self.provider.name
            except (AgentUnavailable, AgentOutputError) as exc:
                # A model that is down or keeps returning bad JSON must not stall a render.
                log.warning("creative director falling back to rules", extra={"data": {
                    "reason": str(exc)}})
        return self._rules(creative_input, analysis, target_format, shots,
                           character), "rule_based"

    @staticmethod
    def _rules(ci: dict[str, Any], analysis: dict[str, Any], target_format: str,
               shots: list[ShotPlan], character: str | None = None) -> CreativeBrief:
        theme = ci.get("theme") or "cinematic stylised animation"
        character = character or ci.get("character_description")
        prompt_parts = [ci.get("prompt") or theme, theme]
        if character:
            prompt_parts.append(character)
        prompt_parts += ["consistent character", "coherent motion", "high detail"]
        prompt = ", ".join(dict.fromkeys(p for p in prompt_parts if p))
        return CreativeBrief(
            style=ci.get("style", theme),
            theme=theme,
            character=character,
            visual_identity=f"{theme}; consistent palette and lighting across shots",
            prompt=prompt,
            style_strength=float(ci.get("style_strength", 0.7)),
            identity_strength=float(ci.get("identity_strength", 0.8)),
            motion_preservation="strict",
            identity_requirements="match the reference character" if ci.get(
                "character_reference_asset") else "",
            background_requirements="stylise background consistently with the theme",
            target_duration=analysis["duration"],
            target_aspect_ratio=_ASPECT.get(target_format, "9:16"),  # type: ignore[arg-type]
            shot_plan=shots,
            rationale="rule-based brief derived from user creative input and shot analysis",
        )


class DirectorOfPhotography:
    """Cinematography pass: one shot size, angle, movement and lighting per shot, chosen only
    from the allowed vocabulary (director.vocabulary). With a vision-capable model it sees the
    middle frame of each source shot, so the words match the composition the render keeps."""

    role = "director_of_photography"
    instructions = (
        "Choose the cinematography for one shot of a short restyled video. Pick exactly one "
        "term from each allowed list for shot_size, camera_angle, camera_movement and "
        "lighting, copied exactly. If an image is attached it is the middle frame of this "
        "shot in the source video, and the render keeps that composition: choose the shot "
        "size and angle you see in it. Base camera_movement on source_motion (a static source "
        "is Static). Keep lighting consistent with the previous shot unless the intent calls "
        "for a change, and fit it to the theme."
    )

    def __init__(self, provider: AgentProvider, *, vision: bool = False,
                 max_calls: int = 24) -> None:
        self.provider, self.vision, self.max_calls = provider, vision, max_calls

    @staticmethod
    def rules(brief: CreativeBrief, shot: ShotPlan) -> ShotFraming:
        return ShotFraming(
            shot_size=DEFAULT_SHOT_SIZE,  # type: ignore[arg-type]
            camera_angle=DEFAULT_ANGLE,  # type: ignore[arg-type]
            camera_movement=MOVEMENT_FOR_MOTION.get(shot.motion_type, "Static"),  # type: ignore[arg-type]
            lighting=lighting_for(" ".join([brief.theme, brief.style,  # type: ignore[arg-type]
                                            brief.visual_identity])))

    def run(self, brief: CreativeBrief, keyframes: dict[str, bytes] | None = None
            ) -> tuple[CreativeBrief, str]:
        """A copy of ``brief`` with framing on every shot, and who chose it."""
        out = brief.model_copy(deep=True)
        use_model = not isinstance(self.provider, RuleBasedProvider)
        calls = 0
        previous: ShotFraming | None = None
        allowed = {"shot_size": SHOT_SIZES, "camera_angle": CAMERA_ANGLES,
                   "camera_movement": CAMERA_MOVEMENTS, "lighting": LIGHTING_STYLES}
        for n, shot in enumerate(out.shot_plan, 1):
            framing: ShotFraming | None = None
            by = "rule_based"
            if use_model and calls < self.max_calls:
                image = (keyframes or {}).get(shot.shot_id) if self.vision else None
                payload = {
                    "theme": brief.theme, "style": brief.style,
                    "visual_identity": brief.visual_identity,
                    "shot": {"shot_id": shot.shot_id, "number": f"{n} of {len(out.shot_plan)}",
                             "seconds": round(shot.end - shot.start, 1), "intent": shot.intent,
                             "subject": shot.subject, "source_motion": shot.motion_type},
                    "previous_shot": previous.model_dump() if previous else None,
                    "image": "attached: middle frame of this shot" if image else "none",
                    "allowed": allowed,
                }
                calls += 1
                try:
                    framing = self.provider.generate(self.role, self.instructions, payload,
                                                     ShotFraming,
                                                     images=[image] if image else None)
                    by = self.provider.name + ("+vision" if image else "")
                except AgentUnavailable as exc:
                    use_model = False  # down: do not wait on it for every remaining shot
                    log.warning("director of photography falling back to rules",
                                extra={"data": {"reason": str(exc)}})
                except AgentOutputError as exc:
                    log.warning("director of photography: bad framing for one shot",
                                extra={"data": {"shot": shot.shot_id, "reason": str(exc)}})
            if framing is None:
                framing = self.rules(out, shot)
            shot.shot_size, shot.camera_angle = framing.shot_size, framing.camera_angle
            shot.camera_movement, shot.lighting = framing.camera_movement, framing.lighting
            shot.framing_by = by
            previous = framing
        sources = {s.framing_by for s in out.shot_plan}
        return out, (sources.pop() if len(sources) == 1 else "mixed") or "rule_based"


class ChannelManager:
    """Writes the YouTube title, description and tags for a finished video.

    The LLM only drafts wording; ``services.publishing`` adds the fixed parts (``#shorts``,
    AI disclosure, source attribution) and validates lengths, so a model cannot drop them.
    """

    role = "channel_manager"
    instructions = (
        "Write YouTube metadata for a short AI-stylised video made from the described source. "
        "Title: under 70 characters, specific, no clickbait, no emoji, no hashtags. Description: "
        "two or three plain sentences about what the viewer sees and the style, no links, no "
        "hashtags, no mention of being made for kids. Tags: 5 to 12 short lowercase phrases."
    )

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    def run(self, creative_input: dict[str, Any], brief: dict[str, Any],
            target_format: str, duration: float) -> tuple[MetadataDraft | None, str]:
        """None means: no model, use the rule-based metadata unchanged."""
        if isinstance(self.provider, RuleBasedProvider):
            return None, "rule_based"
        try:
            draft = self.provider.generate(self.role, self.instructions, {
                "creative_input": creative_input,
                "brief": {k: brief.get(k) for k in ("theme", "style", "prompt", "character",
                                                    "visual_identity")},
                "target_format": target_format, "duration_seconds": round(duration, 1),
            }, MetadataDraft)
        except (AgentUnavailable, AgentOutputError) as exc:
            log.warning("channel manager falling back to rules", extra={"data": {
                "reason": str(exc)}})
            return None, "rule_based"
        return draft, self.provider.name


class RepairPlanner:
    """Maps QC findings for failing shots to minimal parameter changes."""

    def plan(self, qc_report: dict[str, Any], round_: int,
             current: dict[str, dict[str, Any]]) -> RepairPlan:
        actions = []
        for shot in qc_report.get("shots", []):
            if shot.get("decision") == "PASS":
                continue
            recs = list(shot.get("recommendations") or ["CHANGE_SEED"])
            cur = current.get(shot["shot_id"], {})
            changes: dict[str, float | int | str | bool] = {
                "seed": int(cur.get("seed", 0)) + 7919 * round_}
            if "REDUCE_STYLE_STRENGTH" in recs:
                changes["style_strength"] = round(max(0.3, float(cur.get("style_strength",
                                                                          0.7)) - 0.15), 3)
            if "INCREASE_IDENTITY" in recs:
                changes["identity_strength"] = round(min(1.0, float(cur.get(
                    "identity_strength", 0.8)) + 0.1), 3)
            if "ADD_POSE_CONTROL" in recs:
                changes["pose"] = True
            if "ADD_DEPTH_CONTROL" in recs:
                changes["depth"] = True
            actions.append(RepairAction(shot_id=shot["shot_id"], recommendations=recs,
                                        changes=changes, reason="; ".join(shot.get("issues", []))))
        return RepairPlan(round=round_, actions=actions)
