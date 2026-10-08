"""Agent roles. Each produces a validated schema; deterministic fallbacks keep the pipeline
runnable without a model, and LLM output is constrained by deterministic facts."""

from __future__ import annotations

import logging
import re
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
    ObservedShotFraming,
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


def _shot_number(shot_id: str) -> int | None:
    m = re.search(r"(\d+)\D*$", shot_id)
    return int(m.group(1)) if m else None


def _match_story(shots: list[ShotPlan], plans: list[ShotStory],
                 role: str) -> list[ShotStory | None]:
    """The model's story for each analysed shot: by id, then by shot number ("1" for
    shot_001), then, only if no id matched at all and the counts agree, by position."""
    by_id = {p.shot_id: p for p in plans}
    if len(by_id) != len(plans):
        raise AgentOutputError(role, "duplicate shot ids")
    by_number = {n: p for p in plans if (n := _shot_number(p.shot_id)) is not None}

    def find(shot_id: str) -> ShotStory | None:
        n = _shot_number(shot_id)
        return by_id.get(shot_id) or (by_number.get(n) if n is not None else None)

    matched = [find(s.shot_id) for s in shots]
    if not any(matched) and len(plans) == len(shots):
        log.warning("creative director: shot ids did not match; merged by position")
        return list(plans)
    return matched


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
        "nouns. Use commas between details, never full stops. Preserve the source subject, "
        "setting and action unless the user requests a replacement. If source images are "
        "attached, they correspond in order to image_shot_ids. Describe what is visible; "
        "do not replace a concrete animal or person with a generic object. If an image is "
        "unavailable do not claim to have seen it."
    )
    anchor_note = (
        " The main character is fixed by character_anchor and is added to every shot "
        "automatically: do not restate their clothing or appearance in subject, give only "
        "their pose and expression."
    )

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    def run(self, creative_input: dict[str, Any], analysis: dict[str, Any],
            target_format: str, *, character: str | None = None,
            keyframes: dict[str, bytes] | None = None) -> tuple[CreativeBrief, str]:
        shots = [ShotPlan(shot_id=s["shot_id"], start=s["start"], end=s["end"],
                          motion_type=s.get("motion_type", "unknown"),
                          camera=s.get("camera", "unknown")) for s in analysis["shots"]]
        if not isinstance(self.provider, RuleBasedProvider):
            try:
                payload: dict[str, Any] = {"creative_input": creative_input,
                                           "analysis": analysis, "target_format": target_format}
                if character:
                    payload["character_anchor"] = character
                frames = list((keyframes or {}).items())[:4]
                if frames:
                    payload["image_shot_ids"] = [key for key, _ in frames]
                story = self.provider.generate(
                    self.role, self.instructions + (self.anchor_note if character else ""),
                    payload, StoryBrief, images=[value for _, value in frames] or None)
                # Facts win over the model: shot timing comes from the analysis.
                matched = _match_story(shots, story.shot_plan, self.role)
                planned = [m or ShotStory(**s.model_dump(include=set(ShotStory.model_fields)))
                           for s, m in zip(shots, matched, strict=True)]
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
        "for a change, and fit it to the theme. When an image is attached, also return "
        "observed_subject (specific visible subject, appearance, pose and expression) and "
        "observed_background (visible surroundings), as short concrete noun phrases. "
        "Do not describe edges or instructions as the subject. Do not invent unseen objects. "
        "Keep overlapping objects separate: an occluding hand or prop is not the subject's "
        "anatomy or clothing. Leave uncertain details out instead of copying a guess from "
        "the story pass. If there is no image leave both observed fields empty."
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
                payload: dict[str, Any] = {
                    "theme": brief.theme, "style": brief.style,
                    "visual_identity": brief.visual_identity,
                    "shot": {"shot_id": shot.shot_id, "number": f"{n} of {len(out.shot_plan)}",
                             "seconds": round(shot.end - shot.start, 1), "intent": shot.intent,
                             "subject": shot.subject, "source_motion": shot.motion_type},
                    "previous_shot": previous.model_dump() if previous else None,
                    "image": "attached: middle frame of this shot" if image else "none",
                    "allowed": allowed,
                }
                if image:
                    # An earlier model's guessed anatomy can bias the visual pass into
                    # repeating it. Let the image supply subject/pose independently.
                    payload["shot"].pop("subject")
                    payload["shot"].pop("intent")
                    if previous is not None:
                        payload["previous_shot"] = previous.model_dump(
                            exclude={"observed_subject", "observed_background"})
                calls += 1
                try:
                    framing = self.provider.generate(self.role, self.instructions, payload,
                                                     ObservedShotFraming if image else ShotFraming,
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
            if by.endswith("+vision"):
                # With a character anchor the story's subject is only pose and expression; the
                # observed appearance (the source's actor or animal) would fight the anchor.
                if framing.observed_subject.strip() and not brief.character:
                    shot.subject = framing.observed_subject.strip()
                if framing.observed_background.strip():
                    shot.background = framing.observed_background.strip()
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
             current: dict[str, dict[str, Any]], *, supported: set[str] | None = None) -> RepairPlan:
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
            unsupported = []
            if supported is not None:
                mappings = {"seed": "SEED", "style_strength": "STYLE_STRENGTH",
                            "identity_strength": "IDENTITY_STRENGTH",
                            "pose": "POSE_STRENGTH", "depth": "DEPTH_STRENGTH"}
                unsupported = [k for k in changes if mappings[k] not in supported]
                changes = {k: v for k, v in changes.items() if k not in unsupported}
                if "CONTROL_STRENGTH" in supported:
                    old = float(cur.get("control_strength", 1.0))
                    # Layout/motion drift needs firmer guidance; temporal instability alone
                    # gets a gentler constraint. These are bounded, recorded experiments that
                    # stay inside 0.7-1.0 and never push a value the user chose further out:
                    # on a real shot 1.15 raised QC's motion score but made anatomy worse,
                    # which QC cannot see (docs/upgrade-2026-10-07.md).
                    drift = any(r in recs for r in ("ADD_POSE_CONTROL", "ADD_DEPTH_CONTROL"))
                    new = round(min(max(old, 1.0), old + 0.15) if drift
                                else max(min(old, 0.7), old - 0.1), 3)
                    if new != old:
                        changes["control_strength"] = new
                recs = (["ADJUST_CONTROL_STRENGTH"] if "control_strength" in changes else []) + (
                    ["CHANGE_SEED"] if "seed" in changes else [])
                if not changes:
                    continue
            actions.append(RepairAction(shot_id=shot["shot_id"], recommendations=recs,
                                        changes=changes, unsupported=unsupported,
                                        reason="; ".join(shot.get("issues", []))))
        return RepairPlan(round=round_, actions=actions)
