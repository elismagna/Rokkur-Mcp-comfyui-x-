"""Agent roles. Each produces a validated schema; deterministic fallbacks keep the pipeline
runnable without a model, and LLM output is constrained by deterministic facts."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, cast, get_args

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
    QcRecommendation,
    RepairAction,
    RepairPlan,
    ShotFraming,
    ShotPlan,
    ShotStory,
    StoryBrief,
)
from rokkur_studio.director.subject import SubjectLock
from rokkur_studio.director.subject import apply as apply_subject
from rokkur_studio.director.subject import choose as choose_subject
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
from rokkur_studio.pipeline.stabilize import LEVELS, resolve_level

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
        "When a person or animal is visible, observed_subject describes them even if a prop "
        "is bigger or brighter; the prop goes in observed_background. "
        "Do not describe edges or instructions as the subject. Do not invent unseen objects. "
        "Keep overlapping objects separate: an occluding hand or prop is not the subject's "
        "anatomy or clothing. Leave uncertain details out instead of copying a guess from "
        "the story pass. If there is no image leave both observed fields empty."
    )

    def __init__(self, provider: AgentProvider, *, vision: bool = False,
                 max_calls: int = 24) -> None:
        self.provider, self.vision, self.max_calls = provider, vision, max_calls
        self.lock = SubjectLock()
        self.warnings: list[str] = []

    @staticmethod
    def rules(brief: CreativeBrief, shot: ShotPlan) -> ShotFraming:
        return ShotFraming(
            shot_size=DEFAULT_SHOT_SIZE,  # type: ignore[arg-type]
            camera_angle=DEFAULT_ANGLE,  # type: ignore[arg-type]
            camera_movement=MOVEMENT_FOR_MOTION.get(shot.motion_type, "Static"),  # type: ignore[arg-type]
            lighting=lighting_for(" ".join([brief.theme, brief.style,  # type: ignore[arg-type]
                                            brief.visual_identity])))

    def run(self, brief: CreativeBrief, keyframes: dict[str, bytes] | None = None, *,
            user_text: str = "") -> tuple[CreativeBrief, str]:
        """A copy of ``brief`` with framing on every shot, and who chose it.

        ``user_text`` is your own description of the scene; it can name the main subject
        (``director.subject``). The chosen subject and any warnings are left on ``lock`` and
        ``warnings``."""
        out = brief.model_copy(deep=True)
        observed: dict[str, str] = {}
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
                    observed[shot.shot_id] = framing.observed_subject.strip()
                if framing.observed_background.strip():
                    shot.background = framing.observed_background.strip()
            previous = framing
        # One main subject for the whole video, so a prop seen first can't take over a shot.
        self.lock = choose_subject(observed, user_text)
        self.warnings = apply_subject(out.shot_plan, observed, self.lock)
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


# -- repair planning ----------------------------------------------------------------------
# What auto-tuning may change, and the value a shot has when nothing set it. cfg to shift and
# control_strength are workflow inputs; stabilize and smooth_control are applied by the studio
# itself (pipeline/stabilize.py), so every renderer accepts them.
TUNING_DEFAULTS: dict[str, float | int | str] = {
    "cfg": 6.0, "steps": 20, "canny_low": 0.2, "canny_high": 0.5, "shift": 8.0,
    "control_strength": 1.0, "stabilize": "auto", "smooth_control": 0.0}
WORKFLOW_PARAMS = {"seed": "SEED", "style_strength": "STYLE_STRENGTH",
                   "identity_strength": "IDENTITY_STRENGTH", "pose": "POSE_STRENGTH",
                   "depth": "DEPTH_STRENGTH", "control_strength": "CONTROL_STRENGTH",
                   "cfg": "CFG", "steps": "STEPS", "canny_low": "CANNY_LOW",
                   "canny_high": "CANNY_HIGH", "shift": "SHIFT"}
STUDIO_APPLIED = frozenset({"stabilize", "smooth_control"})
# Render params (Render.params) back to setting names, to build a shot's ``tried`` list.
_FROM_PARAMS = {"CFG": "cfg", "STEPS": "steps", "CANNY_LOW": "canny_low",
                "CANNY_HIGH": "canny_high", "SHIFT": "shift",
                "CONTROL_STRENGTH": "control_strength", "_STABILIZE": "stabilize",
                "_SMOOTH_CONTROL": "smooth_control"}

# Bounds of every automatic step (docs/stability.md). A value you set outside them is never
# pushed further out.
CFG_STEP_DOWN, CFG_STEP_UP, CFG_MIN, CFG_MAX = 0.5, 1.0, 4.0, 8.0
STEPS_STEP, STEPS_MAX = 6, 32
SMOOTH_STEP, SMOOTH_MAX = 0.3, 0.9
# Calmer guide: 0.2/0.5 reaches the official 0.4/0.8 in two steps, then stops at 0.5/0.9.
CANNY_CALM = {"canny_low": (0.1, 0.4, 0.5), "canny_high": (0.15, 0.8, 0.9)}  # step, official, max
# Layout drift steps raised thresholds back down (step, default, floor).
CANNY_FIRM = {"canny_low": (0.1, 0.2, 0.15), "canny_high": (0.15, 0.5, 0.4)}
STEADY_LOW, STRUCTURE_LOW = 7.0, 5.0  # where QC itself flags flicker and layout drift
REVIEW_LOW = 5.0                      # prompt adherence and anatomy scores
DETAIL_LOW = 4.0                      # a restyle may rightly be somewhat softer than its source
_STEADY_METRICS = ("temporal_consistency", "stability", "flicker")
_OTHER_METRICS = ("structure", "motion", "detail", "prompt_adherence", "hand_body_deformation")
_OTHER_PROBLEMS = ("ADD_POSE_CONTROL", "ADD_DEPTH_CONTROL", "FOLLOW_PROMPT", "FIX_ANATOMY",
                   "MORE_DETAIL", "CALM_EDGES", "RERENDER_SHOT")
_KNOWN_RECS = frozenset(get_args(QcRecommendation))

Setting = float | int | str


def _score(shot: dict[str, Any], key: str) -> float | None:
    """A QC score (0-10, higher is better, as qc.score_shot writes them) when present."""
    value = shot.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _number(value: Any, default: float) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _level_setting(value: Any) -> str:
    """A ``stabilize`` setting as one of LEVELS; unreadable values behave as the default."""
    if isinstance(value, bool):
        return "auto" if value else "off"
    if isinstance(value, str) and value.strip().lower() in LEVELS:
        return value.strip().lower()
    return "auto"


class RepairPlanner:
    """Maps QC findings for failing shots to bounded, recorded parameter changes.

    With ``auto_tune`` (the default) every recommendation and low score moves its setting one
    bounded step and the action records the rule and why (``RepairAction.tuning``). Without it,
    only the seed, the controls QC names and the control strength change, as before.
    """

    def plan(self, qc_report: dict[str, Any], round_: int,
             current: dict[str, dict[str, Any]], *, supported: set[str] | None = None,
             auto_tune: bool = True) -> RepairPlan:
        """Plan one repair round.

        ``current`` maps a shot id to its settings: ``seed``, the style, identity and control
        strengths and, for auto-tuning, the TUNING_DEFAULTS keys (a missing key counts as its
        default) plus an optional ``tried`` list of earlier attempts' settings, so a round never
        repeats a combination. ``supported`` holds the workflow's parameter names, or None for a
        renderer without a workflow (every change applies).
        """
        if not auto_tune:
            return self._plan_fixed(qc_report, round_, current, supported=supported)
        actions = []
        for shot in qc_report.get("shots", []):
            if shot.get("decision") == "PASS":
                continue
            action = self._tune(shot, round_, current.get(shot["shot_id"], {}), supported)
            if action is not None:
                actions.append(action)
        return RepairPlan(round=round_, actions=actions)

    @staticmethod
    def settings(values: Mapping[str, Any]) -> dict[str, Setting]:
        """The tuning settings in ``values``, with defaults for missing or unreadable ones."""
        out: dict[str, Setting] = {}
        for key, default in TUNING_DEFAULTS.items():
            value = values.get(key)
            if key == "stabilize":
                out[key] = _level_setting(value)
            elif key == "steps":
                out[key] = int(_number(value, float(default)))
            else:
                out[key] = round(_number(value, float(default)), 3)
        return out

    @staticmethod
    def signature(values: Mapping[str, Any]) -> tuple[str, ...]:
        """Comparable form of a setting combination; ``auto`` counts as the level it tries."""
        s = RepairPlanner.settings(values)
        s["stabilize"] = resolve_level(s["stabilize"])
        return tuple(f"{k}={s[k]}" for k in TUNING_DEFAULTS)

    @staticmethod
    def settings_from_params(params: Mapping[str, Any]) -> dict[str, Any]:
        """Setting names for a render's params (``Render.params``), for a shot's ``tried`` list."""
        return {name: params[key] for key, name in _FROM_PARAMS.items() if key in params}

    def _tune(self, shot: dict[str, Any], round_: int, cur: dict[str, Any],
              supported: set[str] | None) -> RepairAction | None:
        recs = [r for r in shot.get("recommendations") or [] if r in _KNOWN_RECS] or ["CHANGE_SEED"]
        score = {k: _score(shot, k) for k in (*_STEADY_METRICS, *_OTHER_METRICS)}
        problems = dict.fromkeys(recs, "QC asked")  # rule -> why it fires

        def low(rule: str, key: str, limit: float) -> None:
            value = score[key]
            if value is not None and value < limit:
                note = f"{key.replace('_', ' ')} {value:g} < {limit:g}"
                problems[rule] = f"{problems[rule]}, {note}" if rule in problems else note

        low("STABILIZE", "stability", STEADY_LOW)
        low("STABILIZE", "temporal_consistency", STEADY_LOW)
        low("DEFLICKER", "flicker", STEADY_LOW)
        low("FOLLOW_PROMPT", "prompt_adherence", REVIEW_LOW)
        low("FIX_ANATOMY", "hand_body_deformation", REVIEW_LOW)
        low("MORE_DETAIL", "detail", DETAIL_LOW)
        drift = [why for why in (
            "QC asked for depth control" if "ADD_DEPTH_CONTROL" in recs else "",
            f"structure {score['structure']:g} < {STRUCTURE_LOW:g}"
            if score["structure"] is not None and score["structure"] < STRUCTURE_LOW else "")
            if why]

        # Today's fixed changes come first: a new seed and the controls QC names.
        seed = int(cur.get("seed") or 0)
        changes: dict[str, float | int | str | bool] = {"seed": seed + 7919 * round_}
        if "REDUCE_STYLE_STRENGTH" in recs:
            changes["style_strength"] = round(max(0.3, _number(cur.get("style_strength"), 0.7)
                                                  - 0.15), 3)
        if "INCREASE_IDENTITY" in recs:
            changes["identity_strength"] = round(min(1.0, _number(cur.get("identity_strength"),
                                                                  0.8) + 0.1), 3)
        if "ADD_POSE_CONTROL" in recs:
            changes["pose"] = True
        if "ADD_DEPTH_CONTROL" in recs:
            changes["depth"] = True
        unsupported = ([k for k in changes if WORKFLOW_PARAMS[k] not in supported]
                       if supported is not None else [])
        changes = {k: v for k, v in changes.items() if k not in unsupported}

        settings = self.settings(cur)
        proposed = dict(settings)
        tuning: list[str] = []
        fired: list[str] = []

        def allowed(key: str) -> bool:
            return supported is None or key in STUDIO_APPLIED or WORKFLOW_PARAMS[key] in supported

        def change(rule: str, key: str, value: Setting, why: str) -> None:
            value = round(value, 3) if isinstance(value, float) else value
            if value == proposed[key]:
                return
            if not allowed(key):
                if key not in unsupported:
                    unsupported.append(key)
                return
            tuning.append(f"{rule}: {key} {proposed[key]} -> {value} ({why})")
            proposed[key] = value
            if rule in _KNOWN_RECS and rule not in fired:
                fired.append(rule)

        if supported is not None and "CONTROL_STRENGTH" in supported:
            old = float(settings["control_strength"])
            # Layout/motion drift needs firmer guidance; temporal instability alone gets a
            # gentler constraint. These are bounded, recorded experiments that stay inside
            # 0.7-1.0 and never push a value the user chose further out: on a real shot 1.15
            # raised QC's motion score but made anatomy worse, which QC cannot see
            # (docs/upgrade-2026-10-07.md).
            firm = bool(drift) or "ADD_POSE_CONTROL" in recs
            change("ADJUST_CONTROL_STRENGTH", "control_strength",
                   min(max(old, 1.0), old + 0.15) if firm else max(min(old, 0.7), old - 0.1),
                   "layout or motion drifted: firmer guidance" if firm
                   else "gentler guidance, within 0.7-1.0")

        steady_rule = next((r for r in ("STABILIZE", "DEFLICKER") if r in problems), None)
        prompt_rule = "FOLLOW_PROMPT" in problems
        if steady_rule:
            why = problems[steady_rule]
            level = str(settings["stabilize"])
            if level == "off":
                tuning.append(f"{steady_rule}: the stabilizer stays off, as set ({why})")
            elif resolve_level(level) == "light":
                change(steady_rule, "stabilize", "strong",
                       f"{why}; light only deflickers, strong also calms boiling texture")
            smooth = float(settings["smooth_control"])
            if smooth < SMOOTH_MAX:
                change(steady_rule, "smooth_control", min(SMOOTH_MAX, smooth + SMOOTH_STEP),
                       f"{why}; a time-smoothed source draws a steadier Canny/depth guide")
            cfg = float(settings["cfg"])
            # Lower CFG boils less but follows the prompt less: only when steadiness is the
            # worst problem and nobody asked for more prompt.
            if not prompt_rule and cfg > CFG_MIN and self._steadiness_is_worst(score, problems):
                change(steady_rule, "cfg", max(CFG_MIN, cfg - CFG_STEP_DOWN),
                       f"{why}; steadiness is the worst problem and lower guidance boils less")

        if "CALM_EDGES" in problems and drift:
            tuning.append("CALM_EDGES: the Canny thresholds stay as they are, because the layout "
                          f"also drifted ({'; '.join(drift)})")
        elif "CALM_EDGES" in problems:
            for key, (step, official, top) in CANNY_CALM.items():
                old = float(settings[key])
                if old < top:  # a value you set above the cap is left alone
                    change("CALM_EDGES", key, min(top, old + step),
                           f"{problems['CALM_EDGES']}; fewer texture edges (official "
                           f"{official:g}, never above {top:g})")
        elif drift:
            for key, (step, default, floor) in CANNY_FIRM.items():
                old = float(settings[key])
                if old > default:  # only thresholds an earlier round (or you) raised
                    change("layout drift", key, max(floor, old - step),
                           f"{'; '.join(drift)}; more guide edges hold the layout")

        if prompt_rule:
            cfg = float(settings["cfg"])
            if cfg < CFG_MAX:
                change("FOLLOW_PROMPT", "cfg", min(CFG_MAX, cfg + CFG_STEP_UP),
                       f"{problems['FOLLOW_PROMPT']}; stronger prompt guidance")

        detail_rules = [r for r in ("FIX_ANATOMY", "MORE_DETAIL") if r in problems]
        if detail_rules and int(settings["steps"]) < STEPS_MAX:
            change(detail_rules[0], "steps", min(STEPS_MAX, int(settings["steps"]) + STEPS_STEP),
                   f"{'; '.join(problems[r] for r in detail_rules)}; more sampling steps")
            if proposed["steps"] != settings["steps"]:
                fired += [r for r in detail_rules if r not in fired]

        tried = {self.signature(t) for t in cur.get("tried") or [] if isinstance(t, Mapping)}
        if self.signature(proposed) in tried:
            for key, value in self._untried_moves(proposed, prompt_rule):
                if allowed(key) and self.signature({**proposed, key: value}) not in tried:
                    change("UNTRIED", key, value, "these settings were already tried in an "
                           "earlier round")
                    break
            else:
                tuning.append("UNTRIED: every nearby setting was tried before; only the seed "
                              "changes")

        changes.update({k: v for k, v in proposed.items() if v != settings[k]})
        if "seed" in changes:
            tuning.insert(0, f"CHANGE_SEED: seed {seed} -> {changes['seed']} "
                          "(every repair round rerolls)")
        if supported is None:
            recs = list(dict.fromkeys([*recs, *fired]))
        else:
            if not changes:
                return None
            named = [r for r, k in (("REDUCE_STYLE_STRENGTH", "style_strength"),
                                    ("INCREASE_IDENTITY", "identity_strength"),
                                    ("ADD_POSE_CONTROL", "pose"),
                                    ("ADD_DEPTH_CONTROL", "depth")) if k in changes]
            recs = list(dict.fromkeys(
                (["ADJUST_CONTROL_STRENGTH"] if "control_strength" in changes else []) + named
                + fired + (["CHANGE_SEED"] if "seed" in changes else [])))
        return RepairAction(shot_id=shot["shot_id"],
                            recommendations=cast(list[QcRecommendation], recs),
                            changes=changes, unsupported=unsupported,
                            reason="; ".join(shot.get("issues", [])), tuning=tuning)

    @staticmethod
    def _steadiness_is_worst(score: dict[str, float | None], problems: dict[str, str]) -> bool:
        steady = [v for k in _STEADY_METRICS if (v := score[k]) is not None]
        others = [v for k in _OTHER_METRICS if (v := score[k]) is not None]
        if steady:
            return all(min(steady) <= o for o in others)
        return not any(r in problems for r in _OTHER_PROBLEMS)

    @staticmethod
    def _untried_moves(s: dict[str, Setting], prompt_rule: bool) -> list[tuple[str, Setting]]:
        """Single extra steps that are safe for any problem, in the order they are tried."""
        moves: list[tuple[str, Setting]] = []
        if s["stabilize"] != "off" and resolve_level(s["stabilize"]) == "light":
            moves.append(("stabilize", "strong"))
        if float(s["smooth_control"]) < SMOOTH_MAX:
            moves.append(("smooth_control",
                          round(min(SMOOTH_MAX, float(s["smooth_control"]) + SMOOTH_STEP), 3)))
        if int(s["steps"]) < STEPS_MAX:
            moves.append(("steps", min(STEPS_MAX, int(s["steps"]) + STEPS_STEP)))
        if float(s["cfg"]) > CFG_MIN and not prompt_rule:
            moves.append(("cfg", round(max(CFG_MIN, float(s["cfg"]) - CFG_STEP_DOWN), 3)))
        return moves

    def _plan_fixed(self, qc_report: dict[str, Any], round_: int,
                    current: dict[str, dict[str, Any]], *,
                    supported: set[str] | None = None) -> RepairPlan:
        """The planner before auto-tuning: a new seed, the controls QC names and control
        strength within 0.7-1.0."""
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
