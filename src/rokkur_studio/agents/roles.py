"""Agent roles. Each produces a validated schema; deterministic fallbacks keep the pipeline
runnable without a model, and LLM output is constrained by deterministic facts."""

from __future__ import annotations

import logging
from typing import Any

from rokkur_studio.agents.providers import AgentProvider, AgentUnavailable, RuleBasedProvider
from rokkur_studio.agents.schemas import CreativeBrief, RepairAction, RepairPlan, ShotPlan

log = logging.getLogger(__name__)

_ASPECT = {"youtube_short": "9:16", "youtube_video": "16:9"}


class CreativeDirector:
    """Turns the user's creative input + video analysis into a production brief."""

    role = "creative_director"
    instructions = (
        "Turn the creative input and the source analysis into a production brief. Keep the "
        "shot boundaries from the analysis; describe style, theme, prompt strategy, identity "
        "and background requirements."
    )

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    def run(self, creative_input: dict[str, Any], analysis: dict[str, Any],
            target_format: str) -> tuple[CreativeBrief, str]:
        shots = [ShotPlan(shot_id=s["shot_id"], start=s["start"], end=s["end"],
                          motion_type=s.get("motion_type", "unknown"),
                          camera=s.get("camera", "unknown")) for s in analysis["shots"]]
        if not isinstance(self.provider, RuleBasedProvider):
            try:
                brief = self.provider.generate(self.role, self.instructions, {
                    "creative_input": creative_input, "analysis": analysis,
                    "target_format": target_format}, CreativeBrief)
                # Facts win over the model: shot timing comes from the analysis.
                brief.shot_plan = [s.model_copy(update={"intent": b.intent}) for s, b in
                                   zip(shots, brief.shot_plan + shots[len(brief.shot_plan):],
                                       strict=False)]
                brief.target_duration = analysis["duration"]
                return brief, self.provider.name
            except AgentUnavailable:
                log.warning("creative director provider unavailable; using rules")
        return self._rules(creative_input, analysis, target_format, shots), "rule_based"

    @staticmethod
    def _rules(ci: dict[str, Any], analysis: dict[str, Any], target_format: str,
               shots: list[ShotPlan]) -> CreativeBrief:
        theme = ci.get("theme") or "cinematic stylised animation"
        character = ci.get("character_description")
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
