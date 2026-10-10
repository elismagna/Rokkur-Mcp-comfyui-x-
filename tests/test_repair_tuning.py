"""RepairPlanner auto-tuning: each rule, its bounds, unsupported controls, never repeating a
setting combination, and the unchanged planner behind ``auto_tune=False``."""
from typing import Any

import pytest

from rokkur_studio.agents.roles import RepairPlanner
from rokkur_studio.agents.schemas import RepairAction, RepairPlan
from rokkur_studio.manifest.schema import ReconstructionManifest
from rokkur_studio.services.projects import latest_document
from tests.test_pipeline import create, run, status

WAN = {"SEED", "CONTROL_STRENGTH", "STEPS", "CFG", "CANNY_LOW", "CANNY_HIGH"}


def shot(recs: list[str] | None = None, **metrics: Any) -> dict[str, Any]:
    return {"shot_id": "a", "decision": "FAIL", "issues": ["test issue"],
            "recommendations": recs or [], **metrics}


def tune(s: dict[str, Any], cur: dict[str, Any] | None = None, *,
         supported: set[str] | None = None, round_: int = 1) -> RepairAction:
    plan = RepairPlanner().plan({"shots": [s]}, round_, {"a": {"seed": 10, **(cur or {})}},
                                supported=supported)
    assert len(plan.actions) == 1
    return plan.actions[0]


# -- steadiness ---------------------------------------------------------------------------
def test_stabilize_turns_on_the_strong_stabilizer_smooths_the_guide_and_lowers_cfg():
    a = tune(shot(["STABILIZE"]), supported=WAN)
    assert a.changes["seed"] != 10
    assert (a.changes["stabilize"], a.changes["smooth_control"], a.changes["cfg"]) == (
        "strong", 0.3, 5.5)
    assert {"STABILIZE", "CHANGE_SEED"} <= set(a.recommendations)
    rules = [t.split(":")[0] for t in a.tuning]
    assert rules.count("STABILIZE") == 3 and "CHANGE_SEED" in rules
    assert any("QC asked" in t and "smooth_control 0.0 -> 0.3" in t for t in a.tuning)


def test_steadiness_steps_stop_at_their_bounds():
    a = tune(shot(["DEFLICKER"]), {"stabilize": "strong", "smooth_control": 0.9, "cfg": 4.0},
             supported=WAN)
    assert not {"stabilize", "smooth_control", "cfg"} & set(a.changes)
    a = tune(shot(["STABILIZE"]), {"smooth_control": 0.75, "cfg": 4.2}, supported=WAN)
    assert (a.changes["smooth_control"], a.changes["cfg"]) == (0.9, 4.0)
    a = tune(shot(["STABILIZE"]), {"cfg": 3.0}, supported=WAN)  # your lower CFG stays
    assert "cfg" not in a.changes


def test_a_stabilizer_you_turned_off_stays_off():
    a = tune(shot(["STABILIZE"]), {"stabilize": "off"}, supported=WAN)
    assert "stabilize" not in a.changes and a.changes["smooth_control"] == 0.3
    assert any("stays off" in t for t in a.tuning)


@pytest.mark.parametrize(("metrics", "rule"), [
    ({"temporal_consistency": 5.2}, "STABILIZE"),   # today's QC report shape
    ({"stability": 6.1, "temporal_consistency": 8.0}, "STABILIZE"),
    ({"flicker": 4.0}, "DEFLICKER"),
])
def test_low_steadiness_scores_tune_without_a_new_recommendation(metrics, rule):
    a = tune(shot(["CHANGE_SEED", "REDUCE_STYLE_STRENGTH"], **metrics))
    assert a.changes["stabilize"] == "strong" and a.changes["smooth_control"] == 0.3
    assert a.changes["style_strength"] == 0.55  # today's change still applies
    assert rule in a.recommendations
    assert any(t.startswith(f"{rule}:") and " < 7" in t for t in a.tuning)


def test_cfg_drops_only_when_steadiness_is_the_worst_problem():
    a = tune(shot(["STABILIZE"], temporal_consistency=6.0, structure=3.0, motion=8.0),
             supported=WAN)
    assert "cfg" not in a.changes and a.changes["smooth_control"] == 0.3
    a = tune(shot(["STABILIZE", "ADD_POSE_CONTROL"]), supported=WAN)  # no scores: rec decides
    assert "cfg" not in a.changes
    a = tune(shot(["STABILIZE"], temporal_consistency=6.0, structure=7.5, motion=8.0),
             supported=WAN)
    assert a.changes["cfg"] == 5.5


def test_more_prompt_wins_over_lower_cfg_for_steadiness():
    a = tune(shot(["STABILIZE", "FOLLOW_PROMPT"]), supported=WAN)
    assert a.changes["cfg"] == 7.0 and a.changes["stabilize"] == "strong"


# -- guide edges --------------------------------------------------------------------------
def test_calm_edges_steps_toward_the_official_thresholds_and_stops_at_the_cap():
    cur: dict[str, Any] = {}
    seen = []
    for round_ in range(1, 5):
        a = tune(shot(["CALM_EDGES"]), cur, supported=WAN, round_=round_)
        cur = {**cur, **{k: a.changes[k] for k in ("canny_low", "canny_high") if k in a.changes}}
        seen.append((cur.get("canny_low", 0.2), cur.get("canny_high", 0.5)))
    assert seen == [(0.3, 0.65), (0.4, 0.8), (0.5, 0.9), (0.5, 0.9)]
    a = tune(shot(["CALM_EDGES"]), {"canny_low": 0.6, "canny_high": 0.95}, supported=WAN)
    assert "canny_low" not in a.changes and "canny_high" not in a.changes  # yours stay


@pytest.mark.parametrize(("old", "new"), [
    ((0.4, 0.8), (0.3, 0.65)),
    ((0.25, 0.6), (0.15, 0.45)),
    ((0.2, 0.5), None),        # at the defaults: never lowered
    ((0.1, 0.3), None),        # your lower values: not pushed further
])
def test_layout_drift_steps_raised_thresholds_back_down(old, new):
    a = tune(shot(["ADD_DEPTH_CONTROL"]), {"canny_low": old[0], "canny_high": old[1]},
             supported=WAN)
    got = (a.changes.get("canny_low"), a.changes.get("canny_high"))
    assert got == (new or (None, None))


def test_drift_from_the_structure_score_alone_also_counts():
    a = tune(shot([], structure=3.2), {"canny_low": 0.4, "canny_high": 0.8,
                                       "control_strength": 0.8}, supported=WAN)
    assert (a.changes["canny_low"], a.changes["control_strength"]) == (0.3, 0.95)
    assert any("structure 3.2 < 5" in t for t in a.tuning)


def test_calm_edges_and_drift_together_leave_the_thresholds_alone():
    a = tune(shot(["CALM_EDGES", "ADD_DEPTH_CONTROL"]), {"canny_low": 0.3}, supported=WAN)
    assert "canny_low" not in a.changes and "canny_high" not in a.changes
    assert any(t.startswith("CALM_EDGES:") and "drifted" in t for t in a.tuning)


# -- picture review -----------------------------------------------------------------------
@pytest.mark.parametrize(("old", "new"), [(6.0, 7.0), (7.5, 8.0), (8.0, None), (9.0, None)])
def test_follow_prompt_raises_cfg_up_to_eight(old, new):
    a = tune(shot(["FOLLOW_PROMPT"]), {"cfg": old}, supported=WAN)
    assert a.changes.get("cfg") == new


def test_low_prompt_adherence_score_counts_as_follow_prompt():
    a = tune(shot([], prompt_adherence=3.0), supported=WAN)
    assert a.changes["cfg"] == 7.0 and "FOLLOW_PROMPT" in a.recommendations


@pytest.mark.parametrize(("recs", "old", "new"), [
    (["FIX_ANATOMY"], 20, 26), (["MORE_DETAIL"], 30, 32), (["FIX_ANATOMY"], 32, None),
    (["FIX_ANATOMY", "MORE_DETAIL"], 20, 26),  # one step, not two
])
def test_anatomy_and_detail_add_sampling_steps_up_to_32(recs, old, new):
    a = tune(shot(recs), {"steps": old}, supported=WAN)
    assert a.changes.get("steps") == new
    if new:
        assert set(recs) <= set(a.recommendations)


def test_low_anatomy_and_detail_scores_count_too():
    assert tune(shot([], hand_body_deformation=2.0)).changes["steps"] == 26
    assert tune(shot([], detail=3.0)).changes["steps"] == 26
    assert "steps" not in tune(shot([], detail=6.0)).changes


# -- control strength keeps today's band ----------------------------------------------------
@pytest.mark.parametrize(("recs", "old", "new"), [
    (["ADD_DEPTH_CONTROL"], 0.95, 1.0),
    (["ADD_DEPTH_CONTROL"], 1.0, None),
    (["ADD_DEPTH_CONTROL"], 1.5, None),
    (["CHANGE_SEED"], 1.0, 0.9),
    (["CHANGE_SEED"], 0.7, None),
    (["CHANGE_SEED"], 0.5, None),
])
def test_control_strength_stays_in_its_band_when_tuning(recs, old, new):
    a = tune(shot(recs), {"control_strength": old}, supported={"SEED", "CONTROL_STRENGTH"})
    assert a.changes.get("control_strength") == new


# -- what the workflow supports -------------------------------------------------------------
def test_unsupported_workflow_inputs_are_dropped_and_listed():
    a = tune(shot(["FOLLOW_PROMPT", "FIX_ANATOMY", "CALM_EDGES", "STABILIZE"]),
             supported={"SEED", "CONTROL_STRENGTH"})
    assert {"cfg", "steps", "canny_low", "canny_high"} <= set(a.unsupported)
    assert not {"cfg", "steps", "canny_low", "canny_high"} & set(a.changes)
    # The studio applies the stabilizer and the guide smoothing itself: any workflow takes them.
    assert a.changes["stabilize"] == "strong" and a.changes["smooth_control"] == 0.3
    assert not any(t.startswith(("FOLLOW_PROMPT", "FIX_ANATOMY", "CALM_EDGES"))
                   for t in a.tuning)


def test_a_workflow_with_no_repair_controls_still_gets_no_actions():
    report = {"shots": [shot(["REDUCE_STYLE_STRENGTH", "ADD_DEPTH_CONTROL"])]}
    assert not RepairPlanner().plan(report, 1, {}, supported=set()).actions
    # Steadiness can still be repaired by the studio's own stabilizer.
    plan = RepairPlanner().plan({"shots": [shot(["STABILIZE"])]}, 1, {}, supported=set())
    assert set(plan.actions[0].changes) == {"stabilize", "smooth_control"}
    assert "seed" in plan.actions[0].unsupported


def test_passing_shots_and_unknown_recommendations_are_ignored():
    report = {"shots": [{"shot_id": "p", "decision": "PASS", "recommendations": ["PASS"]},
                        shot(["SOMETHING_NEW", "STABILIZE"])]}
    plan = RepairPlanner().plan(report, 1, {}, supported=WAN)
    assert [a.shot_id for a in plan.actions] == ["a"]
    assert "SOMETHING_NEW" not in plan.actions[0].recommendations


# -- never repeat a combination -------------------------------------------------------------
def test_a_combination_tried_before_is_not_planned_again():
    first = tune(shot(["STABILIZE"]), supported=WAN)
    planned = {k: v for k, v in first.changes.items() if k != "seed"}
    assert planned["control_strength"] == 0.9  # part of the combination too
    a = tune(shot(["STABILIZE"]), {"tried": [planned]}, supported=WAN)
    # stabilize is already strong in the plan: the next untried move is more guide smoothing.
    assert a.changes["smooth_control"] == 0.6 and a.changes["stabilize"] == "strong"
    assert any(t.startswith("UNTRIED: smooth_control 0.3 -> 0.6") for t in a.tuning)
    again = tune(shot(["STABILIZE"]), {"tried": [planned]}, supported=WAN)
    assert again.changes == a.changes  # deterministic


def test_a_seed_only_reroll_of_tried_settings_escalates():
    cur = {"control_strength": 0.7,  # at its floor: control strength cannot change
           "tried": [{"cfg": 6.0, "steps": 20, "stabilize": "light", "control_strength": 0.7}]}
    a = tune(shot(["CHANGE_SEED"]), cur, supported=WAN)  # "light" was tried, the same as "auto"
    assert set(a.changes) == {"seed", "stabilize"} and a.changes["stabilize"] == "strong"
    strong = {"stabilize": "strong", "control_strength": 0.7}
    a = tune(shot(["CHANGE_SEED"]), {**strong, "tried": [strong]}, supported={"SEED"})
    assert a.changes["smooth_control"] == 0.3  # the next untried move this workflow allows
    assert "steps" not in a.changes and "cfg" not in a.changes


def test_when_every_nearby_setting_was_tried_only_the_seed_changes():
    base = {"stabilize": "strong", "smooth_control": 0.9, "steps": 32, "cfg": 4.0,
            "control_strength": 0.7}
    a = tune(shot(["CHANGE_SEED"]), {**base, "tried": [base]}, supported=WAN)
    assert set(a.changes) == {"seed"}
    assert any("every nearby setting was tried" in t for t in a.tuning)


def test_tried_settings_can_come_from_render_params():
    params = {"CFG": 5.5, "STEPS": 20, "CONTROL_STRENGTH": 1.0, "SEED": 3, "WIDTH": 480,
              "_STABILIZE": "strong", "_SMOOTH_CONTROL": 0.3}
    settings = RepairPlanner.settings_from_params(params)
    assert settings == {"cfg": 5.5, "steps": 20, "control_strength": 1.0,
                        "stabilize": "strong", "smooth_control": 0.3}
    assert RepairPlanner.signature(settings) == RepairPlanner.signature(
        {"cfg": "5.5", "steps": 20.0, "stabilize": "STRONG", "smooth_control": 0.3,
         "canny_low": 0.2})
    assert RepairPlanner.signature({"stabilize": "auto"}) == RepairPlanner.signature({})
    assert RepairPlanner.signature({"stabilize": "off"}) != RepairPlanner.signature({})


# -- documents ------------------------------------------------------------------------------
def test_old_repair_plan_documents_still_validate():
    old = {"round": 2, "actions": [{"shot_id": "shot_001", "recommendations": [
        "ADJUST_CONTROL_STRENGTH", "CHANGE_SEED"], "changes": {"seed": 7929,
        "control_strength": 0.9}, "reason": "temporal flicker (score 5.2)", "unsupported": []}]}
    plan = RepairPlan.model_validate(old)
    assert plan.actions[0].tuning == []
    redo = {"round": 1, "requested_by": "you", "actions": [{
        "shot_id": "shot_002", "changes": {"seed": 1}, "reason": "", "tags": ["blurry"],
        "recommendations": ["CHANGE_SEED"], "unsupported": []}]}
    assert RepairPlan.model_validate(redo).actions[0].changes == {"seed": 1}


# -- auto_tune=False is the planner from before auto-tuning -------------------------------------
def legacy_plan(qc_report: dict[str, Any], round_: int, current: dict[str, dict[str, Any]], *,
                supported: set[str] | None = None) -> RepairPlan:
    """RepairPlanner.plan as of 383fc17, copied verbatim."""
    actions = []
    for s in qc_report.get("shots", []):
        if s.get("decision") == "PASS":
            continue
        recs = list(s.get("recommendations") or ["CHANGE_SEED"])
        cur = current.get(s["shot_id"], {})
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
                drift = any(r in recs for r in ("ADD_POSE_CONTROL", "ADD_DEPTH_CONTROL"))
                new = round(min(max(old, 1.0), old + 0.15) if drift
                            else max(min(old, 0.7), old - 0.1), 3)
                if new != old:
                    changes["control_strength"] = new
            recs = (["ADJUST_CONTROL_STRENGTH"] if "control_strength" in changes else []) + (
                ["CHANGE_SEED"] if "seed" in changes else [])
            if not changes:
                continue
        actions.append(RepairAction(shot_id=s["shot_id"], recommendations=recs,  # type: ignore[arg-type]
                                    changes=changes, unsupported=unsupported,
                                    reason="; ".join(s.get("issues", []))))
    return RepairPlan(round=round_, actions=actions)


REPORT = {"shots": [
    {"shot_id": "a", "decision": "PASS", "recommendations": ["PASS"]},
    {"shot_id": "b", "decision": "FAIL", "issues": ["temporal flicker (score 5.1)"],
     "recommendations": ["CHANGE_SEED", "REDUCE_STYLE_STRENGTH"], "temporal_consistency": 5.1,
     "structure": 7.0},
    {"shot_id": "c", "decision": "FAIL", "issues": ["layout drift"],
     "recommendations": ["REDUCE_STYLE_STRENGTH", "ADD_DEPTH_CONTROL", "INCREASE_IDENTITY",
                         "ADD_POSE_CONTROL"], "structure": 2.0},
    {"shot_id": "d", "decision": "FAIL", "issues": ["soft"],
     "recommendations": ["STABILIZE", "CALM_EDGES", "FOLLOW_PROMPT", "MORE_DETAIL"],
     "stability": 4.0},
    {"shot_id": "e", "decision": "FAIL"},
]}
CURRENT = {"b": {"seed": 5, "control_strength": 0.75, "style_strength": 0.7},
           "c": {"seed": 6, "control_strength": 0.9, "identity_strength": 0.95, "cfg": 7.0},
           "d": {"seed": 7, "canny_low": 0.4, "stabilize": "light", "steps": 30}}


@pytest.mark.parametrize("supported", [None, set(), {"SEED"}, {"SEED", "CONTROL_STRENGTH"},
                                       WAN | {"STYLE_STRENGTH", "POSE_STRENGTH"}])
@pytest.mark.parametrize("round_", [1, 3])
def test_auto_tune_off_is_exactly_the_old_planner(supported, round_):
    got = RepairPlanner().plan(REPORT, round_, CURRENT, supported=supported, auto_tune=False)
    assert got.model_dump() == legacy_plan(REPORT, round_, CURRENT,
                                           supported=supported).model_dump()
    assert all(a.tuning == [] for a in got.actions)


def test_a_flickering_shot_is_repaired_with_recorded_tuning(ctx, sample_video):
    pid = create(ctx, sample_video, test_faults={"shot_002": {"kind": "flicker", "attempts": [1]}})
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    with ctx.db.session() as s:
        plan = RepairPlan.model_validate(latest_document(s, pid, "repair_plan").data)
        manifest = ReconstructionManifest.model_validate(latest_document(s, pid, "manifest").data)
    action = plan.actions[0]
    assert action.shot_id == "shot_002" and "STABILIZE" in action.recommendations
    assert any(t.startswith("STABILIZE: smooth_control") for t in action.tuning)
    overrides = manifest.shot("shot_002").overrides
    assert overrides["stabilize"] == "strong" and overrides["smooth_control"] == 0.3


def test_auto_tune_on_records_why_for_every_shot():
    plan = RepairPlanner().plan(REPORT, 1, CURRENT, supported=WAN)
    assert [a.shot_id for a in plan.actions] == ["b", "c", "d", "e"]
    d = next(a for a in plan.actions if a.shot_id == "d")
    assert d.changes["canny_low"] == 0.5 and d.changes["steps"] == 32
    assert d.changes["cfg"] == 7.0  # FOLLOW_PROMPT; steadiness did not lower it
    assert all(a.tuning and a.tuning[0].startswith("CHANGE_SEED") for a in plan.actions)
