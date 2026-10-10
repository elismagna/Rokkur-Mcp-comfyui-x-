# Stability: coherent renders, and changing the work while it renders

Elis's priority (2026-10-10): coherent output without artifacts comes before variety, and
the workload should be adjustable while a video is being made, for anything predictable and
safe. The studio's stated goal is communication between the person and the studio without
losing or altering detail: every decision is recorded with what produced it, and every
change is visible on the project page.

## Stable mode (New video)

A checkbox on New video (`creative.stable`). What it fixes, and why:

| Fixed | Why |
|---|---|
| One seed for every shot (`shot_seed(project, "stable")`, or the seed you typed) | Different seeds per shot change the texture and the subject's look between shots. |
| The profile's full steps, at least 20 | Fewer steps leave Wan's typical smearing and melted limbs. |
| Source guide (`CONTROL_STRENGTH`) 1.0 | Structure and motion follow the footage, which is what keeps the shot coherent. |
| No per-shot framing or lighting from the Director of Photography | Per-shot framing terms make Wan recompose; the brief's one prompt renders every shot. |
| Reference mode `cutout` when nothing else is chosen | The same subject cutout is shown to Wan for every shot, so identity cannot drift. |
| Quality pass mark +1.0 (capped at 9) | Flicker, black frames and structure loss fail the check and are repaired instead of accepted. |

Values you set yourself on New video (seed, steps, source guide) win over the mode. Stable mode
does not change the model, the profile or the resolution. It is implemented in
`pipeline/stages.py` (`creative_plan`, `compile_stage`, `quality_check`, `stable_overrides`).

Real-world status: the mechanics are tested with the preview renderer and the fake ComfyUI.
Whether Stable mode removes the artifacts Elis sees on the RTX 3070 is the next thing to
check on the PC: render the same clip once normally and once in Stable mode, same prompt,
and compare the QC scores and the shots by eye. Record the outcome in the handoff.

## Adjusting the remaining shots

While a video is between WORKFLOW_READY and REPAIRING (`commands.ADJUSTABLE`), the project
page shows **Adjust the remaining shots**; the API is `POST /projects/{id}/adjust`. Allowed
changes are the ones that cannot break a shot in progress or invalidate a finished one:

- an extra direction appended to the prompts of the shots still to render (`prompt_extra`);
- seed, steps, prompt guidance (`cfg`), source guide (`control_strength`), edge thresholds;
- the appearance reference: a picture from the library, or none.

`commands.adjust_remaining_shots` writes a new manifest version with the changes on the shots
that have no finished render; the render loop (`stages.render`) reloads the manifest before
each shot and continues with the new values, and records a `SHOTS_ADJUSTED` event with the
shots and values. Finished shots are never touched here: use **Redo** on the finished video
for those, which keeps the others as they are.

Not adjustable mid-render, on purpose: the render profile, resolution and frame rate (the
assembled video needs one size and rate), the subject keep/restyle decision (its masks are
computed per shot against the plan) and the theme itself (a different theme is a new video).
