# AI director

Every shot gets its own diffusion prompt, built in three passes from a fixed cinematography
vocabulary and ComfyUI's prompt rules. Models only choose within those limits, and every
model pass has a rule-based fallback, so a render never waits on a model.

```
analysis (shots, motion) ──► 1 Creative Director ──► 2 Director of Photography ──► 3 prompt compiler
                              story: intent,          shot size, angle,            Rule of Nouns order,
                              subject as states,      movement, lighting           action strip-out,
                              background              (allowed terms only;         framing weights,
                                                      sees each shot's middle      character anchor,
                                                      frame if the model can       global look
                                                      read images)                      │
                                                                                        ▼
                                          per-shot STYLE_PROMPT ──► Wan render, one ComfyUI job per shot
                                          Batch Prompt Schedule ──► export for FizzNodes (copy / download)
```

Code: `director/` (vocabulary, rules, assets, prompts, passes), roles in `agents/roles.py`,
the stage in `pipeline/stages.py::creative_plan`. Config: `director:` in `config/studio.yaml`.

## Example

Brief subject for a shot, as a model might write it: `he is walking through the rain,
suddenly turns, thinking about home`. With character `NEO`, framing Close-up / Low-angle /
Slow push-in / Moody neon rim lighting and the default global look, the shot renders with:

```
Cinematic film still, (close-up shot:1.3), (low-angle shot:1.25), slow push-in,
a 20s athletic male, wearing a tattered black hooded jacket, dark denim jeans, intense pale
features, in mid-stride through the rain, head turned sharply, focused expression,
moody neon rim lighting, neon alley, wet asphalt, <theme>, 35mm anamorphic lens, highly
detailed textures, photorealistic cinematic film still, depth of field
```

The dashboard's **Director** page has a "Try the rules" form that compiles a prompt like this
without a model or a render.

## 1. Allowed vocabulary (the DP pass)

| Category | Terms |
|---|---|
| Shot sizes | Extreme Close-up, Close-up, Medium Shot, Cowboy Shot, Full Shot, Extreme Long Shot |
| Camera angles | Low-angle, Eye-level, High-angle, Dutch tilt, Overhead crane, Birds-eye view |
| Camera movement | Static, Slow push-in, Tracking pan, Handheld shake, Jib tilt, Dolly zoom |
| Lighting styles | Volumetric god rays, High-key overhead, Chiaroscuro high-contrast, Moody neon rim lighting, Golden hour diffusion, Cyberpunk bi-color hue, Rembrandt lighting |

They are `Literal` types (`director/vocabulary.py`), so the JSON schema sent to Ollama limits
the model to these strings and pydantic rejects anything else; a rejected answer is retried,
then that one shot falls back to rules. Each term has a prompt phrase ("Close-up" →
`close-up shot`).

The DP runs once per shot with the brief, the shot's intent and subject, the source's
measured motion and the previous shot's choices (for consistent lighting). When the model
reports the `vision` capability (Ollama `/api/show`), it also gets the shot's middle frame,
because the render keeps the source's composition: the framing words should describe what
is actually there. `director.vision_model` can name a separate Ollama vision model for this
pass; `director.vision: off` disables images.

Rule-based framing (no model): Medium Shot, Eye-level, movement from the measured motion
(static → Static, gentle → Slow push-in, moderate → Tracking pan, high → Handheld shake),
lighting from theme keywords (neon → Moody neon rim lighting, noir → Chiaroscuro, forest/fog →
Volumetric god rays, …; default Golden hour diffusion).

## 2. Diffusion parsing rules (the compiler, deterministic)

- **Rule of Nouns.** Order: prefix, weighted shot size, weighted angle, movement, subject
  block, lighting, background, theme, style, extra detail, global style modifiers. The subject
  block is the character anchor followed by the shot's state and is never split by other
  terms. Free text loses full stops, colons, semicolons, brackets and quotes (sentence breaks
  become commas); decimals and ratios like `f/1.8` or `16:9` stay.
- **Action strip-out.** Progressive actions and inner states become physical states:
  `is walking` → `in mid-stride`, `suddenly runs` → `in full sprint`, `thinking about home` →
  `focused expression`, `looking at the camera` → `gaze fixed on the camera`, `feeling sad` →
  `sad expression`. About 40 action verbs and 20 inner-state verbs are mapped
  (`director/rules.py`). Words that only look like verbs stay (`running water`, `waving flag`).
  Phrases it cannot convert, such as `starts to glow`, are listed as a warning on the
  project page.
- **Token weights.** Only the compiler adds weights, and only to framing:
  `(close-up shot:1.3)`, `(low-angle shot:1.25)` (`director.framing_weight`,
  `director.angle_weight`; 1.0 removes the brackets). Weights or brackets typed into briefs,
  characters or the theme are removed so the emphasis stays predictable.

## 3. Asset tracker (characters and the global look)

Stored at `data\director\asset_tracker.json` in the studio folder; edit it on the Director
page, through `GET/PUT /director/assets`, or by hand:

```json
{
  "PROMPT_PREFIX": "Cinematic film still",
  "GLOBAL_STYLE_MODIFIERS": "35mm anamorphic lens, highly detailed textures, photorealistic cinematic film still, depth of field",
  "GLOBAL_NEGATIVE_PROMPT": "blurry, deformed, drawing, cartoon, illustration, distorted hands",
  "CHARACTERS": {
    "NEO": "a 20s athletic male, wearing a tattered black hooded jacket, dark denim jeans, intense pale features"
  }
}
```

- Pick a character on **New video** (or `render ... --character NEO`): its description opens
  the subject of every shot, and the Creative Director is told not to restate clothing or
  appearance, only pose and expression. That is what keeps a character from changing between
  shots. A free-text "character just for this video" works the same way.
- **Apply the global look** (on by default; `--no-global-look` on the command line) adds the
  prefix, the style modifiers and the negative prompt. The defaults are photoreal; for a
  claymation or anime theme, untick it or change the look on the Director page.
- A negative term the video's own theme asks for is left out for that video (a "cartoon
  noir" theme keeps "cartoon" out of its negative prompt) and noted on the project page.
- An invalid tracker file stops the brief stage with a message naming the file, rather than
  silently rendering without your characters.

## Prompt schedule (FizzNodes Batch Prompt Schedule)

The brief stage also writes the shots as keyframes in FizzNodes' format: a keyframe every
`schedule_interval` frames at `schedule_fps` (default 24 at 24 fps, so frame 24 is one
second), plus a keyframe on each side of every cut. The node blends prompts between
keyframes, so without the cut keyframes a hard cut would fade over up to a second.

```
"0": "Cinematic film still, (medium shot:1.3), ... --neg blurry, deformed, ...",
"24": "Cinematic film still, (medium shot:1.3), ... --neg blurry, deformed, ...",
"54": "... last frame of shot 1 ...",
"55": "... first frame of shot 2 ...",
"72": "..."
```

Entries are separated by commas (the node parses the text as JSON), and `--neg` separates
the negative prompt, which FizzNodes splits off (`director.schedule_inline_negative: false`
leaves it out). Set the node's `max_frames` to the number shown on the project page.

Get it from the project page (Copy / Download .txt), `GET /projects/{id}/prompt-schedule`,
`.\scripts\studio.ps1 prompt-schedule <project id>`, or
`data\projects\<id>\prompts\prompt_schedule_v<n>.txt`.

The studio's own render does not use the schedule: Wan 2.1 samples a whole clip from one
prompt, so the studio renders each shot as its own ComfyUI job with that shot's prompt,
which gives hard cuts and per-shot framing directly. The schedule is for workflows that take
per-frame prompts (FizzNodes is a custom node and is not installed by the studio).

## Checking it on the workstation

`.\scripts\studio.ps1 agent-check` now also asks the Director of Photography for each sample
shot, says whether the model can read images (and sends it a test image if so), and prints
the compiled prompt and the schedule size.

## Limits

- Without vision, the DP chooses framing from the brief, not the picture. With the
  video-to-video workflow the source's composition wins anyway, so wrong framing words mostly
  waste prompt space; a vision-capable model fixes this.
- The strip-out is a word list, not a parser. An unlisted verb passes through unchanged
  (only a leading "is"/"are" is dropped: `is waiting` becomes `waiting`).
- Character anchors are text only. Character LoRAs would need a LoRA loader in the workflow
  template; that is not built.
