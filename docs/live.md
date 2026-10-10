# Live view: watch the studio work, decide with the evidence in front of you

**Why it exists.** Elis's main goal for Rökkur (2026-10-10): communication between the person
and the studio without losing or altering detail. The live view is that goal as a page: every
step the studio takes is shown as it happens, every stage lists what it produced and who
produced it (a model, the rules, FFmpeg, you), and every decision that waits for a person is
shown next to the frames and renders it is about, with the same buttons the project page has.

Open it from a project page (**Watch it work**) or at `/ui/projects/{id}/live`. The API gives
the same data as JSON at `GET /projects/{id}/live` (`?events=N`), and the page polls
`/ui/projects/{id}/live.json` every 3 s while the video is being worked on (12 s otherwise).
It is built by `services/live.py:snapshot`, read-only, from what the studio already records.

## What is on it

- **The pipeline as a graph**: rights → ingest → analyse → brief → workflow → render →
  quality → repair → edit → ready → published. Green is done, the glowing node is where the
  studio is, red is where it stopped, amber with a question mark waits for you, a dashed
  node was not needed (no repairs).
- **Waiting for you**: each decision with its evidence and actions. A rights question shows
  the clip's frames. The repair limit shows every failing shot's latest render next to its
  original frame with the quality issues measured, and offers *more repairs*, *check again*,
  *keep these renders* and the way to the shots for rating and redo. The render budget and a
  stopped stage show their numbers and reason with *allow more* or *resume*. A publish
  proposal shows the final cut with *approve and upload*. The buttons are the project page's
  own forms, so a decision here is the same decision there.
- **Now**: the running job and for how long, the shot in progress with its attempt, renderer
  and workflow, ComfyUI's queue (how many prompts run and wait, and whether this shot is the
  one running or its place in line; the studio's prompts are recognised by their `rokkur/`
  input folder), and the jobs queued next. **Used so far**: renders, GPU and cloud minutes,
  repair rounds.
- **Shots**: one card per shot from the moment the footage is analysed (motion type), then
  the brief's intent and prompt, the original frame, the latest render as a playable video
  with the frame as its poster, the state (planned, rendering now, rendered, needs repair,
  failed), attempts, the measured quality and issues, your verdict, and the overrides the
  shot carries (seed, steps, source guide, prompt addition).
- **What each stage produced**: rights category, status and evidence; the source's size,
  rate and length; shots and scene cuts found and which signals were measured; the brief's
  prompt, subject and shot count and who wrote it (the director on Ollama, or the rules);
  the manifest's profile, size, subject decision and reference; renders done and attempts;
  the quality score, pass mark, decision and failing shots; repair rounds and each action's
  changes and reason; the final file, length and soundtrack; the drafted title and who
  wrote it; the upload.
- **Events**: the audit trail in words, newest first, with the numbers that matter
  (`describe_event`).

## Honest limits

- Progress inside one ComfyUI prompt (the sampler's step count) is not shown: the worker
  polls `/history`, not the websocket, and the API process has no channel to the worker. The
  queue position comes straight from ComfyUI's `/queue` and is read on each poll.
- The page does not refresh itself while a video plays or a field has focus, so a decision is
  never pulled out from under you; when the set of decisions changes it reloads once.
- Tested here with the preview renderer and the fake ComfyUI (`tests/test_live.py`), and
  rendered in Chromium at desktop and phone width.
