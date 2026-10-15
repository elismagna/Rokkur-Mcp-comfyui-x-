# Night School: the studio practises and learns what renders well

The studio learns which render settings give better pictures by practising on videos it has
already made. It learns general quality, not anyone's taste: your ratings never feed it (they
still drive the separate Your taste page).

## What a practice batch does

One batch takes one shot of a finished video and renders it several ways on this PC:

1. **Baseline**: exactly as the studio would render it.
2. **Another seed**: only the seed changes. The score difference shows how much luck alone
   moves the score, so a lesson has to beat that.
3. **One change at a time**, rotating through the list below so a night of batches covers
   all of them (`practice.variants` per batch, default 5). Changes the workflow cannot make
   are skipped (the 4-step draft has no CFG or sampler input).

| Change | Overrides |
|---|---|
| control 0.85 / 1.15 | `control_strength` |
| edges 0.4/0.8, 0.1/0.3 | `canny_low` / `canny_high` (default 0.2/0.5) |
| cfg 4.5 / 7.5 | `cfg` (default 6) |
| shift 5 / 12 | `shift` (default 8) |
| 70% steps | `steps` (speed: does it look as good?) |
| sampler euler | `sampler` (default uni_pc) |
| stabilizer off / strong | `stabilize` (default auto) |
| smoothed guide | `smooth_control` 0.5 |
| reference none / cutout / first frame | the character reference Wan gets (`reference_mode`); skipped when the video has an uploaded character reference or already uses that type |

All renders of a batch run in one GPU batch, so the models load once. Then each render is
checked like QC checks a shot, with the AI picture review always on:

- **score** (0 to 10) = the mean of QC's overall score, the mean of the five review scores
  (prompt, style, subject, anatomy, steadiness) and, once trained, the quality judge
  (`docs/judge.md`), which learned botched vs good frames from rated AI images;
- a render whose anatomy scores 4 or less is capped at 5, however steady: on a real shot,
  control 1.15 raised QC while bodies got worse, and the score must not reward that;
- with neither a vision model nor a judge the score is QC alone; such batches are shown but
  never become lessons.

The report (a `practice_report` document on the video) ranks the versions, with links to each
render and its contact sheet. The video itself never changes: its status stays, its shots keep
their renders, and practice renders do not count against its render budget.

Practice always runs after real work. If you start a video while a batch is running, the batch
stops after the current render and the rest of it continues after your video's job.

## Lessons

`lessons` compares each change with the baseline of the same batch, per render profile. A
change becomes:

- **better** after at least `practice.min_batches` (3) reviewed batches, with a mean gain above
  both `practice.min_gain` (0.5) and the seed's average swing, and a win in 3 of 4 batches;
- **worse**: the same, the other way;
- **as good, faster**: no clear difference, at least 15% less render time (the first render of
  a batch loads the models and is left out of the time comparison);
- **no clear difference**, or **needs N more**.

With `practice.apply_lessons: true` in `config/studio.yaml`, new videos start with the
better and as-good-faster changes for their profile. A setting you choose on the New video
form always wins; only values left at the form's default are replaced. A proven reference type
is used only when the form's reference is left on auto and no character image is uploaded. It is off by default:
look at the lessons first.

## Run it

Dashboard: **Night School** in the sidebar. Pick a video (or every finished video in turn) and
a number of batches, then **Practise overnight**.

```powershell
.\scripts\studio.ps1 practice --all --batches 8      # every finished video, in turn
.\scripts\studio.ps1 practice <project id> --batches 4 --shot shot_002
.\scripts\studio.ps1 practice --all --batches 8 --profile RTX3070_DRAFT
.\scripts\studio.ps1 lessons                         # what it has learned; --json for data
```

Time per batch on the RTX 3070 is roughly (variants + 2) shot renders plus a vision review
each: 7 renders of a QUALITY shot, or about a minute each on DRAFT. Leave the worker and Ollama
running; check the Queue page in the morning.

## Limits

- The AI reviewer is the local vision model (qwen3.5:9b by default). It is noisy, which is why
  a lesson needs several batches and must beat the seed's swing. A stronger vision model in
  `quality.vision_model` makes the lessons more trustworthy.
- Lessons are per profile: what helps the 4-step draft may not help QUALITY.
- Changes are tried one at a time; combinations are not tested.
- Practice runs on this PC only, never on the paid cloud GPU.
