# Ratings, redo and your taste

You rate what the studio makes; the studio uses those ratings in two ways. Right away, a
rating changes what happens to that shot. Over time, ratings across projects become a taste
profile that suggests settings on the New video page. Nothing changes a default on its own.

## Three separate signals

| Signal | Who gives it | What it is for |
|---|---|---|
| Your rating | You, per shot and per whole video | What you like. The only input to learning. |
| Measured quality (QC) | `pipeline/qc.py` | Flicker, motion, structure, artifacts, detail, 0 to 10. Decides repairs. |
| AI estimate | A model (`rater="ai"`), not built yet | Kept in the same table but never read by learning. |

They are shown side by side and never merged into one score, so a bad shot can't hide
behind a good average.

## Rating

Four verdicts: super dislike (−2), dislike (−1), like (+1), super like (+2). Pressing your
current verdict again removes it. Optional tags say what the rating is about (main subject,
style, background, continuity, motion, flicker, detail, follows my prompt), and an optional
note says what to keep or change.

- **Shot ratings** belong to the render attempt you were looking at. Each one stores a
  snapshot of what made it: prompt, seed, steps, guidance, source structure
  (control_strength), workflow, quality profile, framing words, subject mode, reference mode
  and the QC result.
- **Video ratings** belong to the final (or assembled) video and snapshot the project's
  settings.

On the dashboard ratings save instantly (`POST /ui/projects/{id}/rate`, JSON); the API has
`GET` and `PUT /projects/{id}/ratings`. Ratings are stored in the `ratings` table
(migration `0002`), and every change is an audit event (`RATING_SET`, `RATING_CLEARED`).

## What a rating changes now

- **A shot you like passes the quality check**, whatever QC measured (`accepted_by: you` in
  the QC report). A liked shot is never sent back for repair.
- **Redo**: on a finished video, or one stopped at the repair limit, pick shots and press
  Redo. Disliked shots are picked for you. Each picked shot gets a new seed; a *motion* tag
  raises source structure by 0.1 (up to 1.0), a *style* or *follows my prompt* tag lowers it
  by 0.1 (down to 0.7). Shots you did not pick keep their renders, a pending publish
  proposal is withdrawn, your title and description edits are kept, and the video is edited
  again. The redo gets its own repair budget.

## Your taste (`/ui/taste`, `studio.ps1 taste`)

Built only from your ratings (`services/taste.py`):

- Every rating is turned into features: the prompt's own terms (framing vocabulary and
  weights left out), and settings such as quality profile, workflow, source structure,
  guidance, steps, framing, lighting, subject and reference mode.
- Ratings are averaged per project first, so twenty shots of one video count as one example.
- A feature's **lift** is how far its projects rate above or below your overall average,
  shrunk toward zero when it comes from few projects.
- Confidence: *too early to tell* under 2 projects, *some* from 2, *strong* from 4 projects
  with a lift of at least 0.5. Terms that sit in nearly everything you rate (|lift| < 0.1)
  are left out of the like/dislike lists.
- **Suggestions** need *some* confidence and a lift of at least 0.4: add a term you like to
  the visual style, avoid a term you dislike, or set a quality profile, subject mode,
  reference mode, source structure, guidance or step count. They appear on the New video
  page with an Apply button and do nothing until you press it.
- **QC against you** shows how often QC agreed with your verdict, and lists the shots QC
  failed but you liked, and the reverse. That is the evidence for tuning QC thresholds.

The page ends with a plain-text summary (no media, paths or ids) to paste to Codex or
another assistant.

## Limits

- Learning is a transparent profile, not model training. It suggests; you decide.
- Redo changes the seed and source structure only. Other changes (prompt wording, a
  different workflow) are made on a new video.
- No AI rater exists yet; the `rater` column is ready for one.
