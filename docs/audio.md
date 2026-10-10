# Sound: music and effects generation, editing, soundtracks at any stage

Dashboard: **Sound** (`/ui/audio`). API: `/audio` and `POST /projects/{id}/soundtrack`. CLI:
`rokkur-studio audio`, `audio-edit`, `audio-list`. Every clip is an `AudioClip` row
(migration `0004`) with its file at `data/audio/<id>/clip.flac`; a clip made from another one
points at it (`parent_id`), a mix records every input in `request`. The page draws a waveform
per clip (`/audio/{id}/waveform.png`, FFmpeg `showwavespic`) so you see what you have.

## Making sound (ComfyUI, GPU)

| Operation | Model and workflow | Input | Limit |
|---|---|---|---|
| Music | `ACE_STEP` → `audio_ace_step` (ACE-Step v1 3.5B, Apache-2.0) | style tags; optional lyrics with `[verse]`/`[chorus]`, empty = instrumental | 240 s per clip |
| Sound effect | `STABLE_AUDIO` → `audio_stable_open` (Stable Audio Open 1.0, Stability AI Community License) | a description, optional "avoid" text | 47 s per clip |

Profiles live in `config/audio_profiles.yaml`; page defaults in `config/studio.yaml` under
`audio:` (`music_profile`, `sound_profile`, `default_seconds`, `max_batch`, `max_seconds`).
`services/audio.py:request_audio` validates and queues one `audio` job per request;
`pipeline/audio.py:audio_job` takes a GPU lease of the profile's class (music `GPU_HEAVY`,
effects `GPU_MEDIUM`), submits, polls and downloads exactly as pictures and video shots do, and
re-encodes the result to FLAC. A CUDA OOM walks the profile's ladder: `clear_cache`,
`single_clip` (one clip per prompt, seeds counting up), `shorter` (half the length, at least
5 s). Cloud mode works the same through the cloud server and records `cloud_gpu_minutes`.

### Models to install

Into ComfyUI's models folder (`scripts/install-models.ps1` moves them from Downloads):

| File | Folder | From | Licence |
|---|---|---|---|
| `ace_step_v1_3.5b.safetensors` | `checkpoints` | huggingface.co/Comfy-Org/ACE-Step_ComfyUI_repackaged (all_in_one) | Apache-2.0 |
| `stable_audio_open_1.0.safetensors` | `checkpoints` | huggingface.co/stabilityai/stable-audio-open-1.0 (gated: accept the licence) | Stability AI Community License (free under US$1M yearly revenue) |
| `t5_base.safetensors` | `text_encoders` | huggingface.co/google-t5/t5-base (`model.safetensors`, renamed) | Apache-2.0 |

Then `scripts\studio.ps1 comfy-check`: both `audio_*` templates are validated against the live
`/object_info` and the model file names. Both graphs use ComfyUI core audio nodes only
(`comfy_extras/nodes_audio.py`, `nodes_ace.py`), no custom node packs. Sources and node-by-node
provenance are in each `params.yaml` header.

## Shaping sound (FFmpeg, CPU, at once)

`services/audio.py:edit_audio` runs one operation and puts the result in the library as a new
clip; the original stays. Trim (start/end), fade (in/out seconds), level (dB), loudness
(`loudnorm` to a LUFS target; -14 is what YouTube plays at), loop (repeat to a length with a
fade at the end), speed (0.25×–4×, pitch kept with `atempo` or changed like a tape), mix (two
or more clips with levels and start offsets, `amix` + limiter), join (one after another),
extract (the sound of a video file). Uploads and a video's sound come in through **Bring a
sound in** (or `POST /audio/upload`); every file needs its rights line, like any source.

## Soundtracks at any stage

A video's added track is `creative.audio_bed_path` (Codex's soundtrack mixing in the edit
stage: looped to the video's length, mixed under the footage's own sound at
`audio_bed_gain`, `keep_source_audio` mutes the footage). The sound studio feeds it in three
places, all through `commands.set_soundtrack`:

- **New video**: a *Soundtrack from the sound library* select next to the file upload. A
  library clip carries its own rights line (generated: "made in the studio"; uploaded or
  extracted: what you wrote), so no second confirmation is asked.
- **The project page, Sound section** and the **Use as soundtrack** form on a clip: before the
  edit stage the choice is saved and used when the edit runs; on a finished video
  (`READY_TO_PUBLISH`) the edit runs again now, the new cut becomes the latest final and the
  earlier one stays in Renders (a pending publish proposal is withdrawn, the metadata you
  wrote is kept). Removing the added track works the same way. Published videos keep their
  sound; `EDITING`/`PUBLISHING` refuse until they finish. Every change is a `SOUNDTRACK_SET`
  event with the values.
- **Take the final cut's / the footage's sound into the library**: the video's sound becomes a
  clip you can shape (normalise, fade, mix a bed under it) and set back as the soundtrack.
- **Make music or effects for this video** opens Sound with the video's length filled in.

Extended videos keep the original's sound under the continuation (padded with silence).

## Honest status

Tested here with the fake ComfyUI (it serves a tone of the requested length, so the tests
check the plumbing: batches, seeds, OOM ladder, FLAC, lengths, leases, costs), real FFmpeg for
every edit, and the whole soundtrack flow end to end with the preview renderer. **No real GPU
has run the two audio graphs yet**; the first ACE-Step and Stable Audio renders on the PC are
the acceptance test (ACE-Step's checkpoint is about 7 GB in fp16 and will offload on the
RTX 3070; expect slow first runs). Not built: stem separation, voice cloning or text to
speech, beat-synced cuts, audio-reactive video; each needs a proven published workflow or
node pack first, per the project rule.
