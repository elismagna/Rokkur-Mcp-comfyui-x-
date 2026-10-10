# Render speed on the RTX 3070

Where a project's wall-clock time goes, what has been cut, and what to test next. Numbers marked
*measured* come from Elis's PC; everything else is an estimate until `timings` says otherwise.

## Measure first: `timings`

```powershell
.\scripts\studio.ps1 timings                 # latest project
.\scripts\studio.ps1 timings <project id>    # a specific one
.\scripts\studio.ps1 timings <id> --json     # the same numbers as JSON
```

It reads what the studio already records, so it works on old projects too. Per stage: runs,
working time and time waiting in the queue. Per shot attempt: wall time, ComfyUI execution time,
the rest (upload, download, FFmpeg), subject-mask time, QC score and your rating. The first shot
of each render run is marked "(loads models)". At the bottom: how much time went to repair
re-renders and how many of them you liked.

## Where the time goes

| Part | Rough share | Notes |
|---|---|---|
| Wan sampling | most of it | 81 frames at 480×832 ≈ 33k tokens per step; 20 steps × CFG 6 = 40 model passes per shot |
| Repair re-renders | up to ×2 | each round re-renders every failing shot; seed rerolls rarely fixed anything (Codex, 6 and 8 rounds) |
| Model loading | per shot, before this change | `/free` after every shot reloaded the 2.8 GB Wan model and 6.7 GB text encoder from disk |
| Director passes (Ollama) | ~18 s per project | *measured* 18.5 s for both passes with vision |
| Subject masks (U²-Net, CPU) | seconds per shot | only in Keep mode, cached per shot |
| QC, FFmpeg, queue | small | QC reads 64×64 grey frames; intermediate encodes are already `veryfast` |

## Shipped (items 1–3 don't change how a render looks)

1. **Models stay loaded between shots.** Every shot took its own GPU lease, and each lease ended
   with ComfyUI `/free`, which also clears ComfyUI's node cache. So shot 2 reloaded the Wan
   model and the umt5 text encoder from disk, then shot 3, and so on. A render stage now runs
   inside one `heavy_batch`: Ollama is unloaded once before the first shot and ComfyUI is
   freed once after the last. Any Ollama call still frees an idle ComfyUI first
   (`free_comfyui_before_agents`), so Ollama never falls back to the CPU.
   Expected: the first shot is as before; every later shot saves the load time. `timings`
   shows it as the gap between the "(loads models)" shot and the rest.
2. **Repairs stop after one round that didn't help.** `render.stall_reports: 2` (was a fixed
   3). If no failing shot gained 0.2 QC points in the last round, the project stops at the same
   choice as the repair limit: keep the renders or grant more rounds. A round that does help
   keeps going, up to `render.max_retries`. Set it to 3 to get the old behaviour back.
3. **ComfyUI is polled every second** instead of every two (two local GETs per poll).

4. **Draft first, quality for the keepers.** PREVIEW and RTX3070_DRAFT name an `upgrade_to`
   profile (RTX3070_QUALITY). On a finished fast video the project page shows **Render in
   quality**: the picked shots, or every shot if none is picked, render again in quality with
   the same prompt, seed and settings, and the other shots stay. API: `POST
   /projects/{id}/upgrade` with `{"shots": [...]}` (empty = all). A different sampler and step
   count do not give identical frames, so the draft shows the direction (look, subject
   handling, framing), not the exact final frames. The saving is every rejected direction no
   longer costs a full-quality render.

## Next: speed that may cost quality (A/B with your ratings)

Run each one on the same clip and seed, rate the shots, and paste `timings` for both runs.

1. **RTX3070_DRAFT** (already built): Self-Forcing DMD LoRA, 4 steps, CFG 1, so 4 model passes
   instead of 40. Needs `Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors` in
   ComfyUI `models/loras` (`scripts/install-models.ps1` offers it), then `comfy-check`.
   If you like DRAFT shots as much as QUALITY ones, it becomes the default; if not, it is
   still the fastest way to see whether a prompt works before a full render.
2. **Fewer frames per shot.** Sampling cost grows faster than linearly with frames. Shots over
   3 s could render at 49 frames and be split, or keep 81; test on a long shot.
3. **ComfyUI launch options** (ComfyUI Desktop settings, no studio change): `--fast` (fp16
   accumulation on PyTorch 2.7+) and SageAttention (`--use-sage-attention`, needs the
   `sageattention` package in ComfyUI's Python) are both reported to speed up Wan on RTX 30
   cards. Not tested on this PC; check `comfy-render` still works and compare a shot.
4. **TeaCache/MagCache** skip near-identical steps (custom node pack, has published Wan 1.3B
   example workflows). Only after 1–3, and starting from the pack's own example workflow.

Not worth it: lowering QC cost (it is already cheap), overlapping mask work with rendering
(seconds per shot, for real complexity), or GPU-encoding intermediates.
