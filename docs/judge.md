# Quality judge: learning what botched and good AI frames look like

The judge is a small model the studio trains itself, on this PC, from examples of good and
botched AI images. It scores every render 0 to 10 (higher: fewer artifacts, better made) and
names the render's worst frame. In QC it only advises, so it never fails a shot on its own.
In Night School it counts as one third of the practice score (`docs/practice.md`), next to
QC's measurements and the AI picture review.

## How it works

- **CLIP ViT-B/16** turns each frame into 512 numbers that describe what it looks like. It is
  an ONNX export (Xenova, MIT), 346 MB, checksum-verified, and runs on the CPU. It downloads to
  `data/models/` on first use.
- A **linear head**, trained here in seconds, turns those numbers into two scores:
  plausibility (no melted hands, extra limbs or warped faces) and overall quality.
- It is trained on:
  - **RichHF-18K** (Google Research, CVPR 2024 best paper). About 18,000 AI-generated images
    from Pick-a-Pic, each rated by people for artifacts and overall quality.
    `references fetch` downloads it one file at a time (about 3 GB in total). It keeps only the
    CLIP features and the scores, about 20 MB, and deletes the images.
  - **Your examples**: images in `data/references/good` and `data/references/bad`. They also
    come from the **Good** / **Botched** buttons on practice renders on the Night School page,
    which save 4 frames and retrain at once. Each of your images counts 5 times as much as a
    public one (`judge.user_weight`). Your examples look like this studio's video frames;
    RichHF's are still images.
- Accuracy is measured on RichHF's test split, which training never sees. The figure is how
  often the judge orders two images the way people did: 50% is a coin flip.

## Measured here

These numbers come from a CPU in the build environment. The judge was trained on the RichHF
validation split plus part of the train split, then tested on the 955 test images:

| Trained on | Artifact pairs ranked like people | Artifact correlation |
|---|---|---|
| 700 images (first try) | 74% | 0.62 |
| 7,000 images | 76% | 0.65 |
| 10,000 images | 77% | 0.67 |

More training images help. Expect a few points more with the whole train split.

## Set it up on the PC

```powershell
.\scripts\studio.ps1 references fetch     # about 18k images, 30-60 min on the CPU; resumable
.\scripts\studio.ps1 references train     # seconds; prints the test accuracy
.\scripts\studio.ps1 references status
```

After that, every QC run and practice batch includes the judge's scores. If you want to stop
using it, set `judge.enabled: false`.

## Feeding it your own examples

- Botched: frames with melted bodies, smeared faces, boiling textures or broken hands. Your
  own failed renders are the most useful, because they are exactly what the judge has to spot.
- Good: frames you consider well made. Ideally use the same kind of content (stylized video
  frames), so the judge does not learn "photo = good".
- Any size of JPEG, PNG or WebP works. Use images you are allowed to use. They stay on this
  PC, are never committed to git and are never published.
- A few dozen of each already move the judge. Run `references train` after adding a folder
  of them.

## Limits

- One frame at a time. The judge sees artifacts in a frame, not flicker over time; QC's
  stability and flicker scores cover that.
- RichHF rates text-to-image stills from 2023 models. Your examples are what teach it this
  studio's look.
- A linear head on CLIP features is coarse. It ranks clearly botched frames below clean ones
  well, but it does not see small details such as a sixth finger at 480p.
