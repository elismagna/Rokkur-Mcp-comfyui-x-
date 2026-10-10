# 3D studio: open, measure, fix, convert and make printable models

Dashboard: **3D** (`/ui/3d`). API: `/models3d`. CLI: `rokkur-studio mesh-info FILE`,
`mesh-convert IN OUT [--fit MM] [--floor] [--repair] [--ascii]` (files only, no database) and
`model3d-list`. Every model is a `Model3D` row (migration `0006`) with its working file at
`data/3d/<id>/model.stl`, the original next to it, and its measurements in `stats`. An edit
makes a new model pointing at the one it came from (`parent_id`), so nothing is overwritten.

## What it can do

| Way | Where it runs | Tool | Notes |
|---|---|---|---|
| Open STL, OBJ, PLY, GLB | CPU, at once | built in (`mesh/io.py`, numpy only) | binary and ASCII STL; OBJ polygons fan-triangulated; binary/ASCII PLY; GLB triangle meshes |
| Measure | CPU | built in | size, triangles, surface, volume, closed/watertight, open edges, flat triangles, inside-out |
| Fit size, scale, rotate, mirror, centre on the plate, flip, repair, combine | CPU, at once | built in (`mesh/ops.py`) | mirror keeps the triangles facing out; repair welds points, drops flat triangles and turns an inside-out model right; real holes are not filled |
| Export STL (binary or ASCII), OBJ, PLY, GLB | CPU | built in | |
| Picture to relief or lithophane | CPU, at once | built in (`ops.relief_from_gray`) | brightness is height on a solid base; inverted for a backlit lithophane; always closed |
| LiDAR or point cloud to a 2.5D solid | CPU, at once | built in (`ops.relief_from_points`) | PLY/OBJ vertices or an XYZ/PTS/CSV list; the highest point per grid cell becomes the top of a closed solid; right for terrain, walls, reliefs; not for the back of an object |
| Point cloud to a full surface | CPU job | Open3D Poisson (MIT), optional | `pip install open3d` in the studio's environment; offered only when importable |
| One picture to a 3D shape | GPU job, this PC or the cloud server | ComfyUI Hunyuan3D 2.0 (`workflows/mesh_hunyuan3d_i2m`) | untextured; about 6 GB VRAM; licence below |
| Photo set to a model | CPU/GPU job | Meshroom `meshroom_batch` (MPL-2.0), optional | 10+ overlapping photos; can take an hour; the largest mesh Meshroom writes is used |
| Video to a model | CPU/GPU job | FFmpeg frames, then Meshroom | `three.video_fps` frames a second, at most `three.max_video_frames`, evenly spread |

A phone LiDAR app (Polycam, 3D Scanner App, Scaniverse) usually exports a mesh: open it
directly. A raw point cloud uses the 2.5D solid or Poisson.

**Local and cloud.** Everything built in runs on the studio's own CPU. The picture-to-3D job
uses `ctx.comfy_for(target)` like pictures, sound and video: this PC's ComfyUI with a GPU
lease, or the cloud server (cost recorded as `cloud_gpu_minutes`). Meshroom and Open3D run
where the worker runs; the cloud server only speaks ComfyUI, so they are local only.

## The picture-to-3D workflow

Adapted node for node from Comfy-Org's template `3d_hunyuan3d_image_to_model` (MIT,
github.com/Comfy-Org/workflow_templates) and its tutorial (docs.comfy.org/tutorials/3d/hunyuan3D-2):
ImageOnlyCheckpointLoader, CLIPVisionEncode (crop none), Hunyuan3Dv2Conditioning,
ModelSamplingAuraFlow 1.0, EmptyLatentHunyuan3Dv2 3072, KSampler euler/normal 20 steps cfg 8,
VAEDecodeHunyuan3D (8000 chunks, octree 256), VoxelToMesh (surface net, 0.6), SaveGLB. Input
names were checked against ComfyUI's `comfy_extras/nodes_hunyuan3d.py` and `nodes_save_3d.py`.
The page offers the octree resolution (128 / 256 / 384) and the seed; a CUDA OOM fails the
model with a hint to lower the octree or use the cloud server.

Model: `checkpoints/hunyuan3d-dit-v2_fp16.safetensors` (4.59 GB) from
huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged (`scripts/install-models.ps1` moves it).
**Licence:** the Tencent Hunyuan 3D 2.0 Community License; its territory excludes the EU, the
UK and South Korea, and large-scale use needs Tencent's permission. Read it before using the
results commercially.

## Verified, and not

- Tested here (`tests/test_models3d.py`, 7 tests): every format round-trips to the exact
  volume; edits; relief, lithophane and scan solids are closed; import refuses a bare point
  cloud with the way forward; export; the picture-to-3D job through a fake ComfyUI that
  returns a GLB (lease, cost, OOM message); photos and video through a fake `meshroom_batch`
  (frames really extracted by FFmpeg); page, API and CLI. The page and its canvas 3D view
  were rendered in Chromium at desktop and phone width with no script errors.
- Not run yet: Hunyuan3D on a real GPU, real Meshroom, real Open3D. Those are the PC
  acceptance tests: `comfy-check` for `mesh_hunyuan3d_i2m`, one picture of a single object on
  a plain background, then print the STL.
