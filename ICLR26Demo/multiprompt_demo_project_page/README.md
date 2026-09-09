# Promptable Spatial Segmentation — web demo

Interactive demo for VGGT-Omega multiprompt segmentation on **static** and
**dynamic** scenes. Click a point on the reference frame, get the object
segmented in every view and lifted into a 3D point cloud you can play back
frame by frame.

```bash
./run_demo.sh          # http://localhost:7860
./run_demo.sh 9000     # different port; viser viewers use 9001+
```

---

## What it does

1. **Point prompts → mask prompts.** Clicks on the reference frame go to a local
   SAM ViT-H (`SamPredictor`), whose mask becomes the reference-view prompt for
   the VGGT-Omega segmentation head. This replaces the automatic
   `sam_top_k_masks` seeding used by the offline static pipeline; everything
   downstream is unchanged.
2. **Segment everything.** A dense SAM point grid decomposes the reference frame
   and keeps the largest non-overlapping regions as prompts — no clicking
   required. `top_k = 1` gives a single-mask run, higher values a multi-mask one.
   The input can be a video, a single image, or several images dropped together
   and treated as an ordered multi-view set.
3. **Static vs dynamic.** Two segmentation regimes, picked in the UI (see below).
4. **3D output.** The masks are backprojected into a coloured point cloud,
   viewable interactively in viser (per-frame playback, follow camera) and
   exported as a composited video: follow camera full-frame, bird's-eye view
   inset top-right, input video inset bottom-right.

## Static vs dynamic

|                | Static | Dynamic |
| -------------- | ------ | ------- |
| Propagation    | 6-frame overlapping clips, each anchored on the reference frame plus a hand-off mask from the previous clip | one forward pass over the whole clip, frames in temporal order |
| Cross-object exclusivity | yes | yes |
| 3D-connectivity filter | yes | **no** |
| Geometry       | sensor depth + poses when the source is a ScanNet scene, otherwise predicted | predicted |

The asymmetry in the geometric filter is the important part. `geo_cc` keeps only
the mask component that is 3D-connected to the seed, which assumes a pixel's
world position depends on the camera pose alone. That holds in a rigid scene and
stops masks from drifting onto neighbouring objects. In a dynamic scene it is
false: a moving subject lands in different world voxels every frame, so the
filter would delete the tracked object instead of protecting it. Dynamic mode
therefore relies on the head's cross-view attention over the full temporal
context, with no geometric post-processing.

## Geometry

Static ScanNet scenes come with sensor depth and poses. Everything else — any
uploaded video — has neither, so the full VGGT-Omega reconstruction head predicts
depth and camera poses. Predicted depth is only defined up to scale, so the
reconstruction is rescaled to put its median depth at 3 m; the renderer's camera
offsets, point size and depth cull are all sized relative to scene depth rather
than to fixed metres, which is what makes the same code frame a ScanNet room and
an outdoor tracking shot. The follow camera also narrows its field of view to
compensate for standing behind the camera that captured the points, so the
subject keeps the same size on screen in both the video and the viser view.

## Layout

```
app/
  paths.py        repo path bootstrap + checkpoint locations
  frames.py       video / image-folder / ScanNet frame sampling
  geometry.py     predicted depth + poses; padded-square → working-resolution
  sam_prompt.py   SamPredictor point prompts + dense-grid segment-everything
  segment.py      static clip tracking vs dynamic one-shot
  static_ref.py   loads the offline static pipeline as a library
  render.py       point cloud, octree, follow/BEV renders, cloud export
  compose.py      picture-in-picture compositing (rounded, translucent insets)
  viser_view.py   interactive viewer: playback + follow camera
  session.py      session state and the single-slot GPU job queue
  server.py       FastAPI routes
web/              single-page frontend (no build step)
tools/            CLI smoke tests and debugging helpers
runs/             per-session artefacts (not tracked)
```

Segmentation, the 3D-connectivity filter, point collection and the LingBot
renderer are **not** reimplemented here. `static_ref.py` imports
`../vggt_omega_multiprompt_static_scene/render_multiprompt_demo.py` and calls it,
so the demo and the paper results come from the same code.

## Command line

The whole pipeline runs without the UI:

```bash
# dense-grid prompts on a dynamic clip
python -m app.pipeline \
    --source ../vggt_omega_multiprompt_dynamic_scene/03_scale_change/04_man_riding_bike_towards_camera.mp4 \
    --scene_mode dynamic --n_frames 24 --prompt_mode dense --top_k 4 \
    --out_dir runs/cli_bike

# a single click (normalised coordinates) on a static ScanNet scene
python -m app.pipeline \
    --source ../../data/dataset/scannet/test/scene0046_00 \
    --scene_mode static --n_frames 60 --prompt_mode points \
    --points '[{"points":[{"x":0.47,"y":0.50,"label":1}]}]'

# interactive viewer for an existing result
python -m app.viser_view --cloud runs/cli_bike/*_cloud.npz --port 7861
```

`tools/smoke_api.py` drives the HTTP API exactly as the browser does, and
`tools/debug_dynamic.py` sweeps window sizes for a single point prompt.

## Outputs

Per session, under `runs/<session-id>/`:

| File | Contents |
| ---- | -------- |
| `*_pip.mp4` | composited result: follow camera + BEV and input insets |
| `*_masks_overlay.mp4` | 2D masks drawn on the input frames |
| `*_hl_follow.mp4`, `*_hl_birdeye.mp4` | raw point-cloud renders |
| `*_cloud.npz` | per-frame point cloud (`xyz`, `rgb`, `seg`, `obj`, `frame`, `c2w`, `K`) |
| `*_pred_masks.npz` | `pred_masks (S,K,H,W)` and the reference-frame `prompt_masks` |
| `*_prompt.png` | reference frame with the seed masks tinted and the clicks marked |
| `*_manifest.json` | everything needed to reproduce the run (see below) |
| `session_log.txt` | the full run log, including timings and any warnings |

Directories are named after the clip and the scene mode —
`runs/04_man_riding_bike_towards_camera_dynamic/` — with a `_2`, `_3` suffix when
that name is taken. A directory is only created once a segmentation actually
runs, so picking clips in the gallery leaves nothing behind, and a second run of
the same clip lands beside the first instead of overwriting it.

The manifest is a complete record of the request, not just of the output. Besides
the result filenames it stores the resolved source path, the sampled
`frame_indices` and working resolution, the **verbatim prompt** — every click's
normalised `x`/`y` and include/exclude `label`, or the grid density and object
count in dense mode — and the switches the run used (`mask_threshold`,
`exclusive`, `geo_cc`, render mode and size). Re-running `app.pipeline` with
those values reproduces the result exactly; `--points` takes the `prompt.objects`
block as-is.

## Requirements

Runs in the `lingbot-map` conda environment (PyTorch 2.8 + CUDA, Open3D,
`segment_anything`, viser, the compiled `render_cuda_ext`). Only FastAPI,
uvicorn and `python-multipart` are extra — see `requirements.txt`. Point
`PYTHON=/path/to/python` at a different interpreter if needed.

Checkpoints (paths in `app/paths.py`):

- `checkpoints/vggt_omega_1b_256_text.pt` — backbone and reconstruction head
- `weights/seg_omega_lora_multiprompt/checkpoint-best.pth` — LoRA + seg head
- `checkpoints/sam/sam_vit_h_4b8939.pth` — SAM ViT-H
