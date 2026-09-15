# Feature Puzzles Data-Material Pipeline

Turns one parent RGB image into publication-ready **Feature Puzzles** training-data
material for two tasks, following the official Puzzles crop + PnP camera
construction:

1. **Promptable segmentation** — globally consistent object instance masks
   (SAM2.1, Grounded-SAM, SAM, or FastSAM) warped into each crop (stable parent
   instance IDs across the sequence).
2. **Open-vocabulary segmentation** — high-resolution DINOv3 teacher features
   (plain DINOv3 or DINOv3 + JAFAR RGB-guided upsampling) warped into each crop,
   visualized with one shared PCA basis (HSV rainbow colour scheme by default).

Geometry comes from **MoGe-2** (metric depth, point map, intrinsics, normal),
with depth discontinuity edges removed so the point cloud has clean object
surfaces. Each crop gets its own world-to-camera pose via `solvePnPRansac`
against the shared crop intrinsic `boxK`, exactly like `Image2Clips.py`.

## Quick start

```bash
cd puzzles/ICLR2027/FeaturePuzzlesPipline
python run_pipeline.py all --image street.png --output outputs/street
```

Run a single stage (cached by config + input hash):

```bash
python run_pipeline.py fastsam --image room1.png --output outputs/room1
python run_pipeline.py moge   --image room1.png --output outputs/room1
python run_pipeline.py puzzles --image room1.png --output outputs/room1
python run_pipeline.py dinov3  --image room1.png --output outputs/room1
python run_pipeline.py viz2d   --image room1.png --output outputs/room1
python run_pipeline.py viz3d   --image room1.png --output outputs/room1
```

Force-rerun a stage (ignores cache): add `--force`.

## Stages

| Stage | Module | Output dir | Key outputs |
|------|--------|-----------|-------------|
| 1 Masks | `sam2_masks.py` / `grounded_sam_masks.py` / `sam_masks.py` / `fastsam_masks.py` | `01_fastsam/` | `masks.npz`, `instance_ids.png`, `mask_overlay.png` |
| 2 MoGe-2 | `moge2_geometry.py` | `02_moge2/` | `geometry.npz` (incl. `depth_edge`), `depth.png`, `normal.png`, `parent_pointcloud.ply` |
| 3 Puzzles | `puzzles_sequence.py` | `03_puzzles/` | `sequence.npz`, `crops/`, `masks/`, `poses/`, `sequence_strip.png`, `crop_regions.png` |
| 4 DINOv3 | `dinov3_features.py` | `04_dinov3/` | `features.npz`, `parent_pca.png`, `parent_pca_hr.png`, `crop_pca/` |
| 5 Viz 2D | `visualize_2d.py` | `05_visualization/` | strips + `pipeline_montage.png` |
| 5 Viz 3D | `visualize_3d.py` | `05_visualization/` | `cameras_*.png`, `scene_*.glb` |
| 6 Aug | `puzzles_sequence.py` | `06_augmentation_comparison/` | rotated-crop comparison strip |

## Coordinate convention

- MoGe outputs a **camera-frame** point map. We treat that camera frame as the
  **world** frame for the whole sequence.
- Every crop shares one intrinsic `boxK` (calibrated from the template box).
- Each crop gets its own **world-to-camera** pose `w2c` from PnP, where the
  3D points are the MoGe camera-frame points and the 2D points are template-box
  pixel coordinates.
- 3D exports place the parent point cloud in the world frame and draw each
  crop's frustum from `c2w = inv(w2c)`.

## Configuration

`configs/default.yaml` controls all stages. Notable options:

- **`masks.backend`** — `sam2` (default; SAM2.1 hiera-large "segment everything"
  via a point grid with stability-score filtering + containment merge for
  object-level units, plus a 2nd hole-fill pass that prompts SAM2 again in
  unmasked regions so segmentable areas the first grid missed still get a mask),
  `grounded_sam` (Grounding DINO + SAM, text-prompted), `sam` (SAM ViT-H), or
  `fastsam`. Weights are local snapshots under `weights/sam2.1-hiera-large` /
  `weights/sam-vit-huge` / `weights/grounding-dino-tiny`.
- **`moge.depth_edge_rtol`** — relative tolerance for depth-edge
  removal (matches the official `--threshold`); edge pixels are dropped from
  the valid mask and the exported point cloud.
- **`puzzles.sampling`** — `coverage` (default; place N crops on a jittered grid
  matched to the image aspect ratio — centres are randomised by
  `coverage_jitter` and sizes target ~`coverage_target` of the image, so the
  crops cover ~90%+ of the scene with some randomness, not a rigid full tile) or
  `trajectory` (ordered overlapping crops, the official Image2Clips behaviour).
  `03_puzzles/crop_regions.png` shows every crop box on the parent image.
- **`dinov3.mode`** — `dinov3` (default; DINOv3-B/16 native dense patch tokens on
  a high-res aspect-preserved parent — the DINOv3 paper Fig 4 recipe, finer
  resolution = crisper features) or `dinov3_jafar` (DINOv3 + JAFAR RGB-guided
  upsampler). `04_dinov3/parent_pca_hr.png` is the full-resolution PCA feature
  image; `parent_pca.png` is the native token-resolution map.
- **`dinov3.pca_color`** — PCA→colour scheme: `cmap` (default; map the 1st PC
  through a matplotlib colormap — `pca_cmap: Spectral`, the MoGe depth-visualisation
  colormap, for a smooth perceptually-pleasing gradient), `hsv` (3 comps → HSV
  rainbow), `best` (search 8 sign-flips × 6 orderings for the most colourful
  direct-RGB, the DINOv3 report recipe), or `rgb` (legacy direct comp→R/G/B).
  `pca_hue_offset` rotates the hue wheel. Crop PCA colours reuse the
  parent's per-component percentile bounds **and the parent's feature scale**, so
  an image region keeps the same colour in the parent map and in every crop that
  contains it (cross-view colour consistency).

CLI `--seed` overrides the seed.

## Tests

```bash
python -m pytest tests/ -q
```

CPU unit tests cover crop geometry, overlap constraints, mask-ID propagation,
PCA consistency, pose conversion, reprojection, and manifest serialization.

## Dependencies

`torch`, `opencv-python`, `numpy`, `ultralytics` (FastSAM), `timm` (DINOv3),
`transformers` (Grounding DINO + SAM), `moge` (MoGe-2, from
`github.com/microsoft/MoGe`), `utils3d` (MoGe dep, also used for depth-edge
removal), `trimesh` (GLB), `matplotlib` (3D PNG), `scikit-learn`, `pyyaml`,
`tqdm`. The JAFAR upsampler is loaded from `puzzles/repo/JAFAR`.

See `DESIGN.md` for the full design.
