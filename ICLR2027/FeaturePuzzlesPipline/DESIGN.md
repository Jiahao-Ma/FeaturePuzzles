# Feature Puzzles Data-Material Pipeline Design

## Goal

Build a reusable, staged CLI that turns one parent RGB image into publication-ready
Feature Puzzles training-data material for:

1. promptable segmentation, using globally consistent FastSAM instance masks; and
2. open-vocabulary segmentation, using high-resolution DINOv3 teacher features.

The pipeline also reconstructs monocular geometry with MoGe-2, follows the official
Puzzles crop and PnP camera construction, and exports static and interactive 3D
visualizations.

All new source code, configuration, tests, documentation, and generated output live
under `puzzles/ICLR2027/FeaturePuzzlesPipline/`.

## Scope

- Provide a generic command-line pipeline.
- Run that pipeline on `room1.png` to generate the initial flowchart material.
- Export intermediate machine-readable arrays as well as editable visual assets.
- Use unaugmented crops for the main sequence.
- Export a separate random-camera-rotation augmentation comparison.
- Do not perform open-vocabulary text classification or feature clustering. The
  open-vocabulary target is the per-crop DINOv3 teacher feature, visualized with one
  shared PCA basis.

## Architecture

The CLI supports individual cached stages and an `all` command:

```bash
python run_pipeline.py all \
  --image room1.png \
  --output outputs/room1
```

### Stage 1: FastSAM parent masks

Run FastSAM once on the parent image. Preserve the complete `(M, H, W)` binary-mask
stack because FastSAM proposals may overlap. Also create a deterministic, mutually
exclusive instance-ID map for visualization. Every crop inherits the same parent
instance IDs; FastSAM is not rerun independently on crops.

### Stage 2: MoGe-2 geometry

Load `Ruicheng/moge-2-vitl-normal` through the official MoGe-2 API and infer metric
depth, valid geometry mask, camera intrinsics, normal map, and camera-frame point
map. Cache the raw arrays and export depth, normal, and parent point-cloud assets.

### Stage 3: Puzzles sequence and cameras

Follow the official Puzzles `Image2Clips.py` geometry:

1. Generate an ordered sequence of six overlapping crop boxes.
2. Crop parent RGB, point map, geometry validity, depth, and FastSAM masks.
3. Resize every crop to the template crop resolution.
4. Calibrate one template intrinsic matrix `boxK`.
5. For each crop, sample valid 2D--3D correspondences and estimate a world-to-camera
   matrix with `cv2.solvePnPRansac`.
6. Reject a crop when PnP quality checks fail and resample it.

The main output uses the clean crop sequence. A separate optional branch applies the
official random camera rotation and point-cloud rasterization to illustrate geometry
augmentation.

### Stage 4A: Promptable-segmentation labels

Crop and nearest-neighbor resize every parent binary mask. Store per-crop binary
masks, visible instance IDs, exclusive label maps, and colored overlays. Parent
instance IDs remain stable across the complete sequence.

### Stage 4B: DINOv3 teacher targets

Run `vit_base_patch16_dinov3.lvd1689m` once on a high-resolution parent image. Warp
the dense parent feature map into each crop using normalized parent-image crop
coordinates and bilinear sampling. L2-normalize per-pixel features.

Fit one three-component PCA transform on sampled parent features, then apply that
same transform and color range to the parent and all crop features. PCA RGB colors
therefore have a consistent meaning across the sequence. PCA is a visualization of
continuous teacher features, not a semantic class label.

### Stage 5: Visualization and export

Export:

- individual transparent-background PNG assets;
- sequence strips and mask/feature panels;
- high-resolution 3D PNGs;
- GLB scenes for interactive inspection;
- RGB-, mask-ID-, and DINO-PCA-colored point clouds;
- crop camera frustums derived from PnP poses; and
- a final flowchart montage:

`RGB -> FastSAM -> MoGe-2 -> Puzzles crops -> PnP cameras -> masks / DINOv3 features`

## Source Layout

```text
FeaturePuzzlesPipline/
├── DESIGN.md
├── README.md
├── run_pipeline.py
├── fastsam_masks.py
├── moge2_geometry.py
├── puzzles_sequence.py
├── dinov3_features.py
├── visualize_2d.py
├── visualize_3d.py
├── pipeline_utils.py
├── configs/
│   └── default.yaml
├── tests/
├── room1.png
└── outputs/
    └── room1/
```

No new implementation code is placed in the repository's shared `src/` tree.
Existing repository code and the official Puzzles repository are references only;
required pipeline behavior is exposed through modules in this directory.

## Output Contract

```text
outputs/room1/
├── manifest.json
├── 01_fastsam/
│   ├── masks.npz
│   ├── instance_ids.png
│   └── mask_overlay.png
├── 02_moge2/
│   ├── geometry.npz
│   ├── depth.png
│   ├── normal.png
│   └── parent_pointcloud.ply
├── 03_puzzles/
│   ├── sequence.npz
│   ├── crops/
│   ├── masks/
│   ├── poses/
│   └── sequence_strip.png
├── 04_dinov3/
│   ├── parent_features.npz
│   ├── crop_features.npz
│   ├── parent_pca.png
│   └── crop_pca/
├── 05_visualization/
│   ├── cameras_rgb.png
│   ├── cameras_mask.png
│   ├── cameras_feature.png
│   ├── scene_rgb.glb
│   ├── scene_mask.glb
│   ├── scene_feature.glb
│   └── pipeline_montage.png
└── 06_augmentation_comparison/
```

`manifest.json` records input identity, model identifiers, parameters, random seed,
crop boxes, shared crop intrinsics, per-crop world-to-camera and camera-to-world
matrices, PnP diagnostics, array paths, and software versions.

## Validation and Error Handling

- FastSAM: fail clearly when no usable masks remain after filtering.
- MoGe-2: validate shapes, finite values, positive depth, and valid-point coverage.
- Crop sampling: resample boxes that violate overlap or valid-point thresholds.
- PnP: require success, enough inliers, bounded reprojection error, and a valid
  rotation matrix; resample failed crops.
- Masks: use nearest-neighbor resize and assert stable parent instance IDs.
- DINOv3: validate feature dimensions and L2 norms; use one PCA transform globally.
- 3D: verify `w2c`/`c2w` inversion, frustum direction, and crop reprojection.
- Cache: invalidate a stage when its model identity, input hash, or relevant
  parameters change.

Failures include the stage, affected crop, relevant metric, and remediation in the
error message. Partial intermediate outputs remain available for inspection.

## Testing

CPU unit tests cover crop geometry, overlap constraints, mask-ID propagation, crop
warps, PCA consistency, pose conversion, reprojection metrics, manifest
serialization, and cache invalidation.

A GPU smoke test runs the complete pipeline on `room1.png` with reduced model
settings where appropriate. Completion requires:

- all expected files;
- six valid crop poses;
- valid prompt masks;
- aligned DINOv3 crop features;
- readable PNG, PLY, NPZ, and GLB outputs; and
- successful static and interactive 3D exports.

