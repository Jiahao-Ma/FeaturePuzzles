# Prompt Matching on VGGT / FastVGGT

Training-free, **prompt-driven patch matching** on top of a VGGT (or
FastVGGT) backbone. Point at one reference view, draw a mask (or a bbox)
over the region you care about, and get back per-patch similarity maps +
binarised masks on every other view in the same scene — no fine-tuning, no
extra heads.

## 1. What this package uses from VGGT

* **Used**: the VGGT `Aggregator` forward pass and its
  `aggregated_tokens_list` (layer-wise `[B, S, P, 2*embed_dim]` tokens), the
  image preprocessor (`vggt.utils.eval_utils.get_vgg_input_imgs` when the
  FastVGGT tree is on `sys.path`, otherwise
  `vggt.utils.load_fn.load_and_preprocess_images` — geometrically
  identical), and the public `VGGT` class (local `.pt` **or**
  `VGGT.from_pretrained(...)`).
* **Not used / not touched**: the camera / depth / point / track heads,
  FastVGGT's token merging (`merging/`). We *instantiate* FastVGGT's
  `VGGT(merging=0, ...)` so the merging hooks stay idle, and we read hidden
  tokens directly — this is similar in spirit to FastVGGT's metric
  computation but never invokes the merging kernel.

No file inside the upstream VGGT / FastVGGT tree is modified.

### FastVGGT-specific quirks that this package handles for you

* FastVGGT's aggregator only stores intermediates at
  `block_num ∈ {4, 11, 17, 23}`, so `aggregated_tokens_list` has length 4
  (not 24 like stock VGGT). `extractor.py` auto-translates
  `--layers 17 23` into the right `output_list` indices on either layout.
* `VGGT(merging=0, merge_ratio=0.9, vis_attn_map=False)` is passed
  automatically when those parameters exist on the constructor.
* `model.update_patch_dimensions(W_p, H_p)` is called before the forward
  pass when the method exists; FastVGGT crashes inside
  `merging.merge.token_merge_bipartite2d` without it.
* Weights are loaded with `strict=False`; extraneous heads
  (`point_head.*`, `track_head.*`) are silently ignored.

## 2. Quick start

```bash
python prompt_matching/prepare_example.py \
    --source_dir /path/to/your/multi_view_images \
    --n 6

python -m prompt_matching.run \
    --image_dir   prompt_matching_data/example/images \
    --prompt_mask prompt_matching_data/example/prompt_mask.png \
    --ref_view_idx 0 \
    --output_dir  prompt_matching_outputs/run1 \
    --layers 17 23 \
    --query_mode mean \
    --threshold  otsu
```

Three outputs land in `--output_dir`:

* `00_ref_view.png` — reference view with the user prompt outlined in red.
* `01_heatmaps_grid.png` — every view overlaid with the cosine-similarity
  heat-map (JET colormap).
* `02_masks_grid.png` — every view with its binarised mask overlaid in
  green.
* `config.json` — full record of arguments + resolved meta (patch grid,
  layers, amp dtype, …).

## 3. Parameters

| Flag | Default | Meaning |
| --- | --- | --- |
| `--image_dir` | — | folder of same-scene multi-view images |
| `--prompt_mask` / `--prompt_bbox` | — | mutually exclusive; either a binary PNG over the reference image, or `x1,y1,x2,y2` in ORIGINAL pixel coords |
| `--ref_view_idx` | 0 | index into `--image_dir` sorted listing |
| `--ckpt_path` | `None` | local VGGT weights `.pt`; if missing, auto-downloads `facebook/VGGT-1B` |
| `--layers` | `17 23` | aggregator block_nums to average. Stock VGGT accepts any index 0..23 (negatives allowed). FastVGGT only exposes `{4, 11, 17, 23}`. |
| `--query_mode` | `mean` | `mean`: pool prompt tokens → single dot product. `max`: per-prompt-token dot product then max |
| `--threshold` | `otsu` | `otsu` or `percentile` (see `--percentile`) |
| `--percentile` | `90` | used when `--threshold percentile` |
| `--abs_threshold` | `None` | absolute cosine floor; patches below are dropped regardless of the per-view percentile/Otsu split. Use 0.45–0.60 to force empty masks on views where the prompted object is not visible. |
| `--coverage_threshold` | `0.3` | min per-patch mask coverage before a patch is kept |
| `--per_view_threshold` / `--global_threshold` | per-view | threshold each map separately or all jointly |
| `--heatmap_alpha`, `--mask_alpha` | `0.55`, `0.45` | overlay opacity |

## 4. Layout of the package

```
prompt_matching/
├── NOTES.md              # Step 0 read-through of VGGT source
├── __init__.py           # re-exports the public functions
├── extractor.py          # Step 2: VGGT forward + normalised patch tokens
├── prompt.py             # Step 3: mask / bbox → patch mask
├── matcher.py            # Step 4: cosine similarity, Otsu, upsample
├── visualize.py          # Step 5: heat-map / mask overlays + grid maker
├── run.py                # Step 6: CLI gluing Steps 2-5 together
├── prepare_example.py    # Step 7: toy data generator
├── generate_masks.py     # YOLOv8-seg (+ FastSAM fallback) auto prompt masks
├── run_scene_batch.py    # batch CLI: all sub-scenes × 5 masks + summary grids
├── weights/              # cached YOLOv8-seg / FastSAM checkpoints
└── README.md             # this file
```

Smoke-test each module individually:

```bash
python -m prompt_matching.prompt       # no GPU needed
python -m prompt_matching.matcher      # no GPU needed
python -m prompt_matching.visualize    # no GPU needed
python -m prompt_matching.extractor --image_dir <dir>   # needs GPU + weights
```

Each of these prints shapes / statistics that the plan asks you to eyeball.

### A concrete smoke run on this repo

From `TrainingFreeVGGTMatching/` with the official VGGT-1B checkpoint at
`/mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt`:

```bash
python prompt_matching/prepare_example.py \
    --n 8 --box 1000,150,1350,500 \
    --dest_dir prompt_matching_data/kong

python -m prompt_matching.run \
    --image_dir   prompt_matching_data/kong/images \
    --prompt_mask prompt_matching_data/kong/prompt_mask.png \
    --ref_view_idx 0 \
    --ckpt_path   /mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt \
    --output_dir  prompt_matching_outputs/run_kong \
    --layers 17 23 --query_mode mean \
    --threshold percentile --percentile 95
```

Expected console output:

```
[extractor] N=8  preprocessed=(350,518)  patch_size=14  grid=(25,37)
[extractor] tokens.shape=(8, 25, 37, 2048)  norm.mean=1.0000
[run] Step 4a self-match OK mean=0.95   # ≥ 0.90 assertion
[run] Step 4b OK  positives_per_view=[47, 47, 47, 47, 47, 47, 47, 47]
```

The resulting `02_masks_grid.png` has the green mask centred on the KONG
poster's gorilla face in every view, demonstrating the full pipeline end
to end.

### Batch validation across all sub-scenes

If you have `examples/<scene>/*.JPG` with multiple sub-scenes,
`run_scene_batch.py` auto-generates 5 prompt masks per scene with
**YOLOv8-seg** (plus **FastSAM** fallback for scenes outside COCO's 80
classes), runs the full pipeline once per scene, and writes a single
`summary.png` per scene.

```bash
pip install ultralytics                       # one-time
python -m prompt_matching.run_scene_batch \
    --examples_dir examples \
    --data_root    prompt_matching_data \
    --out_root     prompt_matching_outputs \
    --n_views 6 --k_masks 5 \
    --ckpt_path /mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt \
    --layers 17 23 --query_mode mean \
    --threshold percentile --percentile 95 \
    --abs_threshold 0.5
```

`--abs_threshold` is crucial when validating across many views: the
per-view percentile always picks the top 5% of some view, even if the
prompted object is not in that view at all. The absolute cosine-similarity
floor forces an empty mask (rendered with a red X + `NO MATCH` banner) on
views whose best patch falls below the floor. `0.5` is a good default
given VGGT-1B's similarity statistics (object-present views tend to peak
at 0.80+; object-absent views peak around 0.35–0.60).

Output layout:

```
prompt_matching_data/<scene>/
    images/           # 6 frames evenly sampled from examples/<scene>
    masks/            # mask_00_<class>.png ...  mask_04_<class>.png (YOLO+FastSAM)
prompt_matching_outputs/<scene>/
    mask_00_<class>.png    # one strip per mask: ref+outline | view1_heat+mask | ...
    mask_01_<class>.png
    ...
    summary.png            # 5-row grid: rows = masks, cols = ref + other views
    report.json            # self-match, per-view sim range, positives, ...
batch_report.json          # aggregated at out_root/
```

Each `summary.png` row is: `ref_view_with_prompt_outlined(col 0) |
view1_heatmap+mask_contour(col 1) | ... | viewN-1_heatmap+mask_contour`,
so for one scene you can visually check all masks at once.

## 5. Known limitations

* **Repeated textures**: dense patch features give similar scores to
  visually similar regions (windows, tiles, leaves). Use tighter prompts
  (`mean` query + small mask) or a higher `--percentile` threshold to
  suppress.
* **Extreme viewpoints**: if an object is severely occluded or only
  partially in frame, Otsu may cut off entirely and the mask will be
  mostly empty on that view. `--global_threshold` sometimes helps.
* **Preprocessing mismatch**: prompts are mapped through the exact same
  resize + centre-crop that `load_and_preprocess_images(mode="crop")`
  performs. If you switch VGGT to `mode="pad"` or a different
  `img_size`, update `prompt._resize_mask_to_model_input` to match.
* **D = 2 * embed_dim**: the aggregator concatenates frame- and
  global-attention intermediates on the channel axis. This is a VGGT
  specific detail, not a bug — see `NOTES.md §3`.
* **FastVGGT merging** changes the global-attention feature *distribution*
  slightly. In practice prompt matching still works, but if you plug in a
  heavily-merged FastVGGT checkpoint and results look worse than vanilla
  VGGT, drop `--layers` to shallower indices (e.g. `11 17`) where merging
  has less effect.

## 6. Troubleshooting

* `Could not locate a 'vggt' package.` → install FastVGGT, or add a VGGT
  checkout to `PYTHONPATH`. The extractor also falls back to
  `/mnt/dataset/marko/code/Depth/vggt` if that directory exists.
* `assert_self_match` fails → your prompt mask does not actually land on
  any patch; check `--coverage_threshold`, or confirm the mask is over the
  correct reference view. The run logs will print the resolved
  `active_patches` count.
* `Token count mismatch` → VGGT changed its layout (extra / fewer register
  tokens). The error message contains the expected-vs-actual counts;
  update `extractor.py` accordingly.
