# Prompt Similarity Computation — Implementation Notes (English)

> Companion code: `prompt_matching/extractor.py`, `prompt_matching/prompt.py`,
> `prompt_matching/matcher.py`, `prompt_matching/run.py`,
> `prompt_matching/run_scene_batch.py`.
> This document does **not** describe VGGT's architecture. It describes
> exactly **how this repo uses VGGT's patch tokens** to perform a
> training-free, prompt-driven, cross-view matching.

---

## 0. Pipeline in one sentence

Given N images of one scene, a reference view index `ref_view_idx`, and a
binary prompt region drawn on that reference view: run VGGT's
`Aggregator.forward` **once**, take the patch tokens from the requested
layers, L2-normalise them, average across layers, re-normalise; pool the
prompt patches into a single unit vector q (or keep them as a set of K
unit vectors); compute a cosine similarity between q and every patch
token to obtain a dense `[H_p, W_p]` map per view; threshold it to a
binary mask; upsample to original resolution for visualisation. **Zero
training, zero new parameters.**

---

## 1. Which VGGT layers and which kind of token

### 1.1 Source: `Aggregator`'s `aggregated_tokens_list`

`extractor.extract_patch_tokens` performs a single
`model.aggregator(images)` call and obtains
`aggregated_tokens_list, patch_start_idx`. These are the per-block
intermediate hidden states that VGGT's `Aggregator` **explicitly stores**,
each with shape `[B, S, P, 2 * embed_dim]`.

Key facts:

- **Not the attention K matrix** — these are the hidden states **after**
  each attention block. FastVGGT's own token-merging code also computes
  cosine similarity over the hidden states (see `merging/merge.py:48-226`,
  `vggt/layers/attention.py:167-209`); we follow the same convention.
  See `NOTES.md §7`.
- `2 * embed_dim`: when storing intermediates VGGT's Aggregator
  **concatenates** the frame-attention tokens and the global-attention
  tokens **along the channel dimension** (see
  `vggt/models/aggregator.py:375-381`). So the feature dimension is
  `2C = 2 * embed_dim = 2048` (for VGGT-1B with `embed_dim=1024`). This
  is a VGGT internal detail; we use the concatenated 2048-d vector as-is.

### 1.2 Which layers

Controlled by the CLI flag `--layers`, default `[17, 23]` (VGGT has 24
blocks; 22/23 are the deepest two). FastVGGT, however, **only appends an
intermediate when `block_num is in {4, 11, 17, 23}`**, so its list has
length 4 instead of 24.

`extractor._resolve_layer_indices` translates user input automatically:

| `--layers` value | Stock VGGT list index | FastVGGT list index |
| --- | --- | --- |
| 4  | 4  | 0 |
| 11 | 11 | 1 |
| 17 | 17 | 2 |
| 23 | 23 | 3 |

### 1.3 Special-token vs patch-token layout

The 3rd dim of every layer tensor is `P = patch_start_idx + H_p * W_p`:

- `patch_start_idx = 5`: 1 camera token + 4 register tokens, returned by
  the aggregator's forward (`vggt/models/aggregator.py:144-150` and
  `aggregator.py:246-254`).
- The remaining `H_p * W_p` entries are the actual **patch tokens**,
  arranged row-major into a `(H_p, W_p)` grid.

```python
patch_tokens = layer_tokens[:, :, patch_start_idx:, :]  # [B, S, H_p*W_p, 2C]
```

`H_p, W_p` come from the preprocessed image size `(H, W)` divided by
`patch_size` (default 14). `patch_size` is **not** hard-coded:
`patch_size = int(model.aggregator.patch_size)`.

### 1.4 Preprocessing and patch-grid alignment

Preprocessing is done by `vggt.utils.eval_utils.get_vgg_input_imgs` (used
when FastVGGT is on the path), with
`vggt.utils.load_fn.load_and_preprocess_images` as fallback. Both are
geometrically identical to `mode='crop'`:

1. resize so that `W_new = 518`,
   `H_new = round(H_orig * 518 / W_orig / 14) * 14`;
2. if `H_new > 518`, centre-crop along height to 518.

Once we have `(H, W)`, we compute `(H_p, W_p) = (H // 14, W // 14)`.

> **FastVGGT-only requirement:** call
> `model.update_patch_dimensions(W_p, H_p)` before the forward pass,
> otherwise the token-merging kernel asserts
> `tokens_per_img * num_imgs == N` and crashes. We call it
> unconditionally and skip when the method is absent (stock VGGT). See
> `NOTES.md §2`.

### 1.5 Inference precision and `no_grad`

```python
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        aggregated_tokens_list, patch_start_idx = model.aggregator(images_in)
```

- sm >= 8 (A100/H100/L40) -> `bfloat16`; older GPUs fall back to
  `float16`; CPU stays in `float32`.
- We immediately upcast to `.float()` before normalisation and cosine to
  avoid low-precision numerics.

---

## 2. Turning patch tokens into prompt-ready features

### 2.1 Single-layer extraction

```python
patch_tokens = layer_tokens[:, :, patch_start_idx:, :]  # [B, S, H_p*W_p, 2C]
patch_tokens = patch_tokens.float()
patch_tokens = F.normalize(patch_tokens, dim=-1)        # L2 along channel
```

### 2.2 Multi-layer averaging + re-normalisation

```python
accum = sum_over_layers( F.normalize(layer_tokens, dim=-1) )
avg = accum / len(resolved_layers)
tokens = F.normalize(avg, dim=-1)                       # MUST renormalise
```

> Why **re-** normalise? The arithmetic mean of unit vectors is generally
> not a unit vector. Downstream cosine similarity is a plain dot product
> (see §4), and a non-unit length silently breaks the cosine
> interpretation. See `NOTES.md` "hard constraints §7".

The final tensor is `tokens : [N, H_p, W_p, D]` with `D = 2 * embed_dim
= 2048`, dtype `float32`, every vector unit-norm (the run log prints
`norm.mean=1.0000  norm.std=0.0000`).

---

## 3. Prompt region -> patch-grid mask

Code: `prompt.encode_mask_prompt` / `encode_bbox_prompt`.

**Key subtlety: the user's prompt is in original image resolution. It
must go through exactly the same resize + centre-crop as the image
before being downsampled to the patch grid.**

### 3.1 Resize + centre-crop (image-aligned)

`prompt._resize_mask_to_model_input` mirrors `mode='crop'`:

```python
new_W = W                                  # 518
intermediate_H = round(orig_H * (new_W / orig_W) / patch_size) * patch_size
resized = cv2.resize(mask, (new_W, intermediate_H), INTER_AREA)
if intermediate_H > H:
    start_y = (intermediate_H - H) // 2
    resized = resized[start_y:start_y + H, :]
elif intermediate_H < H:
    pad_top, pad_bottom = (H - intermediate_H) // 2, ...
    resized = np.pad(resized, ((pad_top, pad_bottom), (0, 0)))
```

We use `INTER_AREA` so that downsizing **preserves the coverage ratio**
of the mask (we threshold on coverage in the next step).

### 3.2 Downsample to patch grid (coverage-based)

```python
coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA)
patch_mask = coverage >= coverage_threshold              # default 0.3
```

Each patch is a 14x14 pixel block, so `INTER_AREA` downsampling yields
the fraction (0..1) of that patch covered by the prompt foreground. The
default `coverage_threshold = 0.3` keeps a patch only if at least 30 %
of its pixels are covered.

> Fallback: if **no patch** meets the coverage threshold (prompt is too
> small or falls between grid cells), the code falls back to
> `argmax(coverage)` to keep one patch and prevent downstream
> divide-by-zero, with a printed warning.

### 3.3 BBox form

`encode_bbox_prompt` reuses the same resize + crop, then converts the
covered min/max pixel coordinates into a patch rectangle via
`floor / ceil`:

```python
px1 = floor(xs.min() / patch_size)
px2 = ceil((xs.max() + 1) / patch_size)
patch_mask[py1:py2, px1:px2] = True
```

`run_scene_batch.py` calls `encode_mask_prompt(..., coverage_threshold=0.3)`.

---

## 4. Prompt feature and cosine similarity

Code: `matcher.compute_similarity`.

### 4.1 Pull out prompt tokens

```python
ref_tokens = tokens[ref_view_idx]                # [H_p, W_p, D]
prompt_tokens = ref_tokens[mask_t]               # [K, D]
```

`K = patch_mask.sum()` is the number of patches selected in §3. Every
`prompt_tokens[k]` is a unit vector.

### 4.2 Two query modes

#### `query_mode = "mean"` (default)

Arithmetic-average the K unit vectors and **renormalise** to obtain a
single unit vector `q in R^D`:

```python
q = prompt_tokens.mean(dim=0)        # [D]
q = F.normalize(q, dim=-1)           # MUST renormalise
sim = torch.einsum("nhwd,d->nhw", tokens, q)
```

So `sim[i, y, x] = <tokens[i, y, x], q>`. Both sides are unit vectors,
hence this **is** the cosine similarity, in `[-1, 1]`.

Interpretation: collapse the prompt region into a single "prototype
vector" and ask each patch how aligned it is with the prototype. `mean`
is robust to within-prompt diversity and is good for "find the same
semantic object".

#### `query_mode = "max"`

No averaging. Compute a dot product between every patch and every prompt
token, then take the maximum across prompt tokens:

```python
sim = torch.einsum("nhwd,kd->nhwk", tokens, prompt_tokens)
sim = sim.amax(dim=-1)               # [N, H_p, W_p]
```

Interpretation: "is this patch very similar to **any** prompt patch".
More texture-sensitive, higher recall but also more false positives on
visually similar background.

### 4.3 Self-match sanity check

`matcher.assert_self_match`: take the prompt mask, apply it to the ref
view's similarity map, require the mean to be `>= min_mean` (default
0.95; `run.py` relaxes to 0.90).

In `mean` mode, the self-match equals the average of `<ref_token_k, q>`,
which is very close to 1 because q itself is the renormalised mean of
those tokens. If this assertion fails the most likely causes are:

- the prompt does not actually land on the ref view in the patch grid
  (preprocessing / coordinate misalignment); or
- `coverage_threshold` is too strict and the §3 argmax fallback fired.

Empirically this repo gets self-match `0.92~0.97` (see `report.json` or
the `[run] Step 4a self-match OK mean=...` log line in `run.py`).

---

## 5. Thresholding: dense sim -> binary mask

Code: `matcher.binarize`.

### 5.1 Statistical threshold (otsu / percentile)

- `--threshold otsu`: 1-D Otsu on each per-view similarity map (a small
  hand-rolled NumPy version, no scikit-image dependency).
- `--threshold percentile --percentile 95`: take the 95th percentile of
  the view's similarity values.

Both are computed **per-view** by default (`--per_view_threshold`); use
`--global_threshold` to share a single threshold across all views.

### 5.2 Absolute floor `--abs_threshold` (important)

The fundamental problem with statistical thresholds: **they always keep
some pixels**, even on a view where the prompted object is not present
at all. In batch validation this leads to "NO-MATCH views pretending to
match".

With `--abs_threshold T` (typically 0.5):

```python
final_thr = max(stat_thr, T)
mask = (sim >= final_thr)
```

A view whose `sim.max() < T` ends up with an empty mask, rendered as a
red diagonal X with a `NO MATCH` banner
(`run_scene_batch.py:226-235`). That is exactly why some views in
`prompt_matching_outputs/<scene>/summary.png` are crossed out.

### 5.3 Upsample back to original resolution

```python
sim_hi  = upsample_patch_map(sim_np[i],  H, W, method="bilinear")  # heatmap
mask_hi = upsample_patch_map(mask_np[i], H, W, method="nearest")   # binary
```

Heatmap uses bilinear (continuous); mask uses nearest (preserves the
binary structure, no holes).

---

## 6. Pipeline at a glance

```
run.py / run_scene_batch.py
  |- extractor.extract_patch_tokens
  |    |- _build_vggt(merging=0)               # disable FastVGGT merging
  |    |- _preprocess_images                   # resize to (H, W), patch grid
  |    |- model.update_patch_dimensions(W_p, H_p)   # FastVGGT requires this
  |    |- model.aggregator(images)             # one forward pass
  |    |    -> aggregated_tokens_list (4 or 24 layers) [B,S,P,2C]
  |    |    -> patch_start_idx = 5
  |    |- pick layers -> drop special tokens -> L2-normalise
  |    |- average across layers -> renormalise
  |    |- return tokens [N, H_p, W_p, D=2048], meta
  |
  |- prompt.encode_mask_prompt(raw_mask, meta, coverage_threshold=0.3)
  |    |- mirror load_and_preprocess_images(crop) into (H, W)
  |    |- INTER_AREA downsample to (H_p, W_p), get coverage in [0,1]
  |    |- return patch_mask = coverage >= 0.3   # [H_p, W_p] bool
  |
  |- matcher.compute_similarity(tokens, ref_view_idx, patch_mask, query_mode)
  |    |- q = mean(prompt tokens) -> renormalise   (mean mode)
  |    |  or keep prompt_tokens as-is              (max mode)
  |    |- sim = einsum/dot -> [N, H_p, W_p]        cosine in [-1, 1]
  |
  |- matcher.assert_self_match (only in mean mode)
  |
  |- matcher.binarize(sim, otsu/percentile, abs_threshold=0.5)
  |    |- per-view threshold; floor at abs_threshold to suppress false positives
  |
  |- matcher.upsample_patch_map -> visualize.* -> summary.png / mask_*.png
```

---

## 7. Default hyperparameters (`run_scene_batch.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--layers` | `[17, 23]` | VGGT block 17 / 23 hidden states (FastVGGT auto-mapped to list indices 2, 3) |
| `--query_mode` | `mean` | how to pool the K prompt tokens |
| `--threshold` | `percentile` | which statistical threshold |
| `--percentile` | `95` | 95th-percentile per-view threshold |
| `--abs_threshold` | `0.5` | absolute cosine floor; NO MATCH below |
| `coverage_threshold` (hardcoded) | `0.3` | min fraction for a patch to enter the prompt |
| `--n_views` | `6` | number of frames sampled per scene |
| `--k_masks` | `5` | number of prompt masks generated per scene by YOLOv8-seg |

---

## 8. Pitfalls observed during debugging

1. **Layer numbering**: `--layers 17 23` are 0..23 indices in stock VGGT,
   but must be members of `{4, 11, 17, 23}` in FastVGGT. Other values
   raise `IndexError` on FastVGGT.
2. **D = 2048, not 1024**: VGGT concatenates frame- and
   global-attention tokens along the channel dim. Treating the result as
   one 2048-d vector for cosine is the correct convention.
3. **Renormalise after layer averaging**: otherwise the dot product is no
   longer a true cosine and `assert_self_match` will fail.
4. **The prompt MUST share the image's resize + centre-crop**: a naive
   `cv2.resize` from original-resolution mask straight to `(H_p, W_p)`
   skips the centre crop and misaligns patches.
5. **`coverage_threshold = 0.3` is empirical**: too high and small
   prompts get wiped out, triggering the argmax fallback; too low and
   border patches drag the self-match down.
6. **`abs_threshold = 0.5` is calibrated for VGGT-1B**: present-object
   views typically peak above 0.8, absent-object views around 0.35-0.60,
   and 0.5 cleanly separates them. Recalibrate when changing backbones.
