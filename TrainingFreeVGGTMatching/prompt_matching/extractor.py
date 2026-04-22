"""Step 2 — Patch token extractor (FastVGGT-aware).

Loads a VGGT / FastVGGT-compatible model, runs a single forward pass on a
folder of images, and returns L2-normalised patch tokens averaged across the
selected aggregator layers.

Design choices (see ../NOTES.md for the full, verified rationale):

* FastVGGT's aggregator only appends intermediates at ``block_num ∈
  {4, 11, 17, 23}`` (``vggt/models/aggregator.py:285,369-381``), so
  ``aggregated_tokens_list`` has length **4** there while the original VGGT
  returns length **24**. The user-facing ``--layers 17 23`` always refers to
  ``block_num`` and we translate it to the correct ``output_list`` index at
  runtime.
* Each tensor has shape ``[B, S, P, 2*embed_dim]``. ``P`` already includes the
  ``patch_start_idx`` special tokens (1 camera + 4 register). We drop those
  and reshape the rest to ``[B, S, H_p, W_p, 2C]``.
* ``(H_p, W_p) = (H // patch_size, W // patch_size)`` and
  ``patch_size`` is read from ``model.aggregator.patch_size`` — never
  hard-coded.
* FastVGGT refuses to run unless
  ``model.update_patch_dimensions(patch_width, patch_height)`` has been
  called (token-merging kernels assert on the grid shape). We call it
  unconditionally when the method is available.
* ``merging`` is forced to 0 so global-attention token count stays equal to
  ``S*(H_p*W_p + 5)`` and our grid alignment is preserved.
* L2-normalise each selected layer, average them, re-normalise. Return
  float32.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_FASTVGGT_BLOCK4DPT_IDX = [4, 11, 17, 23]


def _ensure_vggt_on_path() -> None:
    """Make ``import vggt`` (and ``merging``) resolvable.

    Order of preference:
    1. ``vggt`` already importable on ``sys.path``.
    2. The FastVGGT clone shipped next to this package.
    3. The reference VGGT implementation on the dataset mount.
    """

    try:
        import vggt  # noqa: F401
        return
    except ImportError:
        pass

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "FastVGGT",
        Path("/mnt/dataset/marko/code/puzzles/TrainingFreeVGGTMatching/FastVGGT"),
        Path("/mnt/dataset/marko/code/Depth/vggt"),
    ]
    for cand in candidates:
        if (cand / "vggt").is_dir():
            sys.path.insert(0, str(cand))
            try:
                import vggt  # noqa: F401
                return
            except ImportError:
                sys.path.pop(0)
                continue

    raise ImportError(
        "Could not locate a `vggt` package. Install FastVGGT, put a VGGT "
        "checkout on PYTHONPATH, or make sure FastVGGT/ sits next to the "
        "prompt_matching/ directory."
    )


def _list_image_paths(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths: List[str] = []
    for entry in sorted(os.listdir(image_dir)):
        full = os.path.join(image_dir, entry)
        if os.path.isfile(full) and Path(entry).suffix.lower() in exts:
            paths.append(full)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir!r}.")
    return paths


def _build_vggt(ckpt_path: str | None, device: str):
    """Instantiate VGGT / FastVGGT, load weights, keep ``merging=0``."""

    from vggt.models.vggt import VGGT

    import inspect
    sig = inspect.signature(VGGT.__init__)
    vggt_kwargs = {}
    if "merging" in sig.parameters:
        vggt_kwargs["merging"] = 0
    if "merge_ratio" in sig.parameters:
        vggt_kwargs["merge_ratio"] = 0.9
    if "vis_attn_map" in sig.parameters:
        vggt_kwargs["vis_attn_map"] = False

    model = VGGT(**vggt_kwargs)

    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        incompat = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompat, "missing_keys", []) or [])
        unexpected = list(getattr(incompat, "unexpected_keys", []) or [])
        if missing:
            print(f"[extractor] load_state_dict missing keys (up to 8): {missing[:8]}  (total {len(missing)})")
        if unexpected:
            print(f"[extractor] load_state_dict unexpected keys (up to 8): {unexpected[:8]}  (total {len(unexpected)})")
    else:
        if ckpt_path:
            print(f"[extractor] ckpt_path {ckpt_path!r} not found, falling back to HF Hub.")
        try:
            model = VGGT.from_pretrained("facebook/VGGT-1B", **vggt_kwargs)
        except TypeError:
            model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = model.to(device).eval()
    return model


def _preprocess_images(image_paths: Sequence[str]) -> Tuple[torch.Tensor, int, int]:
    """Run FastVGGT's preprocessing pipeline when available, otherwise fall
    back to ``vggt.utils.load_fn.load_and_preprocess_images`` which is
    geometrically identical to ``get_vgg_input_imgs`` in ``mode='crop'``.

    Returns ``(tensor[N, 3, H, W], patch_width, patch_height)``.
    """

    try:
        from vggt.utils.eval_utils import get_vgg_input_imgs, load_images_rgb
        imgs = load_images_rgb([Path(p) for p in image_paths])
        if not imgs:
            raise RuntimeError("load_images_rgb returned an empty list.")
        arr = np.stack(imgs)
        vgg_input, pw, ph = get_vgg_input_imgs(arr)
        return vgg_input, int(pw), int(ph)
    except ImportError:
        pass

    from vggt.utils.load_fn import load_and_preprocess_images
    vgg_input = load_and_preprocess_images(list(image_paths))
    _, _, H, W = vgg_input.shape
    return vgg_input, W // 14, H // 14


def _resolve_layer_indices(
    requested_layers: Sequence[int],
    num_output_layers: int,
) -> List[int]:
    """Translate ``block_num``-style user input to ``output_list`` indices.

    * If ``num_output_layers == 24`` we assume it's the original VGGT and use
      the requested indices directly.
    * If ``num_output_layers == len(_FASTVGGT_BLOCK4DPT_IDX) == 4`` we assume
      it's FastVGGT and map each layer through ``_FASTVGGT_BLOCK4DPT_IDX``.
    * Anything else triggers a clear error.
    """

    resolved: List[int] = []

    if num_output_layers == 24:
        for l in requested_layers:
            idx = l if l >= 0 else num_output_layers + l
            if not 0 <= idx < num_output_layers:
                raise IndexError(f"layer {l} (→ {idx}) out of range [0, {num_output_layers}).")
            resolved.append(idx)
        return resolved

    if num_output_layers == len(_FASTVGGT_BLOCK4DPT_IDX):
        for l in requested_layers:
            if l in _FASTVGGT_BLOCK4DPT_IDX:
                resolved.append(_FASTVGGT_BLOCK4DPT_IDX.index(l))
                continue
            if 0 <= l < num_output_layers:
                resolved.append(int(l))
                continue
            if l < 0 and -num_output_layers <= l:
                resolved.append(num_output_layers + l)
                continue
            raise IndexError(
                f"FastVGGT only stores intermediates at block_num "
                f"{_FASTVGGT_BLOCK4DPT_IDX}. Requested layer {l} does not match."
            )
        return resolved

    raise RuntimeError(
        f"Unexpected aggregator output list length {num_output_layers}; "
        f"only 4 (FastVGGT) and 24 (official VGGT) are known."
    )


def extract_patch_tokens(
    image_dir: str,
    ckpt_path: str | None = None,
    layers: Sequence[int] = (17, 23),
    device: str = "cuda",
    image_paths: Sequence[str] | None = None,
) -> Tuple[torch.Tensor, Dict]:
    """Run a (Fast)VGGT forward pass and return L2-normalised patch tokens.

    Args:
        image_dir: Folder containing the images. Ignored when ``image_paths``
            is provided.
        ckpt_path: Local ``.pt`` / ``.pth`` weights file. ``None`` falls back
            to ``VGGT.from_pretrained("facebook/VGGT-1B")``.
        layers: ``block_num`` values to combine, e.g. ``(17, 23)``. For
            FastVGGT only values in ``{4, 11, 17, 23}`` are meaningful; for
            plain VGGT any 0–23 index works. Negatives count from the end of
            the aggregator's output list.
        device: Torch device string.
        image_paths: Optional explicit list of image paths. The order becomes
            the view order.

    Returns:
        tokens: ``[N, H_p, W_p, D]`` float32, L2-normalised along ``D``.
        meta:   dict with keys ``N, H_p, W_p, D, H, W, patch_size,
                patch_start_idx, n_aggregator_layers, layers_used,
                block_nums_used, image_paths, orig_sizes, orig_H, orig_W``.
    """

    _ensure_vggt_on_path()
    from PIL import Image

    if image_paths is None:
        image_paths = _list_image_paths(image_dir)
    image_paths = list(image_paths)

    orig_sizes: List[Tuple[int, int]] = []
    for p in image_paths:
        with Image.open(p) as im:
            w, h = im.size
        orig_sizes.append((int(h), int(w)))

    vgg_input, pw, ph = _preprocess_images(image_paths)
    N, C_in, H, W = vgg_input.shape
    if C_in != 3:
        raise RuntimeError(f"Expected 3 channels, got {C_in}.")
    H_p, W_p = H // 14, W // 14
    if (W_p, H_p) != (pw, ph):
        raise RuntimeError(
            f"Patch grid mismatch: input tensor → ({H_p},{W_p}), "
            f"preprocessor → ({ph},{pw})."
        )

    device_t = torch.device(device)
    if device_t.type == "cuda":
        try:
            cap = torch.cuda.get_device_capability(device_t)
            amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        except Exception:
            amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    model = _build_vggt(ckpt_path, device)
    patch_size = int(getattr(model.aggregator, "patch_size", 14))
    if patch_size != 14:
        print(f"[extractor] NOTE: model.aggregator.patch_size={patch_size} (non-default).")
        H_p, W_p = H // patch_size, W // patch_size

    if hasattr(model, "update_patch_dimensions"):
        model.update_patch_dimensions(W_p, H_p)

    if device_t.type == "cuda" and amp_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        vgg_input_dev = vgg_input.to(device).to(torch.bfloat16)
    else:
        vgg_input_dev = vgg_input.to(device)

    print(
        f"[extractor] N={N}  preprocessed=({H},{W})  patch_size={patch_size}  "
        f"grid=({H_p},{W_p})  amp_dtype={amp_dtype}  device={device}  "
        f"requested_layers={list(layers)}"
    )

    images_in = vgg_input_dev.unsqueeze(0)  # [B=1, S=N, 3, H, W]
    with torch.no_grad():
        if device_t.type == "cuda":
            try:
                amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
            except (AttributeError, TypeError):
                amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            with amp_ctx:
                aggregated_tokens_list, patch_start_idx = model.aggregator(images_in)
        else:
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_in)

    n_layers_total = len(aggregated_tokens_list)
    resolved_layers = _resolve_layer_indices(layers, n_layers_total)
    if n_layers_total == len(_FASTVGGT_BLOCK4DPT_IDX):
        block_nums_used = [_FASTVGGT_BLOCK4DPT_IDX[r] for r in resolved_layers]
    else:
        block_nums_used = list(resolved_layers)

    expected_P = patch_start_idx + H_p * W_p
    sample_shape = aggregated_tokens_list[resolved_layers[0]].shape
    if sample_shape[2] != expected_P:
        raise RuntimeError(
            f"Token count mismatch: tensor[2]={sample_shape[2]} but "
            f"patch_start_idx + H_p*W_p = {expected_P}. Aggregator layout "
            f"assumption broke (sample_shape={sample_shape}).")

    accum: torch.Tensor | None = None
    for rid in resolved_layers:
        layer_tokens = aggregated_tokens_list[rid]                 # [B, S, P, 2C]
        patch_tokens = layer_tokens[:, :, patch_start_idx:, :]     # [B, S, H_p*W_p, 2C]
        patch_tokens = patch_tokens.float()
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        accum = patch_tokens if accum is None else accum + patch_tokens

    assert accum is not None
    avg_tokens = accum / float(len(resolved_layers))
    tokens = F.normalize(avg_tokens, dim=-1)                       # re-normalise

    B, S, _, D = tokens.shape
    tokens = tokens.view(B, S, H_p, W_p, D)
    if B != 1:
        raise RuntimeError(f"Expected B=1, got B={B}.")
    tokens = tokens[0].contiguous()                                # [N, H_p, W_p, D]

    meta: Dict = {
        "N": int(N),
        "H_p": int(H_p),
        "W_p": int(W_p),
        "D": int(D),
        "H": int(H),
        "W": int(W),
        "patch_size": int(patch_size),
        "patch_start_idx": int(patch_start_idx),
        "n_aggregator_layers": int(n_layers_total),
        "layers_used": resolved_layers,
        "block_nums_used": block_nums_used,
        "image_paths": image_paths,
        "orig_sizes": orig_sizes,
        "orig_H": int(orig_sizes[0][0]),
        "orig_W": int(orig_sizes[0][1]),
    }

    print(
        f"[extractor] tokens.shape={tuple(tokens.shape)}  "
        f"norm.mean={tokens.norm(dim=-1).mean().item():.4f}  "
        f"norm.std={tokens.norm(dim=-1).std().item():.4f}  "
        f"resolved_layer_indices={resolved_layers}  "
        f"block_nums={block_nums_used}"
    )

    del aggregated_tokens_list
    if device_t.type == "cuda":
        torch.cuda.empty_cache()

    return tokens, meta


if __name__ == "__main__":
    """Smoke test: print the shape and norm statistics."""
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for extract_patch_tokens.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 23])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokens, meta = extract_patch_tokens(
        args.image_dir, args.ckpt_path, args.layers, args.device
    )
    print("tokens:", tokens.shape, "dtype:", tokens.dtype)
    print("norm mean:", tokens.norm(dim=-1).mean().item())
    print("meta:", {k: v for k, v in meta.items() if k != "image_paths"})
