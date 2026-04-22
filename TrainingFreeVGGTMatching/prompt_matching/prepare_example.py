"""Step 7 — Prepare a toy multi-view example if the user has none.

Default behaviour:

* Source images: ``--source_dir`` (defaults to ``examples/`` under the project
  root). Anything with a common image extension is picked up, the first
  ``--n`` files are copied to ``prompt_matching_data/example/images/``.
* Prompt mask: a centred ``--box`` rectangle on the first copied image.
* A tiny ``config.json`` is dropped next to the mask so it is obvious which
  reference view was used.

Run:

    python prompt_matching/prepare_example.py          # copy from ./examples
    python prompt_matching/prepare_example.py \\
        --source_dir /path/to/DSC_images --n 8 --box 400,300,800,700
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "examples"
DEFAULT_DEST = PROJECT_ROOT / "prompt_matching_data" / "example"


def _parse_box(s: str):
    parts = [int(x) for x in s.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box must be 'x1,y1,x2,y2' (4 ints).")
    return tuple(parts)


def _list_images(src: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a toy multi-view example.")
    ap.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--dest_dir", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--n", type=int, default=5, help="Number of images to copy.")
    ap.add_argument("--box", type=_parse_box, default=None,
                    help="Prompt rectangle as 'x1,y1,x2,y2' in ORIG pixels of the first image. "
                         "Default: a 200x200 square centred on the first image.")
    ap.add_argument("--ref_view_idx", type=int, default=0)
    args = ap.parse_args()

    src: Path = args.source_dir
    dst: Path = args.dest_dir

    if not src.is_dir():
        raise SystemExit(f"source_dir {src} does not exist.")

    all_imgs = _list_images(src)
    if not all_imgs:
        raise SystemExit(f"No images found in {src}.")

    picks = all_imgs[: args.n]
    if len(picks) < args.n:
        print(f"[prepare_example] Warning: only {len(picks)} images available, asked for {args.n}.")

    img_dst = dst / "images"
    img_dst.mkdir(parents=True, exist_ok=True)
    for p in picks:
        target = img_dst / p.name
        if not target.exists():
            shutil.copy2(p, target)
    print(f"[prepare_example] Copied {len(picks)} images to {img_dst}")

    first = cv2.imread(str(picks[args.ref_view_idx]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"cv2 failed to open {picks[args.ref_view_idx]}")
    h, w = first.shape[:2]

    if args.box is None:
        cy, cx = h // 2, w // 2
        x1, y1, x2, y2 = cx - 100, cy - 100, cx + 100, cy + 100
    else:
        x1, y1, x2, y2 = args.box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"Box {(x1, y1, x2, y2)} is degenerate for image {(w, h)}.")

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    mask_path = dst / "prompt_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"[prepare_example] Wrote prompt_mask.png ({mask.sum()//255} px) at {mask_path}")

    config = {
        "ref_view_idx": args.ref_view_idx,
        "ref_image": picks[args.ref_view_idx].name,
        "box": [x1, y1, x2, y2],
        "image_count": len(picks),
    }
    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        "\n[prepare_example] Done. Now run, e.g.:\n"
        f"  python -m prompt_matching.run \\\n"
        f"    --image_dir {img_dst} \\\n"
        f"    --prompt_mask {mask_path} \\\n"
        f"    --ref_view_idx {args.ref_view_idx} \\\n"
        f"    --output_dir prompt_matching_outputs/run1\n"
    )


if __name__ == "__main__":
    main()
