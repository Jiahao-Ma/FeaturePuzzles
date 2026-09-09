"""Reorganize per-scene outputs:
  Keep at <scene>/ top level (downloadable):
    - first-frame prompt mask visuals (overlay, grid, npz, 256_*.png)
    - seg side-by-side combined videos (seg_follow/birdeye_pointcloud_combined.mp4)
  Move everything else into <scene>/materials/
"""
import shutil
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent


def reorganize(scene_dir: Path):
    mats = scene_dir / "materials"
    mats.mkdir(exist_ok=True)
    keep_prefixes = (
        "_prompt_masks_firstframe_overlay.png",
        "_prompt_masks_firstframe_grid.png",
        "_prompt_masks_firstframe.npz",
    )
    moved, kept = [], []
    for p in sorted(scene_dir.iterdir()):
        if p.is_dir():
            if p.name == "materials":
                continue
            # move the diag folder into materials/
            target = mats / p.name
            shutil.move(str(p), str(target))
            moved.append(p.name)
            continue
        name = p.name
        # keep: first-frame mask files + seg combined videos
        is_keep = (
            any(name.endswith(suf) for suf in keep_prefixes)
            or name.startswith(f"{scene_dir.name}_prompt_mask_256_")
            or name == f"{scene_dir.name}_seg_follow_pointcloud_combined.mp4"
            or name == f"{scene_dir.name}_seg_birdeye_pointcloud_combined.mp4"
        )
        if is_keep:
            kept.append(name)
        else:
            shutil.move(str(p), str(mats / name))
            moved.append(name)
    print(f"[{scene_dir.name}] kept {len(kept)} at top, moved {len(moved)} to materials/")
    for k in kept:
        print(f"  keep: {k}")
    print(f"  materials/: {sorted(p.name for p in mats.iterdir())[:6]}...")


for s in ["scene0000_01", "scene0046_00"]:
    reorganize(OUT_ROOT / s)
