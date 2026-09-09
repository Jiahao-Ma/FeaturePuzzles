"""End-to-end orchestration, shared by the web server and the CLI.

    source -> frames -> geometry -> prompts -> masks -> point cloud + videos
"""
from __future__ import annotations

import itertools
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from . import paths, sam_prompt
from .compose import compose_pip_video
from .frames import FrameSource, load_any
from .geometry import build_geometry
from .render import export_point_cloud, mask_overlay_video, render_views
from .segment import DYNAMIC, STATIC, run_segmentation
from .static_ref import ref

Log = Callable[[str], None]


@dataclass
class PromptSpec:
    """What the user asked to segment.

    ``mode="points"`` carries per-object clicks; ``mode="dense"`` runs SAM's
    dense grid and takes the largest ``top_k`` regions as prompts instead.
    """

    mode: str = "points"                       # "points" | "dense"
    objects: List[Dict] = field(default_factory=list)
    top_k: int = 10
    points_per_side: int = 32
    ref_view: int = 0

    @property
    def n_objects(self) -> int:
        return len(self.objects) if self.mode == "points" else self.top_k


@dataclass
class PreparedInput:
    src: FrameSource
    geo: Dict[str, np.ndarray]
    scene_mode: str
    source: Optional[Path] = None
    long_side: int = 640

    @property
    def run_name(self) -> str:
        """Default name for this input's artefacts, and for the directory holding them."""
        return f"{self.src.name}_{self.scene_mode}"


def prepare(source: Path, scene_mode: str, n_frames: int = 24,
            long_side: int = 640, log: Log = print) -> PreparedInput:
    """Sample frames and resolve geometry (ground truth if present, else predicted).

    Writes nothing: an output directory is only created once there are results
    to put in it, so browsing clips does not litter ``runs/``.
    """
    src = load_any(Path(source), n_frames, long_side, log=log)
    if scene_mode == STATIC and not src.has_gt_geometry:
        log("[prepare] static mode without sensor depth -> predicting geometry; "
            "the 3D-connectivity filter will run on predicted depth")
    geo = build_geometry(src, scene_mode, log=log)
    return PreparedInput(src=src, geo=geo, scene_mode=scene_mode,
                         source=Path(source), long_side=long_side)


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def unique_run_dir(root: Path, name: str) -> Path:
    """Create ``root/<name>``, falling back to ``<name>_2``, ``_3``, ...

    Naming directories after the clip makes ``runs/`` browsable; the counter is
    what keeps a second run of the same clip from overwriting the first. The
    directory is created here so that concurrent callers cannot both claim the
    same name.
    """
    base = _UNSAFE.sub("_", name).strip("._") or "run"
    for n in itertools.count(1):
        candidate = root / (base if n == 1 else f"{base}_{n}")
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            continue
    raise AssertionError("unreachable")


def build_prompts(prep: PreparedInput, spec: PromptSpec, log: Log = print) -> np.ndarray:
    """Turn a ``PromptSpec`` into (K, H, W) uint8 masks on the reference frame."""
    rgb = prep.src.images[spec.ref_view]
    if spec.mode == "dense":
        # Over-fetch, drop near-duplicate proposals, then take the top-K: two
        # prompts describing the same region make the multi-mask forward fight
        # itself, especially with cross-object exclusivity on.
        masks = sam_prompt.dense_grid_masks(
            rgb, top_k=spec.top_k * 2, points_per_side=spec.points_per_side, log=log)
        masks = sam_prompt.dedupe_masks(masks)
        if spec.top_k > 0:
            masks = masks[:spec.top_k]
    else:
        objs = [o for o in spec.objects if o.get("points")]
        if not objs:
            raise ValueError("no point prompts supplied")
        masks = sam_prompt.masks_from_points(
            rgb, objs, image_id=f"{prep.src.name}:{spec.ref_view}", log=log)
    masks = masks[masks.reshape(masks.shape[0], -1).sum(1) > 0]
    if masks.shape[0] == 0:
        raise ValueError("prompts produced no non-empty mask")
    log(f"[prompt] {masks.shape[0]} object mask(s) on frame {spec.ref_view}; "
        f"areas={[int(m.sum()) for m in masks]}")
    return masks


def save_prompt_image(rgb: np.ndarray, prompt_masks: np.ndarray,
                      obj_bgr: np.ndarray, spec: PromptSpec, path: Path) -> Path:
    """Reference frame with the seed masks tinted and the clicks marked.

    The manifest records the clicks numerically; this is the same record in a
    form you can check at a glance when browsing old runs.
    """
    import cv2

    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    H, W = canvas.shape[:2]
    colours = obj_bgr.astype(np.float32)          # already 0-255
    for k, m in enumerate(prompt_masks):
        sel = m > 0
        if not sel.any():
            continue
        canvas[sel] = 0.55 * canvas[sel] + 0.45 * colours[k]
        edge = cv2.dilate(m, np.ones((3, 3), np.uint8)) - m
        canvas[edge > 0] = colours[k]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    if spec.mode == "points":
        for k, obj in enumerate([o for o in spec.objects if o.get("points")]):
            colour = tuple(int(c) for c in colours[k % len(colours)])
            for p in obj["points"]:
                xy = (int(round(float(p["x"]) * W)), int(round(float(p["y"]) * H)))
                # Filled for include, hollow for exclude.
                cv2.circle(canvas, xy, 7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, xy, 5, colour,
                           -1 if int(p.get("label", 1)) == 1 else 2, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return path


def run(prep: PreparedInput, spec: PromptSpec, out_dir: Path, *,
        name: Optional[str] = None,
        mask_threshold: float = 0.5, exclusive: bool = True, geo_cc: bool = True,
        render_mode: str = "hl", render_size: Sequence[int] = (1920, 1080),
        do_render: bool = True, log: Log = print) -> Dict[str, object]:
    """Prompt -> masks -> artefacts. Returns a manifest of what was written."""
    R = ref()
    t0 = time.time()
    name = name or prep.run_name
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_masks = build_prompts(prep, spec, log=log)
    K = prompt_masks.shape[0]
    obj_rgb = R.object_colors_rgb(K)
    obj_bgr = R.object_colors_bgr(K)

    # SAM and VGGT-Omega both want a lot of VRAM; the prompts are already
    # resolved by this point, so let SAM go.
    sam_prompt.release()

    pred_masks = run_segmentation(
        prep.src.images, prompt_masks, prep.geo, prep.scene_mode,
        ref_view=spec.ref_view, thr=mask_threshold, exclusive=exclusive,
        geo_cc=geo_cc and prep.scene_mode == STATIC, device=device, log=log)

    manifest: Dict[str, object] = {
        "name": name,
        "scene_mode": prep.scene_mode,
        "source_kind": prep.src.kind,
        "n_frames": int(prep.src.size),
        "n_objects": int(K),
        "ref_view": int(spec.ref_view),
        "prompt_mode": spec.mode,
        "fps": float(prep.src.fps),
        "geometry": "ground-truth" if prep.src.has_gt_geometry else "predicted",
        "object_colors": (obj_rgb * 255).astype(int).tolist(),
        # Everything below is what it takes to reproduce this run: the exact
        # input, the exact prompt the user gave, and the switches it ran under.
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "source": str(prep.source) if prep.source else None,
            "long_side": int(prep.long_side),
            "frame_hw": list(prep.src.hw),
            "frame_indices": [int(i) for i in prep.src.frame_indices],
        },
        "prompt": asdict(spec),
        "settings": {
            "mask_threshold": float(mask_threshold),
            "exclusive": bool(exclusive),
            "geo_cc": bool(geo_cc and prep.scene_mode == STATIC),
            "render_mode": render_mode,
            "render_size": [int(render_size[0]), int(render_size[1])],
        },
        "prompt_areas_px": [int(m.sum()) for m in prompt_masks],
    }

    np.savez_compressed(out / f"{name}_pred_masks.npz",
                        pred_masks=pred_masks, prompt_masks=prompt_masks)
    manifest["masks_npz"] = f"{name}_pred_masks.npz"

    manifest["prompt_image"] = save_prompt_image(
        prep.src.images[spec.ref_view], prompt_masks, obj_bgr, spec,
        out / f"{name}_prompt.png").name

    overlay = mask_overlay_video(prep.src.images, pred_masks, obj_bgr,
                                 out / f"{name}_masks_overlay.mp4",
                                 fps=prep.src.fps, log=log)
    manifest["overlay_video"] = overlay.name

    cloud = export_point_cloud(prep.geo, pred_masks, obj_rgb,
                               out / f"{name}_cloud.npz", log=log)
    manifest["cloud_npz"] = cloud.name

    if do_render:
        try:
            videos = render_views(
                prep.geo, pred_masks, obj_rgb, out, name, mode=render_mode,
                cam_modes=("follow", "birdeye"), fps=prep.src.fps,
                width=int(render_size[0]), height=int(render_size[1]), log=log)
            manifest["follow_video"] = videos["follow"].name
            manifest["birdeye_video"] = videos["birdeye"].name
            pip = compose_pip_video(
                videos["follow"], videos["birdeye"], prep.src.images,
                out / f"{name}_pip.mp4", fps=prep.src.fps, log=log)
            manifest["pip_video"] = pip.name
        except Exception as exc:                    # rendering is optional
            log(f"[render] failed: {type(exc).__name__}: {exc}")
            manifest["render_error"] = f"{type(exc).__name__}: {exc}"

    manifest["elapsed_s"] = round(time.time() - t0, 1)
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"[done] {name} in {manifest['elapsed_s']}s -> {out}")
    return manifest


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="video, image folder or ScanNet scene dir")
    ap.add_argument("--scene_mode", choices=[STATIC, DYNAMIC], default=DYNAMIC)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_frames", type=int, default=24)
    ap.add_argument("--long_side", type=int, default=640)
    ap.add_argument("--ref_view", type=int, default=0)
    ap.add_argument("--prompt_mode", choices=["points", "dense"], default="dense")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--points", default=None,
                    help='JSON: [{"points":[{"x":10,"y":20,"label":1}]}, ...]')
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--no_exclusive", action="store_true")
    ap.add_argument("--no_geo_cc", action="store_true")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--render_mode", choices=["hl", "seg", "rgb"], default="hl")
    ap.add_argument("--render_width", type=int, default=1920)
    ap.add_argument("--render_height", type=int, default=1080)
    args = ap.parse_args()

    src_path = Path(args.source)
    prep = prepare(src_path, args.scene_mode,
                   n_frames=args.n_frames, long_side=args.long_side)
    out_dir = (Path(args.out_dir) if args.out_dir
               else unique_run_dir(paths.ensure_runs(), prep.run_name))
    spec = PromptSpec(mode=args.prompt_mode, top_k=args.top_k, ref_view=args.ref_view,
                      objects=json.loads(args.points) if args.points else [])
    m = run(prep, spec, out_dir, mask_threshold=args.mask_threshold,
            exclusive=not args.no_exclusive, geo_cc=not args.no_geo_cc,
            render_mode=args.render_mode, do_render=not args.no_render,
            render_size=(args.render_width, args.render_height))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    _cli()
