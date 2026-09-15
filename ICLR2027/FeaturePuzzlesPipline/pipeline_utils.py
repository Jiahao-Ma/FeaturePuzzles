"""Shared helpers for the Feature Puzzles data-material pipeline.

Dependency-light: only numpy, cv2, torch, pyyaml so the pure-geometry / IO
helpers can be unit-tested on CPU without loading any heavy model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml


# Config
def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


# Reproducibility
def seed_everything(seed: int) -> np.random.Generator:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


# IO helpers
def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_image_rgb(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_png(img: np.ndarray, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        cv2.imwrite(str(path), img)
        return
    if img.shape[2] == 4:
        cv2.imwrite(str(path), img)
        return
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_npz(arrays: Dict[str, np.ndarray], path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(str(path), **arrays)


def file_sha1(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# Color helpers
def distinct_colors(n: int, seed: int = 0) -> np.ndarray:
    import colorsys
    rng = np.random.default_rng(seed)
    cols = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = 0.65 + 0.25 * rng.random()
        v = 0.85 + 0.15 * rng.random()
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return cols


def colorize_id_map(id_map: np.ndarray, num_ids: Optional[int] = None,
                    seed: int = 1, palette: Optional[np.ndarray] = None) -> np.ndarray:
    if palette is None:
        if num_ids is None:
            num_ids = int(id_map.max()) + 1 if id_map.size else 0
        palette = distinct_colors(max(num_ids, 1), seed=seed)
    out = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    valid = id_map >= 0
    out[valid] = palette[np.clip(id_map[valid], 0, len(palette) - 1) % len(palette)]
    return out


def overlay_id_map(image: np.ndarray, id_map: np.ndarray, alpha: float = 0.5,
                   palette: Optional[np.ndarray] = None) -> np.ndarray:
    if palette is None:
        num = int(id_map.max()) + 1 if id_map.size else 0
        palette = distinct_colors(max(num, 1), seed=1)
    out = image.astype(np.float32)
    color = colorize_id_map(id_map, palette=palette).astype(np.float32)
    m = (id_map >= 0)[..., None].astype(np.float32)
    out = out * (1 - alpha * m) + color * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_binary_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.5,
                         palette: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = image.shape[:2]
    if palette is None:
        palette = distinct_colors(masks.shape[0], seed=2)
    out = image.astype(np.float32)
    for i in range(masks.shape[0]):
        if masks[i].shape != (h, w):
            continue
        color = palette[i].astype(np.float32)
        m = masks[i][..., None].astype(np.float32)
        out = out * (1 - alpha * m) + color * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)
