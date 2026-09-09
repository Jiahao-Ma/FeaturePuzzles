"""Repository path bootstrap and checkpoint defaults.

Import this module before anything that pulls in ``models.*``, ``lib`` or the
LingBot ``rgbd_render`` package -- those live outside this directory and are
only importable once the paths below are on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = DEMO_ROOT.parents[1]

VIS_ROOT = RELEASE_ROOT / "repo" / "lingbot-map_vis"

_SYS_PATHS = [
    RELEASE_ROOT / "src",
    RELEASE_ROOT / "video_pipeline",
    RELEASE_ROOT / "repo" / "vggt-omega",
    VIS_ROOT,
    VIS_ROOT / "demo_render",
    VIS_ROOT / "demo_render" / "render_cuda_ext",
]

for _p in _SYS_PATHS:
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

OMEGA_CKPT = RELEASE_ROOT / "checkpoints" / "vggt_omega_1b_256_text.pt"
SEG_CKPT = RELEASE_ROOT / "weights" / "seg_omega_lora_multiprompt" / "checkpoint-best.pth"
SAM_CKPT = RELEASE_ROOT / "checkpoints" / "sam" / "sam_vit_h_4b8939.pth"
RENDER_CONFIG = VIS_ROOT / "demo_render" / "config" / "indoor.yaml"

RUNS_ROOT = DEMO_ROOT / "runs"
WEB_ROOT = DEMO_ROOT / "web"

SCANNET_TEST = RELEASE_ROOT / "data" / "dataset" / "scannet" / "test"
DYNAMIC_VIDEO_ROOT = DEMO_ROOT.parent / "vggt_omega_multiprompt_dynamic_scene"


def ensure_runs() -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return RUNS_ROOT
