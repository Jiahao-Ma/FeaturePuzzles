"""3D positional encoding from back-projected depth + camera pose.

Given the clean-recon-tail geometry (predicted depth + 9-D camera pose
encoding, or externally supplied GT depth/pose), back-project every patch
centre into world coordinates and embed the (normalised) XYZ with a small MLP.

The embedding is *added* to the work-dim tokens right before cross-view /
two-way attention so that object-token matching becomes "feature AND 3D
position must agree", which suppresses geometrically-impossible matches.

The final MLP layer is zero-initialised, so at the start of training the module
is an exact no-op: behaviour is identical to the geometry-free model and the
network learns to deviate. This also makes it robust when the geometry is
unreliable (e.g. SA-1B crops): training can simply leave the contribution near
zero.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_vggt_omega_on_path() -> None:
    try:
        import vggt_omega  # noqa: F401
        return
    except ImportError:
        pass
    project_root = Path(__file__).resolve().parents[2]
    for cand in (project_root / "repo" / "vggt-omega",
                 Path("/mnt/dataset/marko/puzzles/puzzles/repo/vggt-omega")):
        if (cand / "vggt_omega").is_dir():
            sys.path.insert(0, str(cand))
            return


def backproject_patches_to_world(
    depth_patch: torch.Tensor,        # (B, N, H_p, W_p) metric depth at patch centres
    extrinsics: torch.Tensor,         # (B, N, 3, 4) camera-from-world, OpenCV
    intrinsics: torch.Tensor,         # (B, N, 3, 3) for the FULL image resolution
    image_hw: tuple,                  # (H, W) full-res used to build intrinsics
) -> torch.Tensor:
    """Return per-patch world XYZ of shape (B, N, H_p, W_p, 3)."""
    B, N, H_p, W_p = depth_patch.shape
    H, W = image_hw
    device = depth_patch.device
    dtype = torch.float32

    # Patch-centre pixel coordinates in full-resolution image space.
    vs = (torch.arange(H_p, device=device, dtype=dtype) + 0.5) * (H / H_p)
    us = (torch.arange(W_p, device=device, dtype=dtype) + 0.5) * (W / W_p)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")           # (H_p, W_p)
    ones = torch.ones_like(uu)
    pix = torch.stack([uu, vv, ones], dim=-1)                # (H_p, W_p, 3)

    K = intrinsics.to(dtype)
    Kinv = torch.linalg.inv(K)                               # (B, N, 3, 3)
    # rays in camera frame (unit depth)
    rays = torch.einsum("bnij,hwj->bnhwi", Kinv, pix)        # (B, N, H_p, W_p, 3)
    cam = rays * depth_patch.unsqueeze(-1).to(dtype)         # scale by depth

    R = extrinsics[..., :3, :3].to(dtype)                    # (B, N, 3, 3)
    T = extrinsics[..., :3, 3].to(dtype)                     # (B, N, 3)
    # cam = R @ world + T  ->  world = R^T @ (cam - T)
    world = torch.einsum("bnji,bnhwj->bnhwi", R, cam - T.view(B, N, 1, 1, 3))
    return world


class Geometry3DPE(nn.Module):
    """Back-project patch depth+pose to world XYZ and embed with a zero-init MLP."""

    def __init__(self, out_dim: int, hidden: int = 256):
        super().__init__()
        _ensure_vggt_omega_on_path()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        # Zero-init the final layer -> module starts as an exact no-op.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _geometry_to_xyz(
        geometry: Dict[str, torch.Tensor], H_p: int, W_p: int, image_hw: tuple,
    ) -> Optional[torch.Tensor]:
        """Resolve world XYZ (B, N, H_p, W_p, 3) from a geometry dict.

        Accepts either a precomputed ``xyz`` field, or ``depth`` + ``pose_enc``
        (predicted) or ``depth`` + ``extrinsics`` + ``intrinsics`` (GT).
        """
        if geometry is None:
            return None
        if "xyz" in geometry and geometry["xyz"] is not None:
            xyz = geometry["xyz"]
            if xyz.shape[2] != H_p or xyz.shape[3] != W_p:
                xyz = F.interpolate(
                    xyz.permute(0, 1, 4, 2, 3).flatten(0, 1),
                    size=(H_p, W_p), mode="nearest",
                ).reshape(xyz.shape[0], xyz.shape[1], 3, H_p, W_p).permute(0, 1, 3, 4, 2)
            return xyz

        depth = geometry.get("depth")
        if depth is None:
            return None
        if depth.dim() == 5:                     # (B, N, H, W, 1)
            depth = depth[..., 0]
        B, N, H, W = depth.shape
        # pool depth to patch grid
        depth_patch = F.adaptive_avg_pool2d(
            depth.reshape(B * N, 1, H, W), (H_p, W_p),
        ).reshape(B, N, H_p, W_p)

        if "extrinsics" in geometry and "intrinsics" in geometry:
            extr, intr = geometry["extrinsics"], geometry["intrinsics"]
        else:
            from vggt_omega.utils.pose_enc import encoding_to_camera
            extr, intr = encoding_to_camera(geometry["pose_enc"], image_hw, build_intrinsics=True)
        return backproject_patches_to_world(depth_patch, extr, intr, image_hw)

    def forward(
        self,
        geometry: Optional[Dict[str, torch.Tensor]],
        H_p: int,
        W_p: int,
        image_hw: tuple,
    ) -> Optional[torch.Tensor]:
        """Return additive token embedding (B, N, H_p, W_p, out_dim) or None."""
        xyz = self._geometry_to_xyz(geometry, H_p, W_p, image_hw)
        if xyz is None:
            return None
        xyz = xyz.float()
        # Per-sample normalisation: centre by mean over all patches/views,
        # scale by isotropic std. Keeps the shared world frame but bounded.
        flat = xyz.reshape(xyz.shape[0], -1, 3)                 # (B, N*P, 3)
        mean = flat.mean(dim=1, keepdim=True)                   # (B, 1, 3)
        std = flat.std(dim=1, keepdim=True).mean(dim=-1, keepdim=True).clamp_min(1e-3)
        xyz_n = (flat - mean) / std
        xyz_n = xyz_n.reshape(xyz.shape)                        # (B, N, H_p, W_p, 3)
        return self.mlp(xyz_n)
