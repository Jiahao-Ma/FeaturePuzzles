"""Stage-2: jointly train the segmentation head AND a CLIP-aligned feature
projection, using the Stage-1 FeatureSequenceHead as a frozen distillation
target and the cached JAFAR-CLIP teacher as the grounding signal.

Loss = L_seg
       + lambda_feat   * L_feat(align_proj(seg_dense), JAFAR_teacher)
       + lambda_distill* L_feat(align_proj(seg_dense), sg(feature_head(tokens)))
       + lambda_inst   * L_inst(align_proj(seg_dense), gt_masks)
       + lambda_warp   * L_warp(align_proj(seg_dense), crop_boxes)

The backbone is frozen by default; optional backbone LoRA (``--backbone_lora``)
mirrors the seg training. The Stage-1 feature head is loaded from
``--feature_head_ckpt`` and kept frozen.

Multi-GPU:
    torchrun --nproc_per_node=8 train_segmentation_jafar_feature.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sa1b.jafar_feature_dataset import JafarFeatureDataset, jafar_feature_collate  # noqa: E402
from models.vggt_omega_segmentation import build_vggt_omega_segmentation  # noqa: E402
from models.vggt_omega_feature_head import build_feature_sequence_head, channel_l2norm  # noqa: E402
from utils.loss_utils import segmentation_loss  # noqa: E402
from utils.feature_losses import (
    feature_alignment_loss,
    warp_consistency_loss,
    instance_prototype_loss,
)  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature alignment projection: seg dense (C_seg) -> CLIP-aligned (C_f) at Fr.
# ---------------------------------------------------------------------------


class FeatureAlignProj(nn.Module):
    def __init__(self, in_dim: int = 96, out_dim: int = 64, feature_res: int = 64):
        super().__init__()
        self.feature_res = feature_res
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GroupNorm(1, out_dim), nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        """dense: (B, N, C_seg, H, W) -> (B, N, C_f, Fr, Fr), channel-L2-normed."""
        B, N, C, H, W = dense.shape
        x = dense.reshape(B * N, C, H, W)
        x = F.interpolate(x, (self.feature_res, self.feature_res),
                          mode="bilinear", align_corners=False)
        x = self.net(x)
        Cf = x.shape[1]
        x = x.view(B, N, Cf, self.feature_res, self.feature_res)
        return channel_l2norm(x)


# ---------------------------------------------------------------------------
# Joint model: seg backbone+head + frozen feature head + align proj.
# ---------------------------------------------------------------------------


class JointSegFeatureModel(nn.Module):
    def __init__(self, seg_model, feature_head, align_proj, *, feature_head_grad: bool):
        super().__init__()
        self.seg_model = seg_model
        self.feature_head = feature_head
        self.align_proj = align_proj
        # Feature head is frozen (eval, no param grad). Optionally allow its
        # forward to participate in the autograd graph (feature_head_grad=True)
        # so the distillation target tracks backbone LoRA updates.
        self.feature_head_grad = feature_head_grad
        for p in self.feature_head.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.seg_model.backbone.eval()
        self.feature_head.eval()
        return self

    def forward(
        self,
        imgs, prompt_masks, prompt_views,
        *,
        prompt_points=None, prompt_point_labels=None,
        prompt_boxes=None, prompt_box_valid=None,
        return_pre_sim=False,
    ):
        B, N, _, H, W = imgs.shape
        want_geom = getattr(self.seg_model.head, "geom_pe", None) is not None
        tokens_list, geom, _ = self.seg_model.backbone.forward_split(
            imgs, return_geometry=want_geom, external_geometry=None,
        )
        seg_out = self.seg_model.head(
            tokens_list, prompt_masks, prompt_views, H, W,
            return_pre_sim=return_pre_sim,
            prompt_points=prompt_points,
            prompt_point_labels=prompt_point_labels,
            prompt_boxes=prompt_boxes,
            prompt_box_valid=prompt_box_valid,
            geometry=geom,
            return_dense=True,
        )
        if self.feature_head_grad and torch.is_grad_enabled():
            feat_pred = self.feature_head(tokens_list)
        else:
            with torch.no_grad():
                feat_pred = self.feature_head(tokens_list)
        aligned = self.align_proj(seg_out["dense_features"])
        seg_out["feat_pred"] = feat_pred
        seg_out["aligned_features"] = aligned
        return seg_out


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl"); torch.cuda.set_device(local_rank)
        return rank, ws, True
    return 0, 1, False


def _is_main(rank): return rank == 0


def get_args():
    p = argparse.ArgumentParser("JAFAR Stage-2: Seg + Feature joint training")
    p.add_argument("--sa1b_root", default="/mnt/dataset/marko/puzzles/sa1b/data")
    p.add_argument("--jafar_cache_root",
                   default="/mnt/dataset/marko/puzzles/sa1b/data/jafar_clip_cache/pca64_long192_ar")
    p.add_argument("--shard_names", nargs="*", default=None)
    p.add_argument("--checkpoint", default=None, help="VGGT-Omega .pt backbone weights")
    p.add_argument("--feature_head_ckpt", required=True,
                   help="Stage-1 checkpoint-best.pth (frozen distillation source)")
    p.add_argument("--init_from", default=None,
                   help="Optional seg-head checkpoint to init the seg head (Stage-0 best).")
    p.add_argument("--init_skip_backbone_lora", action="store_true",
                   help="With --init_from: do NOT load old backbone LoRA weights.")
    p.add_argument("--resume", default=None)
    p.add_argument("--output_dir", default="output/seg_jafar_stage2")

    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_instances", type=int, default=16)
    p.add_argument("--max_prompts", type=int, default=4)
    p.add_argument("--feature_res", type=int, default=64)
    p.add_argument("--feature_dim", type=int, default=64)
    p.add_argument("--vggt_layers", type=int, nargs="+", default=[4, 11, 17, 23])

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=2)

    p.add_argument("--work_dim", type=int, default=768)
    p.add_argument("--n_obj_tokens", type=int, default=4)
    p.add_argument("--n_twoway_layers", type=int, default=4)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--pixel_out_dim", type=int, default=96)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--use_cross_view", action="store_true", default=True)
    p.add_argument("--no_cross_view", dest="use_cross_view", action="store_false")
    p.add_argument("--use_geom_pe", action="store_true", default=False)
    p.add_argument("--no_geom_pe", dest="use_geom_pe", action="store_false")

    p.add_argument("--backbone_lora", action="store_true")
    p.add_argument("--backbone_lora_rank", type=int, default=16)
    p.add_argument("--backbone_lora_alpha", type=float, default=16.0)
    p.add_argument("--backbone_lora_start_layer", type=int, default=18)
    p.add_argument("--ddp_find_unused", action="store_true")
    p.add_argument("--feature_head_grad", action="store_true",
                   help="Let feature-head forward participate in autograd (so the "
                        "distillation target tracks backbone-LoRA updates).")

    p.add_argument("--lambda_bce", type=float, default=5.0)
    p.add_argument("--lambda_dice", type=float, default=5.0)
    p.add_argument("--lambda_iou", type=float, default=1.0)
    p.add_argument("--lambda_occ", type=float, default=0.2)
    p.add_argument("--lambda_consist", type=float, default=0.01)
    p.add_argument("--lambda_point", type=float, default=1.0)
    p.add_argument("--lambda_feat", type=float, default=1.0)
    p.add_argument("--lambda_distill", type=float, default=0.5)
    p.add_argument("--lambda_inst", type=float, default=0.2)
    p.add_argument("--lambda_warp", type=float, default=0.05)
    p.add_argument("--max_warp_pairs", type=int, default=12)
    p.add_argument("--prompt_modalities", nargs="+", default=["mask", "box", "point"])
    p.add_argument("--max_prompt_points", type=int, default=6)
    p.add_argument("--multiview_prompt_prob", type=float, default=0.5)

    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--vis_freq", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_steps", type=int, default=-1)
    return p.parse_args()


def _cosine_lr(opt, epoch, total, warmup, base, min_lr=1e-6):
    if epoch < warmup:
        lr = base * (epoch + 1) / warmup
    else:
        prog = (epoch - warmup) / max(1, total - warmup)
        lr = min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))
    for pg in opt.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def _mean_iou(pred_logits, gt):
    M = min(pred_logits.shape[2], gt.shape[2])
    if M == 0: return 0.0
    pred = (pred_logits[:, :, :M].sigmoid() > 0.5).float()
    g = gt[:, :, :M]
    inter = (pred * g).sum(dim=(-2, -1))
    union = ((pred + g) > 0).float().sum(dim=(-2, -1))
    return (inter / union.clamp(min=1.0)).mean().item()


@torch.no_grad()
def _mean_cos(pred, teacher, valid):
    cos = (pred * teacher).sum(dim=1, keepdim=True) * valid
    return (cos.sum() / valid.sum().clamp_min(1.0)).item()


def train(args):
    rank, ws, is_ddp = _setup_ddp()
    torch.manual_seed(args.seed)
    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if _is_main(rank) else None

    common_ds = dict(
        sa1b_root=args.sa1b_root, n_views=args.n_views, resolution=args.resolution,
        max_instances=args.max_instances, max_prompts=args.max_prompts,
        prompt_modalities=tuple(args.prompt_modalities),
        multiview_prompt_prob=args.multiview_prompt_prob,
        max_prompt_points=args.max_prompt_points, seed=args.seed,
    )
    train_ds = JafarFeatureDataset(
        **common_ds, jafar_cache_root=args.jafar_cache_root,
        feature_res=args.feature_res, feature_dim=args.feature_dim,
        shard_names=args.shard_names, augment=True,
    )
    val_ds = JafarFeatureDataset(
        **{**common_ds, "prompt_modalities": ("mask",), "multiview_prompt_prob": 0.0},
        jafar_cache_root=args.jafar_cache_root,
        feature_res=args.feature_res, feature_dim=args.feature_dim,
        shard_names=args.shard_names, augment=False,
    )
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    tr_idx, va_idx = random_split(range(len(train_ds)), [n_train, n_val],
                                  generator=torch.Generator().manual_seed(args.seed))
    train_subset = Subset(train_ds, tr_idx.indices)
    val_subset = Subset(val_ds, va_idx.indices)
    train_sampler = DistributedSampler(train_subset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_subset, shuffle=False) if is_ddp else None
    _pf = args.prefetch_factor if args.num_workers > 0 else None
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              prefetch_factor=_pf, persistent_workers=(args.num_workers > 0),
                              collate_fn=jafar_feature_collate, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=_pf,
                            persistent_workers=(args.num_workers > 0),
                            collate_fn=jafar_feature_collate)
    if _is_main(rank):
        logger.info("Train %d | Val %d | GPUs %d", n_train, n_val, ws)

    # -- Build seg model --
    backbone_lora_config = None
    if args.backbone_lora:
        backbone_lora_config = dict(rank=args.backbone_lora_rank,
                                    alpha=args.backbone_lora_alpha,
                                    start_layer=args.backbone_lora_start_layer,
                                    inter_frame_multiplier=2)
    seg_model = build_vggt_omega_segmentation(
        checkpoint_path=args.checkpoint, device=str(device),
        work_dim=args.work_dim, n_obj_tokens=args.n_obj_tokens,
        n_twoway_layers=args.n_twoway_layers, nhead=args.nhead,
        pixel_out_dim=args.pixel_out_dim, use_layers=tuple(args.vggt_layers),
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        use_cross_view=args.use_cross_view, use_geom_pe=args.use_geom_pe,
        backbone_lora_config=backbone_lora_config,
    )

    # -- Build frozen feature head from Stage-1 ckpt --
    feature_head = build_feature_sequence_head(
        device=str(device), n_vggt_layers=len(args.vggt_layers),
        work_dim=args.work_dim, feature_out_dim=args.feature_dim,
        feature_res=args.feature_res, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, nhead=args.nhead,
        use_cross_view=args.use_cross_view,
    )
    fh_ckpt = torch.load(args.feature_head_ckpt, map_location=device, weights_only=False)
    fh_state = fh_ckpt["feature_head_state_dict"]
    missing, unexpected = feature_head.load_state_dict(fh_state, strict=False)
    if _is_main(rank):
        logger.info("Loaded feature head from %s (missing=%d unexpected=%d val_cos=%.4f)",
                    args.feature_head_ckpt, len(missing), len(unexpected),
                    fh_ckpt.get("val_cos", float("nan")))

    # -- Optional seg-head init --
    if args.init_from:
        im = torch.load(args.init_from, map_location=device, weights_only=False)
        m, u = seg_model.head.load_state_dict(im["head_state_dict"], strict=False)
        if _is_main(rank):
            logger.info("Init seg head from %s (missing=%d unexpected=%d val_iou=%.4f)",
                        args.init_from, len(m), len(u), im.get("val_iou", float("nan")))
        if "backbone_lora_state_dict" in im and args.backbone_lora and \
                not args.init_skip_backbone_lora:
            seg_model.backbone.load_state_dict(im["backbone_lora_state_dict"], strict=False)

    align_proj = FeatureAlignProj(
        in_dim=args.pixel_out_dim, out_dim=args.feature_dim,
        feature_res=args.feature_res,
    ).to(device)

    joint = JointSegFeatureModel(seg_model, feature_head, align_proj,
                                 feature_head_grad=args.feature_head_grad).to(device)
    if is_ddp:
        joint = DDP(joint, device_ids=[device.index],
                    find_unused_parameters=args.ddp_find_unused)
        raw = joint.module
    else:
        raw = joint

    trainable = [p for p in joint.parameters() if p.requires_grad]
    if _is_main(rank):
        logger.info("Trainable params: %s", f"{sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = 0.0; global_step = 0; start_epoch = 0
    if args.resume:
        rm = torch.load(args.resume, map_location=device, weights_only=False)
        raw.seg_model.head.load_state_dict(rm["head_state_dict"])
        if "align_proj_state_dict" in rm:
            raw.align_proj.load_state_dict(rm["align_proj_state_dict"])
        if "backbone_lora_state_dict" in rm and hasattr(raw.seg_model, "backbone"):
            raw.seg_model.backbone.load_state_dict(rm["backbone_lora_state_dict"], strict=False)
        optimizer.load_state_dict(rm["optimizer_state_dict"])
        scaler.load_state_dict(rm["scaler_state_dict"])
        start_epoch = rm["epoch"]; best_iou = rm.get("val_iou", 0.0)
        global_step = start_epoch * len(train_loader)
        if _is_main(rank):
            logger.info("Resumed from %s (epoch %d val_iou=%.4f)", args.resume, start_epoch, best_iou)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = _cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)
        joint.train(); optimizer.zero_grad()
        ep_loss = 0.0; ep_iou = 0.0; ep_cos = 0.0; n = 0; t0 = time.time()

        for step, batch in enumerate(train_loader):
            imgs = batch["imgs"].to(device, non_blocking=True)
            pm = batch["prompt_masks"].to(device, non_blocking=True)
            gt = batch["gt_consistent_masks"].to(device, non_blocking=True)
            pv = batch["prompt_views"].to(device, non_blocking=True)
            ppts = batch["prompt_points"].to(device, non_blocking=True)
            plab = batch["prompt_point_labels"].to(device, non_blocking=True)
            pbox = batch["prompt_boxes"].to(device, non_blocking=True)
            pboxv = batch["prompt_box_valid"].to(device, non_blocking=True)
            teacher = batch["teacher_features"].to(device, non_blocking=True)
            valid = batch["teacher_valid"].to(device, non_blocking=True)
            boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
            psize = batch["parent_size"].to(device, non_blocking=True)

            want_vis = (global_step + 1) % args.vis_freq == 0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = joint(imgs, pm, pv, prompt_points=ppts, prompt_point_labels=plab,
                            prompt_boxes=pbox, prompt_box_valid=pboxv, return_pre_sim=want_vis)
                logits = out["segmentation_logits"]
                extras = {k: out[k] for k in ("iou_pred", "occ_logits", "obj_tokens") if k in out}
                l_seg = segmentation_loss(
                    logits, gt, prompt_views=pv,
                    lambda_bce=args.lambda_bce, lambda_dice=args.lambda_dice,
                    extras=extras, lambda_iou=args.lambda_iou, lambda_occ=args.lambda_occ,
                    lambda_consist=args.lambda_consist, lambda_point=args.lambda_point,
                    prompt_points=ppts, prompt_point_labels=plab,
                )
                aligned = out["aligned_features"]            # (B,N,Cf,Fr,Fr) L2-normed
                feat_pred = out["feat_pred"].detach()        # frozen distillation target
                l_feat = feature_alignment_loss(aligned, teacher, valid) \
                    if args.lambda_feat > 0 else aligned.sum() * 0.0
                l_distill = feature_alignment_loss(aligned, feat_pred) \
                    if args.lambda_distill > 0 else aligned.sum() * 0.0
                l_inst = instance_prototype_loss(aligned, gt, pv, valid=valid) \
                    if args.lambda_inst > 0 else aligned.sum() * 0.0
                l_warp = warp_consistency_loss(aligned, boxes, psize, args.feature_res,
                                               max_pairs=args.max_warp_pairs) \
                    if args.lambda_warp > 0 else aligned.sum() * 0.0
                loss = (l_seg
                        + args.lambda_feat * l_feat
                        + args.lambda_distill * l_distill
                        + args.lambda_inst * l_inst
                        + args.lambda_warp * l_warp)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            sl = loss.item() * args.accum_steps
            si = _mean_iou(logits, gt)
            sc = _mean_cos(aligned.float(), teacher.float(), valid.float())
            ep_loss += sl; ep_iou += si; ep_cos += sc; n += 1; global_step += 1
            if _is_main(rank) and writer is not None:
                writer.add_scalar("train/loss_step", sl, global_step)
                writer.add_scalar("train/iou_step", si, global_step)
                writer.add_scalar("train/cos_step", sc, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/l_seg", float(l_seg.detach()), global_step)
                writer.add_scalar("train/l_feat", float(l_feat.detach()), global_step)
                writer.add_scalar("train/l_distill", float(l_distill.detach()), global_step)
                writer.add_scalar("train/l_inst", float(l_inst.detach()), global_step)
                writer.add_scalar("train/l_warp", float(l_warp.detach()), global_step)
            if _is_main(rank) and (step + 1) % args.print_freq == 0:
                logger.info("Epoch %d step %d/%d loss=%.4f iou=%.4f cos=%.4f lr=%.2e",
                            epoch + 1, step + 1, len(train_loader),
                            ep_loss / n, ep_iou / n, ep_cos / n, lr)
            if args.max_steps > 0 and (step + 1) >= args.max_steps:
                break

        tt = time.time() - t0
        if _is_main(rank) and writer is not None:
            writer.add_scalar("train/loss_epoch", ep_loss / max(n, 1), epoch + 1)
            writer.add_scalar("train/iou_epoch", ep_iou / max(n, 1), epoch + 1)
            writer.add_scalar("train/cos_epoch", ep_cos / max(n, 1), epoch + 1)

        # -- Validation (mask-only) --
        joint.eval()
        vl = 0.0; vi = 0.0; vc = 0.0; vs = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["imgs"].to(device, non_blocking=True)
                pm = batch["prompt_masks"].to(device, non_blocking=True)
                gt = batch["gt_consistent_masks"].to(device, non_blocking=True)
                pv = batch["prompt_views"].to(device, non_blocking=True)
                teacher = batch["teacher_features"].to(device, non_blocking=True)
                valid = batch["teacher_valid"].to(device, non_blocking=True)
                boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
                psize = batch["parent_size"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = joint(imgs, pm, pv)
                    logits = out["segmentation_logits"]
                    extras = {k: out[k] for k in ("iou_pred", "occ_logits", "obj_tokens") if k in out}
                    l_seg = segmentation_loss(logits, gt, prompt_views=pv,
                                              lambda_bce=args.lambda_bce, lambda_dice=args.lambda_dice,
                                              extras=extras, lambda_iou=args.lambda_iou,
                                              lambda_occ=args.lambda_occ, lambda_consist=args.lambda_consist)
                    aligned = out["aligned_features"]
                    l_feat = feature_alignment_loss(aligned, teacher, valid) if args.lambda_feat > 0 else aligned.sum() * 0.0
                    l_warp = warp_consistency_loss(aligned, boxes, psize, args.feature_res,
                                                   max_pairs=args.max_warp_pairs) if args.lambda_warp > 0 else aligned.sum() * 0.0
                    loss = l_seg + args.lambda_feat * l_feat + args.lambda_warp * l_warp
                vl += loss.item(); vi += _mean_iou(logits, gt)
                vc += _mean_cos(aligned.float(), teacher.float(), valid.float()); vs += 1
                if args.max_steps > 0 and vs >= args.max_steps:
                    break
        avi = vi / max(vs, 1); avc = vc / max(vs, 1); avl = vl / max(vs, 1)
        if _is_main(rank) and writer is not None:
            writer.add_scalar("val/loss_epoch", avl, epoch + 1)
            writer.add_scalar("val/iou_epoch", avi, epoch + 1)
            writer.add_scalar("val/cos_epoch", avc, epoch + 1)
        if _is_main(rank):
            logger.info("Epoch %d/%d train loss=%.4f iou=%.4f cos=%.4f | val loss=%.4f iou=%.4f cos=%.4f time=%.1fs lr=%.2e",
                        epoch + 1, args.epochs, ep_loss / max(n, 1), ep_iou / max(n, 1), ep_cos / max(n, 1),
                        avl, avi, avc, tt, lr)

        if _is_main(rank):
            head_sd = raw.seg_model.head.state_dict()
            align_sd = raw.align_proj.state_dict()
            blora_sd = None
            if getattr(raw.seg_model.backbone, "_lora_enabled", False):
                from models.lora_utils import lora_state_dict
                blora_sd = lora_state_dict(raw.seg_model.backbone)
            if (epoch + 1) % args.save_freq == 0:
                torch.save({"epoch": epoch + 1, "head_state_dict": head_sd,
                            "align_proj_state_dict": align_sd,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "val_iou": avi, "val_cos": avc}, out_dir / "checkpoint-last.pth")
                if blora_sd is not None:
                    torch.save({"backbone_lora_state_dict": blora_sd}, out_dir / "backbone_lora-last.pth")
            if avi > best_iou:
                best_iou = avi
                best = {"epoch": epoch + 1, "head_state_dict": head_sd,
                        "align_proj_state_dict": align_sd, "val_iou": best_iou, "val_cos": avc}
                if blora_sd is not None:
                    best["backbone_lora_state_dict"] = blora_sd
                torch.save(best, out_dir / "checkpoint-best.pth")
                logger.info("New best val IoU: %.4f — saved checkpoint-best.pth", best_iou)
        if is_ddp:
            dist.barrier()

    if writer is not None: writer.close()
    if is_ddp: dist.destroy_process_group()
    if _is_main(rank):
        logger.info("Stage-2 complete. Best val IoU: %.4f", best_iou)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    train(get_args())
