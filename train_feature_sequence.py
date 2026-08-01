"""Stage-1 training: learn the FeatureSequenceHead to predict JAFAR-CLIP-aligned
dense feature maps from the FROZEN VGGT-Omega aggregator tokens.

Only the FeatureSequenceHead is trained; the backbone is frozen (no LoRA).
Loss = lambda_feat * L_feat + lambda_warp * L_warp (+ optional lambda_inst * L_inst).

Single-GPU:
    python train_feature_sequence.py \
        --sa1b_root /mnt/dataset/marko/puzzles/sa1b/data \
        --jafar_cache_root /mnt/dataset/marko/puzzles/sa1b/data/jafar_clip_cache/pca64_long192_ar \
        --checkpoint puzzles/checkpoints/vggt_omega_1b_256_text.pt \
        --output_dir output/feat_stage1

Multi-GPU:
    torchrun --nproc_per_node=8 train_feature_sequence.py ...
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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sa1b.jafar_feature_dataset import JafarFeatureDataset, jafar_feature_collate  # noqa: E402
from models.vggt_omega_backbone import VGGTOmegaBackbone  # noqa: E402
from models.vggt_omega_feature_head import build_feature_sequence_head  # noqa: E402
from utils.feature_losses import feature_total_loss, feature_alignment_loss  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _setup_ddp() -> tuple[int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    return 0, 1, False


def _is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("JAFAR Stage-1: Feature Sequence Head")

    p.add_argument("--sa1b_root", type=str, default="/mnt/dataset/marko/puzzles/sa1b/data")
    p.add_argument("--jafar_cache_root", type=str,
                   default="/mnt/dataset/marko/puzzles/sa1b/data/jafar_clip_cache/pca64_long192_ar")
    p.add_argument("--shard_names", type=str, nargs="*", default=None)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="VGGT-Omega .pt weights (frozen backbone).")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="output/feat_stage1")

    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_instances", type=int, default=16)
    p.add_argument("--max_prompts", type=int, default=4)
    p.add_argument("--feature_res", type=int, default=64)
    p.add_argument("--feature_dim", type=int, default=64)
    p.add_argument("--vggt_layers", type=int, nargs="+", default=[4, 11, 17, 23])

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=2)

    p.add_argument("--work_dim", type=int, default=768)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--no_cross_view", action="store_true")

    p.add_argument("--lambda_feat", type=float, default=1.0)
    p.add_argument("--lambda_warp", type=float, default=0.1)
    p.add_argument("--lambda_inst", type=float, default=0.0,
                   help="Instance-prototype contrastive weight (needs GT masks).")
    p.add_argument("--max_warp_pairs", type=int, default=12)

    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--vis_freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Smoke-test cap on steps per epoch (-1 = no cap).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mean_cosine(pred, teacher, valid):
    cos = (pred * teacher).sum(dim=1, keepdim=True)        # already L2-normalised
    cos = cos * valid
    return (cos.sum() / valid.sum().clamp_min(1.0)).item()


# ---------------------------------------------------------------------------
# TensorBoard visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_feat_vis(writer, tag, imgs, pred, teacher, valid, global_step,
                 max_views=4, max_ch=6):
    import torchvision.utils as vutils
    i = imgs[0].cpu().float()
    p = pred[0].cpu().float()
    t = teacher[0].cpu().float()
    N = min(max_views, i.shape[0])
    # PCA(3) of predicted vs teacher features for the first view.
    def pca3(feat, out_hw):
        # feat: (C, H, W) -> (3, out_hw, out_hw) via top-3 eigenvectors of CxC cov.
        C, H, W = feat.shape
        x = feat.reshape(C, -1).float()                       # (C, HW)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x - x.mean(dim=1, keepdim=True)
        cov = x @ x.t() / max(x.shape[1], 1)                  # (C, C)
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        eigvals, eigvecs = torch.linalg.eigh(cov)             # ascending
        top3 = eigvecs[:, -3:]                                # (C, 3)
        proj = top3.t() @ x                                   # (3, HW)
        proj = proj - proj.min(dim=1, keepdim=True).values
        proj = proj / proj.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        out = proj.reshape(3, H, W)
        if (H, W) != out_hw:
            out = F.interpolate(out.unsqueeze(0), out_hw, mode="bilinear",
                               align_corners=False).squeeze(0)
        return out
    hw = i.shape[-2:]
    panels = []
    for v in range(N):
        panels.append(i[v].clamp(0, 1))
        panels.append(pca3(p[v], hw))
        panels.append(pca3(t[v], hw))
    grid = vutils.make_grid(panels, nrow=3, padding=4, pad_value=0.2)
    writer.add_image(f"{tag}/pca_pred_teacher", grid, global_step)
    # channel-wise cosine map per view
    cos_maps = []
    for v in range(N):
        cos = (p[v] * t[v]).sum(0, keepdim=True).clamp(-1, 1)   # (1,H,W)
        cm = cos.repeat(3, 1, 1) * 0.5 + 0.5
        if cm.shape[-2:] != hw:
            cm = F.interpolate(cm.unsqueeze(0), hw, mode="bilinear",
                               align_corners=False).squeeze(0)
        cos_maps.append(cm)
    cg = vutils.make_grid(cos_maps, nrow=N, padding=2, pad_value=0.0)
    writer.add_image(f"{tag}/cosine_map", cg, global_step)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    rank, world_size, is_ddp = _setup_ddp()
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

    writer = None
    if _is_main(rank):
        tb_dir = out_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    common_ds = dict(
        sa1b_root=args.sa1b_root,
        n_views=args.n_views,
        resolution=args.resolution,
        max_instances=args.max_instances,
        max_prompts=args.max_prompts,
        seed=args.seed,
    )
    train_ds = JafarFeatureDataset(
        **common_ds,
        jafar_cache_root=args.jafar_cache_root,
        feature_res=args.feature_res, feature_dim=args.feature_dim,
        shard_names=args.shard_names, augment=True,
    )
    val_ds = JafarFeatureDataset(
        **common_ds,
        jafar_cache_root=args.jafar_cache_root,
        feature_res=args.feature_res, feature_dim=args.feature_dim,
        shard_names=args.shard_names, augment=False,
    )

    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    train_indices, val_indices = random_split(
        range(len(train_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_subset = Subset(train_ds, train_indices.indices)
    val_subset = Subset(val_ds, val_indices.indices)
    train_sampler = DistributedSampler(train_subset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_subset, shuffle=False) if is_ddp else None

    _pf = args.prefetch_factor if args.num_workers > 0 else None
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=_pf, persistent_workers=(args.num_workers > 0),
        collate_fn=jafar_feature_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=_pf,
        persistent_workers=(args.num_workers > 0), collate_fn=jafar_feature_collate,
    )
    if _is_main(rank):
        logger.info("Train: %d | Val: %d | GPUs: %d", n_train, n_val, world_size)

    # -- Model: frozen backbone + trainable feature head --
    backbone = VGGTOmegaBackbone(
        ckpt_path=args.checkpoint,
        use_layers=tuple(args.vggt_layers),
        amp_dtype=torch.bfloat16,
    ).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    feature_head = build_feature_sequence_head(
        device=str(device),
        n_vggt_layers=len(args.vggt_layers),
        work_dim=args.work_dim,
        feature_out_dim=args.feature_dim,
        feature_res=args.feature_res,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        nhead=args.nhead,
        use_cross_view=not args.no_cross_view,
    )
    if is_ddp:
        feature_head = DDP(feature_head, device_ids=[device.index])
        raw_head = feature_head.module
    else:
        raw_head = feature_head

    trainable = [p for p in feature_head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if _is_main(rank):
        logger.info("Feature head trainable params: %s", f"{n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_cos = -1.0
    global_step = 0
    start_epoch = 0

    if args.resume:
        rm = torch.load(args.resume, map_location=device, weights_only=False)
        raw_head.load_state_dict(rm["feature_head_state_dict"])
        optimizer.load_state_dict(rm["optimizer_state_dict"])
        scaler.load_state_dict(rm["scaler_state_dict"])
        start_epoch = rm["epoch"]
        best_cos = rm.get("val_cos", -1.0)
        global_step = start_epoch * len(train_loader)
        if _is_main(rank):
            logger.info("Resumed from %s (epoch %d, val_cos=%.4f)",
                        args.resume, start_epoch, best_cos)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = _cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)

        feature_head.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_cos = 0.0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            imgs = batch["imgs"].to(device, non_blocking=True)
            teacher = batch["teacher_features"].to(device, non_blocking=True)
            valid = batch["teacher_valid"].to(device, non_blocking=True)
            boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
            psize = batch["parent_size"].to(device, non_blocking=True)
            gt = batch["gt_consistent_masks"].to(device, non_blocking=True)
            pv = batch["prompt_views"].to(device, non_blocking=True)

            want_vis = (global_step + 1) % args.vis_freq == 0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                with torch.no_grad():
                    tokens_list, _ = backbone(imgs)
                pred = feature_head(tokens_list)
                loss, log = feature_total_loss(
                    pred, teacher, valid, boxes, psize, args.feature_res,
                    lambda_feat=args.lambda_feat, lambda_warp=args.lambda_warp,
                    gt_masks=gt if args.lambda_inst > 0 else None,
                    prompt_views=pv if args.lambda_inst > 0 else None,
                    lambda_inst=args.lambda_inst,
                    max_warp_pairs=args.max_warp_pairs,
                )
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step_loss = loss.item() * args.accum_steps
            step_cos = _mean_cosine(pred.float(), teacher.float(), valid.float())
            epoch_loss += step_loss
            epoch_cos += step_cos
            n_steps += 1
            global_step += 1

            if _is_main(rank) and writer is not None:
                writer.add_scalar("train/loss_step", step_loss, global_step)
                writer.add_scalar("train/cos_step", step_cos, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                for k, v in log.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                if want_vis:
                    log_feat_vis(writer, "train", imgs, pred.float(),
                                 teacher.float(), valid.float(), global_step)

            if _is_main(rank) and (step + 1) % args.print_freq == 0:
                logger.info("Epoch %d  step %d/%d  loss=%.4f  cos=%.4f  lr=%.2e",
                            epoch + 1, step + 1, len(train_loader),
                            epoch_loss / n_steps, epoch_cos / n_steps, lr)

            if args.max_steps > 0 and (step + 1) >= args.max_steps:
                break

        train_time = time.time() - t0
        avg_train_loss = epoch_loss / max(n_steps, 1)
        avg_train_cos = epoch_cos / max(n_steps, 1)
        if _is_main(rank) and writer is not None:
            writer.add_scalar("train/loss_epoch", avg_train_loss, epoch + 1)
            writer.add_scalar("train/cos_epoch", avg_train_cos, epoch + 1)

        # -- Validation --
        feature_head.eval()
        val_loss_sum = 0.0
        val_cos_sum = 0.0
        val_steps = 0
        vis_logged = False
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["imgs"].to(device, non_blocking=True)
                teacher = batch["teacher_features"].to(device, non_blocking=True)
                valid = batch["teacher_valid"].to(device, non_blocking=True)
                boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
                psize = batch["parent_size"].to(device, non_blocking=True)
                gt = batch["gt_consistent_masks"].to(device, non_blocking=True)
                pv = batch["prompt_views"].to(device, non_blocking=True)
                want_vis = not vis_logged
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    tokens_list, _ = backbone(imgs)
                    pred = feature_head(tokens_list)
                    loss, _ = feature_total_loss(
                        pred, teacher, valid, boxes, psize, args.feature_res,
                        lambda_feat=args.lambda_feat, lambda_warp=args.lambda_warp,
                        gt_masks=gt if args.lambda_inst > 0 else None,
                        prompt_views=pv if args.lambda_inst > 0 else None,
                        lambda_inst=args.lambda_inst,
                        max_warp_pairs=args.max_warp_pairs,
                    )
                val_loss_sum += loss.item()
                val_cos_sum += _mean_cosine(pred.float(), teacher.float(), valid.float())
                val_steps += 1
                if _is_main(rank) and writer is not None and not vis_logged:
                    log_feat_vis(writer, "val", imgs, pred.float(),
                                 teacher.float(), valid.float(), epoch + 1)
                    vis_logged = True
                if args.max_steps > 0 and val_steps >= args.max_steps:
                    break

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_val_cos = val_cos_sum / max(val_steps, 1)
        if _is_main(rank) and writer is not None:
            writer.add_scalar("val/loss_epoch", avg_val_loss, epoch + 1)
            writer.add_scalar("val/cos_epoch", avg_val_cos, epoch + 1)
        if _is_main(rank):
            logger.info("Epoch %d/%d  train_loss=%.4f cos=%.4f | val_loss=%.4f cos=%.4f  time=%.1fs lr=%.2e",
                        epoch + 1, args.epochs, avg_train_loss, avg_train_cos,
                        avg_val_loss, avg_val_cos, train_time, lr)

        # -- Checkpoint --
        if _is_main(rank):
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    "epoch": epoch + 1,
                    "feature_head_state_dict": raw_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_cos": avg_val_cos,
                    "args": vars(args),
                }, out_dir / "checkpoint-last.pth")
            if avg_val_cos > best_cos:
                best_cos = avg_val_cos
                torch.save({
                    "epoch": epoch + 1,
                    "feature_head_state_dict": raw_head.state_dict(),
                    "val_cos": best_cos,
                    "args": vars(args),
                }, out_dir / "checkpoint-best.pth")
                logger.info("New best val cos: %.4f — saved checkpoint-best.pth", best_cos)

        if is_ddp:
            dist.barrier()

    if writer is not None:
        writer.close()
    if is_ddp:
        dist.destroy_process_group()
    if _is_main(rank):
        logger.info("Stage-1 complete. Best val cos: %.4f", best_cos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    train(get_args())
