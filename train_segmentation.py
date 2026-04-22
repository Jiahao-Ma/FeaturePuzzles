"""Training script for Pi3X multi-view consistent segmentation.

Single-GPU usage:
    python train_segmentation.py \
        --sa1b_root /mnt/dataset/marko/data/sa1b/data \
        --checkpoint checkpoints/model.safetensors \
        --output_dir output/seg_run1

Multi-GPU (DDP) usage:
    torchrun --nproc_per_node=4 train_segmentation.py \
        --sa1b_root /mnt/dataset/marko/data/sa1b/data \
        --checkpoint checkpoints/model.safetensors \
        --output_dir output/seg_run1

TensorBoard:
    tensorboard --logdir output/seg_run1/tb --port 6006 --bind_all
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
import torch.nn as nn
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "Pi3"))

from sa1b.sa1b_puzzles_dataset import SA1BPuzzlesDataset  # noqa: E402
from models.pi3x_segmentation import build_pi3x_segmentation  # noqa: E402
from utils.loss_utils import segmentation_loss  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _setup_ddp() -> tuple[int, int, bool]:
    """Initialise DDP if launched via torchrun. Returns (rank, world_size, is_ddp)."""
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
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Pi3X Segmentation Training")

    p.add_argument("--sa1b_root", type=str, default="/mnt/dataset/marko/data/sa1b/data")
    p.add_argument("--shard_names", type=str, nargs="*", default=None)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to Pi3X .safetensors checkpoint (None = HF download)")
    p.add_argument("--output_dir", type=str, default="output/seg")

    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--max_instances", type=int, default=16)
    p.add_argument("--max_prompts", type=int, default=4)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)

    p.add_argument("--lambda_bce", type=float, default=5.0)
    p.add_argument("--lambda_dice", type=float, default=5.0)

    p.add_argument("--query_dim", type=int, default=256)
    p.add_argument("--num_decoder_layers", type=int, default=3)
    p.add_argument("--nhead", type=int, default=8)

    p.add_argument("--curriculum", action="store_true",
                   help="Enable curriculum overlap scheduling")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable data augmentation on train set")

    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--vis_freq", type=int, default=100,
                   help="Log visual samples to TensorBoard every N training steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Learning rate schedule (cosine with linear warmup)
# ---------------------------------------------------------------------------

def _cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# IoU metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mean_iou(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> float:
    """Compute mean IoU between predictions and GT, handling M_prompt != M_gt."""
    M_pred = pred_logits.shape[2]
    M_gt = gt_masks.shape[2]
    M = min(M_pred, M_gt)
    if M == 0:
        return 0.0
    pred = (pred_logits[:, :, :M].sigmoid() > 0.5).float()
    gt = gt_masks[:, :, :M]
    intersection = (pred * gt).sum(dim=(-2, -1))
    union = ((pred + gt) > 0).float().sum(dim=(-2, -1))
    iou = intersection / union.clamp(min=1.0)
    return iou.mean().item()


# ---------------------------------------------------------------------------
# TensorBoard visualisation helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    (0.90, 0.30, 0.30), (0.30, 0.80, 0.30), (0.30, 0.50, 0.90),
    (0.95, 0.75, 0.20), (0.80, 0.30, 0.80), (0.20, 0.85, 0.85),
    (1.00, 0.55, 0.30), (0.60, 0.95, 0.40), (0.50, 0.30, 0.70),
    (0.95, 0.50, 0.60), (0.40, 0.70, 0.50), (0.70, 0.70, 0.20),
    (0.30, 0.60, 0.80), (0.85, 0.40, 0.55), (0.55, 0.85, 0.70),
    (0.75, 0.60, 0.90),
]

_TURBO_ANCHORS = torch.tensor([
    [0.19, 0.07, 0.23], [0.27, 0.17, 0.59], [0.18, 0.42, 0.84],
    [0.13, 0.65, 0.73], [0.29, 0.82, 0.42], [0.56, 0.93, 0.25],
    [0.84, 0.88, 0.12], [0.97, 0.65, 0.11], [0.94, 0.37, 0.07],
    [0.74, 0.16, 0.02],
])


def _apply_turbo(prob_map: torch.Tensor) -> torch.Tensor:
    """(H, W) float [0,1] -> (3, H, W) turbo-coloured heatmap."""
    idx = (prob_map.clamp(0, 1) * (_TURBO_ANCHORS.shape[0] - 1)).long()
    return _TURBO_ANCHORS[idx].permute(2, 0, 1)


def _overlay_masks(img: torch.Tensor, masks: torch.Tensor,
                   alpha: float = 0.55, max_inst: int = 8) -> torch.Tensor:
    """Overlay coloured instance masks on a (3,H,W) image."""
    out = img.clone()
    M = min(masks.shape[0], max_inst)
    for m in range(M):
        r, g, b = _PALETTE[m % len(_PALETTE)]
        mask = (masks[m] > 0.5).float()
        cm = torch.stack([mask * r, mask * g, mask * b], dim=0)
        out = out * (1 - alpha * mask.unsqueeze(0)) + cm * alpha
    return out.clamp(0, 1)


@torch.no_grad()
def log_vis_to_tb(
    writer: SummaryWriter,
    tag: str,
    imgs: torch.Tensor,
    prompt_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    global_step: int,
    max_views: int = 4,
    max_inst: int = 5,
) -> None:
    """Log structured visualisations to TensorBoard.

    Shapes coming in (batch dim already present):
      imgs          (B, N, 3, H, W)
      prompt_masks  (B, N, M_prompt, H, W)
      gt_masks      (B, N, M_gt, H, W)
      pred_logits   (B, N, M_pred, H, W)
    """
    i = imgs[0].cpu().float()
    s = prompt_masks[0].cpu().float()
    g = gt_masks[0].cpu().float()
    p = pred_logits[0].cpu().float()

    N = min(max_views, i.shape[0])
    M_pred = p.shape[1]

    row_input, row_prompt, row_gt, row_pred = [], [], [], []
    for v in range(N):
        img_v = i[v].clamp(0, 1)
        row_input.append(img_v)
        row_prompt.append(_overlay_masks(img_v, s[v], max_inst=max_inst))
        row_gt.append(_overlay_masks(img_v, g[v], max_inst=max_inst))
        row_pred.append(_overlay_masks(img_v, p[v].sigmoid(), max_inst=max_inst))

    overview = vutils.make_grid(
        row_input + row_prompt + row_gt + row_pred,
        nrow=N, padding=4, pad_value=0.3,
    )
    writer.add_image(f"{tag}/overview", overview, global_step)

    data_grid = vutils.make_grid(
        [i[v].clamp(0, 1) for v in range(N)],
        nrow=N, padding=2, pad_value=0.0,
    )
    writer.add_image(f"{tag}/data_input", data_grid, global_step)

    gt_pred_panels = []
    for v in range(N):
        img_v = i[v].clamp(0, 1)
        gt_pred_panels.append(_overlay_masks(img_v, g[v], max_inst=max_inst))
        gt_pred_panels.append(_overlay_masks(img_v, p[v].sigmoid(), max_inst=max_inst))
        combined_prob = p[v].sigmoid().max(dim=0).values
        gt_pred_panels.append(_apply_turbo(combined_prob))

    gt_pred_grid = vutils.make_grid(
        gt_pred_panels, nrow=3, padding=4, pad_value=0.15,
    )
    writer.add_image(f"{tag}/gt_vs_pred", gt_pred_grid, global_step)

    M_show = min(max_inst, M_pred)
    if M_show > 0:
        prob_panels = [i[0].clamp(0, 1)]
        for m in range(M_show):
            prob_panels.append(_apply_turbo(p[0, m].sigmoid()))
        prob_grid = vutils.make_grid(
            prob_panels, nrow=M_show + 1, padding=2, pad_value=0.0,
        )
        writer.add_image(f"{tag}/prob_maps_v0", prob_grid, global_step)


# ---------------------------------------------------------------------------
# Collate (handles variable M_prompt / M_gt across samples)
# ---------------------------------------------------------------------------

def _pad_collate(batch):
    # prompt_masks and gt_consistent_masks have the same M per sample (1:1),
    # but M may differ across samples in a batch — pad to max.
    max_m = max(b["prompt_masks"].shape[1] for b in batch)
    max_pv = max(b["prompt_views"].shape[0] for b in batch)

    imgs = torch.stack([b["imgs"] for b in batch])

    prompt_list, gt_list, pv_list = [], [], []
    for b in batch:
        pm = b["prompt_masks"]
        g = b["gt_consistent_masks"]
        pv = b["prompt_views"]
        N, M, H, W = pm.shape
        if M < max_m:
            pm = torch.cat([pm, torch.zeros(N, max_m - M, H, W)], dim=1)
            g = torch.cat([g, torch.zeros(N, max_m - M, H, W)], dim=1)
        if pv.shape[0] < max_pv:
            pv = torch.cat([pv, torch.zeros(max_pv - pv.shape[0], dtype=torch.long)])
        prompt_list.append(pm)
        gt_list.append(g)
        pv_list.append(pv)

    return {
        "imgs": imgs,
        "prompt_masks": torch.stack(prompt_list),
        "gt_consistent_masks": torch.stack(gt_list),
        "prompt_views": torch.stack(pv_list),
    }


# ---------------------------------------------------------------------------
# Training loop
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
        logger.info("TensorBoard logs -> %s", tb_dir)

    # -- Shared dataset args --
    common_ds_kwargs = dict(
        sa1b_root=args.sa1b_root,
        shard_names=args.shard_names,
        n_views=args.n_views,
        resolution=args.resolution,
        max_instances=args.max_instances,
        max_prompts=args.max_prompts,
        seed=args.seed,
    )

    train_dataset = SA1BPuzzlesDataset(
        **common_ds_kwargs, augment=not args.no_augment,
    )
    val_dataset = SA1BPuzzlesDataset(
        **common_ds_kwargs, augment=False,
    )

    n_val = max(1, int(len(train_dataset) * args.val_split))
    n_train = len(train_dataset) - n_val
    train_indices, val_indices = random_split(
        range(len(train_dataset)),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)

    train_sampler = DistributedSampler(train_subset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_subset, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_pad_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_pad_collate,
    )

    if _is_main(rank):
        logger.info("Train: %d samples  |  Val: %d samples  |  GPUs: %d",
                     n_train, n_val, world_size)

    # -- Model --
    model = build_pi3x_segmentation(
        checkpoint_path=args.checkpoint,
        device=str(device),
        query_dim=args.query_dim,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,
    )

    if is_ddp:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
        raw_model = model.module
    else:
        raw_model = model

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if _is_main(rank):
        logger.info("Trainable params: %s  |  Frozen params: %s",
                     f"{n_trainable:,}", f"{n_frozen:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = _cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)

        if args.curriculum:
            progress = epoch / max(1, args.epochs - 1)
            train_dataset.set_curriculum(progress)
            if _is_main(rank) and writer is not None:
                writer.add_scalar("curriculum/min_overlap",
                                  train_dataset.puzzle_kwargs["min_overlap"], epoch + 1)
                writer.add_scalar("curriculum/max_overlap",
                                  train_dataset.puzzle_kwargs["max_overlap"], epoch + 1)
                writer.add_scalar("curriculum/min_box_scale",
                                  train_dataset.puzzle_kwargs["min_box_size_scale"], epoch + 1)
                writer.add_scalar("curriculum/max_box_scale",
                                  train_dataset.puzzle_kwargs["max_box_size_scale"], epoch + 1)
                logger.info(
                    "Curriculum progress=%.3f  overlap=[%.3f, %.3f]  scale=[%.3f, %.3f]",
                    progress,
                    train_dataset.puzzle_kwargs["min_overlap"],
                    train_dataset.puzzle_kwargs["max_overlap"],
                    train_dataset.puzzle_kwargs["min_box_size_scale"],
                    train_dataset.puzzle_kwargs["max_box_size_scale"],
                )

        model.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
        epoch_iou = 0.0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            imgs = batch["imgs"].to(device)
            pm = batch["prompt_masks"].to(device)
            gt = batch["gt_consistent_masks"].to(device)
            pv = batch["prompt_views"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                results = model(imgs, pm)
                logits = results["segmentation_logits"]
                loss = segmentation_loss(
                    logits, gt,
                    prompt_views=pv,
                    lambda_bce=args.lambda_bce,
                    lambda_dice=args.lambda_dice,
                )
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step_loss = loss.item() * args.accum_steps
            step_iou = _mean_iou(logits, gt)
            epoch_loss += step_loss
            epoch_iou += step_iou
            n_steps += 1
            global_step += 1

            if _is_main(rank) and writer is not None:
                writer.add_scalar("train/loss_step", step_loss, global_step)
                writer.add_scalar("train/iou_step", step_iou, global_step)
                writer.add_scalar("train/lr", lr, global_step)

                if global_step % args.vis_freq == 0:
                    log_vis_to_tb(writer, "train", imgs, pm, gt, logits, global_step)

            if _is_main(rank) and (step + 1) % args.print_freq == 0:
                avg = epoch_loss / n_steps
                avg_iou = epoch_iou / n_steps
                logger.info(
                    "Epoch %d  step %d/%d  loss=%.4f  mIoU=%.4f  lr=%.2e",
                    epoch + 1, step + 1, len(train_loader), avg, avg_iou, lr,
                )

        train_time = time.time() - t0
        avg_train_loss = epoch_loss / max(n_steps, 1)
        avg_train_iou = epoch_iou / max(n_steps, 1)

        if _is_main(rank) and writer is not None:
            writer.add_scalar("train/loss_epoch", avg_train_loss, epoch + 1)
            writer.add_scalar("train/iou_epoch", avg_train_iou, epoch + 1)

        # -- Validation --
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_steps = 0
        vis_logged = False

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["imgs"].to(device)
                pm = batch["prompt_masks"].to(device)
                gt = batch["gt_consistent_masks"].to(device)
                pv = batch["prompt_views"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    results = model(imgs, pm)
                    logits = results["segmentation_logits"]
                    loss = segmentation_loss(
                        logits, gt,
                        prompt_views=pv,
                        lambda_bce=args.lambda_bce,
                        lambda_dice=args.lambda_dice,
                    )

                val_loss_sum += loss.item()
                val_iou_sum += _mean_iou(logits, gt)
                val_steps += 1

                if _is_main(rank) and writer is not None and not vis_logged:
                    log_vis_to_tb(writer, "val", imgs, pm, gt, logits, epoch + 1)
                    vis_logged = True

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_val_iou = val_iou_sum / max(val_steps, 1)

        if _is_main(rank) and writer is not None:
            writer.add_scalar("val/loss_epoch", avg_val_loss, epoch + 1)
            writer.add_scalar("val/iou_epoch", avg_val_iou, epoch + 1)

        if _is_main(rank):
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_mIoU=%.4f  "
                "time=%.1fs  lr=%.2e",
                epoch + 1, args.epochs, avg_train_loss, avg_val_loss,
                avg_val_iou, train_time, lr,
            )

        # -- Checkpointing (rank 0 only) --
        if _is_main(rank):
            if (epoch + 1) % args.save_freq == 0:
                ckpt = {
                    "epoch": epoch + 1,
                    "seg_head_state_dict": raw_model.seg_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_iou": avg_val_iou,
                }
                torch.save(ckpt, out_dir / "checkpoint-last.pth")

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                torch.save(
                    {"seg_head_state_dict": raw_model.seg_head.state_dict(),
                     "val_iou": best_iou},
                    out_dir / "checkpoint-best.pth",
                )
                logger.info("New best mIoU: %.4f — saved checkpoint-best.pth", best_iou)

        if is_ddp:
            dist.barrier()

    if writer is not None:
        writer.close()
    if is_ddp:
        dist.destroy_process_group()
    if _is_main(rank):
        logger.info("Training complete.  Best val mIoU: %.4f", best_iou)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = get_args()
    train(args)
