"""Stage-1 (DINOv3 teacher) training with a FROZEN lingbot-map backbone.

Same training思路 as ``train_feature_dinov3.py`` (which uses a frozen
VGGT-Omega aggregator), ported to the lingbot-map streaming architecture:

  - Backbone: frozen ``GCTStream`` aggregator (lingbot-map). Gradients never
    flow into it. We run it in *batch mode* (empty KV cache, all N views fed
    together) so each training step is independent. The first
    ``num_frame_for_scale`` views get the scale token (bidirectional among
    themselves); the rest attend causally to past views — matching streaming
    inference.
  - Teacher: frozen DINOv3 ViT-B/16, generated ON THE FLY (no pre-cached
    features), warped into each Puzzles crop box.
  - Student head: the SAME ``RGBGuidedFeatureHead`` used by the VGGT-Omega
    branch (token_dim=2048, 4 cached layers, work_dim=768, feature_out_dim=768,
    feature_res=128). The lingbot-map aggregator exposes the same 4 cached
    blocks (4, 11, 17, 23) with the same 2*embed_dim=2048 token dim, so the head
    is reused verbatim.
  - Loss = lambda_feat*(1-cos) + lambda_warp*warp_consistency + lambda_gram*gram_anchor
    (identical to the VGGT-Omega branch via ``feature_total_loss_dinov3``).

Streaming adaptation: because lingbot-map is a streaming (temporal-causal)
transformer, we feed MORE views per training sample than the VGGT-Omega branch
(``n_views`` default 16 vs 6) so each view sees a longer causal context. The
input resolution is 224 (= 14*16) to match lingbot-map's patch_size=14 and yield
a 16x16 patch grid.

Multi-GPU:
    torchrun --nproc_per_node=8 train_feature_dinov3_lingbot.py \
        --output_dir output/feat_dinov3_lingbot ...
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sa1b.dinov3_stream_dataset import DINOv3StreamDataset, stream_collate  # noqa: E402
from models.lingbot_backbone import LingBotBackbone  # noqa: E402
from models.vggt_omega_feature_head import build_rgb_guided_feature_head  # noqa: E402
from models.dinov3_teacher import DINOv3Talk2DinoTeacher, warp_parent_to_views  # noqa: E402
from utils.feature_losses import feature_total_loss_dinov3  # noqa: E402

logger = logging.getLogger(__name__)


def _setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    return 0, 1, False


def _is_main(rank):
    return rank == 0


def get_args():
    p = argparse.ArgumentParser("JAFAR Stage-1 (DINOv3 teacher) — lingbot-map backbone")
    p.add_argument("--sa1b_root", type=str, default="/mnt/dataset/marko/puzzles/sa1b/data")
    p.add_argument("--shard_names", type=str, nargs="*", default=None)
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/lingbot-map/lingbot-map.pt",
                   help="Path to the released lingbot-map GCTStream checkpoint.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="output/feat_dinov3_lingbot")

    # Streaming: more views than the VGGT-Omega branch (6) to give the
    # temporal-causal aggregator a longer context window.
    p.add_argument("--n_views", type=int, default=16)
    # patch_size=14 -> resolution must be a multiple of 14. 224 = 14*16 -> 16x16 grid.
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--teacher_long", type=int, default=2048)
    p.add_argument("--feature_res", type=int, default=128)
    p.add_argument("--feature_dim", type=int, default=768)
    p.add_argument("--dino_variant", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    p.add_argument("--lingbot_layers", type=int, nargs="+", default=[4, 11, 17, 23])
    p.add_argument("--num_frame_for_scale", type=int, default=2,
                   help="Leading views that get the scale token (bidirectional). "
                        "Set to n_views for fully bidirectional; small value (e.g. 2) "
                        "stays faithful to streaming inference.")

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=1)

    p.add_argument("--work_dim", type=int, default=768)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--no_cross_view", action="store_true")

    p.add_argument("--lambda_feat", type=float, default=1.0)
    p.add_argument("--lambda_warp", type=float, default=0.1)
    p.add_argument("--lambda_gram", type=float, default=0.5)
    p.add_argument("--gram_samples", type=int, default=512)
    p.add_argument("--max_warp_pairs", type=int, default=16)

    p.add_argument("--val_split", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--vis_freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_val_steps", type=int, default=40,
                   help="cap validation batches per epoch (-1 = all).")
    p.add_argument("--save_steps", type=int, default=2000,
                   help="save checkpoint-last every N global steps (0 = only at epoch end).")
    return p.parse_args()


def _cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def _mean_cosine(pred, teacher):
    return float((pred * teacher).sum(dim=2).mean())


@torch.no_grad()
def _pca_fit(feat):
    C = feat.shape[0]
    x = F.normalize(feat.reshape(C, -1).float(), dim=0)
    x = torch.nan_to_num(x)
    mu = x.mean(dim=1, keepdim=True)
    xc = x - mu
    cov = torch.nan_to_num(xc @ xc.t() / max(xc.shape[1], 1))
    _, ev = torch.linalg.eigh(cov)
    basis = ev[:, -3:].t()
    proj = basis @ xc
    return mu, basis, proj.min(dim=1, keepdim=True).values, proj.max(dim=1, keepdim=True).values


@torch.no_grad()
def _pca_apply(feat, mu, basis, gmin, gmax):
    C, H, W = feat.shape
    x = F.normalize(feat.reshape(C, -1).float(), dim=0)
    x = torch.nan_to_num(x)
    proj = basis @ (x - mu)
    proj = (proj - gmin) / (gmax - gmin).clamp_min(1e-6)
    return proj.clamp(0, 1).reshape(3, H, W)


@torch.no_grad()
def log_vis(writer, tag, imgs, pred, teacher, parent_feat, boxes, psize,
            feature_res, step, max_views=4):
    import torchvision.utils as vutils
    N = min(max_views, imgs.shape[1])
    hw = imgs.shape[-2:]
    mu, basis, gmin, gmax = _pca_fit(parent_feat[0])
    parent_rgb = _pca_apply(parent_feat[0], mu, basis, gmin, gmax)
    t_rgb = warp_parent_to_views(parent_rgb.unsqueeze(0), boxes[:1], psize[:1],
                                 feature_res, normalize=False)[0]
    panels = []
    for v in range(N):
        img = imgs[0, v].float().cpu().clamp(0, 1)
        pp = _pca_apply(pred[0, v], mu, basis, gmin, gmax)
        tp = t_rgb[v]
        pp = F.interpolate(pp.unsqueeze(0), hw, mode="bilinear",
                           align_corners=False).squeeze(0).cpu()
        tp = F.interpolate(tp.unsqueeze(0), hw, mode="bilinear",
                           align_corners=False).squeeze(0).cpu()
        panels += [img, pp, tp]
    writer.add_image(f"{tag}/pca_img_pred_teacher",
                     vutils.make_grid(panels, nrow=3, padding=4, pad_value=0.2), step)
    cos_maps = []
    for v in range(N):
        cos = (pred[0, v] * teacher[0, v]).sum(0, keepdim=True).clamp(-1, 1)
        cm = (cos.repeat(3, 1, 1) * 0.5 + 0.5)
        cm = F.interpolate(cm.unsqueeze(0), hw, mode="bilinear",
                           align_corners=False).squeeze(0).cpu()
        cos_maps.append(cm)
    writer.add_image(f"{tag}/cosine_map",
                     vutils.make_grid(cos_maps, nrow=N, padding=2), step)


def train(args):
    rank, world_size, is_ddp = _setup_ddp()
    torch.manual_seed(args.seed + rank)

    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    writer = None
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        (out_dir / "tb").mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    common = dict(
        sa1b_root=args.sa1b_root, teacher_long=args.teacher_long,
        n_views=args.n_views, resolution=args.resolution,
        shard_names=args.shard_names, seed=args.seed,
    )
    train_ds = DINOv3StreamDataset(**common, augment=True)
    val_ds = DINOv3StreamDataset(**common, augment=False)

    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    tr_idx, va_idx = random_split(
        range(len(train_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_subset = Subset(train_ds, tr_idx.indices)
    val_subset = Subset(val_ds, va_idx.indices)
    train_sampler = DistributedSampler(train_subset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_subset, shuffle=False) if is_ddp else None

    _pf = args.prefetch_factor if args.num_workers > 0 else None
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=False,
        prefetch_factor=_pf, persistent_workers=(args.num_workers > 0),
        collate_fn=stream_collate, drop_last=True)
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=False, prefetch_factor=_pf,
        persistent_workers=(args.num_workers > 0), collate_fn=stream_collate)
    if _is_main(rank):
        logger.info("Train: %d | Val: %d | GPUs: %d | n_views=%d res=%d",
                    n_train, n_val, world_size, args.n_views, args.resolution)

    # -- frozen lingbot-map backbone --
    backbone = LingBotBackbone(
        ckpt_path=args.checkpoint, use_layers=tuple(args.lingbot_layers),
        amp_dtype=torch.bfloat16, use_sdpa=True,
        num_frame_for_scale=args.num_frame_for_scale).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # -- frozen DINOv3 teacher (on the fly) --
    teacher_model = DINOv3Talk2DinoTeacher(
        variant=args.dino_variant, teacher_long=args.teacher_long,
        device=str(device), amp_dtype=torch.bfloat16)
    teacher_model.eval()

    # -- trainable RGB-guided head (same as the VGGT-Omega branch) --
    head = build_rgb_guided_feature_head(
        device=str(device), n_vggt_layers=len(args.lingbot_layers),
        work_dim=args.work_dim, feature_out_dim=args.feature_dim,
        feature_res=args.feature_res, start_res=args.resolution // 14,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, nhead=args.nhead,
        use_cross_view=not args.no_cross_view)
    if is_ddp:
        head = DDP(head, device_ids=[device.index])
        raw_head = head.module
    else:
        raw_head = head

    trainable = [p for p in head.parameters() if p.requires_grad]
    if _is_main(rank):
        logger.info("Head trainable params: %s",
                    f"{sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_cos, global_step, start_epoch = -1.0, 0, 0
    if args.resume:
        rm = torch.load(args.resume, map_location=device, weights_only=False)
        raw_head.load_state_dict(rm["feature_head_state_dict"])
        if "optimizer_state_dict" in rm:
            optimizer.load_state_dict(rm["optimizer_state_dict"])
            scaler.load_state_dict(rm["scaler_state_dict"])
        start_epoch = rm["epoch"]
        best_cos = rm.get("val_cos", -1.0)
        global_step = rm.get("global_step", start_epoch * len(train_loader))
        if _is_main(rank):
            logger.info("Resumed from %s (epoch %d, step %d, cos=%.4f)",
                        args.resume, start_epoch, global_step, best_cos)

    def run_teacher(parent_img, boxes, psize):
        if not parent_img.is_floating_point():
            parent_img = parent_img.to(device).float().div_(255.0)
        parent_feat = teacher_model.encode_parent(parent_img)
        teacher = warp_parent_to_views(parent_feat, boxes, psize, args.feature_res)
        return teacher, parent_feat

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = _cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr)
        head.train()
        optimizer.zero_grad()
        ep_loss = ep_cos = 0.0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            imgs = batch["imgs"].to(device, non_blocking=True)
            parent_img = batch["parent_img"].to(device, non_blocking=True)
            boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
            psize = batch["parent_size"].to(device, non_blocking=True)

            with torch.no_grad():
                teacher, parent_feat = run_teacher(parent_img, boxes, psize)

            want_vis = (global_step + 1) % args.vis_freq == 0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                with torch.no_grad():
                    tokens_list, _ = backbone(imgs)
                pred = head(tokens_list, imgs)
            pred_f = pred.float()
            loss, log = feature_total_loss_dinov3(
                pred_f, teacher, boxes, psize, args.feature_res,
                lambda_feat=args.lambda_feat, lambda_warp=args.lambda_warp,
                lambda_gram=args.lambda_gram, gram_samples=args.gram_samples,
                max_warp_pairs=args.max_warp_pairs)
            loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step_loss = loss.item() * args.accum_steps
            step_cos = _mean_cosine(pred_f, teacher)
            ep_loss += step_loss
            ep_cos += step_cos
            n_steps += 1
            global_step += 1

            if _is_main(rank) and writer is not None:
                writer.add_scalar("train/loss_step", step_loss, global_step)
                writer.add_scalar("train/cos_step", step_cos, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                for k, v in log.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                if want_vis:
                    log_vis(writer, "train", imgs, pred_f, teacher, parent_feat,
                            boxes, psize, args.feature_res, global_step)
            if _is_main(rank) and (step + 1) % args.print_freq == 0:
                logger.info("Epoch %d step %d/%d loss=%.4f cos=%.4f lr=%.2e",
                            epoch + 1, step + 1, len(train_loader),
                            ep_loss / n_steps, ep_cos / n_steps, lr)
            if (_is_main(rank) and args.save_steps > 0
                    and global_step % args.save_steps == 0):
                torch.save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "feature_head_state_dict": raw_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_cos": best_cos, "args": vars(args),
                }, out_dir / "checkpoint-last.pth")
                logger.info("Saved checkpoint-last at step %d (cos=%.4f)",
                            global_step, ep_cos / n_steps)
            if args.max_steps > 0 and (step + 1) >= args.max_steps:
                break

        train_time = time.time() - t0
        avg_loss = ep_loss / max(n_steps, 1)
        avg_cos = ep_cos / max(n_steps, 1)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # -- validation --
        head.eval()
        v_loss = v_cos = 0.0
        v_steps = 0
        vis_done = False
        val_cap = args.max_val_steps if args.max_val_steps > 0 else 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["imgs"].to(device, non_blocking=True)
                parent_img = batch["parent_img"].to(device, non_blocking=True)
                boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
                psize = batch["parent_size"].to(device, non_blocking=True)
                teacher, parent_feat = run_teacher(parent_img, boxes, psize)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    tokens_list, _ = backbone(imgs)
                    pred = head(tokens_list, imgs)
                pred_f = pred.float()
                loss, _ = feature_total_loss_dinov3(
                    pred_f, teacher, boxes, psize, args.feature_res,
                    lambda_feat=args.lambda_feat, lambda_warp=args.lambda_warp,
                    lambda_gram=args.lambda_gram, gram_samples=args.gram_samples,
                    max_warp_pairs=args.max_warp_pairs)
                v_loss += loss.item()
                v_cos += _mean_cosine(pred_f, teacher)
                v_steps += 1
                if _is_main(rank) and writer is not None and not vis_done:
                    log_vis(writer, "val", imgs, pred_f, teacher, parent_feat,
                            boxes, psize, args.feature_res, epoch + 1)
                    vis_done = True
                del pred, pred_f, teacher, parent_feat, tokens_list, imgs, parent_img, boxes, psize
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if val_cap and v_steps >= val_cap:
                    break

        avg_vloss = v_loss / max(v_steps, 1)
        avg_vcos = v_cos / max(v_steps, 1)
        if _is_main(rank) and writer is not None:
            writer.add_scalar("val/loss_epoch", avg_vloss, epoch + 1)
            writer.add_scalar("val/cos_epoch", avg_vcos, epoch + 1)
        if _is_main(rank):
            logger.info("Epoch %d/%d train_loss=%.4f cos=%.4f | val_loss=%.4f cos=%.4f time=%.1fs",
                        epoch + 1, args.epochs, avg_loss, avg_cos,
                        avg_vloss, avg_vcos, train_time)
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    "epoch": epoch + 1,
                    "feature_head_state_dict": raw_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_cos": avg_vcos, "args": vars(args),
                }, out_dir / "checkpoint-last.pth")
            if avg_vcos > best_cos:
                best_cos = avg_vcos
                torch.save({
                    "epoch": epoch + 1,
                    "feature_head_state_dict": raw_head.state_dict(),
                    "val_cos": best_cos, "args": vars(args),
                }, out_dir / "checkpoint-best.pth")
                logger.info("New best val cos: %.4f", best_cos)
        if is_ddp:
            dist.barrier()

    if writer is not None:
        writer.close()
    if is_ddp:
        dist.destroy_process_group()
    if _is_main(rank):
        logger.info("Done. Best val cos: %.4f", best_cos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    train(get_args())
