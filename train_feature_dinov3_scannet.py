"""Fine-tune RGBGuidedFeatureHead on ScanNet (EmbodiedSplat) with Puzzles crops.

Keeps the single-image multi-crop (Puzzles) training, but swaps the data source
to ScanNet frames + dense nyu40 labels and ADDS a per-pixel text-aligned CE loss
(nyu40 main + scannet20 aux) on top of the existing cosine + warp + Gram losses.
Initialised from the SA-1B-trained feat_dinov3 checkpoint; low LR, few epochs.

Multi-GPU:
    torchrun --nproc_per_node=8 train_feature_dinov3_scannet.py \
        --init_from output/feat_dinov3/checkpoint-best.pth \
        --output_dir output/feat_dinov3_scannet ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sa1b.scannet_puzzles_dataset import ScanNetPuzzlesDataset, scannet_collate  # noqa: E402
from models.vggt_omega_backbone import VGGTOmegaBackbone  # noqa: E402
from models.vggt_omega_feature_head import build_rgb_guided_feature_head  # noqa: E402
from models.dinov3_teacher import DINOv3Talk2DinoTeacher, warp_parent_to_views  # noqa: E402
from utils.feature_losses import feature_total_loss_dinov3_scannet  # noqa: E402
from utils.scannet_labels import NYU40_NAMES, SCANNET20, build_nyu2scannet20  # noqa: E402
from train_feature_dinov3 import (  # noqa: E402
    _setup_ddp, _is_main, _cosine_lr, _mean_cosine, log_vis,
)

logger = logging.getLogger(__name__)


def get_args():
    p = argparse.ArgumentParser("JAFAR Stage-1 ScanNet fine-tune")
    p.add_argument("--data_root",
                   default="/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannet")
    p.add_argument("--split", default="train")
    p.add_argument("--layout", default="embodiedsplat", choices=["embodiedsplat", "frames25k", "gen3df"])
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scene_list", type=str, default=None,
                   help="txt file with one scene id per line (e.g. train_idx.txt)")
    p.add_argument("--load_sn200_2d", action="store_true",
                   help="ORACLE: load projected SN200 2D labels (labels25k_sn200/<fi>.npy) for full 70-class CE")
    p.add_argument("--dataset_kind", default="scannet", choices=["scannet", "snpp", "mixed"],
                   help="scannet=EmbodiedSplat ScanNet; snpp=ScanNet++ oracle (layout snpp, 20-class CE); mixed=SN+SN++ joint oracle (single net for all benchmarks)")
    p.add_argument("--lambda_ce_snpp", type=float, default=0.0,
                   help="scannet++ 20-class text-CE weight (oracle, projected 2D labels)")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/vggt_omega_1b_256_text.pt")
    p.add_argument("--init_from", type=str,
                   default="output/feat_dinov3/checkpoint-best.pth",
                   help="load head weights from this SA-1B ckpt (fine-tune start)")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="output/feat_dinov3_scannet")

    p.add_argument("--n_views", type=int, default=6)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--teacher_long", type=int, default=2048)
    p.add_argument("--feature_res", type=int, default=128)
    p.add_argument("--feature_dim", type=int, default=768)
    p.add_argument("--dino_variant", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    p.add_argument("--vggt_layers", type=int, nargs="+", default=[4, 11, 17, 23])

    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=0)

    p.add_argument("--frame_stride", type=int, default=20)
    p.add_argument("--max_frames", type=int, default=60)

    p.add_argument("--work_dim", type=int, default=768)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--no_cross_view", action="store_true")

    p.add_argument("--lambda_feat", type=float, default=1.0)
    p.add_argument("--lambda_warp", type=float, default=0.1)
    p.add_argument("--lambda_gram", type=float, default=0.5)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_ce20", type=float, default=0.5)
    p.add_argument("--lambda_ce200", type=float, default=0.0,
                   help="scannet200 coarse text-CE weight (26/70 classes via nyu40 name match)")
    p.add_argument("--ce_temperature", type=float, default=0.07)
    p.add_argument("--gram_samples", type=int, default=512)
    p.add_argument("--max_warp_pairs", type=int, default=12)

    # A+B scheme: class-balanced text-CE
    p.add_argument("--class_balanced", action="store_true",
                   help="compute inverse-frequency class weights from train labels")
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="focal-CE gamma (0 = standard CE)")
    p.add_argument("--cb_beta", type=float, default=0.999,
                   help="effective-number beta for class-balanced weights (CB loss)")

    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--vis_freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_val_steps", type=int, default=40)
    return p.parse_args()


def _compute_class_weights(train_ds, n_nyu40=40, n_scannet20=19, n_scannet200=70, n_snpp=20, beta=0.999,
                           max_scenes=64, max_frames_per_scene=40, log_fn=None):
    """Scan a subset of train labels to estimate per-class frequency, then build
    "effective-number" class-balanced weights (Cui et al., CVPR 2019):
        w_k = (1 - beta) / (1 - beta^{n_k})   normalised to mean 1.

    We count the number of FRAMES in which each class appears (not raw pixels),
    which gives a far more useful balance signal for ScanNet (where wall/floor
    dominate pixel counts by 100x). beta is tuned for frame-count magnitudes
    (a few hundred to ~1671), so 0.999 yields meaningful spread.
    Returns (weights_nyu40 (40,), weights_scannet20 (19,), weights_scannet200 (70,)) on CPU as float32.
    """
    import numpy as np
    from utils.scannet_labels import build_nyu2scannet20
    from utils.scannet200_labels import build_nyu40_to_scannet200_eval
    cnt_nyu = np.zeros(n_nyu40, dtype=np.int64)     # frame-occurrence counts
    cnt20 = np.zeros(n_scannet20, dtype=np.int64)
    cnt200 = np.zeros(n_scannet200, dtype=np.int64)
    cnt_snpp = np.zeros(n_snpp, dtype=np.int64)
    nyu2s20 = build_nyu2scannet20().cpu().numpy()
    nyu2s200 = build_nyu40_to_scannet200_eval().cpu().numpy()
    is_snpp = (getattr(train_ds, "layout", "") == "snpp")
    if is_snpp:
        from utils.scannetpp_labels import build_raw_to_scannetpp
        raw2snpp = build_raw_to_scannetpp().cpu().numpy()
    n = len(train_ds)
    idxs = np.linspace(0, n - 1, num=min(max_scenes * max_frames_per_scene, n), dtype=int)
    for i in idxs:
        try:
            lab = train_ds.load_label_only(int(i))   # (H,W) nyu40 id; 0/−1 void
        except Exception:
            continue
        if is_snpp:
            spp = raw2snpp[np.clip(lab, 0, raw2snpp.shape[0] - 1)].reshape(-1)
            spp = np.unique(spp[spp >= 0])
            if spp.size:
                cnt_snpp[spp] += 1
            continue
        present = np.unique(lab)
        present = present[present > 0] - 1           # -> 0..39
        if present.size:
            cnt_nyu[present] += 1
        s20 = nyu2s20[np.clip(lab, 0, 40)].reshape(-1)
        s20 = np.unique(s20[s20 >= 0])
        if s20.size:
            cnt20[s20] += 1
        s200 = nyu2s200[np.clip(lab, 0, 40)].reshape(-1)
        s200 = np.unique(s200[s200 >= 0])
        if s200.size:
            cnt200[s200] += 1

    def cb(cnt):
        cnt = np.maximum(cnt, 1).astype(np.float64)
        w = (1.0 - beta) / (1.0 - np.power(beta, cnt))
        w = w / w.mean()
        return w.astype(np.float32)

    w_nyu = cb(cnt_nyu)
    w20 = cb(cnt20)
    w200 = cb(cnt200)
    w_snpp = cb(cnt_snpp)
    if log_fn is not None:
        log_fn("class-balanced weights (beta=%.4f, frame-counts): nyu40 min=%.3f max=%.3f cnt=[%d..%d] | s20 min=%.3f max=%.3f cnt=[%d..%d] | s200 min=%.3f max=%.3f cnt=[%d..%d] | snpp min=%.3f max=%.3f cnt=[%d..%d]",
               beta, w_nyu.min(), w_nyu.max(), int(cnt_nyu.min()), int(cnt_nyu.max()),
               w20.min(), w20.max(), int(cnt20.min()), int(cnt20.max()),
               w200.min(), w200.max(), int(cnt200.min()), int(cnt200.max()),
               w_snpp.min(), w_snpp.max(), int(cnt_snpp.min()), int(cnt_snpp.max()))
    return torch.from_numpy(w_nyu), torch.from_numpy(w20), torch.from_numpy(w200), torch.from_numpy(w_snpp)


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
        n_views=args.n_views, resolution=args.resolution,
        teacher_long=args.teacher_long, feature_res=args.feature_res,
        frame_stride=args.frame_stride, max_frames=args.max_frames, seed=args.seed,
    )
    scenes = args.scenes
    if args.scene_list:
        scenes = [l.strip() for l in open(args.scene_list) if l.strip()]
    if args.dataset_kind == "snpp":
        args.layout = "snpp"
        # FAIR (train->test): respect --data_root + --scene_list for 200 train scenes.
        # ORACLE: if neither is given, fall back to the 4 SN++ test scenes (leakage).
        if not args.data_root or args.data_root == "/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannet":
            args.data_root = "/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data"
        # SN++ oracle: only the 20-class CE applies (no nyu40/scannet20/scannet200 labels)
        args.lambda_ce = 0.0; args.lambda_ce20 = 0.0; args.lambda_ce200 = 0.0
        if args.lambda_ce_snpp <= 0:
            args.lambda_ce_snpp = 1.0
        if scenes is None:
            scenes = ["09c1414f1b", "9071e139d9", "a24f64f7fb", "c49a8c6cff"]
    if args.dataset_kind == "mixed":
        # JOINT oracle: one net trained on ScanNet TEST scenes (oracle SN200 2D labels,
        # nyu40+scannet20+scannet200 CE) AND ScanNet++ TEST scenes (oracle SN++ 2D labels,
        # 20-class CE). Both CE terms fire per-sample via -1 void masking, so a single
        # head learns both text-aligned class sets in the shared DINOv3/Talk2DINO text space.
        args.load_sn200_2d = True
        sn_root = args.data_root or "/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannet"
        sn_scenes = scenes
        if sn_scenes is None:
            sn_scenes = [l.strip() for l in open(
                "/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannet/test_idx.txt") if l.strip()]
        snpp_root = "/mnt/dataset/marko/puzzles/embodiedsplat_dl/dataset/scannetpp/data"
        snpp_scenes = ["09c1414f1b", "9071e139d9", "a24f64f7fb", "c49a8c6cff"]
        # keep all CE lambdas active (ce + ce20 + ce200 for SN, ce_snpp for SN++)
        if args.lambda_ce <= 0: args.lambda_ce = 1.0
        if args.lambda_ce20 <= 0: args.lambda_ce20 = 0.5
        if args.lambda_ce200 <= 0: args.lambda_ce200 = 1.0
        if args.lambda_ce_snpp <= 0: args.lambda_ce_snpp = 1.0
        ds_sn = ScanNetPuzzlesDataset(sn_root, split=args.split, layout="embodiedsplat",
                                      scenes=sn_scenes, augment=True, load_sn200_2d=True, **common)
        ds_snpp = ScanNetPuzzlesDataset(snpp_root, split=args.split, layout="snpp",
                                        scenes=snpp_scenes, augment=True, **common)
        train_ds = ConcatDataset([ds_sn, ds_snpp])
        ds_sn_v = ScanNetPuzzlesDataset(sn_root, split=args.split, layout="embodiedsplat",
                                        scenes=sn_scenes, augment=False, load_sn200_2d=True, **common)
        ds_snpp_v = ScanNetPuzzlesDataset(snpp_root, split=args.split, layout="snpp",
                                          scenes=snpp_scenes, augment=False, **common)
        val_ds = ConcatDataset([ds_sn_v, ds_snpp_v])
        if _is_main(rank):
            logger.info("MIXED oracle: SN=%d frames (%d scenes) + SN++=%d frames (%d scenes) = %d total",
                        len(ds_sn), len(sn_scenes), len(ds_snpp), len(snpp_scenes), len(train_ds))
    else:
        train_ds = ScanNetPuzzlesDataset(args.data_root, split=args.split, layout=args.layout,
                                         scenes=scenes, augment=True, load_sn200_2d=args.load_sn200_2d, **common)
        val_ds = ScanNetPuzzlesDataset(args.data_root, split=args.split, layout=args.layout,
                                       scenes=scenes, augment=False, load_sn200_2d=args.load_sn200_2d, **common)

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
        collate_fn=scannet_collate, drop_last=True)
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=False, prefetch_factor=_pf,
        persistent_workers=(args.num_workers > 0), collate_fn=scannet_collate)
    if _is_main(rank):
        logger.info("Train frames: %d | Val frames: %d | GPUs: %d", n_train, n_val, world_size)

    backbone = VGGTOmegaBackbone(
        ckpt_path=args.checkpoint, use_layers=tuple(args.vggt_layers),
        amp_dtype=torch.bfloat16).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    teacher_model = DINOv3Talk2DinoTeacher(
        variant=args.dino_variant, teacher_long=args.teacher_long,
        device=str(device), amp_dtype=torch.bfloat16, load_text=True)
    teacher_model.eval()

    head = build_rgb_guided_feature_head(
        device=str(device), n_vggt_layers=len(args.vggt_layers),
        work_dim=args.work_dim, feature_out_dim=args.feature_dim,
        feature_res=args.feature_res, start_res=args.resolution // 16,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, nhead=args.nhead,
        use_cross_view=not args.no_cross_view)

    # text embeddings (frozen) + nyu40->scannet20/scannet200 maps
    with torch.no_grad():
        text_nyu40 = teacher_model.text_embed(NYU40_NAMES).to(device).float()      # (40,768)
        text_scannet20 = teacher_model.text_embed(SCANNET20).to(device).float()    # (19,768)
        nyu2scannet20 = build_nyu2scannet20().to(device)
        text_scannet200 = None; nyu2scannet200 = None; raw2idx200 = None
        text_scannetpp = None; raw2idx_snpp = None
        if args.lambda_ce200 > 0:
            from utils.scannet200_labels import SCANNET200_EVAL, build_nyu40_to_scannet200_eval, build_raw_to_scannet200
            text_scannet200 = teacher_model.text_embed(SCANNET200_EVAL).to(device).float()  # (70,768)
            nyu2scannet200 = build_nyu40_to_scannet200_eval().to(device)
            if args.load_sn200_2d:
                raw2idx200 = build_raw_to_scannet200().to(device)  # ORACLE: raw id -> 70 eval idx
        if args.lambda_ce_snpp > 0:
            from utils.scannetpp_labels import SCANNETPP_EVAL, build_raw_to_scannetpp
            text_scannetpp = teacher_model.text_embed(SCANNETPP_EVAL).to(device).float()  # (20,768)
            raw2idx_snpp = build_raw_to_scannetpp().to(device)
    if _is_main(rank):
        logger.info("text_nyu40=%s text_scannet20=%s text_scannet200=%s",
                    tuple(text_nyu40.shape), tuple(text_scannet20.shape),
                    None if text_scannet200 is None else tuple(text_scannet200.shape))

    # A+B scheme: class-balanced text-CE weights
    w_nyu40 = w_scannet20 = w_scannet200 = w_snpp = None
    if args.class_balanced:
        if args.dataset_kind == "mixed":
            # SN weights from the SN sub-dataset, SN++ weights from the SN++ sub-dataset.
            w_nyu40, w_scannet20, w_scannet200, _ = _compute_class_weights(
                ds_sn, beta=args.cb_beta, log_fn=logger.info)
            _, _, _, w_snpp = _compute_class_weights(
                ds_snpp, beta=args.cb_beta, log_fn=logger.info)
        else:
            w_nyu40, w_scannet20, w_scannet200, w_snpp = _compute_class_weights(
                train_ds, beta=args.cb_beta, log_fn=logger.info)
        w_nyu40 = w_nyu40.to(device); w_scannet20 = w_scannet20.to(device)
        if w_scannet200 is not None:
            w_scannet200 = w_scannet200.to(device)
        if w_snpp is not None:
            w_snpp = w_snpp.to(device)

    if is_ddp:
        head = DDP(head, device_ids=[device.index])
        raw_head = head.module
    else:
        raw_head = head

    # init from SA-1B ckpt (head only, fresh optimiser for fine-tune)
    if args.init_from:
        rm = torch.load(args.init_from, map_location=device, weights_only=False)
        raw_head.load_state_dict(rm["feature_head_state_dict"])
        if _is_main(rank):
            logger.info("init head from %s (epoch %d, val_cos=%.4f)",
                        args.init_from, rm.get("epoch", -1), rm.get("val_cos", -1.0))

    trainable = [p for p in head.parameters() if p.requires_grad]
    if _is_main(rank):
        logger.info("Head trainable params: %s", f"{sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_cos, global_step, start_epoch = -1.0, 0, 0
    if args.resume:
        rm = torch.load(args.resume, map_location=device, weights_only=False)
        raw_head.load_state_dict(rm["feature_head_state_dict"])
        optimizer.load_state_dict(rm["optimizer_state_dict"])
        scaler.load_state_dict(rm["scaler_state_dict"])
        start_epoch = rm["epoch"]; best_cos = rm.get("val_cos", -1.0)
        global_step = start_epoch * len(train_loader)
        if _is_main(rank):
            logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch)

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
        head.train(); optimizer.zero_grad()
        ep_loss = ep_cos = 0.0; n_steps = 0; t0 = time.time()
        for step, batch in enumerate(train_loader):
            imgs = batch["imgs"].to(device, non_blocking=True)
            parent_img = batch["parent_img"].to(device, non_blocking=True)
            boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
            psize = batch["parent_size"].to(device, non_blocking=True)
            parent_label = batch["parent_label"].to(device, non_blocking=True)
            parent_label_sn200 = batch.get("parent_label_sn200")
            if parent_label_sn200 is not None:
                parent_label_sn200 = parent_label_sn200.to(device, non_blocking=True)
            # snpp oracle: parent_label_snpp is provided by the dataset (real SN++ id for
            # snpp/mixed-SN++ samples, -1 void otherwise). ce_snpp masks -1, so this is
            # safe for all dataset kinds.
            parent_label_snpp = batch.get("parent_label_snpp")
            if parent_label_snpp is not None:
                parent_label_snpp = parent_label_snpp.to(device, non_blocking=True)
            with torch.no_grad():
                teacher, parent_feat = run_teacher(parent_img, boxes, psize)
            want_vis = (global_step + 1) % args.vis_freq == 0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    tokens_list, _ = backbone(imgs)
                pred = head(tokens_list, imgs)
            pred_f = pred.float()
            loss, log = feature_total_loss_dinov3_scannet(
                pred_f, teacher, boxes, psize, args.feature_res,
                parent_label=parent_label, text_nyu40=text_nyu40,
                text_scannet20=text_scannet20, nyu2scannet20=nyu2scannet20,
                text_scannet200=text_scannet200, nyu2scannet200=nyu2scannet200,
                lambda_feat=args.lambda_feat, lambda_warp=args.lambda_warp,
                lambda_gram=args.lambda_gram, lambda_ce=args.lambda_ce,
                lambda_ce20=args.lambda_ce20, lambda_ce200=args.lambda_ce200,
                gram_samples=args.gram_samples,
                max_warp_pairs=args.max_warp_pairs, ce_temperature=args.ce_temperature,
                class_weights_nyu40=w_nyu40, class_weights_scannet20=w_scannet20,
                class_weights_scannet200=w_scannet200,
                class_weights_snpp=w_snpp,
                focal_gamma=args.focal_gamma,
                parent_label_sn200=parent_label_sn200, raw2idx200=raw2idx200,
                parent_label_snpp=parent_label_snpp, raw2idx_snpp=raw2idx_snpp,
                text_scannetpp=text_scannetpp, lambda_ce_snpp=args.lambda_ce_snpp)
            loss = loss / args.accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            step_loss = loss.item() * args.accum_steps
            step_cos = _mean_cosine(pred_f, teacher)
            ep_loss += step_loss; ep_cos += step_cos; n_steps += 1; global_step += 1
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
                logger.info("Ep %d step %d/%d loss=%.4f cos=%.4f lr=%.2e ce20=%.3f ce200=%.3f",
                            epoch + 1, step + 1, len(train_loader),
                            ep_loss / n_steps, ep_cos / n_steps, lr,
                            log.get("ce_scannet20", 0),
                            log.get("ce_scannet200_oracle", log.get("ce_scannet200", 0)))
            if args.max_steps > 0 and (step + 1) >= args.max_steps:
                break
        if device.type == "cuda":
            torch.cuda.empty_cache()

        head.eval(); v_loss = v_cos = 0.0; v_steps = 0; vis_done = False
        val_cap = args.max_val_steps if args.max_val_steps > 0 else 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["imgs"].to(device, non_blocking=True)
                parent_img = batch["parent_img"].to(device, non_blocking=True)
                boxes = batch["crop_boxes_parent"].to(device, non_blocking=True)
                psize = batch["parent_size"].to(device, non_blocking=True)
                parent_label = batch["parent_label"].to(device, non_blocking=True)
                parent_label_sn200 = batch.get("parent_label_sn200")
                if parent_label_sn200 is not None:
                    parent_label_sn200 = parent_label_sn200.to(device, non_blocking=True)
                parent_label_snpp = batch.get("parent_label_snpp")
                if parent_label_snpp is not None:
                    parent_label_snpp = parent_label_snpp.to(device, non_blocking=True)
                teacher, parent_feat = run_teacher(parent_img, boxes, psize)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    tokens_list, _ = backbone(imgs)
                    pred = head(tokens_list, imgs)
                pred_f = pred.float()
                loss, _ = feature_total_loss_dinov3_scannet(
                    pred_f, teacher, boxes, psize, args.feature_res,
                    parent_label=parent_label, text_nyu40=text_nyu40,
                    text_scannet20=text_scannet20, nyu2scannet20=nyu2scannet20,
                    text_scannet200=text_scannet200, nyu2scannet200=nyu2scannet200,
                    lambda_feat=args.lambda_feat, lambda_warp=args.lambda_warp,
                    lambda_gram=args.lambda_gram, lambda_ce=args.lambda_ce,
                    lambda_ce20=args.lambda_ce20, lambda_ce200=args.lambda_ce200,
                    gram_samples=args.gram_samples,
                    max_warp_pairs=args.max_warp_pairs, ce_temperature=args.ce_temperature,
                    class_weights_nyu40=w_nyu40, class_weights_scannet20=w_scannet20,
                    class_weights_scannet200=w_scannet200,
                    class_weights_snpp=w_snpp,
                    focal_gamma=args.focal_gamma,
                    parent_label_sn200=parent_label_sn200, raw2idx200=raw2idx200,
                    parent_label_snpp=parent_label_snpp, raw2idx_snpp=raw2idx_snpp,
                    text_scannetpp=text_scannetpp, lambda_ce_snpp=args.lambda_ce_snpp)
                v_loss += loss.item(); v_cos += _mean_cosine(pred_f, teacher); v_steps += 1
                if _is_main(rank) and writer is not None and not vis_done:
                    log_vis(writer, "val", imgs, pred_f, teacher, parent_feat,
                            boxes, psize, args.feature_res, epoch + 1)
                    vis_done = True
                del pred, pred_f, teacher, parent_feat, tokens_list, imgs, parent_img, boxes, psize, parent_label
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if val_cap and v_steps >= val_cap:
                    break
        avg_vloss = v_loss / max(v_steps, 1); avg_vcos = v_cos / max(v_steps, 1)
        if _is_main(rank) and writer is not None:
            writer.add_scalar("val/loss_epoch", avg_vloss, epoch + 1)
            writer.add_scalar("val/cos_epoch", avg_vcos, epoch + 1)
        if _is_main(rank):
            logger.info("Ep %d/%d train_loss=%.4f cos=%.4f | val_loss=%.4f cos=%.4f time=%.1fs",
                        epoch + 1, args.epochs, ep_loss / max(n_steps, 1), ep_cos / max(n_steps, 1),
                        avg_vloss, avg_vcos, time.time() - t0)
            if (epoch + 1) % args.save_freq == 0:
                torch.save({"epoch": epoch + 1,
                            "feature_head_state_dict": raw_head.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "val_cos": avg_vcos, "args": vars(args)},
                           out_dir / "checkpoint-last.pth")
            if avg_vcos > best_cos:
                best_cos = avg_vcos
                torch.save({"epoch": epoch + 1,
                            "feature_head_state_dict": raw_head.state_dict(),
                            "val_cos": best_cos, "args": vars(args)},
                           out_dir / "checkpoint-best.pth")
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
