# %% [markdown]
# # Pi3X multi-view segmentation — inference notebook
# 
# Uses `output/seg_4gpu/checkpoint-best.pth` (seg head) + Pi3X weights from `args.json`.
# 
# - **Input**: N views + prompt (folder: mask/bbox on view 0; SA-1B: dataset prompts).
# - **Output**: one mask per view (query channel 0), same as `inference_segmentation.py`.
# - **GT**: SA-1B shows `gt_consistent_masks`; synthetic uses the same filled region on all views; **folder mode has no GT** here (inference-only).
# 
# **Plot titles are English.** Six rows: RGB | prompt | prompt bbox | **GT heatmap** | **RGB+GT (blue)** | pred (green).
# 
# Configure `IMAGES_DIR` / `PROMPT_MASK` in the config cell; if unset and `sa1b_root` exists, one SA-1B sample is used.
# 

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

for pth in (PROJECT_ROOT / "src", PROJECT_ROOT / "repo" / "Pi3"):
    s = str(pth.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)

ARGS_PATH = PROJECT_ROOT / "output" / "seg_4gpu" / "args.json"
with open(ARGS_PATH) as f:
    CFG = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DEVICE:", DEVICE)
print(
    "Train cfg: n_views=%d, resolution=%d, query_dim=%d, layers=%d, nhead=%d"
    % (
        CFG["n_views"],
        CFG["resolution"],
        CFG["query_dim"],
        CFG["num_decoder_layers"],
        CFG["nhead"],
    ),
)



# %%
from models.pi3x_segmentation import build_pi3x_segmentation

pi3_rel = CFG.get("checkpoint") or "checkpoints/model.safetensors"
pi3_ckpt = PROJECT_ROOT / pi3_rel
pi3_arg = str(pi3_ckpt) if pi3_ckpt.is_file() else None
if pi3_arg is None:
    print("Pi3X weights not found locally; will try HuggingFace yyfz233/Pi3X")

model = build_pi3x_segmentation(
    checkpoint_path=pi3_arg,
    device=str(DEVICE),
    query_dim=CFG["query_dim"],
    num_decoder_layers=CFG["num_decoder_layers"],
    nhead=CFG["nhead"],
)

seg_ckpt_path = PROJECT_ROOT / "output" / "seg_4gpu" / "checkpoint-best.pth"
ckpt = torch.load(seg_ckpt_path, map_location=DEVICE, weights_only=True)
model.seg_head.load_state_dict(ckpt["seg_head_state_dict"])
print("Loaded seg checkpoint:", seg_ckpt_path)
print("checkpoint-best val mIoU:", ckpt.get("val_iou", "N/A"))
model.eval()


# %%
# ========= 请按需修改 =========
# 方式 A：多视角图片目录（按文件名排序），prompt 只在「第 0 张」上
IMAGES_DIR: Path | None = None  # 例如 PROJECT_ROOT / "examples/room/rgb"
PROMPT_MASK: Path | None = None  # 二值 mask 图，对应 view0（灰度 >127 为前景）
# 若不用 mask，可改用 bbox（在 resolution 尺度下的像素坐标）：
PROMPT_BBOX: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)

# 方式 B：不填目录时，若 SA-1B 路径存在，则用数据集第 SAMPLE_IDX 条样本（含真实多视角与 prompt）
SAMPLE_IDX = 0
THRESHOLD = 0.5
# =============================

resolution = int(CFG["resolution"])
sa1b_ok = Path(CFG["sa1b_root"]).is_dir()

if IMAGES_DIR is not None:
    mode = "folder"
elif sa1b_ok:
    mode = "sa1b"
else:
    mode = "synthetic"

print("数据模式:", mode)
if mode == "folder":
    if PROMPT_MASK is None and PROMPT_BBOX is None:
        raise ValueError("folder 模式下请设置 PROMPT_MASK 或 PROMPT_BBOX")

# %%
import importlib.util


@torch.no_grad()
def forward_masks(
    model: torch.nn.Module,
    imgs_b_n3hw: torch.Tensor,
    sam_masks: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (logits, 二值 mask)。mask 形状 (N, M, H, W) 与 logits 一致。"""
    imgs_b_n3hw = imgs_b_n3hw.to(device)
    sam_masks = sam_masks.to(device)
    model.eval()
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        out = model(imgs_b_n3hw, sam_masks)
    logits = out["segmentation_logits"].float().cpu()
    prob = logits[0].sigmoid()
    binary = (prob > threshold).to(torch.float32)
    return logits[0], binary


# ---- 准备输入 ----
originals: list | None = None

gt_vis = None  # (N, M_gt, H, W) CPU float; SA-1B / synthetic only

if mode == "folder":
    _spec = importlib.util.spec_from_file_location(
        "inference_segmentation_nb", PROJECT_ROOT / "inference_segmentation.py"
    )
    _inf = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_inf)
    load_images = _inf.load_images
    load_prompt_mask = _inf.load_prompt_mask
    create_bbox_mask = _inf.create_bbox_mask
    infer = _inf.infer

    imgs, originals = load_images(str(IMAGES_DIR), resolution=resolution)
    if PROMPT_MASK is not None:
        prompt = load_prompt_mask(str(PROMPT_MASK), resolution)
    else:
        assert PROMPT_BBOX is not None
        prompt = create_bbox_mask(PROMPT_BBOX, resolution, resolution, resolution)
    pred_uint8 = infer(model, imgs, prompt, DEVICE, threshold=THRESHOLD)
    N = imgs.shape[1]
    logits_v = None
    mask_float = torch.from_numpy(pred_uint8.astype(np.float32) / 255.0).view(N, 1, resolution, resolution)
elif mode == "sa1b":
    from sa1b.sa1b_puzzles_dataset import SA1BPuzzlesDataset

    shard_names = CFG.get("shard_names")
    if isinstance(shard_names, list) and len(shard_names) == 0:
        shard_names = None
    ds = SA1BPuzzlesDataset(
        sa1b_root=CFG["sa1b_root"],
        shard_names=shard_names,
        n_views=CFG["n_views"],
        resolution=resolution,
        max_instances=CFG["max_instances"],
        max_prompts=CFG["max_prompts"],
        seed=CFG["seed"],
        augment=False,
    )
    sample = ds[SAMPLE_IDX]
    imgs = sample["imgs"].unsqueeze(0)
    sam = sample["prompt_masks"].unsqueeze(0)
    logits_v, mask_float = forward_masks(model, imgs, sam, DEVICE, THRESHOLD)
    N = imgs.shape[1]
    pred_uint8 = (mask_float[:, 0].numpy() * 255).astype(np.uint8)
    gt_vis = sample["gt_consistent_masks"].detach().cpu().float()
else:
    # 极简烟雾测试：同一张随机图复制 n_views，中心块 prompt
    N = int(CFG["n_views"])
    g = torch.Generator().manual_seed(int(CFG["seed"]))
    one = torch.rand(1, 3, resolution, resolution, generator=g)
    imgs = one.repeat(1, N, 1, 1, 1)
    pm = torch.zeros(1, 1, 1, resolution, resolution)
    margin = resolution // 8  # large block so prompt row is obvious
    pm[0, 0, 0, margin : resolution - margin, margin : resolution - margin] = 1.0
    sam = torch.zeros(1, N, 1, resolution, resolution)
    sam[:, 0] = pm[:, 0]
    logits_v, mask_float = forward_masks(model, imgs, sam, DEVICE, THRESHOLD)
    pred_uint8 = (mask_float[:, 0].numpy() * 255).astype(np.uint8)
    originals = None
    # Pseudo-GT: same mask on all views (one instance), matches training layout
    gt_vis = pm[0, 0].unsqueeze(0).expand(N, -1, -1, -1).clone()

# Same prompt tensor as fed to the model: (N, M, H, W) on CPU for plotting
_, N_vis, _, H, W = imgs.shape
if mode == "folder":
    sam_vis = torch.zeros(N_vis, prompt.shape[2], H, W)
    sam_vis[0] = prompt[0, 0].cpu().float()
else:
    sam_vis = sam[0].detach().cpu().float()

sam_vis = torch.nan_to_num(sam_vis, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
_pm = sam_vis.reshape(N_vis, -1).sum(dim=1)
print("prompt_pixel_sum per view:", [float(x) for x in _pm.tolist()])
if float(_pm.sum()) < 1.0:
    print("WARN: almost no prompt — try SAMPLE_IDX+=1 or folder+IMAGES_DIR+PROMPT_MASK.")

if gt_vis is not None:
    gt_vis = torch.nan_to_num(gt_vis, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    _g = gt_vis.reshape(N_vis, -1).sum(dim=1)
    print("gt_pixel_sum per view:", [float(x) for x in _g.tolist()])
else:
    print("gt_vis: None (folder mode has no GT in this notebook)")

print("N views:", N, "| pred_uint8 shape:", pred_uint8.shape)


# %%
# English titles only (avoid CJK / mojibake in matplotlib).
from matplotlib.patches import Rectangle

imgs_cpu = imgs[0].detach().cpu().float().clamp(0, 1)
M_show = min(4, mask_float.shape[1]) if mask_float.dim() == 4 else 1

PROMPT_VIZ_THRESH = 0.01


def _mask_union_2d(ten):
    """(M,H,W) tensor -> (H,W) max over channels."""
    x = ten.detach().cpu().float().numpy()
    if x.ndim != 3:
        return x
    return np.max(x, axis=0)


def _bbox_from_mask(p2d, thr=PROMPT_VIZ_THRESH):
    ys, xs = np.where(p2d > thr)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


n_rows = 6
fig, axes = plt.subplots(n_rows, N, figsize=(min(3.2 * N, 20), 2.9 * n_rows))
if N == 1:
    axes = axes.reshape(n_rows, 1)

pred_rgb = np.array([0.15, 0.95, 0.25])
gt_rgb = np.array([0.25, 0.55, 1.0])  # blue GT overlay

for v in range(N):
    im = imgs_cpu[v].permute(1, 2, 0).numpy()
    p_prompt = _mask_union_2d(sam_vis[v])
    has_prompt = p_prompt.max() > PROMPT_VIZ_THRESH

    axes[0, v].imshow(im)
    axes[0, v].set_title("v%d RGB" % v)
    axes[0, v].axis("off")

    axes[1, v].imshow(p_prompt, cmap="magma", vmin=0, vmax=1)
    if not has_prompt:
        axes[1, v].text(
            0.5,
            0.5,
            "NO PROMPT",
            transform=axes[1, v].transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="cyan",
            fontweight="bold",
        )
    axes[1, v].set_title("v%d prompt" % v)
    axes[1, v].axis("off")

    axes[2, v].imshow(im)
    if has_prompt:
        lvl = float(np.clip(p_prompt.max() * 0.5, 0.08, 0.99))
        axes[2, v].contour(
            p_prompt,
            levels=[lvl],
            colors="red",
            linewidths=3.0,
            origin="upper",
        )
        bb = _bbox_from_mask(p_prompt)
        if bb is not None:
            x0, y0, x1, y1 = bb
            rect = Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0 + 1,
                y1 - y0 + 1,
                fill=False,
                edgecolor="yellow",
                linewidth=4.0,
            )
            axes[2, v].add_patch(rect)
    axes[2, v].set_title("v%d prompt bbox" % v)
    axes[2, v].axis("off")

    if gt_vis is not None:
        p_gt = _mask_union_2d(gt_vis[v])
        has_gt = p_gt.max() > PROMPT_VIZ_THRESH
        axes[3, v].imshow(p_gt, cmap="viridis", vmin=0, vmax=1)
        if not has_gt:
            axes[3, v].text(
                0.5,
                0.5,
                "EMPTY GT",
                transform=axes[3, v].transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="bold",
            )
        axes[3, v].set_title("v%d GT mask" % v)
        o_gt = im * (1 - 0.5 * p_gt[..., None]) + p_gt[..., None] * gt_rgb * 0.5
        o_gt = np.clip(o_gt, 0, 1)
        axes[4, v].imshow(o_gt)
        axes[4, v].set_title("v%d RGB+GT" % v)
    else:
        axes[3, v].imshow(np.zeros_like(im[:, :, 0]), cmap="gray", vmin=0, vmax=1)
        axes[3, v].text(
            0.5,
            0.5,
            "NO GT\n(folder)",
            transform=axes[3, v].transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
        axes[3, v].set_title("v%d GT (n/a)" % v)
        axes[4, v].imshow(im)
        axes[4, v].set_title("v%d no GT overlay" % v)
    axes[3, v].axis("off")
    axes[4, v].axis("off")

    m = mask_float[v, 0].numpy() if mask_float.dim() == 4 else mask_float[v].numpy()
    o_pred = im * (1 - 0.55 * m[..., None]) + m[..., None] * pred_rgb * 0.55
    o_pred = np.clip(o_pred, 0, 1)
    axes[5, v].imshow(o_pred)
    axes[5, v].set_title("v%d prediction" % v)
    axes[5, v].axis("off")

plt.suptitle(
    "RGB | prompt | prompt bbox | GT heatmap | RGB+GT (blue) | pred (green)",
    fontsize=10,
)
plt.tight_layout()
plt.show()

if M_show > 1:
    fig2, ax2 = plt.subplots(1, M_show, figsize=(3 * M_show, 3))
    if M_show == 1:
        ax2 = [ax2]
    for mi in range(M_show):
        if logits_v is not None:
            disp = logits_v[0, mi].sigmoid().numpy()
            ttl = "v0 q%d prob" % mi
        else:
            disp = mask_float[0, mi].numpy()
            ttl = "v0 q%d" % mi
        ax2[mi].imshow(disp, cmap="magma", vmin=0, vmax=1)
        ax2[mi].set_title(ttl)
        ax2[mi].axis("off")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # 验证：cross-attention 是否已经把 view 0 的 query 灌进了所有视角
#
# **要验证的两个论断**：
#
# 1. `MultiViewQueryDecoder` 输出的 `decoded_queries`（256-D）里，
#    view 1..N-1 的 query 几乎完全相同（无位置编码 → 占位 query 经同一套 attention 后塌缩到同一向量）。
# 2. 把所有视角的 64-D query 替换成 view 0 的 query 重做 einsum，
#    得到的 mask 和原始 mask 应当**几乎一致**（说明 query 已经是"全局物体表示"，
#    各 view 的 mask 差异主要来自 `pixel_feats` 的差异，而非 query 本身的差异）。

# %%
@torch.no_grad()
def _capture_intermediates(model, imgs_b, sam_b, device):
    """Run model.forward once with hooks; return (pixel_feats, decoded_queries_256, logits)."""
    cap: dict = {}

    def hook_pix(_mod, _inp, out):
        cap["pix"] = out.detach()

    def hook_mvd(_mod, _inp, out):
        cap["mvd"] = out.detach()

    h1 = model.seg_head.pixel_decoder.register_forward_hook(hook_pix)
    h2 = model.seg_head.mv_decoder.register_forward_hook(hook_mvd)
    try:
        model.eval()
        if device.type == "cuda":
            cap_dev = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if cap_dev[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model(imgs_b.to(device), sam_b.to(device))
        logits_orig = out["segmentation_logits"].float().cpu()
    finally:
        h1.remove()
        h2.remove()
    return cap["pix"], cap["mvd"], logits_orig


# 复用之前 cell 准备好的 imgs / sam_vis（与模型实际看到的输入一致）
sam_in = sam_vis.unsqueeze(0).contiguous()  # (1, N, M, H, W) on CPU
B = 1
N = N_vis
M_sam = sam_in.shape[2]

pix_out, mvd_out, logits_orig = _capture_intermediates(model, imgs, sam_in, DEVICE)
print("pixel_decoder output (B*N, 64, H, W):", tuple(pix_out.shape))
print("mv_decoder   output (B, N*M, 256)  :", tuple(mvd_out.shape))
print("orig logits        (B, N, M, H, W) :", tuple(logits_orig.shape))


# ---------- (1) 256-D decoded query 之间的 L2 距离矩阵 ----------
mvd_b_n_m_d = mvd_out.float().reshape(B, N, M_sam, -1).cpu()  # (B, N, M, 256)
print("\ndecoded_queries reshape -> (B, N, M, 256):", tuple(mvd_b_n_m_d.shape))

INSTANCE = 0  # 只看第 0 个 instance（你的场景 M_sam=1，就是它）
q = mvd_b_n_m_d[0, :, INSTANCE, :]  # (N, 256)
print("\n‖query[v_i] - query[v_j]‖_2  on instance #%d  (256-dim):" % INSTANCE)
print("       " + "  ".join("v%-6d" % j for j in range(N)))
for i in range(N):
    row = " ".join("%7.4f" % (q[i] - q[j]).norm().item() for j in range(N))
    print("  v%d  %s" % (i, row))

# 期望:
#   - v0 vs v1..vN-1: 显著大于 0（v0 含 prompt 信息）
#   - v1..vN-1 之间: ≈ 0（占位 query 经同一套 attention 之后输出相同）


# ---------- (2) 把所有视角的 query 替换成 view 0, 重算 einsum 比较 ----------
# mask_proj 是 nn.Linear(256 -> 64)
mask_proj = model.seg_head.mask_proj
true_q64 = mask_proj(mvd_out.float()).reshape(B, N, M_sam, -1).cpu()  # (B, N, M, 64)

fake_q64 = true_q64.clone()
fake_q64[:, 1:, :, :] = true_q64[:, 0:1, :, :]  # view 1..N-1 全部用 view 0 的 64-D query

H, W = sam_in.shape[-2:]
pix_b_n_c_h_w = pix_out.float().reshape(B, N, -1, H, W).cpu()  # (B, N, 64, H, W)
print("\npixel_feats reshape -> (B, N, 64, H, W):", tuple(pix_b_n_c_h_w.shape))

logits_true = torch.einsum("bnmd,bndhw->bnmhw", true_q64, pix_b_n_c_h_w)
logits_fake = torch.einsum("bnmd,bndhw->bnmhw", fake_q64, pix_b_n_c_h_w)

diff = (logits_true - logits_fake).abs()
print("\nlogits_true range : [%.3f, %.3f]" % (logits_true.min().item(), logits_true.max().item()))
print("logits_fake range : [%.3f, %.3f]" % (logits_fake.min().item(), logits_fake.max().item()))
print("|true - fake|      mean=%.6f  max=%.6f" % (diff.mean().item(), diff.max().item()))


def _iou(a, b, thr=0.0):
    A = (a > thr).float()
    Bm = (b > thr).float()
    inter = (A * Bm).sum().item()
    union = ((A + Bm) > 0).float().sum().clamp(min=1).item()
    return inter / union


per_view_iou = [_iou(logits_true[0, n, INSTANCE], logits_fake[0, n, INSTANCE]) for n in range(N)]
print("\nper-view mask IoU(orig vs broadcast-v0-query):", ["%.4f" % x for x in per_view_iou])
# 期望: 每个视角 IoU ≈ 1.0（说明 view 1..N-1 的 query 内容已等价于 view 0 propagated 后的 query）

# Sanity check: 自己重算的 logits_true 与模型原始输出 logits_orig 应当几乎相同
# (差异主要来自 autocast bf16 vs 这里的 fp32, 一般 < 1e-2)
sanity = (logits_true - logits_orig).abs()
print("\nsanity |recon - orig| mean=%.6f  max=%.6f" % (sanity.mean().item(), sanity.max().item()))


# %%
# 把 (1) 中的距离矩阵和 (2) 中的 logits 差异画出来, 直观感受
fig_v, axv = plt.subplots(1, 2, figsize=(11, 4.2))

# --- 左: query 距离矩阵 (热力图)
dist_mat = torch.zeros(N, N)
for i in range(N):
    for j in range(N):
        dist_mat[i, j] = (q[i] - q[j]).norm()
im0 = axv[0].imshow(dist_mat.numpy(), cmap="viridis")
axv[0].set_title("‖decoded_query[v_i] - decoded_query[v_j]‖_2  (256-D)")
axv[0].set_xticks(range(N))
axv[0].set_yticks(range(N))
axv[0].set_xticklabels(["v%d" % i for i in range(N)])
axv[0].set_yticklabels(["v%d" % i for i in range(N)])
for i in range(N):
    for j in range(N):
        axv[0].text(j, i, "%.2f" % dist_mat[i, j].item(),
                    ha="center", va="center",
                    color="white" if dist_mat[i, j] < dist_mat.max() * 0.6 else "black",
                    fontsize=9)
fig_v.colorbar(im0, ax=axv[0], fraction=0.046, pad=0.04)

# --- 右: 每个视角 mask IoU(orig vs broadcast-v0-query)
axv[1].bar(range(N), per_view_iou, color="#4caf50")
axv[1].set_ylim(0, 1.05)
axv[1].set_xticks(range(N))
axv[1].set_xticklabels(["v%d" % i for i in range(N)])
axv[1].set_ylabel("IoU(orig mask, broadcast-v0-query mask)")
axv[1].set_title("Mask 一致性: 用 v0 的 query 替换所有 view 后的 IoU")
axv[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
for i, v in enumerate(per_view_iou):
    axv[1].text(i, v + 0.02, "%.3f" % v, ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()


# %%



