"""Picture-in-picture compositing for the result video.

Layout: the follow-camera render fills the frame, with two insets stacked down
the right edge -- bird's-eye view on top, the input video below. Insets get
rounded corners, a feathered (slightly transparent) edge, a soft drop shadow and
a caption chip, so they read as floating panels rather than pasted rectangles.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Supersampling factor for drawing the rounded-rect alpha mask. Drawing at 4x
# and downscaling is what gives the corners clean anti-aliased edges.
_SS = 4


@dataclass
class InsetStyle:
    """Geometry and look of one floating panel."""

    width_frac: float = 0.26        # inset width as a fraction of canvas width
    margin_frac: float = 0.018      # gap to the canvas edge
    gap_frac: float = 0.016         # vertical gap between the two insets
    radius_frac: float = 0.030      # corner radius, relative to inset width
    edge_alpha: float = 0.72        # opacity at the very edge of the panel
    feather_px: int = 6             # width of the edge fade
    border_px: int = 2
    border_rgb: Tuple[int, int, int] = (235, 238, 245)
    border_alpha: float = 0.55
    shadow_px: int = 18
    shadow_alpha: float = 0.45


def _font(size: int):
    from PIL import ImageFont
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _rounded_alpha(h: int, w: int, radius: int, feather: int, edge_alpha: float) -> np.ndarray:
    """Anti-aliased rounded-rectangle alpha in [0,1], fading toward the edge."""
    big = np.zeros((h * _SS, w * _SS), np.uint8)
    r = max(1, radius * _SS)
    cv2.rectangle(big, (r, 0), (w * _SS - r, h * _SS), 255, -1)
    cv2.rectangle(big, (0, r), (w * _SS, h * _SS - r), 255, -1)
    for cx, cy in ((r, r), (w * _SS - r, r), (r, h * _SS - r), (w * _SS - r, h * _SS - r)):
        cv2.circle(big, (cx, cy), r, 255, -1)
    alpha = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    if feather > 0:
        # Distance from the panel edge, in pixels, capped at the feather width.
        inside = (alpha > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        ramp = np.clip(dist / float(feather), 0.0, 1.0)
        alpha = alpha * (edge_alpha + (1.0 - edge_alpha) * ramp)
    else:
        alpha = alpha * edge_alpha
    return alpha


def _fit_cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize + centre-crop so ``img`` exactly fills ``w x h`` without distortion."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


def _paste_panel(canvas: np.ndarray, panel: np.ndarray, x: int, y: int,
                 style: InsetStyle) -> None:
    """Alpha-composite one rounded panel (with shadow and border) in place."""
    h, w = panel.shape[:2]
    radius = max(4, int(round(w * style.radius_frac)))
    alpha = _rounded_alpha(h, w, radius, style.feather_px, style.edge_alpha)

    # Drop shadow: blur the panel silhouette and darken the canvas under it.
    if style.shadow_px > 0:
        pad = style.shadow_px * 2
        sh = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        sh[pad:pad + h, pad:pad + w] = (alpha > 0.05).astype(np.float32)
        ksz = style.shadow_px * 2 + 1
        sh = cv2.GaussianBlur(sh, (ksz, ksz), style.shadow_px * 0.55)
        sy0, sx0 = y - pad + int(style.shadow_px * 0.35), x - pad
        ry0, rx0 = max(0, sy0), max(0, sx0)
        ry1 = min(canvas.shape[0], sy0 + sh.shape[0])
        rx1 = min(canvas.shape[1], sx0 + sh.shape[1])
        if ry1 > ry0 and rx1 > rx0:
            s = sh[ry0 - sy0:ry1 - sy0, rx0 - sx0:rx1 - sx0] * style.shadow_alpha
            region = canvas[ry0:ry1, rx0:rx1].astype(np.float32)
            canvas[ry0:ry1, rx0:rx1] = (region * (1.0 - s[..., None])).astype(np.uint8)

    # Border stroke, drawn into the panel before compositing so the rounded
    # alpha clips it exactly.
    if style.border_px > 0:
        stroke = np.zeros((h * _SS, w * _SS), np.uint8)
        r = max(1, radius * _SS)
        _rounded_stroke(stroke, w * _SS, h * _SS, r, style.border_px * _SS)
        sa = (cv2.resize(stroke, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
              / 255.0 * style.border_alpha)
        col = np.array(style.border_rgb, np.float32)[::-1]      # RGB -> BGR
        panel = (panel.astype(np.float32) * (1 - sa[..., None]) + col * sa[..., None])
        panel = panel.astype(np.uint8)

    region = canvas[y:y + h, x:x + w].astype(np.float32)
    a = alpha[..., None]
    canvas[y:y + h, x:x + w] = (region * (1 - a) + panel.astype(np.float32) * a).astype(np.uint8)


def _rounded_stroke(img: np.ndarray, w: int, h: int, r: int, thickness: int) -> None:
    cv2.line(img, (r, thickness // 2), (w - r, thickness // 2), 255, thickness)
    cv2.line(img, (r, h - thickness // 2), (w - r, h - thickness // 2), 255, thickness)
    cv2.line(img, (thickness // 2, r), (thickness // 2, h - r), 255, thickness)
    cv2.line(img, (w - thickness // 2, r), (w - thickness // 2, h - r), 255, thickness)
    for cx, cy, a0, a1 in ((r, r, 180, 270), (w - r, r, 270, 360),
                            (w - r, h - r, 0, 90), (r, h - r, 90, 180)):
        cv2.ellipse(img, (cx, cy), (r - thickness // 2, r - thickness // 2), 0, a0, a1, 255, thickness)


def _draw_caption(canvas: np.ndarray, text: str, x: int, y: int, font_px: int,
                  anchor: str = "tl", pad: int = 10, bg_alpha: float = 0.55) -> None:
    """Draw a rounded translucent chip with white text at (x, y)."""
    from PIL import Image, ImageDraw

    font = _font(font_px)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    cw, ch = tw + 2 * pad + 4, th + 2 * pad
    if anchor == "tr":
        x = x - cw
    if x < 0 or y < 0 or y + ch > canvas.shape[0] or x + cw > canvas.shape[1]:
        x = int(np.clip(x, 0, canvas.shape[1] - cw))
        y = int(np.clip(y, 0, canvas.shape[0] - ch))

    chip = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(chip)
    d.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=ch // 2,
                        fill=(12, 14, 20, int(255 * bg_alpha)))
    d.text((pad + 2 - l, pad - t), text, font=font, fill=(245, 247, 252, 255))
    chip_np = np.array(chip)
    a = chip_np[..., 3:4].astype(np.float32) / 255.0
    rgb = chip_np[..., :3][..., ::-1].astype(np.float32)          # RGBA -> BGR
    region = canvas[y:y + ch, x:x + cw].astype(np.float32)
    canvas[y:y + ch, x:x + cw] = (region * (1 - a) + rgb * a).astype(np.uint8)


class _VideoFrames:
    """Random-ish sequential access to a video's frames, with graceful EOF."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last: Optional[np.ndarray] = None

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if ok:
            self.last = frame
        return self.last

    def release(self):
        self.cap.release()


def compose_pip_video(follow_video: Path, birdeye_video: Path,
                      input_frames: np.ndarray, out_path: Path, fps: float = 10.0,
                      labels: Sequence[str] = ("Follow camera",
                                               "Bird's-eye view", "Input video"),
                      style: Optional[InsetStyle] = None,
                      width: Optional[int] = None, log=print) -> Path:
    """Composite follow + BEV + input into one video.

    Args:
        follow_video: main view, defines the canvas size unless ``width`` is set.
        birdeye_video: top-right inset.
        input_frames: (S, H, W, 3) uint8 RGB, the bottom-right inset.
    """
    style = style or InsetStyle()
    main = _VideoFrames(follow_video)
    bev = _VideoFrames(birdeye_video)

    first = main.read()
    if first is None:
        raise RuntimeError(f"no frames in {follow_video}")
    Hc, Wc = first.shape[:2]
    if width and width != Wc:
        Hc = int(round(Hc * width / Wc)); Wc = width
    main.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    iw = int(round(Wc * style.width_frac))
    margin = int(round(Wc * style.margin_frac))
    gap = int(round(Wc * style.gap_frac))
    # Each inset keeps its own source aspect ratio, so nothing looks stretched.
    bev_first = bev.read()
    bev_ar = (bev_first.shape[0] / bev_first.shape[1]) if bev_first is not None else 9 / 16
    bev.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    in_ar = input_frames.shape[1] / input_frames.shape[2]
    ih_bev = int(round(iw * bev_ar))
    ih_in = int(round(iw * in_ar))

    x_in = Wc - margin - iw
    y_bev = margin
    y_in = y_bev + ih_bev + gap
    # If the stack would overflow, shrink both insets to fit the canvas height.
    overflow = (y_in + ih_in + margin) - Hc
    if overflow > 0:
        shrink = (Hc - 2 * margin - gap) / float(ih_bev + ih_in)
        iw = int(round(iw * shrink)); ih_bev = int(round(ih_bev * shrink))
        ih_in = int(round(ih_in * shrink))
        x_in = Wc - margin - iw
        y_in = y_bev + ih_bev + gap

    font_px = max(14, int(round(Wc * 0.0135)))
    n_frames = max(main.count, len(input_frames))
    tmp_dir = Path(tempfile.mkdtemp(prefix="pip_"))
    log(f"[compose] {Wc}x{Hc}, {n_frames} frames, inset {iw}x{ih_bev}/{iw}x{ih_in}")

    try:
        for i in range(n_frames):
            f = main.read()
            if f is None:
                break
            canvas = cv2.resize(f, (Wc, Hc), interpolation=cv2.INTER_AREA) \
                if f.shape[:2] != (Hc, Wc) else f.copy()

            bframe = bev.read()
            if bframe is not None:
                _paste_panel(canvas, _fit_cover(bframe, iw, ih_bev), x_in, y_bev, style)

            src_i = min(i * len(input_frames) // max(n_frames, 1), len(input_frames) - 1)
            rgb = cv2.cvtColor(input_frames[src_i], cv2.COLOR_RGB2BGR)
            _paste_panel(canvas, _fit_cover(rgb, iw, ih_in), x_in, y_in, style)

            _draw_caption(canvas, labels[0], margin, margin, font_px)
            _draw_caption(canvas, labels[1], x_in + 12, y_bev + 12, int(font_px * 0.85))
            _draw_caption(canvas, labels[2], x_in + 12, y_in + 12, int(font_px * 0.85))

            cv2.imwrite(str(tmp_dir / f"{i:06d}.png"), canvas)

        _encode(tmp_dir, out_path, fps, log=log)
    finally:
        main.release(); bev.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    log(f"[compose] wrote {out_path}")
    return out_path


def _encode(frame_dir: Path, out_path: Path, fps: float, log=print) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
               "-i", str(frame_dir / "%06d.png"), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return
        log(f"[compose] ffmpeg failed ({r.stderr.strip()[:200]}), falling back to OpenCV")
    files = sorted(frame_dir.glob("*.png"))
    if not files:
        raise RuntimeError("no composited frames to encode")
    h, w = cv2.imread(str(files[0])).shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in files:
        vw.write(cv2.imread(str(f)))
    vw.release()
