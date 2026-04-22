"""Per-crop data augmentation for multi-view segmentation training.

Three augmentation categories with distinct mask-handling strategies:

  1. Geometric  — image, gt_masks, prompt_mask all transformed synchronously
  2. Pixel      — only image modified; masks untouched
  3. Occlusion  — image filled; gt_masks untouched (learn completion);
                  prompt_mask erased in occluded region (simulate partial SAM)
"""

from __future__ import annotations

import cv2
import numpy as np


class CropAugmentor:
    """Augment a single crop (image + gt_masks + prompt_mask).

    Args:
        geo_prob:   overall probability of applying any geometric transform
        pixel_prob: overall probability of applying any pixel transform
        occ_prob:   overall probability of applying occlusion augmentation
    """

    def __init__(
        self,
        geo_prob: float = 0.5,
        pixel_prob: float = 0.8,
        occ_prob: float = 0.4,
    ) -> None:
        self.geo_prob = geo_prob
        self.pixel_prob = pixel_prob
        self.occ_prob = occ_prob

    def __call__(
        self,
        image: np.ndarray,
        gt_masks: np.ndarray,
        prompt_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            image:       (H, W, 3) uint8
            gt_masks:    (M_gt, H, W) float32
            prompt_mask: (M_prompt, H, W) float32
            rng:         numpy Generator for reproducibility

        Returns:
            (image, gt_masks, prompt_mask) — augmented copies
        """
        image = image.copy()
        gt_masks = gt_masks.copy()
        prompt_mask = prompt_mask.copy()

        if rng.random() < self.geo_prob:
            image, gt_masks, prompt_mask = self._apply_geometric(
                image, gt_masks, prompt_mask, rng
            )

        if rng.random() < self.pixel_prob:
            image = self._apply_pixel(image, rng)

        if rng.random() < self.occ_prob:
            image, gt_masks, prompt_mask = self._apply_occlusion(
                image, gt_masks, prompt_mask, rng
            )

        return image, gt_masks, prompt_mask

    # ------------------------------------------------------------------
    # A. Geometric transforms  (image + all masks synchronously)
    # ------------------------------------------------------------------

    @staticmethod
    def _warp_masks(masks: np.ndarray, M: np.ndarray, W: int, H: int) -> np.ndarray:
        for i in range(masks.shape[0]):
            masks[i] = cv2.warpAffine(
                masks[i], M, (W, H),
                flags=cv2.INTER_NEAREST, borderValue=0,
            )
        return masks

    @staticmethod
    def _warp_masks_perspective(
        masks: np.ndarray, M: np.ndarray, W: int, H: int
    ) -> np.ndarray:
        for i in range(masks.shape[0]):
            masks[i] = cv2.warpPerspective(
                masks[i], M, (W, H),
                flags=cv2.INTER_NEAREST, borderValue=0,
            )
        return masks

    def _apply_geometric(
        self,
        image: np.ndarray,
        gt_masks: np.ndarray,
        prompt_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = image.shape[:2]

        if rng.random() < 0.5:
            image = np.fliplr(image).copy()
            gt_masks = np.flip(gt_masks, axis=-1).copy()
            prompt_mask = np.flip(prompt_mask, axis=-1).copy()

        if rng.random() < 0.2:
            image = np.flipud(image).copy()
            gt_masks = np.flip(gt_masks, axis=-2).copy()
            prompt_mask = np.flip(prompt_mask, axis=-2).copy()

        if rng.random() < 0.4:
            angle = float(rng.uniform(-15, 15))
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (W, H), borderValue=0)
            gt_masks = self._warp_masks(gt_masks, M, W, H)
            prompt_mask = self._warp_masks(prompt_mask, M, W, H)

        if rng.random() < 0.3:
            tx = float(rng.uniform(-0.1, 0.1)) * W
            ty = float(rng.uniform(-0.1, 0.1)) * H
            sx = float(rng.uniform(0.8, 1.2))
            sy = float(rng.uniform(0.8, 1.2))
            shear_x = float(rng.uniform(-0.17, 0.17))
            cos_a, sin_a = 1.0, 0.0
            M = np.float32([
                [sx * cos_a + shear_x * sin_a,
                 sx * (-sin_a) + shear_x * cos_a, tx],
                [sy * sin_a, sy * cos_a, ty],
            ])
            image = cv2.warpAffine(image, M, (W, H), borderValue=0)
            gt_masks = self._warp_masks(gt_masks, M, W, H)
            prompt_mask = self._warp_masks(prompt_mask, M, W, H)

        if rng.random() < 0.3:
            d = float(rng.uniform(0.0, 0.2))
            pts_src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            pts_dst = pts_src + rng.uniform(-d * min(W, H), d * min(W, H),
                                            size=(4, 2)).astype(np.float32)
            M_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)
            image = cv2.warpPerspective(image, M_persp, (W, H), borderValue=0)
            gt_masks = self._warp_masks_perspective(gt_masks, M_persp, W, H)
            prompt_mask = self._warp_masks_perspective(prompt_mask, M_persp, W, H)

        if rng.random() < 0.4:
            scale = float(rng.uniform(0.75, 1.0))
            cw, ch = int(W * scale), int(H * scale)
            x0 = int(rng.integers(0, max(1, W - cw)))
            y0 = int(rng.integers(0, max(1, H - ch)))
            image = cv2.resize(image[y0:y0+ch, x0:x0+cw], (W, H),
                               interpolation=cv2.INTER_LINEAR)
            for i in range(gt_masks.shape[0]):
                gt_masks[i] = cv2.resize(gt_masks[i, y0:y0+ch, x0:x0+cw],
                                         (W, H), interpolation=cv2.INTER_NEAREST)
            for i in range(prompt_mask.shape[0]):
                prompt_mask[i] = cv2.resize(prompt_mask[i, y0:y0+ch, x0:x0+cw],
                                            (W, H), interpolation=cv2.INTER_NEAREST)

        return image, gt_masks, prompt_mask

    # ------------------------------------------------------------------
    # B. Pixel transforms  (image only, masks untouched)
    # ------------------------------------------------------------------

    def _apply_pixel(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if rng.random() < 0.8:
            image = self._color_jitter(image, rng)

        if rng.random() < 0.1:
            image = cv2.cvtColor(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )

        if rng.random() < 0.3:
            k = int(rng.choice([3, 5, 7]))
            sigma = float(rng.uniform(0.1, 2.0))
            image = cv2.GaussianBlur(image, (k, k), sigma)

        if rng.random() < 0.2:
            blur = cv2.GaussianBlur(image, (0, 0), 3.0)
            strength = float(rng.uniform(0.5, 1.5))
            image = cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)
            image = np.clip(image, 0, 255).astype(np.uint8)

        if rng.random() < 0.3:
            quality = int(rng.integers(50, 96))
            _, buf = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if rng.random() < 0.2:
            sigma = float(rng.uniform(0.0, 0.03))
            noise = rng.normal(0, sigma * 255, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image

    @staticmethod
    def _color_jitter(
        image: np.ndarray,
        rng: np.random.Generator,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
    ) -> np.ndarray:
        img = image.astype(np.float32)

        b_factor = float(rng.uniform(max(0, 1 - brightness), 1 + brightness))
        img = np.clip(img * b_factor, 0, 255)

        c_factor = float(rng.uniform(max(0, 1 - contrast), 1 + contrast))
        mean = img.mean()
        img = np.clip((img - mean) * c_factor + mean, 0, 255)

        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

        s_factor = float(rng.uniform(max(0, 1 - saturation), 1 + saturation))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)

        h_shift = float(rng.uniform(-hue, hue)) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ------------------------------------------------------------------
    # C. Occlusion transforms
    # ------------------------------------------------------------------

    def _apply_occlusion(
        self,
        image: np.ndarray,
        gt_masks: np.ndarray,
        prompt_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random erasing / cutout style occlusion.

        - image: occluded region filled with random colour
        - gt_masks: untouched (network must learn completion from other views)
        - prompt_mask: occluded region erased (simulates SAM seeing partial object)
        """
        H, W = image.shape[:2]

        aug_type = int(rng.integers(0, 3))

        if aug_type == 0:
            n_blocks = int(rng.integers(1, 4))
            for _ in range(n_blocks):
                bh = int(rng.integers(max(1, int(H * 0.05)), max(2, int(H * 0.15))))
                bw = int(rng.integers(max(1, int(W * 0.05)), max(2, int(W * 0.15))))
                y = int(rng.integers(0, max(1, H - bh)))
                x = int(rng.integers(0, max(1, W - bw)))
                fill = rng.integers(0, 256, size=3).astype(np.uint8)
                image[y:y+bh, x:x+bw] = fill
                prompt_mask[:, y:y+bh, x:x+bw] = 0

        elif aug_type == 1:
            grid_size = int(rng.integers(30, 61))
            ratio = float(rng.uniform(0.4, 0.6))
            hole = int(grid_size * ratio)
            occ = np.ones((H, W), dtype=bool)
            for gy in range(0, H, grid_size):
                for gx in range(0, W, grid_size):
                    occ[gy:gy+hole, gx:gx+hole] = False
            fill = rng.integers(0, 256, size=3).astype(np.uint8)
            image[~occ] = fill
            prompt_mask[:, ~occ] = 0

        else:
            n_holes = int(rng.integers(3, 9))
            for _ in range(n_holes):
                size = int(rng.integers(8, 33))
                y = int(rng.integers(0, max(1, H - size)))
                x = int(rng.integers(0, max(1, W - size)))
                fill = rng.integers(0, 256, size=3).astype(np.uint8)
                image[y:y+size, x:x+size] = fill
                prompt_mask[:, y:y+size, x:x+size] = 0

        return image, gt_masks, prompt_mask
