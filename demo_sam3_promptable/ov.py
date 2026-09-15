"""Open-vocabulary instance segmentation: text prompt -> all matching masks.

Preferred backend is SAM 3 (facebook/sam3). The official checkpoint is HF-gated
and is not on this machine; if ``SAM3_CKPT`` / a local ``sam3.pt`` is present
we use it. Otherwise we run the same PCS contract with Grounding DINO + SAM2
(text boxes refined into instance masks), which is what the local weights
already provide.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GDINO_DIR = ROOT / "ICLR2027/FeaturePuzzlesPipline/weights/grounding-dino-tiny"
SAM2_CKPT = ROOT / "third_party/sam2_ckpts/sam2_hiera_large.pt"
SAM2_CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_REPO = ROOT / "third_party/IGGT_official"

# synonyms so GDINO / SAM3 can fire on nearby words
QUERY = {
    "sofa":    ["sofa", "couch"],
    "table":   ["table", "desk", "dining table"],
    "cabinet": ["cabinet", "cupboard", "wardrobe"],
    "chair":   ["chair", "office chair", "armchair"],
}

# structure words we detect only to veto target masks that leak onto them
NEG_QUERY = ["wall", "floor", "ceiling", "door", "window"]

# a cabinet legitimately has doors and glass fronts, so those words may not veto it
NEG_EXEMPT = {"cabinet": {"door", "window"}}


def _map_label(text, targets):
    t = text.lower().strip().rstrip(".")
    for i, name in enumerate(targets):
        for q in QUERY.get(name, [name]):
            if q in t or t in q:
                return i
    return None


def _map_neg(text, exempt=()):
    """Structure word inside ``text``, ignoring the ones this class may contain."""
    t = text.lower().strip().rstrip(".")
    for q in NEG_QUERY:
        if q in exempt:
            continue
        if q in t or t in q:
            return q
    return None


class OVSegmenter:
    def __init__(self, targets, device="cuda", box_thr=0.22, text_thr=0.18,
                 min_area=80, nms_iou=0.7, use_neg=True, neg_drop=0.55,
                 neg_strong=0.40):
        self.targets = list(targets)
        self.device = torch.device(device)
        self.box_thr = box_thr
        self.text_thr = text_thr
        self.min_area = min_area
        self.nms_iou = nms_iou
        self.use_neg = use_neg
        self.neg_drop = neg_drop      # target dropped when this much sits on structure
        self.neg_strong = neg_strong  # confident targets are never carved by structure
        self.last_neg = None          # structure mask of the most recent segment()
        self.backend, self._impl = self._load()
        print(f"[ov] backend={self.backend} targets={self.targets}")

    def _load(self):
        ckpt = os.environ.get("SAM3_CKPT") or os.environ.get("SAM3_MODEL")
        candidates = [Path(p) for p in
                      ([ckpt] if ckpt else []) + [
                          ROOT / "weights/sam3/sam3.pt",
                          ROOT.parent / "weights/sam3/sam3.pt",
                      ]]
        for p in candidates:
            if p and p.is_file():
                try:
                    return "sam3", self._load_sam3(p)
                except Exception as e:
                    print(f"[ov] SAM3 load failed ({p}): {e}")
        return "grounded_sam2", self._load_gsam2()

    def _load_sam3(self, ckpt):
        sys.path.insert(0, str(ROOT / "third_party/sam3"))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(checkpoint_path=str(ckpt))
        model = model.to(self.device).eval()
        return Sam3Processor(model)

    def _load_gsam2(self):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        sys.path.insert(0, str(SAM2_REPO))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        proc = AutoProcessor.from_pretrained(str(GDINO_DIR))
        dino = GroundingDinoForObjectDetection.from_pretrained(str(GDINO_DIR))
        dino = dino.to(self.device).eval()
        sam = build_sam2(SAM2_CFG, str(SAM2_CKPT), device=str(self.device))
        pred = SAM2ImagePredictor(sam)
        return dict(proc=proc, dino=dino, pred=pred)

    @torch.no_grad()
    def segment(self, rgb: np.ndarray):
        """rgb uint8 HWC -> list of {cls, mask(H,W) bool, score, box xyxy}."""
        if self.backend == "sam3":
            return self._seg_sam3(rgb)
        return self._seg_gsam2(rgb)

    def _suppress_neg(self, inst, negs):
        """Veto target masks that live on wall / floor / ceiling / door / window.

        A target sitting inside a structure mask loses only to a structure
        detection that is at least as confident, so a real cabinet survives the
        wall behind it while a door called "cabinet" does not survive "door".
        Whatever structure is left then trims the targets that kept their box.
        """
        import cv2
        self.last_neg = None
        if not self.use_neg or not negs or not inst:
            return inst
        kept = []
        for d in inst:
            a = int(d["mask"].sum())
            exempt = NEG_EXEMPT.get(self.targets[d["cls"]], ())
            loser = False
            for n in negs:
                if n.get("word") in exempt:
                    continue
                inter = int(np.logical_and(d["mask"], n["mask"]).sum())
                mask_hit = inter / max(a, 1) >= self.neg_drop
                box_hit = (_box_iou(d.get("box"), n.get("box")) >= 0.5
                           if d.get("box") is not None and n.get("box") is not None
                           else False)
                if (mask_hit or box_hit) and n["score"] >= 0.75 * d["score"]:
                    loser = True
                    break
            if not loser:
                kept.append(d)
        if not kept:
            return kept
        strong = np.zeros_like(kept[0]["mask"])
        for d in kept:
            if d["score"] >= self.neg_strong:
                strong |= d["mask"]
        by_word = {}
        for n in negs:
            w = n.get("word") or "wall"
            u = by_word.setdefault(w, np.zeros_like(strong))
            u |= n["mask"]
        for w in by_word:
            by_word[w] = np.logical_and(by_word[w], ~strong)
        hard = np.zeros_like(strong)
        for w, u in by_word.items():
            if not any(w in ex for ex in NEG_EXEMPT.values()):
                hard |= u
        self.last_neg = hard
        out = []
        for d in kept:
            exempt = NEG_EXEMPT.get(self.targets[d["cls"]], ())
            neg = hard.copy()
            for w, u in by_word.items():
                if w not in exempt:
                    neg |= u
            if not neg.any():
                out.append(d)
                continue
            m = d["mask"]
            a = int(m.sum())
            frac = int(np.logical_and(m, neg).sum()) / max(a, 1)
            if frac >= self.neg_drop:
                continue
            if frac > 0.05:
                m = np.logical_and(m, ~neg)
                nlab, lab = cv2.connectedComponents(m.astype(np.uint8))
                if nlab > 2:  # keep the dominant piece only
                    sizes = np.bincount(lab.ravel())
                    sizes[0] = 0
                    m = lab == int(np.argmax(sizes))
                if m.sum() < self.min_area:
                    continue
                d = dict(d, mask=m, box=_mask_box(m))
            out.append(d)
        return out

    @torch.no_grad()
    def refine(self, rgb: np.ndarray, seeds, min_area=20, max_points=3, neg=None):
        """Re-run SAM2 on coarse masks so propagated blobs become whole objects.

        ``seeds`` is a list of bool masks in ``rgb`` resolution; returns a list
        of (mask, score) with ``None`` where refinement was not possible.
        """
        out = [None] * len(seeds)
        if self.backend != "grounded_sam2":
            return out
        idx = [i for i, s in enumerate(seeds)
               if s is not None and int(np.asarray(s).sum()) >= min_area]
        if not idx:
            return out
        import cv2
        pred = self._impl["pred"]
        pred.set_image(rgb)
        for i in idx:
            S = np.asarray(seeds[i]).astype(bool)
            ys, xs = np.nonzero(S)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            # points from the mask interior so thin fragments still vote inside
            core = cv2.erode(S.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
            src = core if core.sum() >= 3 else S
            cy, cx = np.nonzero(src)
            take = np.linspace(0, len(cx) - 1, min(max_points, len(cx))).astype(int)
            pts = np.stack([cx[take], cy[take]], 1).astype(np.float32)
            lbl = np.ones(len(pts), np.int32)
            try:
                masks, ious, _ = pred.predict(
                    box=box, point_coords=pts, point_labels=lbl,
                    multimask_output=True)
            except Exception:
                continue
            masks = np.asarray(masks).astype(bool)
            ious = np.asarray(ious).reshape(-1)
            ssum = int(S.sum())
            best = None
            for k in range(len(masks)):
                mk = masks[k]
                a = int(mk.sum())
                if a < min_area or a > 0.9 * mk.size:
                    continue
                if neg is not None and neg.any():
                    grown = np.logical_and(mk, ~S)
                    gsz = int(grown.sum())
                    # refinement may not spill onto wall / floor / ceiling
                    if gsz and int(np.logical_and(grown, neg).sum()) / gsz >= 0.5:
                        continue
                inter = int(np.logical_and(mk, S).sum())
                cover = inter / max(ssum, 1)          # seed explained by candidate
                if cover < 0.35:                     # drifted away from the seed
                    continue
                sc = float(ious[k]) + 0.5 * cover
                if best is None or sc > best[0]:
                    best = (sc, mk, float(ious[k]))
            if best is not None:
                out[i] = (best[1], best[2])
        return out

    def _seg_sam3(self, rgb):
        proc = self._impl
        img = Image.fromarray(rgb)
        state = proc.set_image(img)
        out = []
        negs = []
        for i, name in enumerate(list(self.targets)
                                + (list(NEG_QUERY) if self.use_neg else [])):
            prompt = " . ".join(QUERY.get(name, [name]))
            o = proc.set_text_prompt(state=state, prompt=prompt)
            masks = o.get("masks")
            scores = o.get("scores")
            boxes = o.get("boxes")
            if masks is None:
                continue
            masks = np.asarray(masks)
            if masks.ndim == 4:
                masks = masks[:, 0]
            scores = np.asarray(scores).reshape(-1)
            boxes = np.asarray(boxes).reshape(-1, 4) if boxes is not None else None
            for k in range(len(masks)):
                m = masks[k].astype(bool)
                if m.sum() < self.min_area:
                    continue
                if i >= len(self.targets):   # structure prompt: veto material only
                    negs.append(dict(mask=m, score=float(scores[k])))
                    continue
                box = boxes[k] if boxes is not None else _mask_box(m)
                out.append(dict(cls=i, mask=m, score=float(scores[k]),
                                box=np.asarray(box, np.float32)))
        return _nms_inst(self._suppress_neg(out, negs), self.nms_iou)

    def _dino(self, img, words, H, W):
        """One Grounding DINO pass -> [(phrase, score, box)]."""
        impl = self._impl
        text = " . ".join(words) + "."
        inputs = impl["proc"](images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            raw = impl["dino"](**inputs)
        try:
            det = impl["proc"].post_process_grounded_object_detection(
                raw, inputs.get("input_ids"),
                threshold=self.box_thr, text_threshold=self.text_thr,
                target_sizes=[(H, W)])[0]
        except TypeError:
            det = impl["proc"].post_process_grounded_object_detection(
                raw, inputs.get("input_ids"),
                box_threshold=self.box_thr, text_threshold=self.text_thr,
                target_sizes=[(H, W)])[0]
        boxes, scores = det.get("boxes"), det.get("scores")
        labels = det.get("labels") or det.get("text_labels") or []
        if boxes is None or len(boxes) == 0:
            return []
        boxes = boxes.detach().cpu().numpy().astype(np.float32)
        scores = scores.detach().cpu().numpy().astype(np.float32)
        if hasattr(labels, "cpu"):
            labels = labels.cpu().tolist()
        return [(str(l), float(s), b) for l, s, b in zip(labels, scores, boxes)]

    def _seg_gsam2(self, rgb):
        impl = self._impl
        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb)
        # targets and structure words need separate passes: in a single pass DINO
        # fuses adjacent prompt words into phrases like "cupboard wardrobe door",
        # and then the two groups can no longer be told apart
        tgt_hits = self._dino(img, [q for n in self.targets
                                    for q in QUERY.get(n, [n])], H, W)
        neg_hits = self._dino(img, list(NEG_QUERY), H, W) if self.use_neg else []
        tgt, amb = [], []
        for lab, sc, box in tgt_hits:
            cid = _map_label(lab, self.targets)
            if cid is None:
                continue
            # DINO answered with a structure word as well -> count it as structure
            word = (_map_neg(lab, NEG_EXEMPT.get(self.targets[cid], ()))
                    if self.use_neg else None)
            if word:
                amb.append((word, sc, box))
            else:
                tgt.append((cid, sc, box))
        if not tgt:
            self.last_neg = None
            return []
        impl["pred"].set_image(rgb)

        def _mask(box):
            masks, iou, _ = impl["pred"].predict(box=box, multimask_output=False)
            return masks[0].astype(bool), float(iou[0])

        negs = []
        for lab, sc, box in list(neg_hits) + amb:
            word = _map_neg(lab) or "wall"
            m, iou = _mask(box)
            if m.sum() >= self.min_area:
                negs.append(dict(mask=m, score=sc * iou, box=box, word=word))
        inst = []
        for cid, sc, box in tgt:
            m, iou = _mask(box)
            if m.sum() < self.min_area:
                continue
            inst.append(dict(cls=int(cid), mask=m, score=sc * iou, box=box))
        return _nms_inst(self._suppress_neg(inst, negs), self.nms_iou)


def _mask_box(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def _box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / max(ar, 1e-6))


def _nms_inst(inst, iou):
    if len(inst) < 2:
        return inst
    order = np.argsort([-x["score"] for x in inst])
    keep = []
    for i in order:
        a = inst[i]
        dup = False
        for j in keep:
            b = inst[j]
            inter = np.logical_and(a["mask"], b["mask"]).sum()
            union = np.logical_or(a["mask"], b["mask"]).sum()
            if union and inter / union >= iou:
                dup = True
                break
        if not dup:
            keep.append(i)
    return [inst[i] for i in keep]
