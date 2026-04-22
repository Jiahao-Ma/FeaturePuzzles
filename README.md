# FeaturePuzzles

Training-free / training-based prompt matching with VGGT (and FastVGGT)
on multi-view "puzzle" crops generated from SA-1B.

This repository contains **only the source code and markdown documents**.
Checkpoints, generated puzzle data, training logs, and demo example assets
are intentionally excluded (see [`.gitignore`](.gitignore)).

---

## Repository layout

```
.
├── process.md                          # Project progress + soft-mode design notes
├── cursor_inference_segmentation_py.md # Inference pipeline notes
├── requirements_seg.txt                # Python deps for the segmentation pipeline
├── train_segmentation.py               # Training entry-point (DPT segmentation head)
├── inference_segmentation.py           # Inference entry-point
├── debug_infer.py / debug_infer.ipynb  # Debug utilities for the inference path
├── debug.ipynb / infer.ipynb           # Misc. exploration notebooks
├── sa1b_demo.ipynb                     # SA-1B + puzzles dataset demo
├── doc/                                # Extra design / data documentation
├── scripts/                            # Setup scripts (env + 3rd-party clones)
├── src/
│   ├── sa1b/                           # SA1BPuzzlesDataset + puzzles bridge
│   ├── models/                         # Segmentation heads
│   └── utils/                          # Loss helpers
└── TrainingFreeVGGTMatching/
    ├── prompt_matching/                # Core: extractor / prompt / matcher / runners
    │   ├── extractor.py                # VGGT/FastVGGT patch-token extraction
    │   ├── prompt.py                   # hard / soft / subpatch prompt encoding
    │   ├── matcher.py                  # similarity / binarization / upsampling
    │   ├── run.py / run_scene_batch.py # CLI runners
    │   ├── README.md / NOTES.md
    │   └── SIMILARITY_{en,zh}.md       # Soft-mode similarity write-up
    ├── compare_prompt_token.ipynb      # Hard / soft / subpatch comparison
    ├── infer_prompt_matching.ipynb     # Soft-mode inference + visualization
    ├── prompt_matching_plan.md         # Roadmap
    └── scripts.md                      # Run recipes
```

> Notebooks are committed with **outputs cleared** so that no embedded image
> data or large base64 payloads end up in git history. Re-run them locally
> to reproduce the figures.

---

## Third-party dependencies (NOT in this repo)

The pipeline depends on three external repositories that you should clone
yourself into the workspace root. They are intentionally git-ignored.

```bash
# 1. Puzzles (multi-view crop generation, used by src/sa1b/puzzles_bridge.py)
git clone https://github.com/Jiahao-Ma/puzzles-code.git puzzles

# 2. FastVGGT (fast VGGT backbone used by TrainingFreeVGGTMatching/prompt_matching)
git clone https://github.com/mystorm16/FastVGGT.git TrainingFreeVGGTMatching/FastVGGT

# 3. (Optional) Pi3 — only needed by the older Pi3 segmentation experiments
mkdir -p repo && git clone https://github.com/yyfz/Pi3.git repo/Pi3
```

You will also need to download the VGGT checkpoint and place it at the path
referenced by the notebooks (`CKPT_PATH`, e.g.
`/path/to/vggt/checkpoints/vggt.pt`).

---

## Quick start

```bash
# 1. Install Python dependencies
pip install -r requirements_seg.txt
pip install -r TrainingFreeVGGTMatching/prompt_matching/requirements.txt

# 2. Clone third-party deps (see above)

# 3. Open the soft-mode inference notebook
jupyter lab TrainingFreeVGGTMatching/infer_prompt_matching.ipynb
```

For the full design discussion (soft-mode prompt token extraction, puzzles
data construction, project progress, and the planned LoRA + DPT stage),
see [`process.md`](process.md).
