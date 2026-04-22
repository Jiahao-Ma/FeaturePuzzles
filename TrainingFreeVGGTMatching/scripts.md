conda activate puzzles
cd /mnt/dataset/marko/code/puzzles/TrainingFreeVGGTMatching




# 1. 单场景：prompt_matching/run.py
适合给定一组多视角图片 + 一个 prompt（mask 或 bbox），想看跨视角的相似度热图和二值 mask。

准备示例数据（可选）
python prompt_matching/prepare_example.py \
    --source_dir /path/to/your/multi_view_images \
    --n 6

运行可视化
python -m prompt_matching.run \
    --image_dir   prompt_matching_data/example/images \
    --prompt_mask prompt_matching_data/example/prompt_mask.png \
    --ref_view_idx 0 \
    --output_dir  prompt_matching_outputs/run1 \
    --layers 17 23 \
    --query_mode mean \
    --threshold  otsu \
    --prompt_pool soft

# 2. 批量场景：prompt_matching/run_scene_batch.py
python -m prompt_matching.run_scene_batch \
    --examples_dir examples \
    --data_root    prompt_matching_data \
    --out_root     prompt_matching_outputs_soft \
    --n_views 6 --k_masks 5 \
    --ckpt_path /mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt \
    --layers 17 23 --query_mode mean \
    --threshold percentile --percentile 95 \
    --abs_threshold 0.8 \
    --prompt_pool soft

python -m prompt_matching.run_scene_batch \
  --examples_dir examples --data_root prompt_matching_data \
  --out_root prompt_matching_outputs_subpatch \
  --n_views 6 --k_masks 5 \
  --ckpt_path /mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt \
  --layers 17 23 --query_mode mean \
  --threshold percentile --percentile 95 --abs_threshold 0.8 \
  --prompt_pool subpatch \
  --subpatch_mode dense --subpatch_stride 2

# --subpatch_mode : dense (default); dense_gauss; corners; mixed