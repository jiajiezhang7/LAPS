# LAPS on Breakfast — split1 cam01 实验手册（含小数据集分支）

## 实验概述
- 数据集：Breakfast (BreakfastII_15fps_qvga_sync, 15fps, QVGA)
- 视角：仅 cam01（单视角）
- 划分：split1（train 341, test 92）
- 关键预处理：CoTracker2（reinit=true, horizon=16, n_tracks=400），target_fps=10，resize_shorter=320
- 训练输入：仅轨迹 keys_to_load=["tracks"]（不使用图像）
- 当前进度：
  - Step 0.1/0.2/1 已完成
  - Step 2 仅完成 30/341 个训练视频的 HDF5（决定先用这 30 个做小规模训练验证）

## 任务清单（Task List）
- [x] Step 0.1 生成 cam01 过滤 split 文件（train.split1_cam01.bundle=341, test.split1_cam01.bundle=92）
- [x] Step 0.2 创建 cam01 视频符号链接目录（Videos_train.split1_cam01=341, Videos_test.split1_cam01=92）
- [x] Step 1 转换帧级 GT → 段级 JSON（train 341, test 92）
- [ ] Step 2 预处理训练视频（CoTracker→HDF5）（已完成 30/341）
- [ ] Step 3 训练 Motion Tokenizer（小数据集，30 个 HDF5）
- [ ] Step 4 导出能量（train）
- [ ] Step 5 测试集推理+分割（test）
- [ ] Step 6 使用 GT 搜索阈值（train）
- [ ] Step 7 评估（test）

## 关键决策记录
- 使用 cam01 单视角
- 使用 split1
- resize_shorter=320
- keys_to_load=["tracks"]（仅轨迹，不使用图像）
- 新增：先用 30 个 HDF5 进行小规模训练验证（subset30）

---

## Step 0：准备划分与视角（已完成）

### Step 0.1 生成 cam01 过滤的 split 文件
- 输入：
  - /home/johnny/action_ws/online_datasets/breakfast/breakfast/splits/train.split1.bundle
  - /home/johnny/action_ws/online_datasets/breakfast/breakfast/splits/test.split1.bundle
- 输出：
  - /home/johnny/action_ws/online_datasets/breakfast/breakfast/splits/train.split1_cam01.bundle（341 行）
  - /home/johnny/action_ws/online_datasets/breakfast/breakfast/splits/test.split1_cam01.bundle（92 行）
- 命令：
```bash
conda run -n laps bash -lc '
set -e
S=/home/johnny/action_ws/online_datasets/breakfast/breakfast/splits
awk "/_cam01_/" "$S/train.split1.bundle" > "$S/train.split1_cam01.bundle"
awk "/_cam01_/" "$S/test.split1.bundle"  > "$S/test.split1_cam01.bundle"
wc -l "$S/train.split1_cam01.bundle" "$S/test.split1_cam01.bundle"
'
```

### Step 0.2 创建 cam01 视频符号链接目录
- 源视频根：/home/johnny/action_ws/online_datasets/breakfast/breakfast/raw_videos/BreakfastII_15fps_qvga_sync（多级子目录）
- 输出目录：
  - /home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01
  - /home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_test.split1_cam01
- 命令（幂等，若已存在则跳过/覆盖符号链接）：
```bash
conda run -n laps bash -lc '
set -euo pipefail
ROOT=/home/johnny/action_ws/online_datasets/breakfast/breakfast
SRC="$ROOT/raw_videos/BreakfastII_15fps_qvga_sync"
OUT_TRAIN="$ROOT/Videos_train.split1_cam01"
OUT_TEST="$ROOT/Videos_test.split1_cam01"
mkdir -p "$OUT_TRAIN" "$OUT_TEST"
# train
while IFS= read -r stem; do bn="${stem%.txt}.avi"; src=$(find "$SRC" -type f -name "$bn" -print -quit);
  if [[ -n "$src" ]]; then ln -sfn "$src" "$OUT_TRAIN/$bn"; else echo "[MISS] $bn"; fi
done < "$ROOT/splits/train.split1_cam01.bundle"
# test
while IFS= read -r stem; do bn="${stem%.txt}.avi"; src=$(find "$SRC" -type f -name "$bn" -print -quit);
  if [[ -n "$src" ]]; then ln -sfn "$src" "$OUT_TEST/$bn"; else echo "[MISS] $bn"; fi
done < "$ROOT/splits/test.split1_cam01.bundle"
# 统计
ls -1 "$OUT_TRAIN"/*.avi | wc -l || true
ls -1 "$OUT_TEST"/*.avi  | wc -l || true
'
```

---

## Step 1：将帧级 GT 转为段级 JSON（已完成）
- 输入：
  - videos-dir（train）：/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01
  - videos-dir（test）：/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_test.split1_cam01
  - gt-dir：/home/johnny/action_ws/online_datasets/breakfast/breakfast/groundTruth
  - split-file：对应 cam01 过滤后的 bundle
- 输出：
  - /home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/train.split1_cam01
  - /home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/test.split1_cam01
- 命令：
```bash
# train
conda run -n laps python /home/johnny/action_ws/tools/convert_breakfast_gt_to_segments.py \
  --videos-dir /home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01 \
  --gt-dir     /home/johnny/action_ws/online_datasets/breakfast/breakfast/groundTruth \
  --split-file /home/johnny/action_ws/online_datasets/breakfast/breakfast/splits/train.split1_cam01.bundle \
  --out-dir    /home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/train.split1_cam01
# test
conda run -n laps python /home/johnny/action_ws/tools/convert_breakfast_gt_to_segments.py \
  --videos-dir /home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_test.split1_cam01 \
  --gt-dir     /home/johnny/action_ws/online_datasets/breakfast/breakfast/groundTruth \
  --split-file /home/johnny/action_ws/online_datasets/breakfast/breakfast/splits/test.split1_cam01.bundle \
  --out-dir    /home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/test.split1_cam01
```

---

## Step 2：预处理训练视频（CoTracker → HDF5）（部分完成：30/341）
- 输入目录（train）：/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01（期望 341 个 .avi）
- 输出目录：/home/johnny/action_ws/data/preprocessed_breakfast_cam01_m10/split1
- 关键参数：target_fps=10, resize_shorter=320, n_tracks=400, horizon=16, reinit=true, view_name=default
- 说明：目前仅完成约 30 个 HDF5，结构已核验（tracks=(T,16,400,2), vis=(T,16,400)）。
- 命令（按需继续或跳过）：
```bash
conda run -n laps python -m amplify.preprocessing.preprocess_my_segments \
  source=/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01 \
  output_dir=/home/johnny/action_ws/data/preprocessed_breakfast_cam01_m10/split1 \
  dataset_name=breakfast target_fps=10 resize_shorter=320 \
  n_tracks=400 init_queries=uniform reinit=true horizon=16 view_name=default \
  recursive=false skip_exist=true verbose=true
```

---

## Step 3：训练 Motion Tokenizer（小数据集，30 个 HDF5）（待执行）
- 训练数据根：/home/johnny/action_ws/data/preprocessed_breakfast_cam01_m10/split1（仅使用已生成的约 30 个文件）
- 关键参数：keys_to_load=[tracks], true_horizon=16, track_pred_horizon=16, num_tracks=400
- 小数据集建议：num_epochs 可先设 5~10 以避免过拟合；batch_size 视显存调整
- 输出 checkpoint：/home/johnny/action_ws/checkpoints/motion_tokenizer/epochs10_breakfast_split1_cam01_m10_subset30/best.pt
- 命令：
```bash
conda run -n laps python /home/johnny/action_ws/amplify/train_motion_tokenizer.py \
  root_dir=/home/johnny/action_ws/data/preprocessed_breakfast_cam01_m10/split1 \
  train_datasets=[custom_segments:traj0.8] \
  val_datasets=[custom_segments:traj0.2] \
  cond_cameraviews=[default] keys_to_load=[tracks] \
  true_horizon=16 track_pred_horizon=16 num_tracks=400 \
  batch_size=8 gpu_max_bs=8 num_epochs=10 lr=1e-4 \
  quick=false num_workers=4 log_interval=16 \
  resume=false save_interval=5 \
  run_name=epochs10_breakfast_split1_cam01_m10_subset30 \
  use_wandb=false
```

---

## Step 4：导出训练集能量（用于阈值搜索）（待执行）
- 基础配置：/home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01.yaml
- 生成 subset30 版本并替换 checkpoint_path：
```bash
conda run -n laps bash -lc '
set -e
cp /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01.yaml \
   /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_subset30.yaml
sed -E -i \
  "s#^(checkpoint_path:).*$#\1 \"/home/johnny/action_ws/checkpoints/motion_tokenizer/epochs10_breakfast_split1_cam01_m10_subset30/best.pt\"#" \
  /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_subset30.yaml
'
```
- 运行能量导出（train）：
```bash
conda run -n laps python /home/johnny/action_ws/video_action_segmenter/stream_inference.py \
  --params /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_subset30.yaml
```
- 主要 I/O：
  - 输入：/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01
  - 输出能量 JSONL 根：/home/johnny/action_ws/output/breakfast/energy_split1_cam01
  - 能量文件名：每视频目录下 stream_energy_quantized_token_diff_l2_mean.jsonl

---

## Step 5：测试集推理与分割（使用报告阈值）（待执行）
- 基础配置：/home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_test.yaml
- 生成 subset30 版本并替换 checkpoint_path：
```bash
conda run -n laps bash -lc '
set -e
cp /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_test.yaml \
   /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_test_subset30.yaml
sed -E -i \
  "s#^(checkpoint_path:).*$#\1 \"/home/johnny/action_ws/checkpoints/motion_tokenizer/epochs10_breakfast_split1_cam01_m10_subset30/best.pt\"#" \
  /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_test_subset30.yaml
'
```
- 推理+分割（test）：
```bash
conda run -n laps python /home/johnny/action_ws/video_action_segmenter/stream_inference.py \
  --params /home/johnny/action_ws/video_action_segmenter/params_breakfast_split1_cam01_test_subset30.yaml
```
- 主要 I/O：
  - 输入：/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_test.split1_cam01
  - 输出：/home/johnny/action_ws/output/breakfast/segments_split1_cam01/{video}/segmented_videos/{video}_segments.json
  - 同时写出能量 JSONL 到同一根目录（便于评估脚本读取）

---

## Step 6：使用 GT 搜索最佳阈值（train）（待执行）
- 说明：已改造 tools/threshold_search_with_gt.py 支持 --view 可选、动态 JSON key（格式：{source}_{mode}_best）
- 输入：
  - energy-root：/home/johnny/action_ws/output/breakfast/energy_split1_cam01
  - gt-dir：/home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/train.split1_cam01
- 输出（报告）：/home/johnny/action_ws/output/breakfast/thresholds/split1_cam01/best_threshold_quantized_token_diff.json
- 命令：
```bash
conda run -n laps python /home/johnny/action_ws/tools/threshold_search_with_gt.py \
  --energy-root /home/johnny/action_ws/output/breakfast/energy_split1_cam01 \
  --gt-dir      /home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/train.split1_cam01 \
  --source quantized --mode token_diff_l2_mean \
  --target-fps 10 --stride 4 \
  --hysteresis-ratio 0.95 --up-count 2 --down-count 2 --cooldown-windows 1 \
  --max-duration-seconds 2.0 \
  --tolerance-sec 2.0 \
  --output /home/johnny/action_ws/output/breakfast/thresholds/split1_cam01/best_threshold_quantized_token_diff.json
```
- 报告 JSON key：quantized_token_diff_l2_mean_best.best_f1.thr（与 Step 5 的 report_key 一致）

---

## Step 7：评估（test）（待执行）
- 输入：
  - 预测根：/home/johnny/action_ws/output/breakfast/segments_split1_cam01
  - GT 目录：/home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/test.split1_cam01
- 输出：/home/johnny/action_ws/output/breakfast/eval/split1_cam01_eval.json
- 指标：F1@2s（主），附加 F1@5s；mAP@IoU 0.5/0.75
- 命令：
```bash
conda run -n laps python /home/johnny/action_ws/tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/output/breakfast/segments_split1_cam01 \
  --gt-dir    /home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/test.split1_cam01 \
  --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --iou-thrs 0.5 0.75 \
  --output /home/johnny/action_ws/output/breakfast/eval/split1_cam01_eval.json
```

---

## 附：当前小数据集设置与风险提示（Step 3 相关）
- 仅 30 个 HDF5，易过拟合：建议先 num_epochs=5~10，观察 val loss 与 codebook 指标（perplexity/entropy/unique codes）
- 若训练不稳定：适当减小 lr（如 5e-5）或 batch_size；必要时减小模型容量（如 codebook_size=1024）
- keys_to_load=[tracks] + cond_on_img=false 下，img_shape 不影响训练核心逻辑（保持默认即可）；如需强一致可显式设定 img_shape

