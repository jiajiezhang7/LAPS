# LAPS: 潜在动作原语分割（Latent Action-based Primitive Segmentation）

> 面向工业 VLA 的“数据瓶颈”问题，LAPS 提供从原始视频到“动作原语（action primitives）”的端到端、无监督自动化数据流水线。核心输出是片段级动作原语与其潜在表示；核心技术包括基于 FSQ 码本的 Motion Tokenizer 与“潜在动作能量（Latent Action Energy）”的边界检测度量。

- 代码根目录：`/home/johnny/action_ws`
- 主环境：`conda` 环境 `laps`
- 对比基线：ABD（无训练离线）、OTAS（无监督分割）

---

## 流水线总览（Three Phases）

- Phase 1｜关键点跟踪与运动表征
  - 从原始视频经 CoTracker 提取关键点轨迹与速度（可视化与预处理）。
- Phase 2｜运动离散化与在线动作检测
  - 训练 Motion Tokenizer（FSQ 码本）；在线/离线流式推理，输出“潜在动作能量”，经滞回阈值分割为动作片段并导出代码索引。
- Phase 3｜无监督发现与可视化验证
  - 轻量 Transformer（冻结推理）提取段级向量，UMAP 可视化与聚类质量评估；导出代表性片段以人工快速审阅。

参考图示：论文 Figure（示意）；实现入口见“模块地图”。

---

## 安装与环境

- 建议 Python 3.10（`laps` 环境）
- 先激活环境（遵循项目规则：所有命令前均需先激活 `laps`）：

```bash
conda activate laps
```

- 基本依赖（含 Motion Tokenizer 训练）

```bash
pip install -r amplify/requirements.txt
```

- 可选依赖
  - 轨迹提取（CoTracker，若直接使用 `stream_inference.py` 内置 hub 下载可不装源码）：
    - 仅下载权重（推荐）：
      ```bash
      wget -O ~/.cache/torch/hub/checkpoints/cotracker2.pth \
        "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"
      wget -O ~/.cache/torch/hub/checkpoints/scaled_online.pth \
        "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth"
      ```
    - 或安装源码：`pip install git+https://github.com/facebookresearch/co-tracker.git`
  - 可视化/降维：`pip install umap-learn plotly scikit-learn`

---

## 复现指南（一步到位）

以下步骤均默认在 `conda activate laps` 后执行。

### 0) 数据准备与快速可视化

- 原始视频目录（示例）：`/home/johnny/action_ws/datasets/gt_raw_videos/{D01,D02}`
- 可选：窗口化跟踪可视化（验证 CoTracker 轨迹覆盖与密度）

```bash
python -m video_action_segmenter.window_track_and_save \
  --output-dir /home/johnny/action_ws/video_action_segmenter/inference_outputs/windows \
  --target-fps 20 --resize-shorter 480 --grid-size 20 \
  --T 16 --stride 4 --device auto --trail 15
```

### 1) 预处理与训练 Motion Tokenizer

1.1 预处理（生成训练用 HDF5；配置见 `amplify/cfg/preprocessing/preprocess_my_segments.yaml`）

```bash
# 在 amplify 目录下运行（模块名为 preprocessing）
python -m preprocessing.preprocess_my_segments
```

1.2 训练（Hydra 配置驱动；更多说明见 `amplify/motion_tokenizer_overview.md`）

- 快速冒烟：
```bash
python amplify/train_motion_tokenizer.py \
  root_dir=/home/johnny/action_ws/data/preprocessed_data_d01 \
  cond_cameraviews=[default] keys_to_load=[tracks,images] \
  true_horizon=16 track_pred_horizon=16 \
  batch_size=16 gpu_max_bs=16 num_epochs=1 \
  quick=true num_workers=4 log_interval=16 \
  resume=false run_name=smoke_d01 \
  use_wandb=true lr_schedule=null
```

- 完整训练示例（D01）：
```bash
python amplify/train_motion_tokenizer.py \
  root_dir=/media/johnny/48FF-AA60/preprocessed_data_d01_m10 \
  train_datasets=[custom_segments:traj0.8] val_datasets=[custom_segments:traj0.2] \
  cond_cameraviews=[default] keys_to_load=[tracks,images] \
  true_horizon=16 track_pred_horizon=16 \
  batch_size=8 gpu_max_bs=8 num_epochs=5 \
  quick=false num_workers=4 log_interval=8 \
  resume=false run_name=epochs5_complete500_d01_m10 \
  use_wandb=true lr_schedule=null
```

### 2) 在线/离线动作分割（流式推理）

- 推荐参数 YAML：`video_action_segmenter/params.yaml`（或 `params_d01_label.yaml` / `params_d02_label.yaml`）
- 入口：`video_action_segmenter/stream_inference.py`

```bash
python -m video_action_segmenter.stream_inference \
  --params video_action_segmenter/params.yaml
```

- 能量源与模式（`stream_utils/energy.py`）：
  - `source`: `quantized | prequant | velocity`
  - `mode`: `l2_mean | token_diff_l2_mean`（当前实验最佳：`quantized + token_diff_l2_mean`）
- 分割（`stream_utils/segmentation.py`）：双阈值滞回、去抖、冷却、最长时长护栏；导出片段与 `code_indices/*.codes.json`

#### 2.1 阈值扫参与报告（跨设备/场景复用）

```bash
conda run -n laps python -m video_action_segmenter.compute_best_threshold \
  --quantized-jsonl video_action_segmenter/energy_sweep_out/stream_energy_quantized_token_diff_l2_mean.jsonl \
  --velocity-jsonl  video_action_segmenter/energy_sweep_out/stream_energy_velocity_token_diff_l2_mean.jsonl \
  --label-threshold auto \
  --smooth --smooth-method ema --smooth-alpha 0.7 --smooth-window 3 \
  --output-json video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff_smoothed.json
```

- 在线推理读取报告阈值：在参数 YAML 中设置
  - `segmentation.mode: "report"`
  - `segmentation.report_path: video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff_smoothed.json`
  - `segmentation.report_key: quantized.token_diff_l2_mean`

#### 2.2 统一评估（F1@tol、mAP@IoU）

```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_LAPS \
  --gt-dir    /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_LAPS.json
```

> 评估脚本会在每个视频目录下自动发现 `stream_energy_*.jsonl`，以片段内平均能量作为置信度计算 mAP。

### 3) 无监督发现与可视化（UMAP + 聚类）

- 入口：`umap_vis/scripts/sequence_model_embedding.py`
- 输入：LAPS 推理导出的 `code_indices/*.codes.json`（含 `quantized_windows`）
- 输出：
  - 指标 CSV：`umap_vis/statistics/cluster_metrics_seq_model_cosine.csv`
  - 2D/3D 图：`umap_vis/figure/umap_2d_*.png`、`umap_vis/figure/umap_3d_*.html`
  - 可选：按簇采样导出代表性片段视频（便于人工快速审阅）

```bash
conda run -n laps python umap_vis/scripts/sequence_model_embedding.py \
  --data-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_LAPS \
  --fig-dir umap_vis/figure --stats-dir umap_vis/statistics \
  --metric cosine --neighbors 15 --min-dist 0.1 \
  --use-best-grid-config --d-model 256 --n-layers 4 --n-heads 4 --pooling mean \
  --device cpu --k-min 2 --k-max 10 --k-analysis-max 15 --export-video-samples
```

---

## 对比基线与评估协议

### ABD（无训练离线算法）

- 特征：采用 HOF（OpenCV/CPU）片段级特征（不改代码，`--features-dir` 指向 HOF 路径；元数据中 `feature-source` 仍为 `i3d`）
- 入口：`comapred_algorithm/ABD/run_abd.py`

D01 分割：
```bash
conda run -n laps python -m comapred_algorithm.ABD.run_abd \
  --view D01 \
  --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D01 \
  --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_ABD_HOF \
  --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/hof_features/D01 \
  --feature-source i3d --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4
```

D02 分割：
```bash
conda run -n laps python -m comapred_algorithm.ABD.run_abd \
  --view D02 \
  --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D02 \
  --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_ABD_HOF \
  --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/hof_features/D02 \
  --feature-source i3d --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4
```

评估（一致协议）：
```bash
# D01
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_ABD_HOF \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_ABD.json
# D02
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_ABD_HOF \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D02 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_ABD.json
```

> 可选最小代码修改：在 `run_abd.py` 将 `meta_params["source"]` 设置为 `hof` 以与特征来源一致（不影响算法本身，仅影响元数据）。

### OTAS（无监督动作分割）

- 训练/验证环境：`otas`（训练用）；后处理与评估：`laps`
- 入口：`comapred_algorithm/OTAS/code/` 下的 `main.py`、`detect_bdy.py`

训练（示例）：
```bash
# (建议单独在 conda: otas 环境中)
CUDA_VISIBLE_DEVICES=0 conda run -n otas python comapred_algorithm/OTAS/code/main.py \
  --mode train --dataset BF --feature_model tf --gpu 0 \
  --batch_size 16 --num_workers 8 \
  --output-path /home/johnny/action_ws/datasets/output/otas_out/
```

验证生成 mean_error：
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n otas python comapred_algorithm/OTAS/code/main.py \
  --mode val --dataset BF --feature_model tf --gpu 0 \
  --batch_size 16 --num_workers 8 \
  --output-path /home/johnny/action_ws/datasets/output/otas_out/
```

边界检测（laps）：
```bash
conda run -n laps python comapred_algorithm/OTAS/code/detect_bdy.py \
  --output-path /home/johnny/action_ws/datasets/output/otas_out/ \
  --dataset BF --feature_model tf
```

结果适配为统一评估格式：
```bash
# D01
conda run -n laps python tools/adapt_otas_to_segments.py \
  --otas-pred /home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf \
  --raw-dir /home/johnny/action_ws/datasets/gt_raw_videos/D01 \
  --output /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_OTAS
# D02
conda run -n laps python tools/adapt_otas_to_segments.py \
  --otas-pred /home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf \
  --raw-dir /home/johnny/action_ws/datasets/gt_raw_videos/D02 \
  --output /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_OTAS
```

评估（一致协议）：
```bash
# D01
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_OTAS \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_OTAS_trained.json
# D02
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_OTAS \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D02 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_OTAS_trained.json
```

---

## 模块地图（Modules & Entry Points）

- `amplify/`
  - 训练入口：`train_motion_tokenizer.py`
  - 配置：`cfg/`（含 `preprocessing/` 与训练 YAML）
  - 模型：`amplify/models/`；概览：`motion_tokenizer_overview.md`
- `video_action_segmenter/`
  - 推理入口：`stream_inference.py`
  - 参数：`params*.yaml`
  - 工具：`stream_utils/energy.py`、`stream_utils/segmentation.py`、`stream_utils/gating.py`
  - 可视化/窗口跟踪：`window_track_and_save.py`
  - 阈值搜索：`compute_best_threshold.py`
- `umap_vis/`
  - 无监督发现与可视化：`scripts/sequence_model_embedding.py`
- `comapred_algorithm/ABD/`
  - 离线基线：`run_abd.py`、文档：`docs/abd_experiment.md`
- `comapred_algorithm/OTAS/`
  - 无监督分割：`code/{main.py, detect_bdy.py, ...}`、文档：`otas_experiment.md`
- `tools/`
  - 统一评测：`eval_segmentation.py`
  - 适配 OTAS：`adapt_otas_to_segments.py`

---

## 数据与输出（I/O 约定）

- 预处理（训练用）：`root/<view>/{tracks, vis}`（HDF5 与可视化）
- LAPS 推理输出：
  - `segmented_videos/*.mp4` 与 `code_indices/*.codes.json`
  - 可选能量：`stream_energy_*.jsonl`（用于 mAP 置信度）
- 阈值报告：`video_action_segmenter/energy_sweep_report/*.json`
- 基线输出：`/home/johnny/action_ws/datasets/output/segmentation_outputs/{D01_*,D02_*}`
- 评估结果：`/home/johnny/action_ws/datasets/output/stats/seg_eval/*.json`

---

## 关键结果（指引）

- 能量对比与最佳阈值：`video_action_segmenter/energy_sweep_report/`
- UMAP 聚类可视化与指标：`umap_vis/figure/`、`umap_vis/statistics/`
- 论文表格与图示：见 `docs/experiment_paper.md` 与 `docs/experiment_detailed_todo.md`

---

## 常见问题（Troubleshooting）

- 码本“坍塌”与使用度监控：
  - 关注训练日志：`train_codebook_perplexity`、`train_unique_codes_step` 等（详见 `amplify/train_motion_tokenizer.py`）。
  - 可在 `cfg/train_motion_tokenizer.yaml` 调整 `codebook_size`、`batch_size/gpu_max_bs`、`attn_pdrop` 等参数。
- 实时性能：
  - 预算约为 `stride/target_fps` 秒/窗；使用 `pre_gate` 与 `motion_gate` 以降低负载。
- 阈值跨设备迁移：
  - 少量样本重跑 `compute_best_threshold.py` 更新报告阈值，优于在线自适应。

---

## 引用与许可证

- 引用（BibTeX）：请根据论文最终信息补充。
- 许可证：
  - `amplify/` 目录遵循其上游 LICENSE；
  - 其余自研部分遵循仓库默认协议。
