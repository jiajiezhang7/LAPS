# 项目总览

该项目围绕“从视频中学习潜在动作表示（latent action sequences）”构建了完整流水线：

- 使用 `amplify/` 中的 Motion Tokenizer 进行轨迹序列离散化（FSQ 码本）。
- 使用 `video_action_segmenter/` 对长视频或实时流进行窗口化编码、能量分析与阈值分割，导出片段与对应的 code indices。
- 使用 `action_classification/` 对导出的 latent action sequences 做无监督聚类与可视化分析。

核心痛点记录于 `remember.md`：Motion Tokenizer 训练阶段存在“codebook 坍塌”的风险，已在训练日志中补充码本使用度监控指标以便诊断。

---

## 目录结构

- `amplify/`
  - Motion Tokenizer 训练与模型实现（Hydra 配置驱动）。
  - 关键文件：`train_motion_tokenizer.py`、`amplify/models/motion_tokenizer.py`、`cfg/train_motion_tokenizer.yaml`、`preprocessing/`。
- `video_action_segmenter/`
  - 流式推理、能量曲线分析、阈值分割与片段导出；支持批处理目录与多 GPU 并行。
  - 关键文件：`stream_inference.py`、`params.yaml`、`stream_utils/energy.py`、`stream_utils/segmentation.py`、`compute_best_threshold.py`。
- `action_classification/`
  - 针对分割得到的 code indices 做特征工程与无监督聚类；支持 LSTM 序列嵌入训练与评估。
  - 关键文件：`embedding/train.py`、`scripts/infer_sequence_embed_lstm.py`、`evaluation/cluster_eval.py`、`evaluation/embed_cluster_eval.py`、`configs/*.yaml`。
- `amplify_motion_tokenizer/`
  - 独立的推理与模型封装，供分割器直接调用（`video_action_segmenter/stream_inference.py` 使用）。
- 其他
  - `pipeline.md` 使用说明与常用命令。
  - `requirements.txt` 项目依赖清单（详见 `REQUIREMENTS_SUMMARY.md`）。
  - `preprocessed_data/` 数据缓存与预处理产物。

---

## 端到端流程（Quickstart）

> 所有命令默认在 conda 环境中执行：
>
> ```bash
> conda activate laps
> ```

- 分割长视频为 40s 片段（可选，参考 LIBERO 做法）
  ```bash
  python amplify/scripts/split_segments_to_chunks.py \
    /path/to/segments \
    /path/to/segments_40s_complete \
    --workers 64
  ```

- 预处理视频 → 轨迹（CoTracker）
  - 检查/清理：
    ```bash
    python amplify/scripts/check_and_clean_segments.py --dry-run --verbose
    python amplify/scripts/check_and_clean_segments.py
    ```
  - 生成训练数据（在 `amplify/` 下执行）：
    ```bash
    python -m preprocessing.preprocess_my_segments  # 配置见 amplify/cfg/preprocessing/preprocess_my_segments.yaml
    ```

- 训练 Motion Tokenizer（冒烟与正式示例见 `pipeline.md`）
  ```bash
  python amplify/train_motion_tokenizer.py \
    root_dir=/path/to/preprocessed_data \
    train_datasets=[custom_segments:traj1.0] \
    val_datasets=null \
    cond_cameraviews=[default] \
    keys_to_load=[tracks,images] \
    img_shape=[898,1442] \
    true_horizon=16 track_pred_horizon=16 \
    batch_size=8 gpu_max_bs=8 num_epochs=1 \
    quick=true num_workers=2 log_interval=16 \
    resume=false run_name=smoke_custom_d01 \
    use_wandb=true lr_schedule=null
  ```

- 流式/离线推理（实时窗口：T=16，stride=4，20Hz 重采样）
  ```bash
  python -m video_action_segmenter.stream_inference \
    --params video_action_segmenter/params.yaml
  ```
  - 能量源与模式建议：`energy.source=quantized`，`energy.mode=token_diff_l2_mean`（在当前实验中表现最佳）。
  - 可选：输出窗口级能量 JSONL 与 per-video JSONL；可选：导出未量化 latent（prequant）。

- 能量阈值扫参与报告
  ```bash
  conda run -n laps python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl video_action_segmenter/energy_sweep_out/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl  video_action_segmenter/energy_sweep_out/stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --smooth --smooth-method ema --smooth-alpha 0.7 --smooth-window 3 \
    --output-json video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff_smoothed.json
  ```
  - 在线分割时可将 `segmentation.mode=report` 并指定 `report_path` 与 `report_key`（见 `params.yaml`）。

- 序列嵌入与无监督聚类（针对导出的 code indices）
  - 训练 LSTM 序列嵌入（NTP）：
    ```bash
    python -m action_classification.embedding.train \
      --json-root /path/to/stream_outputs/json_by_class \
      --config action_classification/configs/sequence_embed.yaml
    ```
  - 导出嵌入：
    ```bash
    python -m action_classification.scripts.infer_sequence_embed_lstm \
      --json-root /path/to/json \
      --model-pt /path/to/model_best.pt \
      --out-dir action_classification/analysis/seq_embed
    ```
  - 评估与可视化：
    ```bash
    python -m action_classification.evaluation.cluster_eval \
      --json-root /path/to/json \
      --config action_classification/configs/eval_config.yaml

    python -m action_classification.evaluation.embed_cluster_eval \
      --embed-dir action_classification/analysis/seq_embed/<timestamp> \
      --config action_classification/configs/eval_config.yaml
    ```

---

## 组件详解

### 1) Motion Tokenizer（`amplify/`）

- 训练入口：`amplify/train_motion_tokenizer.py`（Hydra 配置：`cfg/train_motion_tokenizer.yaml`）。
- 模型实现：`amplify/models/motion_tokenizer.py`
  - 编码器 `VAEEncoder` 与解码器 `VAEDecoder`，Transformer 结构。
  - 量化器：`vector_quantize_pytorch.FSQ`。支持码本大小：16/64/256/512/1024/2048/4096（内部映射至不同 `levels`）。
  - 前向：`forward(x, cond)` → `x_recon, codebook_indices, rel_logits`。
- 训练日志关键指标（用于监控码本健康度与“坍塌”）：
  - `train_codebook_perplexity`（标准困惑度）。
  - `train_entropy_step`、`train_entropy_norm_step`、`train_unique_codes_step`（基于 indices 的使用统计）。
- 数据预处理：`amplify/preprocessing/preprocess_my_segments.py`
  - 基于 CoTracker2/3（`torch.hub` 加载）提取网格关键点轨迹，输出 HDF5（`root/<view>/{tracks,vis}`）。
  - 多 GPU 与预取 IO 可选；视频读取与重采样内置。

### 2) 视频动作分割器（`video_action_segmenter/`）

- 入口：`video_action_segmenter/stream_inference.py`
  - 输入源：`camera|file|rtsp|folder`；目录模式支持多 GPU 并行批处理。
  - 窗口采样：20Hz 重采样、`T=16`、`stride=4`。
  - 快速门控：
    - `pre_gate`（像素差分，低成本筛除静止窗口）。
    - `motion_gate`（基于 CoTracker 速度的静止判定，跳过后续前向）。
  - 能量计算：`stream_utils/energy.py`（`quantized|prequant|velocity` × `l2_mean|token_diff_l2_mean`）。
  - 可视化：基础/增强样式与主题，支持阈值线与统计框叠加。
  - 分割与导出：`stream_utils/segmentation.py`
    - 双阈值滞回、去抖、冷却、最长时长等护栏。
    - 片段视频导出与对应 codes 的 sidecar JSON（支持“允许重叠”与最小重叠比例）。

### 3) 无监督聚类（`action_classification/`）

- 数据与公共组件：`embedding/common.py`（`SeqDataset`、`SeqEncoder`、`scan_samples`、`flatten_codes`）。
- 序列嵌入训练：`embedding/train.py`（NTP 目标，早停与长度统计、自适应 `max_len`）。
- 聚类评估：
  - `evaluation/cluster_eval.py`（特征：BoW/Avg；KMeans/GMM；UMAP 可视化；支持按类平衡拟合）。
  - `evaluation/embed_cluster_eval.py`（基于 LSTM 嵌入的 KMeans/HDBSCAN 多指标对比与 UMAP）。
- 配置：`configs/sequence_embed.yaml`、`configs/eval_config.yaml`。

---

## 安装与环境

- 基础依赖：
  ```bash
  conda activate laps
  pip install -r requirements.txt
  ```
- 可选依赖：
  - W&B（实验跟踪）：`pip install wandb`
  - CoTracker（轨迹提取）：`pip install git+https://github.com/facebookresearch/co-tracker.git`
- 版本与兼容性：Python 3.10，PyTorch ≥ 2.1，详情见 `REQUIREMENTS_SUMMARY.md`。

---

## 常见问题与排查

- Codebook 使用度/坍塌：
  - 关注训练日志中的 `train_codebook_perplexity`、`train_entropy_norm_step` 与 `train_unique_codes_step`；
  - 可在 `cfg/train_motion_tokenizer.yaml` 中调整 `codebook_size`、`hidden_dim`、`attn_pdrop`、`batch_size/gpu_max_bs` 等资源相关项以便稳定训练（按需在你机器上验证）。
- 流式推理性能：
  - 实时预算约为 `stride/target_fps` 秒/窗；可下调 `resize_shorter`、启用 `pre_gate/motion_gate` 以降低负载。
- 能量阈值跨设备迁移：
  - 建议抽取少量样本重新运行 `compute_best_threshold.py` 更新报告阈值，而非在线自适应。

---

## 数据与产物

- 预处理 HDF5：`root/<view>/{tracks,vis}`，与 `amplify/utils/train.py` 期望一致。
- 流式推理输出：
  - 片段视频与 `code_indices/*.codes.json` sidecar；
  - 可选 per-video 能量 JSONL 与窗口级 prequant `.npy`。
- 聚类与可视化：
  - `action_classification/analysis/` 下保存嵌入、UMAP、指标 JSON 等。

---

## 参考与更多文档

- `pipeline.md`：端到端命令与详细参数示例。
- `amplify/motion_tokenizer_overview.md`：Motion Tokenizer 代码结构与训练/推理细节（强烈推荐）。

---

## License

- `amplify/` 为开源上游代码（保留其原始 LICENSE）。其余自研部分遵循仓库默认协议。
