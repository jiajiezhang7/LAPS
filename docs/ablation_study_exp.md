# 🔬 论文消融实验 (Ablation Studies) 关键要点总结

本文档总结了 LAPS 管道（Latent Action-based Primitive Segmentation）消融实验（第 4.6 节/表 4）的核心发现，旨在指导关键设计选择的实践落地。

## 一、 核心组件及其验证目标

消融实验验证了三个关键设计对分割性能 (Seg. F1) 和语义一致性（ICSS）的影响。

| 实验组 | 关键设计 | 验证目标 | 核心结果指标 |
| :--- | :--- | :--- | :--- |
| **信号源消融** | Latent Action Energy ($E_{\text{action}}$) 的计算域 | **分割**：确定最鲁棒的语义变化信号源 | Seg. F1 |
| **表征消融** | Domain-specific Motion Tokenizer ($M_\theta$) | **分割/聚类**：验证专用特征对精细任务的重要性 | Seg. F1, ICSS |
| **编码器消融** | Frozen Transformer Encoder | **聚类**：验证时序动态建模对语义聚类的必要性 | ICSS |

注：本文中的 Seg. F1 明确指 F1@2s（边界容差=2 秒）；除非另行说明，所有评估均在 D01 数据集上进行。

## 二、 关键发现与实践指导

| 消融配置 | Seg. F1 | ICSS | 关键发现及实践指导 |
| :--- | :--- | :--- | :--- |
| **Full Pipeline (Ours)** | **46.3** | **0.65** | **基准**：代表最佳性能配置。 |
| **信号源消融** | | | |
| $\quad E_{\text{action}}$ from Pre-Quant. Latents | 41.7 | -- | **结论**：量化 (Quantization) 是必要的，它抽象了噪声并使潜在向量更具判别性。 |
| $\quad E_{\text{action}}$ from Raw Velocities | 30.2 | -- | **结论**：**必须**在抽象的潜在动作空间 (Latent Space) 而非低级物理空间（Raw Velocities）进行分割，以捕获语义意图变化。 |
| **表征消融** | | | |
| $\quad$ w/o $M_\theta$ (e.g., CLIP) | 25.8 | 0.21 | **结论**：**必须**训练领域专用的 Motion Tokenizer ($M_\theta$) 来捕捉工业任务的精细动作。通用视觉特征效果不佳。 |
| **编码器消融** | | | |
| $\quad$ w/o Transformer (Mean-pool) | -- | 0.38 | **结论**：**必须**使用 Frozen Transformer 或类似时序模型来编码动作序列，简单的平均池化无法实现高质量的语义聚类。 |



## 实验执行计划

本计划基于现有代码与脚本，覆盖论文中三类消融：信号源、表征、编码器。严格不修改源码，仅通过复制与微调已有 YAML、按步骤运行脚本完成。所有实验输出统一存放至：/home/johnny/action_ws/datasets/output/paper_ablation_study/

### 1) 实验环境配置
- Conda 环境：laps（依据 .augment/rules/conda_env.md）
- 主要脚本与配置：
  - 训练：amplify/train_motion_tokenizer.py（Hydra 配置：amplify/cfg/train_motion_tokenizer.yaml）
  - 在线分割与能量导出：video_action_segmenter/stream_inference.py（参数文件示例：video_action_segmenter/params_d01_quant.yaml, params_d01_vel.yaml）
  - 光流能量：video_action_segmenter/scripts/compute_optical_flow_energy.py
  - 基于能量的阈值搜索：tools/threshold_search_with_gt.py；量化vs速度自动阈值：video_action_segmenter/compute_best_threshold.py
  - 离线阈值分割：tools/segment_from_energy.py
  - 分割评估（F1/mAP）：tools/eval_segmentation.py
  - ICSS 语义一致性评估：umap_vis/scripts/vlm_icss_evaluation.py
  - （可选）聚类可视化与补充指标：umap_vis/scripts/sequence_model_embedding.py, umap_vis/scripts/segment_umap_cluster_analysis.py, umap_vis/scripts/codes_umap_cluster_analysis.py
- 数据路径：
  - 原始视频：/home/johnny/action_ws/datasets/gt_raw_videos/D01
  - 标注（GT）：/home/johnny/action_ws/datasets/gt_annotations/D01
  - Motion Tokenizer 权重：/home/johnny/action_ws/checkpoints/motion_tokenizer/epochs5_complete500_d01_m10/best.pt

注意：如需为某些消融新建 YAML，仅做“复制+修改”的最小变更，不改原文件；所有输出目录统一指定到 paper_ablation_study 下各子文件夹。

### 2) 消融实验列表（逐项可执行方案）

为便于追溯，统一命名：expXX_<简述>；每项包含：目的、改动点、执行步骤、输出与评估。

—— A. Full Pipeline（Ours，量化latent能量，含代码导出）
- 目的：作为基准，生成分割与代码侧信息以支持聚类分析。
- 改动点：使用现有 video_action_segmenter/params_d01_quant.yaml（segmentation.enable=true、energy.source=quantized）。将 segmentation.output_dir 改为本实验子目录（通过复制 YAML 再修改）。
- 执行：
  1) 复制配置
     - 拷贝 video_action_segmenter/params_d01_quant.yaml 为 video_action_segmenter/params_d01_quant_paper.yaml，并仅修改：
       - segmentation.output_dir: /home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/segmentation
       - energy.jsonl_path 保持不变（实际输出会随 seg_enable 重定向至每视频子目录）
  2) 在线推理与分割（自动按 report.threshold）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python -m video_action_segmenter.stream_inference \
  --params video_action_segmenter/params_d01_quant_paper.yaml
````
     </augment_code_snippet>
  3) 分割评估（F1@2s + mAP）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/segmentation \
  --gt-dir   /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --tolerance-sec 2.0 \
  --iou-thrs 0.5 0.75 \
  --output /home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/seg_eval.json
````
     </augment_code_snippet>
  4) ICSS 评估（CLIP ViT-B/32，语义一致性）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/vlm_icss_evaluation.py \
  --data-roots /home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/segmentation \
  --out-dir   /home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/icss \
  --clip-model-dir /home/johnny/action_ws/clip-vit-base-patch32 \
  --device cuda --k-min 3 --k-max 10 --metric cosine \
  --seq-d-model 256 --seq-n-layers 2 --seq-n-heads 4 --seq-pooling mean \
  --sample-n 8 --mid-cap 12 --max-cap 16 --sampling-mode cap \
  --weight-by-norm --cluster-pairs-cap 100000 --baseline-R 5 --seed 42
````
     </augment_code_snippet>
- 输出：segmentation/{video}/segmented_videos 与 code_indices；icss/{summary.json, cluster_icss_stats.csv, segments_with_clusters.csv, labels.npy, clip_features.npy, figs/*}
- 评估数据回填：F1@2s_mean、mAP@{0.5,0.75} 来自 seg_eval.json；ICSS 来自 icss/summary.json（字段 icss_overall.overall_mean 等，详见“数据提取与回填方案”）。

—— B. 信号源消融：Pre-Quant Latents（prequant + token_diff）
- 目的：验证去量化前的连续 latent 是否弱于量化后表示。
- 改动点：复制 A 的 YAML，改 energy.source=prequant；建议分两步离线：先导出能量，再用 GT 搜阈值，再离线分割（保证阈值公平）。
- 执行：
  1) 复制配置为 params_d01_prequant_paper.yaml，修改：
     - energy.source: prequant
     - segmentation.enable: false（仅导出能量）；
     - 保留 stride/target_fps，与 A 一致；
  2) 在线遍历导出能量（每视频生成 stream_energy_prequant_token_diff_l2_mean.jsonl）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python -m video_action_segmenter.stream_inference \
  --params video_action_segmenter/params_d01_prequant_paper.yaml
````
     </augment_code_snippet>
     导出的 JSONL 实际位于（segmentation 关闭时）：/home/johnny/action_ws/datasets/output/energy_sweep_out/D01/{video}/
  3) 基于 GT 搜索阈值并保存：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python tools/threshold_search_with_gt.py \
  --tolerance-sec 2.0 \
  --view D01 \
  # 预量化能量所在根目录
  --energy-root /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --source prequant --mode token_diff_l2_mean \
  --output /home/johnny/action_ws/datasets/output/paper_ablation_study/exp02_prequant/best_threshold_prequant.json
````
     </augment_code_snippet>
  4) 离线分割并写入统一输出目录：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python tools/segment_from_energy.py \
  --view D01 \
  --energy-root /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
  --threshold-json /home/johnny/action_ws/datasets/output/paper_ablation_study/exp02_prequant/best_threshold_prequant.json \
  --output-root /home/johnny/action_ws/datasets/output/paper_ablation_study/exp02_prequant/segmentation \
  --source prequant --mode token_diff_l2_mean --target-fps 10 --stride 4
````
     </augment_code_snippet>
  5) 评估（同 A，pred-root 指向本实验 segmentation）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/paper_ablation_study/exp02_prequant/segmentation \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --tolerance-sec 2.0 \
  --iou-thrs 0.5 0.75 \
  --output /home/johnny/action_ws/datasets/output/paper_ablation_study/exp02_prequant/seg_eval.json
````
     </augment_code_snippet>
  6) ICSS 评估（同 A，指向本实验 segmentation）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/vlm_icss_evaluation.py \
  --data-roots /home/johnny/action_ws/datasets/output/paper_ablation_study/exp02_prequant/segmentation \
  --out-dir   /home/johnny/action_ws/datasets/output/paper_ablation_study/exp02_prequant/icss \
  --clip-model-dir /home/johnny/action_ws/clip-vit-base-patch32 \
  --device cuda --k-min 3 --k-max 10 --metric cosine \
  --seq-d-model 256 --seq-n-layers 2 --seq-n-heads 4 --seq-pooling mean \
  --sample-n 8 --mid-cap 12 --max-cap 16 --sampling-mode cap \
  --weight-by-norm --cluster-pairs-cap 100000 --baseline-R 5 --seed 42
````
     </augment_code_snippet>

- 输出：exp02_prequant/ 下含 best_threshold_prequant.json、segmentation/* 与评估 JSON

—— C. 信号源消融：Raw Velocities（velocity + token_diff）
- 目的：验证在速度域进行能量检测的劣势。
- 改动点：复制 A 的 YAML，改 energy.source=velocity，segmentation.enable=false，仅导出能量；其余与 B 相同流程。
- 执行（给出与 B 对称的关键命令）：
  - 在线导出能量：与 B 的第 2 步相同，仅改 params 指向 params_d01_vel_paper.yaml（energy.source=velocity）。
  - 搜索阈值：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python tools/threshold_search_with_gt.py \
  --tolerance-sec 2.0 \
  --view D01 \
  --energy-root /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --source velocity --mode token_diff_l2_mean \
  --output /home/johnny/action_ws/datasets/output/paper_ablation_study/exp03_velocity/best_threshold_velocity.json
````
     </augment_code_snippet>
  - 离线分割：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python tools/segment_from_energy.py \
  --view D01 \
  --energy-root /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
  --threshold-json /home/johnny/action_ws/datasets/output/paper_ablation_study/exp03_velocity/best_threshold_velocity.json \
  --output-root /home/johnny/action_ws/datasets/output/paper_ablation_study/exp03_velocity/segmentation \
  --source velocity --mode token_diff_l2_mean --target-fps 10 --stride 4
````
     </augment_code_snippet>
  - 评估：同 B，改 pred-root 与 output 路径。
  - ICSS 评估：同 A/B，指向本实验 segmentation。

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/vlm_icss_evaluation.py \
  --data-roots /home/johnny/action_ws/datasets/output/paper_ablation_study/exp03_velocity/segmentation \
  --out-dir   /home/johnny/action_ws/datasets/output/paper_ablation_study/exp03_velocity/icss \
  --clip-model-dir /home/johnny/action_ws/clip-vit-base-patch32 \
  --device cuda --k-min 3 --k-max 10 --metric cosine \
  --seq-d-model 256 --seq-n-layers 2 --seq-n-heads 4 --seq-pooling mean \
  --sample-n 8 --mid-cap 12 --max-cap 16 --sampling-mode cap \
  --weight-by-norm --cluster-pairs-cap 100000 --baseline-R 5 --seed 42
````
     </augment_code_snippet>


—— D. 表征消融：w/o M_θ（使用光流能量作为非专用表征替代，分割指标）
- 目的：在不依赖 Motion Tokenizer 的前提下，使用传统光流能量实现动作分割，作为“无专用表征”的替代基线。
- 执行：

  1) 批量计算光流能量（Dual TV-L1）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python -m video_action_segmenter.scripts.compute_optical_flow_energy \
  --view D01 \
  --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D01 \
  --output-root /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
  --target-fps 10 --ema-alpha 0.7 --resize-shorter 480
````
     </augment_code_snippet>
  2) 基于 GT 搜索阈值并离线分割到本实验目录：

     <augment_code_snippet mode="EXCERPT">
````bash
# 阈值
conda run -n laps python tools/threshold_search_with_gt.py \
  --tolerance-sec 2.0 \
  --view D01 \
  --energy-root /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --source optical_flow --mode mag_mean \
  --output /home/johnny/action_ws/datasets/output/paper_ablation_study/exp04_no_mtheta_of/best_threshold_optflow.json
# 分割
conda run -n laps python tools/segment_from_energy.py \
  --view D01 \
  --energy-root /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
  --threshold-json /home/johnny/action_ws/datasets/output/paper_ablation_study/exp04_no_mtheta_of/best_threshold_optflow.json \
  --output-root /home/johnny/action_ws/datasets/output/paper_ablation_study/exp04_no_mtheta_of/segmentation \
  --source optical_flow --mode mag_mean --target-fps 10 --stride 4
````
     </augment_code_snippet>
  3) 评估：同 A，pred-root 指向 exp04_no_mtheta_of/segmentation。
  4) ICSS 评估（同 A/B/C，指向本实验 segmentation）：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/vlm_icss_evaluation.py \
  --data-roots /home/johnny/action_ws/datasets/output/paper_ablation_study/exp04_no_mtheta_of/segmentation \
  --out-dir   /home/johnny/action_ws/datasets/output/paper_ablation_study/exp04_no_mtheta_of/icss \
  --clip-model-dir /home/johnny/action_ws/clip-vit-base-patch32 \
  --device cuda --k-min 3 --k-max 10 --metric cosine \
  --seq-d-model 256 --seq-n-layers 2 --seq-n-heads 4 --seq-pooling mean \
  --sample-n 8 --mid-cap 12 --max-cap 16 --sampling-mode cap \
  --weight-by-norm --cluster-pairs-cap 100000 --baseline-R 5 --seed 42
````
     </augment_code_snippet>
—— D(b). 表征消融：w/o M_θ（CLIP特征+ICSS，聚类指标）
- 目的：以通用视觉特征（CLIP ViT-B/32）替代 Motion Tokenizer，评估“无专用表征”下的语义一致性（ICSS）。
- 前置：优先使用 A 实验的分割片段作为输入：/home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/segmentation
- 输出目录：/home/johnny/action_ws/datasets/output/paper_ablation_study/exp04b_no_mtheta_clip/
- 执行：
  1) 提取片段CLIP特征并计算 ICSS：

     <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/vlm_icss_evaluation.py \
  --data-roots /home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/segmentation \
  --out-dir   /home/johnny/action_ws/datasets/output/paper_ablation_study/exp04b_no_mtheta_clip \
  --clip-model-dir /home/johnny/action_ws/clip-vit-base-patch32 \
  --device cuda --k-min 3 --k-max 10 --metric cosine \
  --seq-d-model 256 --seq-n-layers 2 --seq-n-heads 4 --seq-pooling mean \
  --sample-n 8 --mid-cap 12 --max-cap 16 --sampling-mode cap \
  --weight-by-norm --cluster-pairs-cap 100000 --baseline-R 5 --seed 42
````
     </augment_code_snippet>
- 输出：exp04b_no_mtheta_clip/{summary.json, cluster_icss_stats.csv, segments_with_clusters.csv, labels.npy, clip_features.npy, figs/*}
- 评估数据回填：ICSS 数值来自 summary.json.icss_overall.overall_mean；差值Δ= overall_mean - baseline.mean_of_means。



—— E. 编码器消融：w/o Transformer（Mean-pool，仅 ICSS）
- 目的：验证无时序建模（mean 池化）相较序列模型的语义一致性（ICSS）下降。
- 前置：使用 A 实验生成的分割片段作为输入：/home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/segmentation
- 执行：
  1) 以 mean 池化替代时序编码，计算 ICSS：

  <augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/vlm_icss_evaluation.py \
  --data-roots /home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/segmentation \
  --out-dir   /home/johnny/action_ws/datasets/output/paper_ablation_study/exp05_no_transformer/icss \
  --clip-model-dir /home/johnny/action_ws/clip-vit-base-patch32 \
  --device cuda --k-min 3 --k-max 10 --metric cosine \
  --seq-d-model 256 --seq-n-layers 2 --seq-n-heads 4 --seq-pooling mean \
  --sample-n 8 --mid-cap 12 --max-cap 16 --sampling-mode cap \
  --weight-by-norm --cluster-pairs-cap 100000 --baseline-R 5 --seed 42
````
  </augment_code_snippet>
- 输出：/home/johnny/action_ws/datasets/output/paper_ablation_study/exp05_no_transformer/icss/ 下包含 summary.json、cluster_icss_stats.csv、segments_with_clusters.csv、labels.npy、clip_features.npy、figs/*
- 评估数据回填：ICSS 来自 summary.json.icss_overall.overall_mean；可选报告Δ与随机基线差值：overall_mean - baseline.mean_of_means。

### 3) 实验执行顺序与并行建议
- 顺序建议：先 A（生成 code_indices 供 E 使用），再并行跑 B/C/D（独立）；E 仅依赖 A。
- 并行：同一 GPU 资源允许下，B/C/D 的能量导出可并行；评估脚本（CPU/轻量）也可并行。

### 4) 数据提取与回填方案
- 分割指标（F1@2s、mAP）：来自 eval_segmentation.py 的输出 JSON。
  - 关键字段：summary.F1@2.0s_mean、summary.Precision@2.0s_mean、summary.Recall@2.0s_mean、summary.mAP@0.5、summary.mAP@0.75。
  - 样例路径：/home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/seg_eval.json
  - ICSS 样例路径：/home/johnny/action_ws/datasets/output/paper_ablation_study/exp01_full_quant/icss/summary.json；/home/johnny/action_ws/datasets/output/paper_ablation_study/exp04b_no_mtheta_clip/summary.json

  - ICSS                     
- 回填到 ablation_study_exp.md：按实验行更新“Seg. F1”与“ICSS”。
- 如何定位结果：
  - 分割：直接打开对应 expXX_*/seg_eval.json；或用 jq/python 读取 summary 字段。
  - ICSS：打开对应实验的 icss/summary.json（或 exp04b_no_mtheta_clip/summary.json），提取字段：icss_overall.overall_mean、icss_overall.overall_std、baseline.mean_of_means；可计算 Δ = overall_mean - baseline.mean_of_means。

### 5) 验证检查清单（运行完每项后逐条自查）
- 输出目录结构：paper_ablation_study/ 下每个实验均有 segmentation 子目录（如适用）与 ICSS 输出（icss/summary.json；对 exp04b_no_mtheta_clip 则为根目录下 summary.json）。
- 分割 JSON 命名：{video}/{segmented_videos}/{video}_segments.json 格式正确；必要时存在 {video}/stream_energy_*.jsonl 供置信度计算。
- 评估：seg_eval.json 存在且 summary 字段完整。
- 阈值：B/C/D 的 best_threshold_*.json 存在且被 segment_from_energy.py 正确引用。
- ICSS：各实验的 icss/summary.json 存在且包含顶层字段 config、clustering、icss_overall、baseline；cluster_icss_stats.csv 与 segments_with_clusters.csv 存在；figs/ 下生成 KDE/CI/综合图。

### 6) 备注与注意
- 所有命令均基于现有脚本与路径，未引入新依赖；如遇到 opencv-contrib 缺失（光流），需在 laps 环境中安装后再运行（本计划不执行安装，仅提示）。
- 本计划不执行实验，仅提供可直接落地的指令与路径；必要的 YAML 仅做“复制后少量字段修改”。
