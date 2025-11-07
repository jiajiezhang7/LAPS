# 实验方案与进度追踪（D01/D02）

更新时间：请在每次进展后更新。所有命令均在 conda 环境 laps 下运行（OTAS 在独立的 otas_env 下）。

全局约定：
- 根目录：/home/johnny/action_ws （本文档路径：docs/experiment_detailed_todo.md）
- 视角数据：
  - D01：/home/johnny/action_ws/datasets/gt_raw_videos/D01
  - D02：/home/johnny/action_ws/datasets/gt_raw_videos/D02
- 模型权重：各自 params.yaml 的 checkpoint_path 已确认
- GPU：1 块（ID=0）；允许批处理（单卡并发建议=1）
- 输出组织（统一实验根）：/home/johnny/action_ws/datasets/output
  - energy_sweep_out/{D01,D02}
  - energy_sweep_report/{D01,D02}
  - segmentation_outputs/{D01,D02}
  - umap_vis/{figure,statistics}
  - stats/{seg_eval,icss}
  - tables/{table1,table2,table3,table4}
  - figures/{figure3,figure5,...}

- 存储迁移与符号链接（重要）：
  - /home/johnny/action_ws/datasets/output → 符号链接 → /media/johnny/48FF-AA60/output
  - /home/johnny/action_ws/comapred_algorithm/OTAS/data → 符号链接 → /media/johnny/48FF-AA60/OTAS/data
  - /home/johnny/action_ws/comapred_algorithm/ABD/i3d_features → 符号链接 → /media/johnny/48FF-AA60/ABD/i3d_features
  - 说明：文档与脚本中的路径仍以 /home/johnny/action_ws/... 为准（符号链接透明生效）。若需直接访问外部盘，请替换为右侧实际路径。

说明：为避免反复改 YAML，建议为不同视角与能量源复制参数文件（最小侵入，仅改少量字段）：
- video_action_segmenter/params_d01_quant.yaml
- video_action_segmenter/params_d01_vel.yaml
- video_action_segmenter/params_d02_quant.yaml
- video_action_segmenter/params_d02_vel.yaml
（由 AI 创建，确保 energy.source / energy.jsonl_path / segmentation.output_dir / input.dir 正确）

---
## 总体进度快照
- 阶段0（环境准备）：[x] 已完成
- 阶段1 Q1（能量信号）：
  - quantized 能量：[x] 已完成
  - velocity 能量：[x] 已完成
  - optical_flow 能量：[x] 已完成（TV-L1）
- 阶段2（LAPS 分割）：[x] 已完成（D01/D02 各 6/6）
- 阶段3 Q2（评估）：[ ] 部分待实现（LAPS 评估/Optical Flow baseline/OTAS）

---


## 阶段0 环境与数据（准备/验证）
目标：确认输入/输出路径、环境可用；设置独立输出目录，避免覆盖。

输入：
- D01/D02 输入根目录（已确认）
- Motion Tokenizer checkpoint（已确认）

操作与命令：
1) 快速冒烟（不保存分割，验证推理与写能量 JSONL 路径权限）
- 确保 YAML 中 device: "cuda" 或 "auto"；input.batch.gpu_ids: [0]
- D01（量化能量）
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d01_quant.yaml
- D02（量化能量）
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d02_quant.yaml

预期输出与检查点：
- 能量 JSONL 文件可写入（见各 YAML 的 energy.jsonl_path）
- 运行日志无错误；GPU 正常占用；速度与显存在可接受范围

---

## 阶段1 Q1：能量信号有效性（quantized vs velocity）
目标：在 D01/D02 上分别导出 quantized/token_diff 与 velocity/token_diff 的能量 JSONL，做统计分析与论文风格可视化，产出阈值报告。
注意（两阶段流程）：
- 第一轮（仅能量）：禁用分割（segmentation.enable=false），分别导出 quantized 与 velocity 的能量 JSONL，不读取 report/threshold
- 第二轮（分割）：仅对 quantized 源执行在线分割，使用阶段1生成的 best_threshold_quantized_token_diff.json（segmentation.mode=report）


输入：
- D01 YAML：/home/johnny/action_ws/video_action_segmenter/params_d01_quant.yaml、/home/johnny/action_ws/video_action_segmenter/params_d01_vel.yaml
- D02 YAML：/home/johnny/action_ws/video_action_segmenter/params_d02_quant.yaml、/home/johnny/action_ws/video_action_segmenter/params_d02_vel.yaml
- 建议 energy.jsonl_path：
注意：实际输出为每视频一份 JSONL。\n- segmentation.enable=false 时：/home/johnny/action_ws/datasets/output/energy_sweep_out/{VIEW}/{video_name}/stream_energy_{source}_{mode}.jsonl\n- segmentation.enable=true 时：/home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW}/{video_name}/stream_energy_{source}_{mode}.jsonl\n
  - D01 quant：/home/johnny/action_ws/datasets/output/energy_sweep_out/D01/stream_energy_quantized_token_diff_l2_mean.jsonl
  - D01 vel：/home/johnny/action_ws/datasets/output/energy_sweep_out/D01/stream_energy_velocity_token_diff_l2_mean.jsonl
  - D02 quant：/home/johnny/action_ws/datasets/output/energy_sweep_out/D02/stream_energy_quantized_token_diff_l2_mean.jsonl
  - D02 vel：/home/johnny/action_ws/datasets/output/energy_sweep_out/D02/stream_energy_velocity_token_diff_l2_mean.jsonl

操作与命令：
1) 生成能量 JSONL（四次运行）
- D01 quantized
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d01_quant.yaml
- D01 velocity
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d01_vel.yaml
- D02 quantized
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d02_quant.yaml
- D02 velocity
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d02_vel.yaml

2) 分析与报告（分别对 D01、D02）
注：quantized 的 JSONL 位于 segmentation_outputs/{VIEW}/{video_name}/；velocity 的 JSONL 位于 energy_sweep_out/{VIEW}/{video_name}/。建议分别对两者运行 analyze_energy_jsonl 生成各自报告（本次已执行）。

- D01 分析报告
  conda run -n laps python -m video_action_segmenter.analyze_energy_jsonl \
    --input-dir /home/johnny/action_ws/datasets/output/energy_sweep_out/D01 \
    --output-dir /home/johnny/action_ws/datasets/output/energy_sweep_report/D01 \
    --title "Energy Analysis D01"
- D02 分析报告
  conda run -n laps python -m video_action_segmenter.analyze_energy_jsonl \
    --input-dir /home/johnny/action_ws/datasets/output/energy_sweep_out/D02 \
    --output-dir /home/johnny/action_ws/datasets/output/energy_sweep_report/D02 \
    --title "Energy Analysis D02"

3) 阈值自动搜索（以 velocity 生成伪标签，推荐 quantized 的阈值）
- D01 阈值
  conda run -n laps python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl /home/johnny/action_ws/datasets/output/energy_sweep_out/D01/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl /home/johnny/action_ws/datasets/output/energy_sweep_out/D01/stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --output-json /home/johnny/action_ws/datasets/output/energy_sweep_report/D01/best_threshold_quantized_token_diff.json
- D02 阈值
  conda run -n laps python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl /home/johnny/action_ws/datasets/output/energy_sweep_out/D02/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl /home/johnny/action_ws/datasets/output/energy_sweep_out/D02/stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --output-json /home/johnny/action_ws/datasets/output/energy_sweep_report/D02/best_threshold_quantized_token_diff.json

4) 论文风格曲线（可选）
- 示例（D01 quantized）：
  conda run -n laps python -m video_action_segmenter.scripts.plot_energy_segment_from_jsonl \
    --jsonl /home/johnny/action_ws/datasets/output/energy_sweep_out/D01/stream_energy_quantized_token_diff_l2_mean.jsonl \\
    --params /home/johnny/action_ws/video_action_segmenter/params_d01_quant.yaml \
    --segment-length 2000 --seed 42

预期输出与检查点：
- energy_sweep_out/{D01,D02}/*.jsonl 四个文件存在
- energy_sweep_report/{D01,D02}/*（CSV/JSON/图/HTML）存在
- best_threshold_quantized_token_diff.json 含 best_j/best_f1 的 thr 字段


### 补充：Optical Flow 基线（TV-L1）
- 算法选择确认：仅使用 OpenCV Dual TV-L1 作为 optical flow 算法（不使用 Farneback/RAFT/FlowFormer 等）。
- 三路对比关系：quantized（主方法） vs optical_flow（传统光流基线） vs velocity（稳健运动信号补充）。

技术细节：
- 能量定义：逐帧计算 TV-L1 光流 (u,v)，幅值 mag = sqrt(u^2 + v^2)，对全图取均值，作为该帧/窗口的能量值。
- 帧率对齐：target_fps=10（与现有 pipeline 一致）。
- 平滑策略：EMA 平滑（alpha 待定，建议与 quantized/velocity 保持一致）。
- 输出 JSONL 字段：source="optical_flow"，mode="mag_mean"，window，energy。
- 存储路径：/home/johnny/action_ws/datasets/output/energy_sweep_out/{VIEW}/{video_name}/stream_energy_optical_flow_mag_mean.jsonl

待办（Q1-OpticalFlow）：
- [x] 实现 compute_optical_flow_energy.py 脚本（TV-L1）。
- [x] 为 D01 和 D02 全量生成 optical_flow 能量 JSONL 文件。
- [x] 更新能量分析报告，纳入 optical_flow 数据。
- [x] 生成 Q1 论文所需的 60 秒代表性片段可视化图（E_action vs optical_flow + GT 边界叠加）。

已完成（Q1 相关）：
- [x] quantized 与 velocity 的能量 JSONL 已生成并统一存放在 energy_sweep_out/{VIEW}/{video_name}/。
- [x] D01/D02 的 quant 与 velocity 分析报告已生成。
- [x] D01/D02 的 阈值 JSON 已生成并用于第二轮分割。

- [x] D01/D02 各 6 个视频的 optical_flow 能量 JSONL 已生成；参数：target_fps=10，ema_alpha=0.7，resize_shorter=480；算法：OpenCV Dual TV-L1；完成时间：2025-11-06。
- [x] 能量分析报告已更新，包含 optical_flow 数据（路径：/home/johnny/action_ws/datasets/output/energy_sweep_report/{D01,D02}/report.html）。

---

## 阶段2 Q2：无监督动作分割性能（与 OTAS 对比）
目标：读取阶段1阈值进行在线分割，导出片段与 codes，并用人工 GT 评估 F1@2s 与类无关 mAP@IoU。

- 进展：LAPS（quantized）分割已完成（D01 6/6，D02 6/6）。

输入：
- 分别使用 D01/D02 的 best_threshold_quantized_token_diff.json
- YAML：params_d01_quant.yaml / params_d02_quant.yaml（segmentation.mode=report，report_path 指向上面的 JSON）
- GT（见“GT 标注规范”）

操作与命令：
1) 在线分割并导出
- 建议 segmentation.output_dir：
  - D01：/home/johnny/action_ws/datasets/output/segmentation_outputs/D01
  - D02：/home/johnny/action_ws/datasets/output/segmentation_outputs/D02
- D01：
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d01_quant.yaml
- D02：
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params /home/johnny/action_ws/video_action_segmenter/params_d02_quant.yaml
（输出结构：{video}/segmented_videos/*.mp4 与 code_indices/*.codes.json）

2) 评估（AI 将新增工具，不改核心）
- 目标：类无关 mAP@IoU=[0.5,0.75]，F1@2s
- 命令：
  conda run -n laps python tools/eval_segmentation.py \
    --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/{D01,D02} \
    --gt-dir /home/johnny/action_ws/datasets/gt_annotations/{D01,D02} \
    --iou-thrs 0.5 0.75 --tolerance-sec 2.0 \
    --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_{VIEW}.json

3) OTAS 基线（独立环境，不影响 laps）
- 环境准备（仅文档，不立即执行）：
  git clone https://github.com/yl596/OTAS.git /home/johnny/action_ws/external/OTAS
  conda create -y -n otas_env python=3.9
  conda run -n otas_env pip install -r /home/johnny/action_ws/external/OTAS/requirements.txt
- 推理与适配（AI 将编写适配脚本）：
  - 产出 OTAS 预测片段后，运行：
    conda run -n laps python tools/adapt_otas_to_segments.py \
      --otas-pred /home/johnny/action_ws/external/OTAS/outputs/{VIEW} \
      --output /home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW}_OTAS
  - 使用相同 eval_segmentation.py 评估，生成对照表（tables/table1.csv）



### ABD 基线（Q2）
目标：复现 ABD（CVPR'22，无训练）离线算法，直接读取 D01/D02 原始视频，产出与 LAPS 兼容的分割结果，用同一评估脚本比较。

数据与接口约定：
- 输入：/home/johnny/action_ws/datasets/gt_raw_videos/{D01,D02} 下各视频
- 特征：使用 I3D 特征（通过 abd_env 批量提取）；不再使用 LAPS latent（prequant/quantized）
- 时间映射：sec ≈ clip_idx × clip_stride（默认 clip_stride=0.4s）
- 输出：/home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW}_ABD/{video_stem}/segmented_videos/{video_stem}_segments.json
  - 段 JSON 元数据包含：video、segments[{start_sec,end_sec}]、video_duration_sec、segmentation_params（{source:"i3d", alpha, k, stride, target_fps, view, clip_duration, clip_stride}）、processed_at

运行：
- D01：
  conda run -n laps python -m comapred_algorithm.ABD.run_abd \
    --view D01 \
    --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D01 \
    --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_ABD \
    --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/i3d_features/D01 \
    --feature-source i3d --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4
- D02：
  conda run -n laps python -m comapred_algorithm.ABD.run_abd \
    --view D02 \
    --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D02 \
    --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_ABD \
    --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/i3d_features/D02 \
    --feature-source i3d --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4

评估：与 LAPS/OTAS 使用相同脚本
  conda run -n laps python tools/eval_segmentation.py \
    --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW}_ABD \
    --gt-dir /home/johnny/action_ws/datasets/gt_annotations/{VIEW} \
    --iou-thrs 0.5 0.75 --tolerance-sec 2.0 \
    --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_{VIEW}_ABD.json

环境：
- 推荐复用 laps 环境（ABD 仅依赖 numpy/scipy）；如后续引入额外特征提取器，可新建 abd_env（此处仅文档记录，暂不创建）

### Optical Flow Baseline（Q2）
待办：
- [x] 实现 optical_flow vs GT 的阈值搜索与视角级聚合脚本。
- [x] 实现 optical_flow 分割 baseline runner（复用现有状态机参数：hysteresis / up_down_count / cooldown / max_duration 等）。
- [x] 对 D01 和 D02 运行 optical_flow 分割。
- [x] 计算 optical_flow baseline 的 mAP@IoU 与 F1@2s 指标。
- [ ] 生成 Table 1 对比表格（LAPS vs Optical Flow Baseline vs OTAS）。

预期输出与检查点：
- segmentation_outputs/{D01,D02} 下有分割视频与 codes.json
- stats/seg_eval/seg_eval_{VIEW}.json（本方法与 OTAS 各一份）存在且指标合理
- tables/table1.csv 生成

---

## 阶段3 Q3：原语语义一致性（聚类/可视化/VLM）
目标：段级聚类与 UMAP 可视化，计算聚类质量指标；用 CLIP 提取帧嵌入，计算簇内语义相似度（ICSS）。

输入：
- segmentation_outputs/{D01,D02}

操作与命令：
1) 无训练聚合 + UMAP + 聚类指标
- 示例（以 D01 为例；D02 同理修改 --data-dir）：
  conda run -n laps python /home/johnny/action_ws/umap_vis/scripts/segment_umap_cluster_analysis.py \
    --data-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D01 \
    --output-dir /home/johnny/action_ws/datasets/output/umap_vis/figure \
    --aggs attn_norm mean mean_std first_last max \
    --k-min 2 --k-max 10 --metric cosine

2) 冻结序列模型（轻量 Transformer）
  conda run -n laps python /home/johnny/action_ws/umap_vis/scripts/sequence_model_embedding.py \
    --data-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D01 \
    --fig-dir /home/johnny/action_ws/datasets/output/umap_vis/figure \
    --stats-dir /home/johnny/action_ws/datasets/output/umap_vis/statistics \
    --use-best-grid-config --metric cosine --k-min 2 --k-max 10

3) 语义一致性（CLIP ICSS；AI 新增工具）
  conda run -n laps python tools/icss_eval.py \
    --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW} \
    --sample-per-seg 3 --clip-backbone ViT-B-32 \
    --output /home/johnny/action_ws/datasets/output/stats/icss/icss_{VIEW}.json

预期输出与检查点：
- /home/johnny/action_ws/datasets/output/umap_vis/figure 下生成 UMAP 图；/home/johnny/action_ws/datasets/output/umap_vis/statistics 下生成聚类指标 CSV
- /home/johnny/action_ws/datasets/output/stats/icss/icss_{VIEW}.json 存在
- /home/johnny/action_ws/datasets/output/tables/table2.csv（聚类指标汇总）、/home/johnny/action_ws/datasets/output/tables/table3.csv（ICSS）生成

---

## 阶段4 消融实验（能量/平滑/门控/状态机/聚合/序列模型）
目标：验证设计选择的稳定性与优势；自动化批跑与汇总。

操作与命令（AI 新增最小侵入工具）：
- 能量/平滑/门控/状态机 sweep：
  conda run -n laps python tools/sweep_energy.py \
    --views D01 D02 --sources prequant quantized velocity \
    --modes l2_mean token_diff_l2_mean \
    --smoothing "ema:0.5,0.7|ma:3,5" \
    --gate on off --hysteresis 0.9 0.95 \
    --output /home/johnny/action_ws/datasets/output/tables/table4.csv
- 聚合与序列模型 sweep：
  conda run -n laps python tools/sweep_seg.py \
    --data-roots /home/johnny/action_ws/datasets/output/segmentation_outputs/D01 \
    --aggs attn_norm mean mean_std first_last \
    --model-grid "d=128,256;n_layers=2,4;n_heads=2,4" \
    --output /home/johnny/action_ws/datasets/output/tables/table4_seq.csv

预期输出与检查点：
- tables/table4*.csv 存在；绘制趋势图（figures/*）可复现

---

## GT 标注规范（Q2 评估）
目标：类无关的“动作段”边界，用于 IoU/F1 评估。建议范围兼顾多样性与可行性。

建议标注规模：
- 每个视角（D01、D02）建议 60 分钟（理想），至少 30 分钟（最小可行）
- 优先选择 6 个视频文件 × 每个 10 分钟（或 3 个 × 10 分钟，最小可行）
- 覆盖不同场景/工况/时间段，包含静止—运动—静止的多样模式

标注粒度与规则：
- 段定义：同一原语/动作持续的时间区间（可类无关，只需边界）；
- 最小持续：建议 ≥ 1.0s，短于 1s 的碎片可合并到相邻段；
- 时间单位：秒（以原视频时间线）；
- 文件命名：D01_{video_stem}_segments.json 或 D02_{video_stem}_segments.json（与输入视频同名 stem）
- 存放路径：/home/johnny/action_ws/datasets/gt_annotations/{D01,D02}
- JSON 示例：
  {
    "video": "D01_xxx.mp4",
    "segments": [
      {"start_sec": 12.0, "end_sec": 15.6},
      {"start_sec": 30.2, "end_sec": 33.1}
    ]
  }

注意：
- 评估时按类无关匹配（不需要标签名），IoU 与 2s 容差匹配；
- 预测段的“置信度”将用段内平滑能量均值（用于 AP 排序）；
- 窗口到秒的换算：sec ≈ window_idx * stride / target_fps（默认 stride=4, target_fps=10）。

---

## 详细 ToDoList（按阶段组织）

### 阶段0
- [x] 配置输入目录 D01/D02（User 完成）
- [x] 确认 Motion Tokenizer checkpoint 路径（User 完成）
- [x] 为 D01/D02 复制量化/速度 YAML（AI 完成；产出：params_d01_quant.yaml、params_d01_vel.yaml、params_d02_quant.yaml、params_d02_vel.yaml）
- [x] 设置 segmentation.output_dir 为本次实验专用路径（AI 完成；已更新至 /home/johnny/action_ws/datasets/output/segmentation_outputs/{D01,D02}）
- [x] 环境冒烟跑通（AI；依赖：GPU 可用；产出：日志 + 能量 JSONL）

### 阶段1（Q1）
- [x] 运行 D01 quant 生成 JSONL（AI；产出：energy_sweep_out/D01/quantized_*_token_diff_l2_mean.jsonl）
- [x] 运行 D02 quant 生成 JSONL（AI；产出：energy_sweep_out/D02/quantized_*_token_diff_l2_mean.jsonl）
- [x] 运行 D01 vel 生成 JSONL（AI；完成；产出：energy_sweep_out/D01/velocity_*_token_diff_l2_mean.jsonl）
- [x] 运行 D02 vel 生成 JSONL（AI；完成；产出：energy_sweep_out/D02/velocity_*_token_diff_l2_mean.jsonl）
- [x] 生成 D01 分析报告（AI；依赖：JSONL；产出：report/HTML/CSV/图）
- [x] 生成 D02 分析报告（AI；依赖：JSONL；产出：report/HTML/CSV/图）
- [x] 计算 D01 阈值 JSON（AI；依赖：quant/vel JSONL）
- [x] 计算 D02 阈值 JSON（AI；依赖：quant/vel JSONL）
- [x] 可选绘图（AI）
- [x] 计算 D01/D02 最佳阈值 JSON（AI；依赖：两类 JSONL；产出：best_threshold*.json）
- [x] 生成论文风格曲线（AI；可选；产出：figures/*）

Optical Flow（TV-L1）待办：
- [x] 实现 compute_optical_flow_energy.py（TV-L1）。
- [x] 生成 D01/D02 optical_flow JSONL。
- [x] 更新 Q1 报告包含 optical_flow。
- [x] 生成 60 秒叠图（E_action vs optical_flow + GT）。

### 阶段2（Q2）
- [x] 在线分割导出 D01（AI；依赖：best_threshold；产出：segmented_videos+codes）
- [x] 在线分割导出 D02（AI；依赖：best_threshold；产出：segmented_videos+codes）
- [x] 编写 tools/eval_segmentation.py（AI；依赖：GT；产出：seg_eval_{VIEW}.json + tables/table1.csv）
- [x] OTAS 环境搭建与推理（AI；依赖：仓库；产出：OTAS outputs）
- [x] 编写 tools/adapt_otas_to_segments.py 并评估（AI；依赖：OTAS outputs；产出：对照结果）

Optical Flow Baseline（Q2）待办：
- [x] 实现 optical_flow vs GT 的阈值搜索与视角级聚合脚本。
- [x] 实现 optical_flow 分割 baseline runner（复用当前状态机参数）。
- [x] 运行 D01/D02 optical_flow 分割。
- [x] 计算 mAP@IoU 与 F1@2s。
- [] 生成 Table 1（LAPS vs Optical Flow vs OTAS）。

### 阶段3（Q3）
- [x] 运行 segment_umap_cluster_analysis.py（AI；依赖：segmentation_outputs；产出：UMAP 图 + 指标 CSV）
- [x] 运行 sequence_model_embedding.py（AI；依赖：segmentation_outputs；产出：UMAP + 指标 CSV）
- [ ] 编写 tools/icss_eval.py（AI；依赖：CLIP；产出：icss_{VIEW}.json + tables/table3.csv）

### 阶段4（消融）
- [ ] 编写 tools/sweep_energy.py / tools/sweep_seg.py（AI；依赖：阶段1/2/3 管线；产出：table4*.csv + figures/*）

---

## 进度追踪
- 当前状态：阶段2 已完成（D01/D02 quantized 在线分割完成：各 6/6）；阶段1（quantized/velocity/optical_flow）已完成；阶段3（Q2 评估）部分待做（LAPS 评估 / Optical Flow baseline / OTAS）
- 已完成里程碑：
  - 输入/输出路径更新并同步至文档；输入目录 D01/D02 确认；checkpoint 路径确认；GPU/并行策略确认；YAML 生成与输出目录写权限验证
- OTAS 进展（在 otas 环境执行）：
  - [x] 新建 conda 环境 otas 并安装依赖（torch/torchvision/opencv/numpy/scipy/pandas/tqdm/Pillow）
  - [x] 按 OTAS 目录约定建立/链接视频：data/breakfast/videos/{D01,D02}/{D01,D02}/*.mp4
  - [x] D01 全量抽帧至 data/breakfast/frames/{P}_{cam}_{act}/Frame_%06d.jpg（已验证样例目录存在且帧数正常）
  - [x] 生成/更新 data/breakfast/video_info.pkl（当前包含 D01 条目；D02 将在其抽帧后合并）
  - [x] 通过 make_video_info_from_frames.py 从帧目录重建（D01 共 6 条目），输出：comapred_algorithm/OTAS/data/breakfast/video_info.pkl

  - [ ] 运行 OTAS 验证推理（TF）：进行中（D01）；输出将写入 /home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf/mean_error/*.pkl（通过符号链接落到外部盘）
  - [x] 数据装载与切窗已完成（window_lists 已生成：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf/window_lists/*）

  - [ ] 边界检测 detect_bdy.py（待上一阶段完成后运行），输出 detect_seg/*.pkl（同目录根：/home/johnny/action_ws/datasets/output/otas_out/...）
  - [ ] 结果适配与评估（laps 环境）：将 OTAS pkl→segments.json 并评估 F1@2s、mAP@IoU
  - [!] 2025-11-05 重启恢复：已检查可能的中断风险，清理 window_lists/mean_error/detect_seg 后重新启动 OTAS 推理（laps 环境；--batch_size 64, --num_workers 16, --gpu 0）。预计 10–30 分钟完成，完成后将继续 detect_bdy 与结果适配/评估。

- 问题与解决：
  - [x] 本地备份目录已由用户手动强制清理（datasets/output.backup_*、OTAS/data.backup_*、ABD/i3d_features.backup_*）
  - [x] 输出路径统一为 /home/johnny/action_ws/datasets/output/otas_out（通过符号链接落到外部盘），现有脚本无需改动
- 下一步行动：
- 流程约束：阶段1第一轮仅能量不分割（segmentation.enable=false），阶段2读取阶段1阈值执行分割（quantized 源使用 mode=report）

  1) 依次运行 D01-quant → D01-vel → D02-quant → D02-vel 生成 JSONL
  2) 生成 D01/D02 分析报告与最佳阈值 JSON
  3) （可选）生成论文风格曲线图

---

## 关键决策记录（持续更新）
- GT 标注需求：每视角建议 60 分钟，最小 30 分钟；类无关边界；路径 /home/johnny/action_ws/datasets/gt_annotations/{D01,D02}；命名 D01_{video_stem}_segments.json / D02_{video_stem}_segments.json（User 已提供）
- 阈值选择：默认使用 best_j.thr（如需更偏向 F1，可切换 best_f1.thr）
- 聚类参数：UMAP metric=cosine, neighbors=15, min-dist=0.1；序列模型使用 best-grid（d=256, n_layers=4, n_heads=4, pooling=mean）
- 基线方法：OTAS 官方实现；独立环境 otas_env；统一使用相同评估脚本；预测段置信度=段内平滑能量均值

---

## 验证检查点速查
- 阶段0：stream_inference 最小跑通，无异常日志；能量 JSONL 路径可写
- 阶段1：四个 JSONL 存在；分析报告与阈值 JSON 生成；曲线图外观合理
- 阶段2：segmented_videos 与 codes.json 结构完整；评估 JSON 与 Table1 生成
- 阶段3：UMAP 图与统计 CSV 生成；ICSS JSON 与 Table3 生成
- 阶段4：Table4 与趋势图生成，可复现



---

## 阶段2 Q2：Optical Flow 基线分割与评估（完成）
- [x] 评估脚本支持 noisy 后缀 GT：tools/eval_segmentation.py 已兼容 *_segments.json 与 *_segments_noisy.json（非 noisy 优先）
- [x] 阈值搜索（公平性：仅用 GT 边界；状态机参数与 LAPS 一致：hysteresis_ratio=0.95, up=2, down=2, cooldown=1, max_dur=2s, target_fps=10, stride=4）
  - 工具：tools/threshold_search_with_gt.py
  - D01 输出：/home/johnny/action_ws/datasets/output/energy_sweep_report/D01/best_threshold_optical_flow_mag_mean.json
    - best_f1.thr=1.1571，F1@2s=0.2758，Prec=0.2284，Rec=0.3542
  - D02 输出：/home/johnny/action_ws/datasets/output/energy_sweep_report/D02/best_threshold_optical_flow_mag_mean.json
    - best_f1.thr=1.0520，F1@2s=0.2442，Prec=0.1659，Rec=0.4735
- [x] 离线分割（复用与 LAPS 一致的状态机）
  - 工具：tools/segment_from_energy.py（预测段 JSON：{output_root}/{stem}/segmented_videos/{stem}_segments.json；并复制能量 JSONL 到 {output_root}/{stem}/）
  - D01 输出根：/home/johnny/action_ws/datasets/output/segmentation_outputs/D01_optical_flow
  - D02 输出根：/home/johnny/action_ws/datasets/output/segmentation_outputs/D02_optical_flow
- [x] 评估（F1@2s/5s 与 mAP@IoU=[0.5,0.75]）
  - D01 结果：/home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_optical_flow.json
    - F1@2s=0.2766，P=0.2288，R=0.3558，mAP@0.5=0.0112，mAP@0.75=0.0007
    - F1@5s=0.3604，P=0.2981，R=0.4637

  - D02 结果：/home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_optical_flow.json
    - F1@2s=0.2435，P=0.1652，R=0.4735，mAP@0.5=0.0131，mAP@0.75=0.0023
    - F1@5s=0.2739，P=0.1858，R=0.5329


备注：上述 optical_flow 基线严格遵循公平性要求；不依赖 quantized/velocity 伪标签，仅使用 GT 时间边界用于阈值搜索；分割状态机参数与 LAPS 完全一致。


### LAPS 分割评估（noisy GT，完成）
- 评估脚本：tools/eval_segmentation.py（支持 *_segments.json 与 *_segments_noisy.json；本次按 noisy GT）
- 指标口径：F1@2s/5s 与 mAP@IoU=[0.5,0.75]

- 评估所用命令：

  conda run -n laps python tools/eval_segmentation.py --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01 --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01.json

  conda run -n laps python tools/eval_segmentation.py --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D02 --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D02 --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02.json


- D01 结果：/home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01.json

  {
  "num_videos": 6,
  "F1@2.0s_mean": 0.8111758474576272,
  "Precision@2.0s_mean": 0.79458648989899,
  "Recall@2.0s_mean": 0.8302469135802468,
  "F1@5.0s_mean": 0.8140007062146893,
  "Precision@5.0s_mean": 0.7971906565656566,
  "Recall@5.0s_mean": 0.8333333333333334,
  "mAP@0.5": 0.7554228434987613,
  "mAP@0.75": 0.3578861238546849
  }

- D02 结果：/home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02.json

  {
  "num_videos": 6,
  "F1@2.0s_mean": 0.9961685823754789,
  "Precision@2.0s_mean": 0.9925093632958801,
  "Recall@2.0s_mean": 1.0,
  "F1@5.0s_mean": 0.9961685823754789,
  "Precision@5.0s_mean": 0.9925093632958801,
  "Recall@5.0s_mean": 1.0,
  "mAP@0.5": 0.9507687428447819,
  "mAP@0.75": 0.41550276728875457
  }

### 2025-11-06 · OTAS 实验 · 任务 1（video_info.pkl 完整性校验）
- 读取：/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info.pkl
- 条目数：12
- 前 3 条：
  - comapred_algorithm/OTAS/data/breakfast/videos/D01/D01/D01_sample_1_seg001.mp4 → true_vlen=14994, lose=[]
  - comapred_algorithm/OTAS/data/breakfast/videos/D01/D01/D01_sample_2_seg001.mp4 → true_vlen=1286, lose=[]
  - comapred_algorithm/OTAS/data/breakfast/videos/D01/D01/D01_sample_2_seg002.mp4 → true_vlen=14996, lose=[]
- 路径格式校验：未通过。当前键（vpath）为 {VIEW}/{VIEW}/{stem}.mp4（如 D01/D01/D01_sample_1_seg001.mp4），与本次任务期望的 {P}/{cam}/{P}_{act}.mp4 不一致；但与文档中“按 OTAS 目录约定建立/链接视频：.../{D01,D02}/{D01,D02}/*.mp4”一致。
- 处理：按任务要求，已停止后续任务 2–4。请确认采用哪一种目录/键名规范：
  1) 继续沿用 {VIEW}/{VIEW}/{stem}.mp4（据此调整后续校验与脚本）；或
  2) 切换到 {P}/{cam}/{P}_{act}.mp4（需重建 video_info.pkl 及相关目录）。


### 2025-11-06 · OTAS 实验 · 任务 2（帧目录与 video_info 对应性校验）
- 目录：/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames
- 子目录总数：12（与 video_info.pkl 条目一致）
- 命名规范：均符合 {VIEW}_{VIEW}_{stem}/（如 D01_D01_sample_1_seg001/）
- 帧命名规范：抽查 3 个目录，均存在 Frame_000001.jpg，文件命名符合 Frame_%06d.jpg
  - D01_D01_sample_1_seg001：Frame_000001.jpg 存在；frame_count=14994
  - D01_D01_sample_2_seg001：Frame_000001.jpg 存在；frame_count=1286
  - D02_D02_sample_1_seg001：Frame_000001.jpg 存在；frame_count=15002
- 结论：任务 2 通过（无缺失与命名错误）

### 2025-11-06 · OTAS 实验 · 任务 3（OTAS 推理状态检查）
- 检查目录：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf/mean_error/
- 结果：存在且包含 12 个 .pkl 文件 → 推理已完成
- 动作：按规范跳过任务 3.3（不重复运行推理），直接进入任务 4

### 2025-11-06 · OTAS 实验 · 任务 4（detect_bdy → 适配 → 评估）
- 4.1 边界检测（laps 环境）：
  命令：
    conda run -n laps python comapred_algorithm/OTAS/code/detect_bdy.py \
      --output-path /home/johnny/action_ws/datasets/output/otas_out/ \
      --dataset BF \
      --feature_model tf
  输出检查：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf/detect_seg/
  - .pkl 数量：12（与 mean_error 对齐）

- 4.2 结果适配为 LAPS 评估格式（segments.json）：
  - D01：
    conda run -n laps python tools/adapt_otas_to_segments.py \
      --otas-pred /home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf \
      --raw-dir /home/johnny/action_ws/datasets/gt_raw_videos/D01 \
      --output /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_OTAS
    结果：写入 6 个 JSON（OK 提示），示例：D01_sample_1_seg001_segments.json（segments=104）
  - D02：
    conda run -n laps python tools/adapt_otas_to_segments.py \
      --otas-pred /home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf \
      --raw-dir /home/johnny/action_ws/datasets/gt_raw_videos/D02 \
      --output /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_OTAS
    结果：写入 6 个 JSON（OK 提示），示例：D02_sample_1_seg001_segments.json（segments=107）
  - JSON 结构校验：通过（包含 video、segments[{start_sec,end_sec}] 等字段）

- 4.3 评估（F1@2s/5s，mAP@0.5/0.75）：
  - D01：
    conda run -n laps python tools/eval_segmentation.py \
      --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_OTAS \
      --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
      --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
      --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_OTAS.json
    指标：
    {
      "num_videos": 6,
      "F1@2.0s_mean": 0.1543,
      "Precision@2.0s_mean": 0.1469,
      "Recall@2.0s_mean": 0.2220,
      "F1@5.0s_mean": 0.2720,
      "Precision@5.0s_mean": 0.2580,
      "Recall@5.0s_mean": 0.3869,
      "mAP@0.5": 0.00072,
      "mAP@0.75": 0.00001
    }
  - D02：
    conda run -n laps python tools/eval_segmentation.py \
      --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_OTAS \
      --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D02 \
      --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
      --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_OTAS.json
    指标：
    {
      "num_videos": 6,
      "F1@2.0s_mean": 0.1984,
      "Precision@2.0s_mean": 0.1489,
      "Recall@2.0s_mean": 0.3052,
      "F1@5.0s_mean": 0.2965,
      "Precision@5.0s_mean": 0.2227,
      "Recall@5.0s_mean": 0.4560,
      "mAP@0.5": 0.00379,
      "mAP@0.75": 0.00005
    }

- 小结：任务 4 完成（detect_bdy → 适配 → 评估）。OTAS 在本数据与当前协议下显著弱于 LAPS 与 Optical Flow 基线，后续在 Table 1 中对比呈现。

### 2025-11-06 · OTAS 规范性/公平性核查与改进计划（阶段性结论）
- 规范性核查（结论：部分符合）
  - 数据准备：视频键与帧目录命名符合本仓库实现要求（{VIEW}/{VIEW}/{stem}.mp4；frames/{P}_{cam}_{act}/Frame_%06d.jpg），video_info.pkl=12 条，window_lists 已生成。
  - 推理流程：main.py 按代码逻辑在 mode=val 时会尝试加载模型权重（/OTAS/BF_tf/model/lr_{lr}_best.pth.tar）；若不存在则使用随机权重进行“验证”。当前输出目录下未检测到 model/ckpt，mean_error 可能来自未训练权重，属非标准流程（会显著拉低性能）。
  - 边界检测：detect_bdy.py 采用默认参数（order=15，threshold=51，combine=False），与官方代码默认一致。
  - 后处理与适配：adapt_otas_to_segments.py 使用原视频 FPS 将帧索引→秒，生成 segments.json；逻辑正确。
- 公平性核查（结论：部分不公平）
  - 输入一致性：是（D01/D02 各 6 个视频）。
  - 评估协议一致性：是（同 GT 目录、同评估脚本、同指标）。
  - 超参数调优公平性：否。LAPS/Optical Flow 使用 GT 搜索阈值；OTAS 未做 order/threshold/平滑的调优，且可能未训练模型 → 建议给予等价的“验证集调参”机会。
  - 特征可比性：部分可比（方法本质不同，保持统一评估协议已足够；但 OTAS 若未训练将严重失衡）。
- 改进建议与最小实验计划（待确认后执行）
  1) 先训练 ED_tf 预测器（无监督）→ 再生成 mean_error：
     - 训练：python comapred_algorithm/OTAS/code/main.py --mode train --dataset BF --feature_model tf --gpu 0 --batch_size 32 --num_workers 8 --output-path /home/johnny/action_ws/datasets/output/otas_out/
     - 验证：python comapred_algorithm/OTAS/code/main.py --mode val   （同参）
  2) 边界检测参数小网格（按视角独立调）：order∈{7,11,15,19,23}，threshold∈{45,51,60}，EMA α∈{0.5,0.7,0.85}（对 mean_error 先平滑），并加入最短段约束（min_seg_len∈{0.5,1.0}s）。
  3) 公平性：与 LAPS/OptFlow 一致，使用同一 GT 在“视角级”调参（如 3/3 拆分或留一法），报告验证集与测试集性能。
  4) 目标：优先提升 F1@2s；若计算预算有限，先在 D01 上试验，再迁移到 D02。


### 2025-11-06 · OTAS 训练修复与启动（otas 环境）
- 代码修复：
  - main.py 最优模型保存逻辑（已修复）
    - 问题：初次保存时 `state` 未定义；`best` 保存逻辑被错误地置于 `if not os.path.exists(model_save_path)` 内，目录存在后不再保存
    - 修复：使用 `os.makedirs(model_save_path, exist_ok=True)`；在保存前重建 `state={'epoch':epoch+1,'state_dict':...}` 并将保存逻辑移出目录判断
  - model.py 位置编码长度（已修复）
    - 问题：`PositionalEncoding(max_len=5)` 导致训练路径序列长度 N*SL-1=9 与 PE 长度不匹配，RuntimeError(9 vs 5)
    - 修复：改为 `PositionalEncoding(d_model=512, max_len=5000)`，覆盖训练/验证两种序列长度
- 训练启动（cwd=OTAS/code；env=otas；GPU=0）：
  - 首次尝试：`--batch_size 32` → CUDA OOM（已按规范降为 16）
  - 当前运行命令：
    - CUDA_VISIBLE_DEVICES=0 conda run -n otas python main.py \
      --mode train --dataset BF --feature_model tf --gpu 0 \
      --batch_size 16 --num_workers 8 \
      --output-path /home/johnny/action_ws/datasets/output/otas_out/
  - 训练日志：/home/johnny/action_ws/datasets/output/otas_out/OTAS/train_BF_tf_20251106_112035.log
- 说明：为避免路径相对性问题（arg_pars 默认 ../data/...），训练/验证均在 `comapred_algorithm/OTAS/code/` 目录下执行，保持与代码假设一致。
- 后续计划：训练完成后将以相同参数运行 `--mode val` 生成新的 mean_error，并在 laps 环境执行 detect_bdy → 适配 → 评估；结果与随机权重基线对比将整理至 docs/otas_experiment.md。
