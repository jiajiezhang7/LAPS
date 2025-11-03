# 实验方案与进度追踪（D01/D02）

更新时间：请在每次进展后更新。所有命令均在 conda 环境 laps 下运行（OTAS 在独立的 otas_env 下）。

全局约定：
- 根目录：/home/johnny/action_ws （本文档路径：docs/experiment_detailed_todo.md）
- 视角数据：
  - D01：/media/johnny/48FF-AA60/raw_videos/raw_videos_d01_subset
  - D02：/media/johnny/48FF-AA60/raw_videos/raw_videos_d02_subset
- 模型权重：各自 params.yaml 的 checkpoint_path 已确认
- GPU：1 块（ID=0）；允许批处理（单卡并发建议=1）
- 输出组织（建议新建统一实验根）：/media/johnny/48FF-AA60/exp_results
  - energy_sweep_out/{D01,D02}
  - energy_sweep_report/{D01,D02}
  - segmentation_outputs/{D01,D02}
  - umap_vis/{figure,statistics}
  - stats/{seg_eval,icss}
  - tables/{table1,table2,table3,table4}
  - figures/{figure3,figure5,...}

说明：为避免反复改 YAML，建议为不同视角与能量源复制参数文件（最小侵入，仅改少量字段）：
- video_action_segmenter/params_d01_quant.yaml
- video_action_segmenter/params_d01_vel.yaml
- video_action_segmenter/params_d02_quant.yaml
- video_action_segmenter/params_d02_vel.yaml
（由 AI 创建，确保 energy.source / energy.jsonl_path / segmentation.output_dir / input.dir 正确）

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

输入：
- D01 YAML：/home/johnny/action_ws/video_action_segmenter/params_d01_quant.yaml、/home/johnny/action_ws/video_action_segmenter/params_d01_vel.yaml
- D02 YAML：/home/johnny/action_ws/video_action_segmenter/params_d02_quant.yaml、/home/johnny/action_ws/video_action_segmenter/params_d02_vel.yaml
- 建议 energy.jsonl_path：
  - D01 quant：/home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D01/stream_energy_quantized_token_diff_l2_mean.jsonl
  - D01 vel：/home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D01/stream_energy_velocity_token_diff_l2_mean.jsonl
  - D02 quant：/home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D02/stream_energy_quantized_token_diff_l2_mean.jsonl
  - D02 vel：/home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D02/stream_energy_velocity_token_diff_l2_mean.jsonl

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
- D01 分析报告
  conda run -n laps python -m video_action_segmenter.analyze_energy_jsonl \
    --input-dir /home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D01 \
    --output-dir /home/johnny/action_ws/video_action_segmenter/energy_sweep_report/D01 \
    --title "Energy Analysis D01"
- D02 分析报告
  conda run -n laps python -m video_action_segmenter.analyze_energy_jsonl \
    --input-dir /home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D02 \
    --output-dir /home/johnny/action_ws/video_action_segmenter/energy_sweep_report/D02 \
    --title "Energy Analysis D02"

3) 阈值自动搜索（以 velocity 生成伪标签，推荐 quantized 的阈值）
- D01 阈值
  conda run -n laps python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl /home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D01/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl /home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D01/stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --output-json /home/johnny/action_ws/video_action_segmenter/energy_sweep_report/D01/best_threshold_quantized_token_diff.json
- D02 阈值
  conda run -n laps python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl /home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D02/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl /home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D02/stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --output-json /home/johnny/action_ws/video_action_segmenter/energy_sweep_report/D02/best_threshold_quantized_token_diff.json

4) 论文风格曲线（可选）
- 示例（D01 quantized）：
  conda run -n laps python -m video_action_segmenter.scripts.plot_energy_segment_from_jsonl \
    --jsonl /home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D01/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --params /home/johnny/action_ws/video_action_segmenter/params_d01_quant.yaml \
    --segment-length 2000 --seed 42

预期输出与检查点：
- energy_sweep_out/{D01,D02}/*.jsonl 四个文件存在
- energy_sweep_report/{D01,D02}/*（CSV/JSON/图/HTML）存在
- best_threshold_quantized_token_diff.json 含 best_j/best_f1 的 thr 字段

---

## 阶段2 Q2：无监督动作分割性能（与 OTAS 对比）
目标：读取阶段1阈值进行在线分割，导出片段与 codes，并用人工 GT 评估 F1@2s 与类无关 mAP@IoU。

输入：
- 分别使用 D01/D02 的 best_threshold_quantized_token_diff.json
- YAML：params_d01_quant.yaml / params_d02_quant.yaml（segmentation.mode=report，report_path 指向上面的 JSON）
- GT（见“GT 标注规范”）

操作与命令：
1) 在线分割并导出
- 建议 segmentation.output_dir：
  - D01：/media/johnny/48FF-AA60/exp_results/segmentation_outputs/D01
  - D02：/media/johnny/48FF-AA60/exp_results/segmentation_outputs/D02
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
    --pred-root /media/johnny/48FF-AA60/exp_results/segmentation_outputs/{D01,D02} \
    --gt-dir /home/johnny/action_ws/datasets/gt_annotations/{D01,D02} \
    --iou-thrs 0.5 0.75 --tolerance-sec 2.0 \
    --output /home/johnny/action_ws/stats/seg_eval/seg_eval_{VIEW}.json

3) OTAS 基线（独立环境，不影响 laps）
- 环境准备（仅文档，不立即执行）：
  git clone https://github.com/yl596/OTAS.git /home/johnny/action_ws/external/OTAS
  conda create -y -n otas_env python=3.9
  conda run -n otas_env pip install -r /home/johnny/action_ws/external/OTAS/requirements.txt
- 推理与适配（AI 将编写适配脚本）：
  - 产出 OTAS 预测片段后，运行：
    conda run -n laps python tools/adapt_otas_to_segments.py \
      --otas-pred /home/johnny/action_ws/external/OTAS/outputs/{VIEW} \
      --output /media/johnny/48FF-AA60/exp_results/segmentation_outputs/{VIEW}_OTAS
  - 使用相同 eval_segmentation.py 评估，生成对照表（tables/table1.csv）

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
    --data-dir /media/johnny/48FF-AA60/exp_results/segmentation_outputs/D01 \
    --output-dir /home/johnny/action_ws/umap_vis/figure \
    --aggs attn_norm mean mean_std first_last max \
    --k-min 2 --k-max 10 --metric cosine

2) 冻结序列模型（轻量 Transformer）
  conda run -n laps python /home/johnny/action_ws/umap_vis/scripts/sequence_model_embedding.py \
    --data-dir /media/johnny/48FF-AA60/exp_results/segmentation_outputs/D01 \
    --fig-dir /home/johnny/action_ws/umap_vis/figure \
    --stats-dir /home/johnny/action_ws/umap_vis/statistics \
    --use-best-grid-config --metric cosine --k-min 2 --k-max 10

3) 语义一致性（CLIP ICSS；AI 新增工具）
  conda run -n laps python tools/icss_eval.py \
    --pred-root /media/johnny/48FF-AA60/exp_results/segmentation_outputs/{VIEW} \
    --sample-per-seg 3 --clip-backbone ViT-B-32 \
    --output /home/johnny/action_ws/stats/icss/icss_{VIEW}.json

预期输出与检查点：
- umap_vis/figure 下生成 UMAP 图；umap_vis/statistics 下生成聚类指标 CSV
- stats/icss/icss_{VIEW}.json 存在
- tables/table2.csv（聚类指标汇总）、tables/table3.csv（ICSS）生成

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
    --output /home/johnny/action_ws/tables/table4.csv
- 聚合与序列模型 sweep：
  conda run -n laps python tools/sweep_seg.py \
    --data-roots /media/johnny/48FF-AA60/exp_results/segmentation_outputs/D01 \
    --aggs attn_norm mean mean_std first_last \
    --model-grid "d=128,256;n_layers=2,4;n_heads=2,4" \
    --output /home/johnny/action_ws/tables/table4_seq.csv

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
- 文件命名：gt_{video_stem}.json（与输入视频同名 stem）
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
- [ ] 为 D01/D02 复制量化/速度 YAML（AI；依赖：现有 params.yaml/params_d02.yaml；产出：4 个 YAML）
- [ ] 设置 segmentation.output_dir 为本次实验专用路径（AI；依赖：输出组织策略；产出：不覆盖既有数据）
- [ ] 环境冒烟跑通（AI；依赖：GPU 可用；产出：日志 + 能量 JSONL）

### 阶段1（Q1）
- [ ] 运行 D01 quant/vel 生成 JSONL（AI；依赖：YAML；产出：2 个 JSONL）
- [ ] 运行 D02 quant/vel 生成 JSONL（AI；依赖：YAML；产出：2 个 JSONL）
- [ ] 生成 D01 分析报告（AI；依赖：JSONL；产出：report/HTML/CSV/图）
- [ ] 生成 D02 分析报告（AI；依赖：JSONL；产出：report/HTML/CSV/图）
- [ ] 计算 D01/D02 最佳阈值 JSON（AI；依赖：两类 JSONL；产出：best_threshold*.json）
- [ ] 生成论文风格曲线（AI；可选；产出：figures/*）

### 阶段2（Q2）
- [ ] 在线分割导出 D01（AI；依赖：best_threshold；产出：segmented_videos+codes）
- [ ] 在线分割导出 D02（AI；依赖：best_threshold；产出：segmented_videos+codes）
- [ ] 编写 tools/eval_segmentation.py（AI；依赖：GT；产出：seg_eval_{VIEW}.json + tables/table1.csv）
- [ ] OTAS 环境搭建与推理（AI；依赖：仓库；产出：OTAS outputs）
- [ ] 编写 tools/adapt_otas_to_segments.py 并评估（AI；依赖：OTAS outputs；产出：对照结果）

### 阶段3（Q3）
- [ ] 运行 segment_umap_cluster_analysis.py（AI；依赖：segmentation_outputs；产出：UMAP 图 + 指标 CSV）
- [ ] 运行 sequence_model_embedding.py（AI；依赖：segmentation_outputs；产出：UMAP + 指标 CSV）
- [ ] 编写 tools/icss_eval.py（AI；依赖：CLIP；产出：icss_{VIEW}.json + tables/table3.csv）

### 阶段4（消融）
- [ ] 编写 tools/sweep_energy.py / tools/sweep_seg.py（AI；依赖：阶段1/2/3 管线；产出：table4*.csv + figures/*）

---

## 进度追踪
- 当前状态：阶段0 进行中（输入与 checkpoint 已确认；待创建 YAML 与设定输出目录）
- 已完成里程碑：
  - 输入目录 D01/D02 确认；checkpoint 路径确认；GPU/并行策略确认
- 问题与解决：
  - 暂无
- 下一步行动：
  1) AI 生成四个 YAML 并设置输出目录
  2) 跑通阶段1四个 JSONL 并生成报告与阈值
  3) User 按规范完成 GT 标注的首批样例（每视角≥3 个视频×10 分钟）

---

## 关键决策记录（持续更新）
- GT 标注需求：每视角建议 60 分钟，最小 30 分钟；类无关边界；路径 /home/johnny/action_ws/datasets/gt_annotations/{D01,D02}；命名 gt_{video_stem}.json（待 User 最终确认）
- 阈值选择：默认使用 best_f1.thr（可切换 best_j.thr 做召回/精度权衡）
- 聚类参数：UMAP metric=cosine, neighbors=15, min-dist=0.1；序列模型使用 best-grid（d=256, n_layers=4, n_heads=4, pooling=mean）
- 基线方法：OTAS 官方实现；独立环境 otas_env；统一使用相同评估脚本；预测段置信度=段内平滑能量均值

---

## 验证检查点速查
- 阶段0：stream_inference 最小跑通，无异常日志；能量 JSONL 路径可写
- 阶段1：四个 JSONL 存在；分析报告与阈值 JSON 生成；曲线图外观合理
- 阶段2：segmented_videos 与 codes.json 结构完整；评估 JSON 与 Table1 生成
- 阶段3：UMAP 图与统计 CSV 生成；ICSS JSON 与 Table3 生成
- 阶段4：Table4 与趋势图生成，可复现

