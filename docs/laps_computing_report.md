# LAPS 在线推理计算效率报告（持续更新）

更新时间：2025-11-12

## 实验摘要（TL;DR）
- 管线吞吐（端到端，含追踪）：约 2.33 FPS（target_fps=10，stride=4，T≈15，320p，grid_size=20）
- 每窗总时延 t_total：均值 1.661 s；p95 1.662 s
- 追踪（CoTracker）耗时 t_track：均值 1.659 s（≈99.8% 占比）
- 模型前向（编码+量化）t_forward：均值 2.79 ms/窗；折算 ≈0.186 ms/帧（模型-only 理论 >5.3k FPS）
- 端到端延迟 latency：均值 3.01 s；p95 3.29 s
- GPU 峰值显存：≈ 2.04 GB；CPU 峰值 RSS：≈ 2.05 GB
- 预算达标率 ok_ratio：0.0（瓶颈在追踪模块）

---

## 实验目的
对 LAPS 方法在“在线推理阶段”的计算效率进行量化评估，用于论文中佐证“轻量化、低计算成本”的主张。特别区分：
- Pipeline（端到端，含追踪）性能：贴近实际在线应用体验；
- 模型本体（编码+量化）性能：体现 LAPS 的轻量化本质，与外部追踪模块解耦。

## 数据与输入
- 数据集：D01
- 示例视频：/home/johnny/action_ws/datasets/gt_raw_videos/D01/D01_sample_1_seg001.mp4
- 单视频配置：video_action_segmenter/params_perf_single_d01.yaml（max_windows=60，便于快速采样）

## 运行环境与关键配置
- Conda 环境：laps（已用于运行）
- 设备：CUDA（amp: true）
- 模型：motion tokenizer（参数量 30.23M）
- 关键流式参数：
  - target_fps = 10，stride = 4，窗口长度 T ≈ 15（由 checkpoint 推断）
  - resize_shorter = 320，grid_size = 20（CoTracker 网格）
  - gating：pre_gate 与 motion_gate 均启用
- 计时准确性：在追踪与前向计时点加入 torch.cuda.synchronize()，确保 GPU 计时可靠

## 指标采集与落盘
- 逐窗 JSONL：datasets/output/paper_ablation_study/exp01_full_quant/segmentation/D01_sample_1_seg001/stream_perf.jsonl
  - 字段：t_track、t_forward、t_total、budget、slack、status(OK/LAG)、latency
- 汇总 JSON：datasets/output/paper_ablation_study/exp01_full_quant/segmentation/D01_sample_1_seg001/D01_sample_1_seg001_perf_summary.json
  - 字段：各时延分布统计（mean/p50/p95/max）、吞吐（windows/s 与 FPS 估计）、预算达标率、GPU/CPU 峰值内存、参数量等

## 结果（来自汇总 JSON）
- windows_done：60；wall_time：102.99 s；budget_per_window：0.400 s；ok_ratio：0.0
- realized_windows_per_sec：0.5826；throughput_fps_estimate：2.3303
- t_total_s：mean 1.6615 | p95 1.6625 | max 2.3788
- t_track_s：mean 1.6587 | p95 1.6592 | max 2.3788
- t_forward_s：mean 0.00279 | p95 0.00126 | max 0.1441（秒/窗）
- latency_s：mean 3.0054 | p95 3.2858 | max 3.7751
- gpu_max_mem_allocated_mb：≈ 2038.9；gpu_max_mem_reserved_mb：≈ 2374.0；cpu_max_rss_mb：≈ 2048.0
- 模型参数量 param_count：30,232,805

## 关键解读
1) 端到端瓶颈在追踪：t_track_mean / t_total_mean ≈ 1.6587 / 1.6615 ≈ 99.8%。
2) LAPS 模型前向极轻量：t_forward_mean ≈ 2.79 ms/窗；按 T≈15 折算 ≈ 0.186 ms/帧 ⇒ 模型-only 理论吞吐 >5.3k FPS。
3) Pipeline 吞吐仅约 2.33 FPS 的主要原因是当前追踪配置（320p、grid_size=20）的计算负担；若替换/简化追踪或降低分辨率/网格密度，端到端 FPS 有明显上升空间。
4) 预算达标率为 0（budget=0.4 s/窗），与追踪主导时延一致；可通过更激进的 gating 与更少的追踪点进一步降低触发频率与均摊成本。

## 逐窗记录示例（JSONL 节选）
- {"win": 10, "t_track_s": 1.6523, "t_forward_s": 0.1441, "t_total_s": 1.7964, "budget_s": 0.4, "slack_s": -1.3964, "latency_s": 3.3621}
- {"win": 21, "t_track_s": 1.6580, "t_forward_s": 0.0012, "t_total_s": 1.6591, "budget_s": 0.4, "slack_s": -1.2591, "latency_s": 3.2308}

## 复现实验
1) 配置文件：video_action_segmenter/params_perf_single_d01.yaml（已包含单视频路径与参数）
2) 运行命令：
   - conda activate laps
   - export PYTHONPATH=.
   - python -u video_action_segmenter/stream_inference.py --params video_action_segmenter/params_perf_single_d01.yaml

## 备注与变更记录
- 新增性能监控：逐窗 JSONL 与整段汇总 JSON；记录时延/内存/吞吐/延迟/参数量
- 计时修正：在追踪与前向计时点加入 CUDA 同步，避免 GPU 异步导致的低估
- 输出路径与分段结果对齐，便于按视频聚合与论文引用

（本报告将随后续实验配置与优化同步更新，始终保留“模型本体 vs 端到端”的对照口径，以突出 LAPS 轻量化本质与工程提升空间。）

