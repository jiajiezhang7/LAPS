
## 任务总览与关键结论
- 当前 50salads 数据在本地为 HuggingFace 版本，仅包含预提取特征（2048-d .npy）、逐帧文本标注和 5 折划分文件，不含原始视频。LAPS 需要“原始 RGB 视频”做点迹跟踪与能量计算，才能完成训练/推理/评估。
- 因此，需先准备 50salads 的原始视频（建议放在 /home/johnny/action_ws/online_datasets/50salads/videos，文件名与 groundTruth 中的 stem 保持一致，如 rgb-01-1.mp4）。
- 评估将采用 F1@2s 与 F1@5s，统一用仓库内 tools/eval_segmentation.py。

如您确认“已有原始视频（或允许我下载）”，以下计划即可直接执行；若暂无原始视频，请先确认是否需要我加入“下载并重命名整理”的步骤。

---

## 1. 数据集检查结果（/home/johnny/action_ws/online_datasets/50salads）
- 文件结构
  - 50salads/features: 2048 维特征 .npy（文件名形如 rgb-01-1.npy）
  - 50salads/groundTruth: 对应逐帧标签 .txt（形如 rgb-01-1.txt，每行 1 帧标签）
  - 50salads/splits: 5 折划分（train.splitK.bundle/test.splitK.bundle）
  - 50salads/mapping.{txt,csv}: 类别映射
- online_datasets/50salads/50salads.py（HF 数据集脚本）
  - 加载逻辑以特征为主：T×2048 的 feature 与帧级标签对齐（采取 min(len(feature), len(gt))）
  - 无原视频时，无法将“帧索引”精确转换为“秒”，而我们的评估指标是以秒为容忍度（F1@t 秒）
- 结论
  - 要跑 LAPS，必须对“原始视频”进行点迹跟踪与能量计算；现有 .npy 特征不能直接用于 LAPS。
  - 逐帧标签需转换为“片段 JSON（start_sec/end_sec）”才能用于仓库内统一评估脚本。

---

## 2. LAPS 算法与本仓库流程理解（与代码完全对齐）
- 预处理（amplify/preprocessing/preprocess_my_segments.py）
  - 输入：原始视频目录
  - 用 CoTracker（cotracker2 or cotracker3_online）在窗口内跟踪点迹，输出 HDF5（root/<view>/{tracks,images}）
  - 关键参数：n_tracks=400, horizon=16, target_fps=10, resize_shorter=480, stride=4
- Motion Tokenizer 训练（amplify/train_motion_tokenizer.py + amplify/cfg/train_motion_tokenizer.yaml）
  - 输入：上述 HDF5 轨迹数据
  - 训练 FSQ 码本，输出 best.pt
  - 关键参数：true_horizon=16, track_pred_horizon=16, batch_size/gpu_max_bs/num_epochs 等
- 推理与分割（video_action_segmenter/stream_inference.py + params_*.yaml）
  - 对原始（长）视频流式处理：重采样、跟踪、量化（用训练好的 tokenizer），计算“量化码差分能量”（最佳：quantized + token_diff_l2_mean）
  - 基于滞回阈值+去抖+冷却的状态机进行分割，导出 {video, segments[{start_sec,end_sec}]} JSON
- 阈值选择
  - 两种方式都已支持：
    1) 无 GT：video_action_segmenter/compute_best_threshold.py（基于能量分布的启发式）
    2) 有 GT：tools/threshold_search_with_gt.py（基于训练集 GT 边界最大化 F1@tol 搜索阈值）
- 统一评估（tools/eval_segmentation.py）
  - 输入：预测 JSON 根目录 + GT 段 JSON 目录
  - 输出：每视频与总体 F1@t，以及可选 mAP@IoU

---

## 3. 面向 50salads 的完整实验计划（按 5 折交叉验证）

说明：
- 所有命令均在 conda 环境 laps 下执行：前缀使用 conda run -n laps
- 变量占位：请按实际路径替换

### 3.1 准备原始视频（一次性）
- 将 50salads 原始 RGB 视频（命名与 groundTruth 同 stem）放在：
  - /home/johnny/action_ws/online_datasets/50salads/videos
  - 例如：/home/johnny/action_ws/online_datasets/50salads/videos/rgb-01-1.mp4
- 若缺少 CoTracker 权重（首次用时需要），可按 pipeline.md 的提示下载（可选）：
  - cotracker2.pth 与 scaled_online.pth 放到 ~/.cache/torch/hub/checkpoints/

### 3.2 逐帧标签 → 段 JSON（每个 split 单独一份）
目的：把 groundTruth 的帧级标签转成 LAPS 评估所需的 segments.json（以“秒”为单位）
- 输出建议：
  - 训练 GT：/home/johnny/action_ws/online_datasets/50salads/gt_segments_json/train.splitK
  - 测试 GT：/home/johnny/action_ws/online_datasets/50salads/gt_segments_json/test.splitK
- 转换逻辑（核心思路，9 行参考）：
````python mode=EXCERPT
def labels_to_segments(labels, fps, n_frames):
    # 将逐帧标签压成片段边界（秒）
    segs, s = [], 0
    for i in range(1, n_frames + 1):
        if i == n_frames or labels[i] != labels[i - 1]:
            segs.append((s / fps, i / fps))
            s = i
    return segs
````
- 实施方式（推荐新增一个小工具脚本 tools/convert_50salads_gt_to_segments.py，读取：
  - groundTruth/*.txt（每行一个标签）
  - 原始视频获取 fps 与总帧数（用 OpenCV 读取）
  - splits/train.splitK.bundle 与 test.splitK.bundle 逐个生成 {stem}_segments.json
）——如您同意，我后续可直接为您补上该脚本。

### 3.3 预处理训练视频→HDF5（用于训练 Motion Tokenizer）
- 不需要处理测试视频（为了公平，若您希望“严格不看测试”，训练只用 train.splitK 的视频；若允许无监督地用所有视频也可以进一步稳健码本）
- 命令（按需调整 source/output_dir/dataset_name）：
  - 在项目根目录执行
  - 仅处理训练集视频时，建议先将 train.splitK.bundle 中列出的文件建立软链接到一个 train.videos.splitK 目录
  - 统一命令（示例，处理一个目录，target_fps=10）：
    conda run -n laps python -m amplify.preprocessing.preprocess_my_segments \
      source=/home/johnny/action_ws/online_datasets/50salads/videos_train.splitK \
      output_dir=/home/johnny/action_ws/data/preprocessed_50salads_m10/splitK \
      dataset_name=50salads target_fps=10 resize_shorter=480 \
      n_tracks=400 init_queries=uniform reinit=true horizon=16 view_name=default \
      skip_exist=true verbose=true
- 产物（期望）：
  - /home/johnny/action_ws/data/preprocessed_50salads_m10/splitK/<相对结构>/*.hdf5
  - HDF5 内包含 root/default/{tracks,images}，tracks 形状约为 (T, horizon, N, 2)

### 3.4 训练 Motion Tokenizer（每个 split 一次）
- 以 3.3 的 HDF5 为根目录
- 命令（示例，5–20 epoch 视资源而定；quick=false 才是正式训练）：
  conda run -n laps python amplify/train_motion_tokenizer.py \
    root_dir=/home/johnny/action_ws/data/preprocessed_50salads_m10/splitK \
    train_datasets=[custom_segments:traj0.8] \
    val_datasets=[custom_segments:traj0.2] \
    cond_cameraviews=[default] keys_to_load=[tracks,images] \
    true_horizon=16 track_pred_horizon=16 \
    batch_size=8 gpu_max_bs=8 num_epochs=10 \
    quick=false num_workers=4 log_interval=8 \
    resume=false run_name=epochs10_50salads_splitK_m10 use_wandb=false
- 产物（期望）：
  - /home/johnny/action_ws/checkpoints/motion_tokenizer/epochs10_50salads_splitK_m10/best.pt

### 3.5 提取训练集能量 JSONL（用于阈值搜索）
- 复制一份推理参数模板：
  - cp video_action_segmenter/params_d02_label.yaml video_action_segmenter/params_50salads_splitK.yaml
- 修改关键字段（仅列出需要改动）：
  - checkpoint_path: 指向 3.4 的 best.pt
  - input.type: "folder"
  - input.dir: /home/johnny/action_ws/online_datasets/50salads/videos_train.splitK
  - visualize: false
  - target_fps: 10, stride: 4, resize_shorter: 320 或 480
  - energy:
    - enable: true
    - source: "quantized"
    - mode: "token_diff_l2_mean"
    - jsonl_output: true
    - jsonl_path: /home/johnny/action_ws/output/50salads/energy_splitK
  - segmentation:
    - enable: false（此步只导出能量，先不做分割）
- 运行：
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params video_action_segmenter/params_50salads_splitK.yaml

- 产物（期望）：
  - /home/johnny/action_ws/output/50salads/energy_splitK/{stem}/stream_energy_quantized_token_diff_l2_mean.jsonl

### 3.6 用训练 GT 搜索最佳阈值（F1@2s 为主）
- 前提：已完成 3.2（train GT 段 JSON）与 3.5（能量 JSONL）
- 命令（tools/threshold_search_with_gt.py；注意它的 --view 仅作占位，传 D01 即可）：
  conda run -n laps python tools/threshold_search_with_gt.py \
    --view D01 \
    --energy-root /home/johnny/action_ws/output/50salads/energy_splitK \
    --gt-dir /home/johnny/action_ws/online_datasets/50salads/gt_segments_json/train.splitK \
    --source quantized --mode token_diff_l2_mean \
    --target-fps 10 --stride 4 \
    --hysteresis-ratio 0.95 --up-count 2 --down-count 2 --cooldown-windows 1 \
    --max-duration-seconds 2.0 \
    --tolerance-sec 2.0 \
    --output /home/johnny/action_ws/output/50salads/thresholds/splitK/best_threshold_quantized_token_diff.json
- 产物（期望）：
  - /home/johnny/action_ws/output/50salads/thresholds/splitK/best_threshold_quantized_token_diff.json
  - JSON 内含 “best_f1.thr” 等字段

（如不想用 GT 搜索，也可用 video_action_segmenter/compute_best_threshold.py 对能量分布启发式选阈值，但与 F1 指标未必最优）

### 3.7 测试集推理与分割（生成待评估的 segments.json）
- 复用 3.5 的 YAML，另存为 params_50salads_splitK_test.yaml，修改：
  - input.dir: /home/johnny/action_ws/online_datasets/50salads/videos （或 test.splitK 的子集目录）
  - segmentation:
    - enable: true
    - mode: "report"
    - report_path: /home/johnny/action_ws/output/50salads/thresholds/splitK/best_threshold_quantized_token_diff.json
    - output_dir: /home/johnny/action_ws/output/50salads/segments_splitK
    - export_segments_json: true
- 运行：
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params video_action_segmenter/params_50salads_splitK_test.yaml
- 产物（期望）：
  - /home/johnny/action_ws/output/50salads/segments_splitK/{stem}/segmented_videos/{stem}_segments.json
  - 同目录保留对应能量 JSONL 便于评估计算置信度（可选）

### 3.8 评估（F1@2s 与 F1@5s）
- 前提：已完成 3.2（test GT 段 JSON）与 3.7（预测段 JSON）
- 命令：
  conda run -n laps python tools/eval_segmentation.py \
    --pred-root /home/johnny/action_ws/output/50salads/segments_splitK \
    --gt-dir /home/johnny/action_ws/online_datasets/50salads/gt_segments_json/test.splitK \
    --iou-thrs 0.5 0.75 \
    --tolerance-sec 2.0 \
    --tolerance-secs 5.0 \
    --output /home/johnny/action_ws/output/50salads/stats/seg_eval_splitK.json
- 输出（期望）：
  - JSON：包含每视频指标与 summary（F1@2.0s_mean、F1@5.0s_mean 等）

---

## 4. 统一命令清单（示例路径，均在 laps 环境）
- 预处理训练视频（见 3.3）
- 训练（见 3.4）
- 训练集能量（见 3.5）
- 阈值搜索（见 3.6）
- 测试集推理分割（见 3.7）
- 评估（见 3.8）

如需我把上述 YAML（params_50salads_splitK*.yaml）与 GT 转换脚本（tools/convert_50salads_gt_to_segments.py）直接加到仓库，我可以一次性提交，确保开箱即用。

---

## 5. 预期输出位置与格式
- HDF5（训练数据）：/home/johnny/action_ws/data/preprocessed_50salads_m10/splitK/.../*.hdf5
- Tokenizer checkpoint：/home/johnny/action_ws/checkpoints/motion_tokenizer/epochs10_50salads_splitK_m10/best.pt
- 能量 JSONL（训练/测试）：/home/johnny/action_ws/output/50salads/energy_splitK/{stem}/stream_energy_quantized_token_diff_l2_mean.jsonl
- 预测 segments（测试）：/home/johnny/action_ws/output/50salads/segments_splitK/{stem}/segmented_videos/{stem}_segments.json
- GT segments（训练/测试）：/home/johnny/action_ws/online_datasets/50salads/gt_segments_json/{train.splitK|test.splitK}/{stem}_segments.json
- 评估结果：/home/johnny/action_ws/output/50salads/stats/seg_eval_splitK.json

---

## 6. 注意事项
- 环境：所有步骤都在 conda env “laps” 下执行
- 文件名匹配：原始视频文件名必须与 groundTruth 的 stem 一致（rgb-XX-Y）
- FPS 与持续时间：GT 转段时请用“原始视频的 fps 与总帧数”计算秒，保证与推理得到的“秒”对齐
- 阈值：推荐在训练集上做基于 GT 的阈值搜索（F1@2s），测试时复用该阈值；评估同时给出 F1@5s
- 性能与显存：n_tracks=400、resize_shorter=480 较稳妥；若显存吃紧可降到 320 或减少 n_tracks
- 可复现性：stream_inference 会在每视频子目录记录 segments.json 与能量 JSONL，评估脚本会直接读取
- 不创建无关文档：本计划仅提出必要的配置与代码步骤，不额外生成说明性文档

---

## 需要您确认的事项
1) 是否已经具备 50salads 原始视频？若没有，是否需要我加上“下载与重命名整理”的步骤并执行？
2) 是否同意我在仓库中添加“GT 转换小工具（tools/convert_50salads_gt_to_segments.py）”与 2 个参数 YAML（params_50salads_splitK*.yaml）？这将显著降低后续执行成本。
