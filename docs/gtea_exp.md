# GTEA 实验记录（split1）

- 环境：conda env "laps"
- 数据根：/home/johnny/action_ws/online_datasets/gtea/gtea
- 本文档用于记录每一步的执行命令、进度与关键产出路径。

## 任务清单与状态
- [x] 0. 创建文档与基础配置
- [x] 1. 创建工具脚本与配置（convert 脚本 + 2 个 YAML）
- [x] 2. 准备训练集视频软链接目录（Videos_train.split1）
- [x] 3. 转换 GT（train/test → 段 JSON 秒）
- [x] 4. 预处理训练视频（CoTracker → HDF5）
- [x] 4.5 HDF5 读取问题诊断与修复（读侧低层回退）
- [x] 5. 训练 Motion Tokenizer（FSQ 码本）
- [x] 6. 提取训练集能量（JSONL）
- [x] 7. 基于训练 GT 搜索最佳阈值（F1@2s）
- [x] 8. 测试集推理与分割（使用最佳阈值）
- [x] 9. 评估（F1@2s、F1@5s）

## 里程碑与产出路径
- 预处理输出（HDF5）：/home/johnny/action_ws/data/preprocessed_gtea_m10/split1
- Tokenizer ckpt：/home/johnny/action_ws/checkpoints/motion_tokenizer/epochs10_gtea_split1_m10/best.pt
- 训练能量：/home/johnny/action_ws/output/gtea/energy_split1/{stem}/stream_energy_quantized_token_diff_l2_mean.jsonl
- 阈值报告：/home/johnny/action_ws/output/gtea/thresholds/split1/best_threshold_quantized_token_diff.json
- 测试分割：/home/johnny/action_ws/output/gtea/segments_split1/{stem}/segmented_videos/{stem}_segments.json
- GT 段 JSON：/home/johnny/action_ws/online_datasets/gtea/gt_segments_json/{train.split1|test.split1}/{stem}_segments.json
- 评估结果：/home/johnny/action_ws/output/gtea/stats/seg_eval_split1.json

## 运行日志
- 2025-11-10: 初始化文档，待创建脚本与YAML。


- 2025-11-10 11:20: 已创建脚本与配置：tools/convert_gtea_gt_to_segments.py、params_gtea_split1{,_test}.yaml；创建训练集软链接 Videos_train.split1。
- 2025-11-10 11:35: 完成 GT 段标注转换（train/test），输出至 online_datasets/gtea/gt_segments_json/{train.split1,test.split1}/。
- 2025-11-10 12:05: 启动预处理（CoTracker tracks）。命令：
  conda run -n laps python -m amplify.preprocessing.preprocess_my_segments \
    mode=tracks source=/home/johnny/action_ws/online_datasets/gtea/gtea/Videos_train.split1 \
    output_dir=/home/johnny/action_ws/data/preprocessed_gtea_m10/split1 dataset_name=gtea \
    target_fps=10 resize_shorter=480 n_tracks=400 init_queries=uniform reinit=true horizon=16 \
    view_name=default recursive=false skip_exist=true verbose=true
- 2025-11-10 12:30: 预处理报错：TypeError: expected np.ndarray (got numpy.ndarray)（PyTorch 与 NumPy 在 laps 环境的兼容性问题）。
- 2025-11-10 12:40: 应急修复：将 amplify/utils/preprocessing_utils.py 中 torch.from_numpy(video) 替换为 torch.tensor(video, dtype=torch.float32) 以规避 ABI 检查；重启预处理，继续运行（skip_exist=true）。

- 2025-11-10 17:50: 预处理进行中，已完成 11/21 个 HDF5，持续生成中（单GPU）。将待其完成后进入 Step 5 训练。

- 2025-11-10 18:54: 预处理阶段已生成 HDF5 共 21/21 个文件（/home/johnny/action_ws/data/preprocessed_gtea_m10/split1）。
  - 完整性检查：发现读入错误，示例（S2_Cheese_C1.hdf5）存在 root/default/{tracks, vis} 两个数据集，但在 h5py 读取时报错：
    ValueError: Insufficient precision in available types to represent (31, 23, 8, 0, 23)
  - 环境版本：h5py=3.11.0，HDF5 lib=1.14.2，numpy=1.26.4，torch=2.5.1+cu124。
  - 影响：Step 5–9（训练→能量→阈值→测试→评估）暂被阻塞，需先修复 HDF5 读取问题。

## 当前问题与处置计划
- 问题描述
  - HDF5 在 h5py 高层读取时报错：ValueError: Insufficient precision in available types to represent (31, 23, 8, 0, 23)。
- 当前状态
  - 已解决（通过读侧兼容层）。
- 解决方案
  - 在 amplify/amplify/loaders/custom_segments_dataset.py 的 load_tracks() 中实现低层 HDF5 回退读取：当高层 .dtype/.read 失败时，使用 h5s/h5t 选择超块并以 IEEE_F32LE 作为内存类型直接读入 np.float32 缓冲，成功读出 tracks 与 vis。
- 证据
  - 诊断脚本对 3 个样本均复现高层报错；smoke 测试成功读取 split1/S2_Cheese_C1.hdf5（tracks (1,16,400,2)）。
- 后续建议
  - 如后续观察到 I/O 性能瓶颈或兼容性问题，再评估是否批量重生成 HDF5；当前无需重生成。



- 2025-11-10 19:20: 第一阶段-诊断：编写并运行 HDF5 诊断脚本（tools/diagnose_gtea_hdf5.py）。对 3 个文件（含 S2_Cheese_C1.hdf5）检查：能列出键与 shape，但访问 dtype/读取即报错。
  - 命令：
    - conda run -n laps python tools/diagnose_gtea_hdf5.py --dir /home/johnny/action_ws/data/preprocessed_gtea_m10/split1 --files S2_Cheese_C1.hdf5
    - conda run -n laps python tools/diagnose_gtea_hdf5.py --dir /home/johnny/action_ws/data/preprocessed_gtea_m10/split1 --pick 3
  - 代表性输出（S2_Cheese_C1.hdf5）：
    - root/default/tracks shape=(317,16,400,2); dtype_error=ValueError: Insufficient precision in available types to represent (31, 23, 8, 0, 23)
    - 低层类型：h5t_class=1, h5t_size=4, h5t_order=0（4字节小端浮点），但高层 dtype 映射失败；读取 dset[0] 与切片均失败。

- 2025-11-10 19:40: 第一阶段-样例重生成：按新输出目录 split1_test 重生成 2 个样例文件，仍复现相同报错，排除“单文件损坏/压缩过滤器”因素。
  - 命令：
    conda run -n laps python -m amplify.preprocessing.preprocess_my_segments \
      mode=tracks source=/home/johnny/action_ws/online_datasets/gtea/gtea/Videos_train.split1 \
      output_dir=/home/johnny/action_ws/data/preprocessed_gtea_m10/split1_test \
      dataset_name=gtea target_fps=10 resize_shorter=480 n_tracks=400 \
      init_queries=uniform reinit=true horizon=16 view_name=default \
      recursive=false skip_exist=false verbose=true max_files=2

- 2025-11-10 20:00: 第一阶段-修复：在读侧实现 h5py 高层失败时的低层 HDF5 回退读取，避免 dtype 映射问题。
  - 修改：amplify/amplify/loaders/custom_segments_dataset.py 的 load_tracks()
    - 高层读取失败时，使用 h5py 低层 API（h5s/h5t）对选定超块进行读取（mem dtype=float32, mtype=IEEE_F32LE），成功读出 tracks/vis。
  - 验证：tools/smoke_read_custom_segments.py 可从 split1 目录读出样例（tracks 形状 (1,16,400,2)，vis 存在）。
  - 结论：不需重生成 21 个 HDF5，读侧兼容层已解决“Insufficient precision …”报错。

- 2025-11-10 20:05: 第一阶段结论：HDF5 可读性问题已解除，进入第二阶段（Motion Tokenizer 训练）。


- 2025-11-10 19:18: 第二阶段-训练前排障：DataLoader 在 collate 阶段报错（TypeError: default_collate: batch must contain tensors ... / RuntimeError: Could not infer dtype of numpy.float32）。
  - 原因：laps 环境下 PyTorch 与 NumPy 的 ABI/类型推断不兼容，`torch.as_tensor(np.ndarray)` 与 `torch.from_numpy()` 出现异常。
  - 修复：在 amplify/amplify/utils/train.py 增加 `safe_collate`，对 batch 中的 numpy 数组显式 `np.stack` 后用 `torch.tensor(..., dtype=*)` 构造，规避 ABI 检查；并在 DataLoader 中设置 `collate_fn=safe_collate`。
  - 验证（smoke）：`num_epochs=1 quick=true num_workers=0` 成功完成 1 epoch，保存 ckpt：checkpoints/motion_tokenizer/smoke_epochs1_gtea_split1_m10_2/best.pt；Val Loss=5.1539。

- 2025-11-10 19:22: 第二阶段-启动正式训练：按 10 epoch 配置启动 Motion Tokenizer 训练。
  - 命令：
    conda run -n laps python amplify/train_motion_tokenizer.py \
      root_dir=/home/johnny/action_ws/data/preprocessed_gtea_m10/split1 \
      train_datasets='[custom_segments:traj0.8]' \
      val_datasets='[custom_segments:traj0.2]' \
      cond_cameraviews='[default]' \
      keys_to_load='[tracks,images]' \
      img_shape='[480,771]' \
      true_horizon=16 track_pred_horizon=16 \
      batch_size=8 gpu_max_bs=8 num_epochs=10 \
      quick=false num_workers=4 log_interval=50 \
      resume=false run_name=epochs10_gtea_split1_m10 \
      use_wandb=false video_root=/home/johnny/action_ws/online_datasets/gtea/gtea/Videos_train.split1
  - 期望输出：/home/johnny/action_ws/checkpoints/motion_tokenizer/epochs10_gtea_split1_m10/best.pt（训练完成后生成）。

- 2025-11-10 19:23: 第二阶段-按需重启训练：应用户要求启用 Weights & Biases 记录（use_wandb=true），重新启动 10-epoch 训练。
  - 命令（变更处：use_wandb=true）：
    conda run -n laps python amplify/train_motion_tokenizer.py \
      root_dir=/home/johnny/action_ws/data/preprocessed_gtea_m10/split1 \
      train_datasets='[custom_segments:traj0.8]' \
      val_datasets='[custom_segments:traj0.2]' \
      cond_cameraviews='[default]' \
      keys_to_load='[tracks,images]' \
      img_shape='[480,771]' \
      true_horizon=16 track_pred_horizon=16 \
      batch_size=8 gpu_max_bs=8 num_epochs=10 \
      quick=false num_workers=4 log_interval=50 \
      resume=false run_name=epochs10_gtea_split1_m10 \
      use_wandb=true video_root=/home/johnny/action_ws/online_datasets/gtea/gtea/Videos_train.split1
  - 说明：如线上凭证未配置，可能进入 wandb 离线模式；训练过程与 checkpoint 保存不受影响（将按实际创建目录记录）。



- 2025-11-10 22:xx: 第二阶段-中断与切换快速路径（原因与方案）
  - 原训练已中断（terminal ID 63）。原因：诊断确认 `keys_to_load=[tracks,images]` 且设置了 `video_root` 导致 DataLoader 每样本实时解码原视频帧（cv2.VideoCapture），成为主瓶颈（~39× 慢于仅轨迹）。
  - 方案：下次训练移除 `video_root`（或设为 `null`），使 `load_images()` 走黑图占位符路径；并降低可视化频次，将 `log_interval` 调大。

- 2025-11-10 22:xx: 第二阶段-拟重启命令（快速路径，移除 video_root，log_interval=200，run_name=epochs10_gtea_split1_m10_fast）
  - 命令：
    conda run -n laps python amplify/train_motion_tokenizer.py \
      root_dir=/home/johnny/action_ws/data/preprocessed_gtea_m10/split1 \
      train_datasets='[custom_segments:traj0.8]' \
      val_datasets='[custom_segments:traj0.2]' \
      cond_cameraviews='[default]' \
      keys_to_load='[tracks,images]' \
      img_shape='[480,771]' \
      true_horizon=16 track_pred_horizon=16 \
      batch_size=8 gpu_max_bs=8 num_epochs=10 \
      quick=false num_workers=4 log_interval=200 \
      resume=false run_name=epochs10_gtea_split1_m10_fast \
      use_wandb=true
  - 变更说明：
    1) 移除 `video_root` 参数，避免训练期实时视频解码；
    2) 将 `log_interval` 从 50 调整为 200，降低可视化带来的额外开销（如需更稀疏可用 500）；
    3) 其余参数保持不变；checkpoint 将保存到 checkpoints/motion_tokenizer/epochs10_gtea_split1_m10_fast/best.pt。
  - 速度验证：待启动后记录“同等 batch 配置的前若干 step 平均耗时”；预计从 ~2 s/step 降至 ~0.1–0.3 s/step（含模型与 4 workers 开销）。


- 2025-11-10 23:21:44: 第二阶段-中断当前训练（WandB run=7w0ja207，run_name=epochs10_gtea_split1_m10_fast）。
  - 中断原因：根因定位为 DataLoader 在 `keys_to_load=[tracks,images]` 情况下，每步构造并堆叠大尺寸 `images`（480×771×3，bs=8≈35MB），且在 `safe_collate` 内进行 numpy→torch 深拷贝，成为主要耗时；与是否进行视频实时解码无关。
  - 操作记录：终止进程 PID=1090252/1090236。

- 2025-11-10 23:21:44: 第二阶段-提交最小代码改动（仅影响可视化分支）。
  - 修改文件：amplify/train_motion_tokenizer.py
  - 修改内容：
    - 训练可视化处：`if train_global_iter % cfg.log_interval == 0 and 'images' in batch:`
    - 验证可视化处：`if val_global_iter % cfg.log_interval == 0 and 'images' in batch:`
  - 目的：允许 `keys_to_load=['tracks']` 的“tracks-only”训练安全跳过图像可视化，避免 KeyError。

- 2025-11-10 23:21:44: 第二阶段-准备重启命令（tracks-only 方案）。
  - 命令：
    ```bash
    conda run -n laps python amplify/train_motion_tokenizer.py \
      root_dir=/home/johnny/action_ws/data/preprocessed_gtea_m10/split1 \
      train_datasets='[custom_segments:traj0.8]' \
      val_datasets='[custom_segments:traj0.2]' \
      cond_cameraviews='[default]' \
      keys_to_load='[tracks]' \
      img_shape='[480,771]' \
      true_horizon=16 track_pred_horizon=16 \
      batch_size=8 gpu_max_bs=8 num_epochs=10 \
      quick=false num_workers=4 log_interval=200 \
      resume=false run_name=epochs10_gtea_split1_m10_tracks_only \
      use_wandb=true video_root=null
    ```
  - 参数变更说明：
    1) `keys_to_load` 改为 `['tracks']`（去除 images，避免 CPU 端堆叠与大拷贝开销）；
    2) 显式 `video_root=null`（避免被默认配置覆盖为非空路径）；
    3) `log_interval=200` 保持，减少可视化频率。
  - 预期速度：从 ~0.86 s/step（tracks+images/黑图占位）降至 ~0.04–0.08 s/step（tracks-only），约 10–20× 加速；验证阶段步时同步下降。



---

- 2025-11-11 02:20: Step 7–9 执行与结果（方案B：基于训练GT阈值搜索）

### Step 7 阈值搜索（已完成）
- 输出路径：/home/johnny/action_ws/output/gtea/thresholds/split1/best_threshold_quantized_token_diff.json
- 关键结果：thr=0.0, F1_mean=0.4016, Precision_mean=0.8749, Recall_mean=0.2668
- 执行命令：
  ```bash
  conda run -n laps python tools/threshold_search_with_gt.py \
    --view D01 \
    --energy-root /home/johnny/action_ws/output/gtea \
    --gt-dir /home/johnny/action_ws/online_datasets/gtea/gt_segments_json/train.split1 \
    --source quantized --mode token_diff_l2_mean \
    --target-fps 10 --stride 4 \
    --hysteresis-ratio 0.95 --up-count 2 --down-count 2 \
    --cooldown-windows 1 --max-duration-seconds 2.0 \
    --tolerance-sec 2.0 \
    --output /home/johnny/action_ws/output/gtea/thresholds/split1/best_threshold_quantized_token_diff.json
  ```
- 注意事项：训练能量文件的实际结构为 `/home/johnny/action_ws/output/gtea/{video_stem}/energy_split1`（文件名为 `energy_split1`）。脚本默认查找 `{energy_root}/{stem}/stream_energy_quantized_token_diff_l2_mean.jsonl`，已在每个 `{stem}` 目录创建符号链接 `stream_energy_quantized_token_diff_l2_mean.jsonl -> energy_split1` 以兼容脚本。

### Step 8 测试集分割（已完成）
- 输出路径：`/home/johnny/action_ws/output/gtea/segments_split1/{video_stem}/segmented_videos/{video_stem}_segments.json`
- 测试视频数量：7 个
  - S1_Cheese_C1, S1_Coffee_C1, S1_CofHoney_C1, S1_Hotdog_C1, S1_Pealate_C1, S1_Peanut_C1, S1_Tea_C1
- 配置修改：`video_action_segmenter/params_gtea_split1_test.yaml` 中 `input.dir` 改为 `/home/johnny/action_ws/online_datasets/gtea/gtea/Videos_test.split1`
- 执行命令：
  ```bash
  conda run -n laps python -m video_action_segmenter.stream_inference \
    --params video_action_segmenter/params_gtea_split1_test.yaml
  ```

### Step 9 评估（已完成）
- 输出路径：/home/johnny/action_ws/output/gtea/stats/seg_eval_split1.json
- 评估指标（汇总）：
  - F1@2s = 0.6008
  - F1@5s = 0.7173
  - mAP@0.5 = 0.0920
  - mAP@0.75 = 0.0041
- 执行命令：
  ```bash
  conda run -n laps python tools/eval_segmentation.py \
    --pred-root /home/johnny/action_ws/output/gtea/segments_split1 \
    --gt-dir /home/johnny/action_ws/online_datasets/gtea/gt_segments_json/test.split1 \
    --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
    --output /home/johnny/action_ws/output/gtea/stats/seg_eval_split1.json
  ```
- 评估脚本输出（摘要）：
  ```json
  {"summary": {"num_videos": 7, "F1@2.0s_mean": 0.6008172140705382, "Precision@2.0s_mean": 0.5808637710411036, "Recall@2.0s_mean": 0.650293056616586, "F1@5.0s_mean": 0.7172847421904588, "Precision@5.0s_mean": 0.6934365045396351, "Recall@5.0s_mean": 0.7765333709241272, "mAP@0.5": 0.09195867336805284, "mAP@0.75": 0.004127212618910809 }}
  ```

### 备注与观察
- 阈值搜索的最佳阈值为 0.0，结合 `max_duration=2s` 与冷却窗口，召回得到较大提升；在测试集上 F1@2s=0.6008，F1@5s=0.7173。
- mAP 在 0.5/0.75 IoU 下分别为 ~0.092/~0.004，说明段定位（IoU）还有提升空间，后续可考虑：
  1) 细化状态机（如自适应持续时间上限、冷却时间）；
  2) 能量平滑或多尺度能量融合；
  3) 在训练侧做留一或交叉验证式阈值选择以提升泛化稳健性。


### Step 10 训练集离线分割（已完成）
- 输出路径（预测根）：`/home/johnny/action_ws/output/gtea/segments_train_split1/{video_stem}/segmented_videos/{video_stem}_segments.json`
- 输入能量：`/home/johnny/action_ws/output/gtea/{video_stem}/(stream_energy_quantized_token_diff_l2_mean.jsonl | energy_split1)`
- 使用参数：target_fps=10, stride=4, dt=0.4s, hysteresis_ratio=0.95, up_count=2, down_count=2, cooldown=1, max_duration=2.0s, smoothing=ema(alpha=0.7, window=3), thr=0.0（来自 Step 7 报告）
- 执行命令：
  ```bash
  conda run -n laps python tools/offline_segment_from_energy.py \
    --energy-root /home/johnny/action_ws/output/gtea \
    --pred-root /home/johnny/action_ws/output/gtea/segments_train_split1 \
    --threshold-json /home/johnny/action_ws/output/gtea/thresholds/split1/best_threshold_quantized_token_diff.json \
    --threshold-key optical_flow_mag_mean_best.best_f1.thr \
    --target-fps 10 --stride 4 --hysteresis-ratio 0.95 --up-count 2 --down-count 2 --cooldown-windows 1 \
    --max-duration-seconds 2.0 --stem-prefixes S2_ S3_ S4_ --use-smoothing --smooth-method ema --smooth-alpha 0.7 --smooth-window 3
  ```
- 日志摘录（每视频段数）：
  - S2_Cheese_C1: n_segments=13；S2_Coffee_C1: 47；S2_Peanut_C1: 45；…（共21个视频，全部成功）
- 说明：在预测目录为每个视频放置能量文件符号链接 `stream_energy_quantized_token_diff_l2_mean.jsonl -> {energy源}`，以便 mAP 评估使用能量均值作为置信度。

### Step 11 训练集评估（已完成）
- 输出路径：`/home/johnny/action_ws/output/gtea/stats/seg_eval_train_split1.json`
- 执行命令：
  ```bash
  conda run -n laps python tools/eval_segmentation.py \
    --pred-root /home/johnny/action_ws/output/gtea/segments_train_split1 \
    --gt-dir /home/johnny/action_ws/online_datasets/gtea/gt_segments_json/train.split1 \
    --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
    --output /home/johnny/action_ws/output/gtea/stats/seg_eval_train_split1.json
  ```
- 评估指标（汇总）：
  - F1@2s = 0.6320
  - F1@5s = 0.7312
  - mAP@0.5 = 0.0924
  - mAP@0.75 = 0.0045
- 评估脚本输出（摘要）：
  ```json
  {"num_videos": 21, "F1@2.0s_mean": 0.6319710881952554, "Precision@2.0s_mean": 0.7646233557092472, "Recall@2.0s_mean": 0.5608493175789032, "F1@5.0s_mean": 0.7311512482631692, "Precision@5.0s_mean": 0.8777716392675846, "Recall@5.0s_mean": 0.6520124456607108, "mAP@0.5": 0.09237037314840263, "mAP@0.75": 0.004515526285149273 }
  ```

### 训练/测试对比与观察
- 指标对比：
  - 训练集 vs 测试集（F1@2s）：0.6320 vs 0.6008（+0.0312）
  - 训练集 vs 测试集（F1@5s）：0.7312 vs 0.7173（+0.0139）
  - mAP@0.5：0.0924 vs 0.0920（几乎一致）
  - mAP@0.75：0.0045 vs 0.0041（几乎一致）
- 结论与分析：
  - 训练集 F1 略高于测试集，差距不大，未见明显过拟合迹象；在 IoU 指标上，训练/测试几乎一致，说明段定位的形状/边界一致性仍是主要瓶颈。
  - 后续可考虑：
    1) 调整状态机（动态持续时间上限、冷却窗口）或引入多阈值后处理；
    2) 能量多尺度融合或自适应平滑；
    3) 针对边界偏移优化（例如对齐策略/边界微调）。
