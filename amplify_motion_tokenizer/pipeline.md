## 青春修炼手册


### 使用复习


#### 准备数据
- 把视频放在 amplify_motion_tokenizer/data/videos下
- 调用 amplify_motion_tokenizer/data_preprocessing/split_videos.py，在amplify_motion_tokenizer/data/video_segments/依子文件夹下生成分割截取后的视频
  python data_preprocessing/split_videos.py --config /home/johnny/johnny_ws/motion_tokenizer/amplify_motion_tokenizer/configs/tokenizer_config.yaml --jobs 16 --skip-if-exists

- 调用 preprocess_segments.py，也即调用CoTracker，生成速度序列

  conda activate amplify_mt
  python -m amplify_motion_tokenizer.data_preprocessing.preprocess_segments \
    --config amplify_motion_tokenizer/configs/tokenizer_config.yaml \
    --use-segments \
    --device auto


#### 分析训练所用数据

    - python -m amplify_motion_tokenizer.analysis.analyze_label_distribution --dir /media/johnny/Data/data_motion_tokenizer/processed_velocities_short --config amplify_motion_tokenizer/configs/tokenizer_config.yaml --recursive --max-files 0

- 分析速度质量
  python -m amplify_motion_tokenizer.analysis.analyze_velocities \
  --input-dir /media/johnny/Data/data_motion_tokenizer/velocities_d02_for_train \
  --config amplify_motion_tokenizer/configs/tokenizer_config.yaml \
  --report-dir amplify_motion_tokenizer/analysis/velocities_d02_for_train_report \
  --max-files 50000 \
  --seed 123 \
  --angle-min-mag 0.001 \
  --zero-thresholds 0.005 0.01 0.02 0.05


#### 训练
- 调用train，开始motion tokenizer的训练
    CUDA_VISIBLE_DEVICES=0,1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n amplify_mt accelerate launch \
    --num_processes 2 \
    -m amplify_motion_tokenizer.train \
    --config amplify_motion_tokenizer/configs/tokenizer_config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 16 \
    --device cuda \
    --mixed-precision bf16 \
    --grad-accum-steps 2 \
    --log-interval 10



#### 监控训练（TensorBoard）
- 默认日志目录：总是写入到“checkpoint_dir/train_{timestamp}/tensorboard”子目录
- 启动命令：
    ```bash
    # 安装依赖（若尚未安装）
    conda run -n amplify_mt pip install -r requirements.txt

    # 启动 TensorBoard（建议绑定到 0.0.0.0 便于远程查看）
    tensorboard --logdir ./checkpoints/train_20250926_124412/tensorboard --port 6006 --bind_all
    ```
- 训练脚本会记录以下指标（仅主进程写日志，兼容 Accelerate 多卡训练）：
    - `train/loss_step`：每 `log_interval` 步的训练损失（可通过 `training.log_interval` 或 `--log-interval` 配置）。
    - `train/acc_step`：每 `log_interval` 步的训练准确率（基于所有点与时间步的分类正确率）。
    - `train/lr`：学习率（取第一个参数组）。
    - `train/loss_epoch`：每个 epoch 的平均训练损失（全局聚合）。
    - `train/acc_epoch`：每个 epoch 的训练准确率（全局聚合）。
    - `train/epoch_time_sec`：每个 epoch 用时（秒）。
    - `train/samples_per_sec`：吞吐（样本/秒，已按进程总和计算）。
    - `train/samples_per_epoch`：每个 epoch 见到的样本总数（全局）。
    - `hparams/*`：训练结束写入的超参与最终指标（便于对比实验）。

#### 验证与最优模型
- 你可以通过以下两种方式启用验证集（两者二选一）：
  1) 在 `configs/tokenizer_config.yaml` 中设置 `data.val_preprocess_output_dir` 为独立的验证数据目录（与训练数据格式相同的 `.pt` 文件）。
  2) 若未指定独立目录，可设置 `data.val_split`（0~1）按比例从训练数据中随机划分一部分作为验证集（默认 0.1）。
- 训练脚本会在每个 epoch 结束后评估并记录：
  - `val/loss_epoch`、`val/acc_epoch`
- 检查点策略：
  - `checkpoints/best.pth`：按验证集平均损失（val/loss_epoch）最优自动保存；若无验证集，则按训练集平均损失（train/loss_epoch）最优保存。
  - `checkpoints/last.pth`：始终保存最后一个 epoch 的权重。

#### 混合精度与梯度累积
- 在配置文件 `training` 段新增：
  - `mixed_precision`: `no` / `fp16` / `bf16`（默认 `no`）。
  - `gradient_accumulation_steps`: 整数（默认 1）。
- 也可在命令行覆盖：
    ```bash
    # 例如：启用 BF16 与 2 步梯度累积（全局 batch = 每卡 batch × GPU 数 × 累积步数）
    CUDA_VISIBLE_DEVICES=0,1 conda run -n amplify_mt accelerate launch \
      --num_processes 2 \
      -m amplify_motion_tokenizer.train \
      --config amplify_motion_tokenizer/configs/tokenizer_config.yaml \
      --batch-size 16 \
      --mixed-precision bf16 \
      --grad-accum-steps 2
    ```
  - 提示：A6000 支持 BF16，通常比 FP16 更稳；梯度累积可在不增大单卡显存的情况下扩大全局 batch。


#### CoTracker可视化

- 对一个动作视频，分T=16滑动窗口，进行CoTracker跟踪（每窗口重采样） —— 更符合当前方法论 （prefered）

  python -m video_action_segmenter.window_track_and_save \
    --output-dir /home/johnny/johnny_ws/motion_tokenizer/video_action_segmenter/inference_outputs/windows \
    --target-fps 20 \
    --resize-shorter 480 \
    --grid-size 20 \
    --T 16 \
    --stride 4 \
    --device auto \
    --trail 15

#### Inference 

##### From Segmented Action Clip Video

- 命令：
  ```bash
  conda run -n amplify_mt python -m amplify_motion_tokenizer.inference_short_clip \
    --config amplify_motion_tokenizer/configs/inference_config.yaml \
    --batch-size 8 \
    --amp
  ```

##### From raw long videos or live video streams

- 命令：
  ```bash
    python -m video_action_segmenter.stream_inference \
      --params video_action_segmenter/params.yaml
  ```
##### 统计目前已有的 code indices 的数量：用于提前作分类

- 命令：
  ```bash
    python amplify_motion_tokenizer/scripts/count_code_indices.py \
      --root /media/johnny/Data/data_motion_tokenizer/online_inference_results_codeboook2048_stride4
  ```

##### Action Energy Analysis （运动能量分析，探究指标优劣 - best: quantized + token_diff_l2_mean）

- 1. 先跑若干样本得到能量 JSONL (输出在energy_sweep_out文件夹下)
- 2. 用 compute_best_threshold.py 产出报告（输出在energy_sweep_report文件夹下）
- 3. 再在流式推理时读取报告阈值，在params中设置：
  - mode: "report"                # fixed | report；report 模式会从报告JSON读取阈值
  - report_path: "./video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff.json"

- 如后续更换设备或场景，建议抽取少量样本再跑一次 compute_best_threshold.py 更新报告阈值，而不是在线做自适应

- Compute best threshold (using smooth参数)
  conda run -n amplify_mt python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl video_action_segmenter/energy_sweep_out/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl  video_action_segmenter/energy_sweep_out/stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --smooth --smooth-method ema --smooth-alpha 0.7 --smooth-window 3 \
    --output-json video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff_smoothed.json


##### Delete static videos

- 检测+预览删除 （dry-run 预览）

  python detect_static_videos.py \
      --root /path/to/videos \
      --output-json empty_videos.json \
      --dry-run --delete-min-motion-ratio 0.02 \
      --delete-max-mean-diff 0.01 \
      --delete-verbose

- 检测+实际删除

  python detect_static_videos.py \
      --root /path/to/videos \
      --delete \
      --delete-min-motion-ratio 0.02 \
      --delete-max-mean-diff 0.01