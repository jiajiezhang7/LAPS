# 整个项目的使用说明


### Split Videos into 40s segments (same as LIBERO)
- python amplify/scripts/split_segments_to_chunks.py \
  /media/johnny/Data/data_motion_tokenizer/whole_d02_videos_segments \
  /media/johnny/Data/data_motion_tokenizer/whole_d02_videos_segments_40s_complete \
  --workers 64



### Preprocess Videos -> Velocities for Training Motion Tokenizer

- 预处理视频片段（清理损坏文件、清理静态视频）
    - 预演（搞清楚状况）
      `python amplify/scripts/check_and_clean_segments.py --dry-run --verbose`
    - 正式执行
      `python amplify/scripts/check_and_clean_segments.py`
    - 参数配置
      修改 `amplify/cfg/preprocessing/check_and_clean_segments.yaml`，可调整根目录、ffmpeg 超时、静止检测阈值等。

- 生成训练数据 - velocities (已跑通)
    - (请注意: preprocesssing 并不是amplify的子模块，而是顶层独立包)

    - 具体配置请修改 `amplify/cfg/preprocessing/preprocess_my_segments.yaml`
    - 在 `amplify` 目录下运行 `python -m preprocessing.preprocess_my_segments`

  - 重新生成FPS=10的数据，为了尝试解决codebook坍塌问题：
  python -m amplify.preprocessing.preprocess_my_segments \
    mode=tracks \
    source=/home/jay/action_ws/data/raw_video_d01 \
    output_dir=/home/jay/action_ws/data/preprocessed_data_d01_m10 \
    n_tracks=400 init_queries=uniform reinit=true \
    horizon=16 target_fps=10 resize_shorter=480 \
    skip_exist=true verbose=true



### Train

- quick test 
  python amplify/train_motion_tokenizer.py \
  root_dir=/home/jay/action_ws/data/preprocessed_data_d01 \
  cond_cameraviews=[default] \
  keys_to_load=[tracks,images] \
  true_horizon=16 track_pred_horizon=16 \
  batch_size=16 gpu_max_bs=16 num_epochs=30 \
  quick=true num_workers=4 log_interval=16 \
  resume=false run_name=new_quick_train_1022_d01_bs8 \
  use_wandb=true lr_schedule=null

- complete test
  python amplify/train_motion_tokenizer.py \
  root_dir=/home/jay/action_ws/data/preprocessed_data_d01 \
  cond_cameraviews=[default] \
  keys_to_load=[tracks,images] \
  true_horizon=16 track_pred_horizon=16 \
  batch_size=16 gpu_max_bs=16 num_epochs=30 \
  quick=false num_workers=4 log_interval=16 \
  resume=false run_name=new_complete_train_1022_d01_bs8 \
  use_wandb=true lr_schedule=null

- codebook collapse test (我希望用到少量部分数据，而不是全部数据)
  python amplify/train_motion_tokenizer.py \
    root_dir=/home/jay/action_ws/data/preprocessed_data_d01 \
    train_datasets=[custom_segments:traj0.1] \
    val_datasets=[custom_segments:traj0.05] \
    cond_cameraviews=[default] \
    keys_to_load=[tracks,images] \
    true_horizon=16 track_pred_horizon=16 \
    batch_size=8 gpu_max_bs=8 num_epochs=2 \
    quick=false num_workers=4 log_interval=16 \
    resume=false run_name=codebook_collapse_regularization_focal_test_d01 \
    use_wandb=true lr_schedule=null

#### CoTracker可视化

- 对一个动作视频，分T=16滑动窗口，进行CoTracker跟踪（每窗口重采样） —— 更符合当前方法论 （prefered）

  python -m video_action_segmenter.window_track_and_save \
    --output-dir /home/jay/action_ws/video_action_segmenter/inference_outputs/windows \
    --target-fps 20 \
    --resize-shorter 480 \
    --grid-size 20 \
    --T 16 \
    --stride 4 \
    --device auto \
    --trail 15

#### Inference 

##### From raw long videos or live video streams

- 命令：
  ```bash
    python -m video_action_segmenter.stream_inference \
      --params video_action_segmenter/params.yaml
  ```

##### Action Energy Analysis （运动能量分析，探究指标优劣 - best: quantized + token_diff_l2_mean）

- 1. 先跑若干样本得到能量 JSONL (输出在energy_sweep_out文件夹下)
- 2. 用 compute_best_threshold.py 产出报告（输出在energy_sweep_report文件夹下）
- 3. 再在流式推理时读取报告阈值，在params中设置：
  - mode: "report"                # fixed | report；report 模式会从报告JSON读取阈值
  - report_path: "./video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff.json"

- 如后续更换设备或场景，建议抽取少量样本再跑一次 compute_best_threshold.py 更新报告阈值，而不是在线做自适应

- Compute best threshold (using smooth参数)
  conda run -n laps python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl video_action_segmenter/energy_sweep_out/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl  video_action_segmenter/energy_sweep_out/stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --smooth --smooth-method ema --smooth-alpha 0.7 --smooth-window 3 \
    --output-json video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff_smoothed.json