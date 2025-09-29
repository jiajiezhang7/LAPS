# 青春修炼手册


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


### Train

- 冒烟测试成功
  python amplify/train_motion_tokenizer.py \
  root_dir=/media/johnny/Data/data_motion_tokenizer/preprocessed_data_d01 \
  train_datasets=[custom_segments:traj1.0] \
  val_datasets=null \
  cond_cameraviews=[default] \
  keys_to_load=[tracks,images] \
  img_shape=[898,1442] \
  true_horizon=16 track_pred_horizon=16 \
  batch_size=16 gpu_max_bs=16 num_epochs=1 \
  quick=true num_workers=2 log_interval=16 \
  resume=false run_name=smoke_custom_d01 \
  use_wandb=true lr_schedule=null

- 第二次测试
  python amplify/train_motion_tokenizer.py \
  root_dir=/media/johnny/Data/data_motion_tokenizer/preprocessed_data_d01_train_partly \
  train_datasets=[custom_segments:traj1.0] \
  val_datasets=null \
  cond_cameraviews=[default] \
  keys_to_load=[tracks,images] \
  img_shape=[898,1442] \
  true_horizon=16 track_pred_horizon=16 \
  batch_size=16 gpu_max_bs=16 num_epochs=30 \
  quick=true num_workers=2 log_interval=16 \
  resume=false run_name=smoke_custom_d01_moredata \
  use_wandb=true lr_schedule=null