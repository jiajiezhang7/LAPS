# OTAS · Breakfast split1 & cam01 实验记录

本实验目标：在 Breakfast 数据集 split1 & cam01 配置上运行 OTAS 算法，产出 F1@2s 与 F1@5s 指标，并完整记录流程与命令。

- 数据集：Breakfast (split1，cam01 视角)
- 评估指标：F1@2s、F1@5s
- 约定 Conda 环境：
  - otas（训练/推理/帧准备）
  - laps（评估）
- 重要绝对路径：
  - 测试帧输出：/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_test
  - 训练帧输出：/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_train
  - video_info_test.pkl：/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_test.pkl
  - video_info_train.pkl：/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_train.pkl
  - 原始测试视频：/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_test.split1_cam01/*.avi
  - 原始训练视频：/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01/*.avi

## 步骤1/2 简述（已完成）
- 阅读 ABD/OTAS 相关文档与代码，明确 OTAS 依赖“逐帧图片”而非 I3D 特征。
- 确认 split1_cam01 划分文件与测试/训练数量：92/341。
- 规划两处最小化代码修改：
  1) tools/adapt_otas_to_segments.py 增加 .avi 回退探测（已完成，见下）
  2) code/main.py 为 pkl_folder_name 拼接 video_info 的 stem 用于隔离缓存（暂缓，原因见“设计与取舍”）

### 已修改代码
- tools/adapt_otas_to_segments.py：允许从 .mp4/.avi 自动回退，保证读取 fps/帧数正确。

## 步骤3 数据准备（已完成）

### 3.1 提帧结果核验
- 目录统计：frames_test 下共 92 个子目录（预期=92）
- 帧文件检查：92/92 有帧文件，0 空目录
- 总帧数：187,095
- 抽样：P03_cam01_cereals=834，P03_cam01_coffee=919，P03_cam01_friedegg=4268，P03_cam01_milk=1160，P03_cam01_salat=4451

### 3.2 生成 video_info_test.pkl
- 命令：
  - conda run -n otas python comapred_algorithm/OTAS/code/make_video_info_from_frames.py \
    --frames-dir /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_test \
    --videos-dir /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/videos \
    --out /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_test.pkl
- 结果：生成 92 条条目 → /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_test.pkl

### 3.3 数据集 dry-run 验证（绕过模型初始化，无需 GPU）
- 说明：main.py 的 --dry_run_dataset 在模型初始化之后执行，会触发 CUDA 检查；为避免 GPU 依赖，本次直接在 otas 环境下调用 dataset.Breakfast 构建 window 列表完成 dry-run。
- 命令（精简）：
  - conda run -n otas python -c "import sys; sys.path.insert(0,'/home/johnny/action_ws/comapred_algorithm/OTAS/code'); from arg_pars import opt; from dataset import Breakfast; opt.frame_path='/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_test'; opt.video_info_file='/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_test.pkl'; opt.dataset='BF'; opt.feature_model='tf'; opt.view='cam01'; opt.pkl_folder_name=opt.output_path+'OTAS/'+'BF_tf_cam01'; print('build...'); ds=Breakfast(seq_len=opt.seq_len,num_seq=opt.num_seq,downsample=opt.ds,pred_step=opt.pred_step,mode='val',view_filter=opt.view); print('size',len(ds))"
- 结果：已构建 window 列表，val 集 window 数=61,504；输出目录：/home/johnny/action_ws/comapred_algorithm/OTAS/output/OTAS/BF_tf_cam01/window_lists

### 3.4 小结
- 提帧完成：92/92 成功
- video_info_test.pkl：生成成功（92 条）
- dry-run：通过（窗口总数 61,504，无异常）

---

## 步骤4~7 进度

### 步骤4：训练集提帧与 video_info_train.pkl（进行中）
- 训练视频数：341（Videos_train.split1_cam01）
- 提帧命令：在 otas 环境运行 Python+ffmpeg 批处理脚本（逐视频生成 {P}_cam01_{act}/Frame_%06d.jpg）
- 日志：/home/johnny/action_ws/output/otas_logs/step4_extract_train.log（持续写入 [OK]/[SKIP]/[FAIL] 与最终统计）
- 当前状态：进行中（frames_train 子目录计数：193/341；正在遍历 *.avi 追加 [SKIP]/[OK]；监控日志：/home/johnny/action_ws/output/otas_logs/step4_extract_train.log；本轮进程 terminal_id=62）
- 下一步：提帧完成后，立即生成 /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_train.pkl，并对训练集执行一次 dataset 构建验证

### 步骤5（计划）：OTAS 训练与测试集推理（10 epochs）
- 训练命令（otas）：
  - cd /home/johnny/action_ws/comapred_algorithm/OTAS/code && \
    python main.py --mode train --num_epoch 10 \
      --frame-path /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_train \
      --video-info-file /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_train.pkl \
      --view cam01 --dataset BF --feature_model tf
- 验证/推理命令（otas）：
  - python main.py --mode val \
      --frame-path /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_test \
      --video-info-file /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/video_info_test.pkl \
      --view cam01 --dataset BF --feature_model tf

### 步骤6（计划）：后处理与评估（laps）
- 边界→分段：python tools/adapt_otas_to_segments.py ... （支持 .mp4/.avi 自动回退）
- 评估：在 laps 环境运行 tools/eval_segmentation.py，统计 F1@2s / F1@5s

### 步骤7（计划）：汇总
- 汇总训练/推理/评估的所有命令、参数与指标，整理为最终结果段落
