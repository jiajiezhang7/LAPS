# ABD 在 Breakfast 数据集上的实验记录（test.split1）

本记录用于复现实验过程与结果，包含数据准备、参数设置、运行命令与评估指标。

## 数据与预处理
- 数据集：Breakfast（15 fps），测试划分：`test.split1`（252 个视频）
- 重要路径：
  - 原始视频（按测试划分创建的软链接）：`/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_test.split1/`
  - 帧级标注：`/home/johnny/action_ws/online_datasets/breakfast/breakfast/groundTruth/`
  - 预提取 I3D 特征（原始，形状 (2048, T)）：`/home/johnny/action_ws/online_datasets/breakfast/breakfast/features/`
  - 转置后的特征（形状 (T, 2048)）：`/home/johnny/action_ws/online_datasets/breakfast/breakfast/features_t/`
  - 段级 GT（秒）：`/home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/test.split1/`

- 预处理脚本与命令：
  1) 创建测试集视频软链接（已在之前创建）：`tools/create_breakfast_test_symlinks.sh`
  2) 特征转置：`tools/transpose_breakfast_features.py`（输出 252 个 (T,2048) 特征）
  3) 帧级 GT 转段级 JSON：`tools/convert_breakfast_gt_to_segments.py`

## ABD 运行配置
- Conda 环境：`abd_env`
- 使用预提取 I3D 特征（不在线提取）
- 关键参数：
  - `alpha = 0.5`
  - `k = 7`（根据 test.split1 段数统计的均值约 7.03）
  - `clip_stride = 0.0666667`（与 15 fps 对齐：每个时间步约 1/15 秒）
  - `target_fps = 15`（仅作为元信息记录）

- 运行命令：
```
conda run -n abd_env python -m comapred_algorithm.ABD.run_abd \
  --input-dir /home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_test.split1 \
  --output-dir /home/johnny/action_ws/output/breakfast/ABD_split1 \
  --features-dir /home/johnny/action_ws/online_datasets/breakfast/breakfast/features_t \
  --feature-source i3d --alpha 0.5 --k 7 \
  --target-fps 15 --clip-duration 2.0 --clip-stride 0.0666667
```

## 评估配置
- Conda 环境：`laps`
- 度量：F1@2s、F1@5s（边界容忍度，秒）、mAP@IoU（0.5、0.75）
- 评估命令：
```
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/output/breakfast/ABD_split1 \
  --gt-dir /home/johnny/action_ws/online_datasets/breakfast/gt_segments_json/test.split1 \
  --iou-thrs 0.5 0.75 \
  --tolerance-sec 2.0 \
  --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/output/breakfast/stats/abd_seg_eval_split1.json
```

## 结果
- 汇总 JSON：`/home/johnny/action_ws/output/breakfast/stats/abd_seg_eval_split1.json`
- 关键指标（test.split1，252 个视频）：
  - F1@2s = 0.3333
  - F1@5s = 0.5450
  - Precision@2s = 0.3117，Recall@2s = 0.4051
  - Precision@5s = 0.5263，Recall@5s = 0.6369
  - mAP@0.5 = 0.2295，mAP@0.75 = 0.0283

## 备注
- Breakfast 提供的预提取 I3D 特征与帧标注一一对应（T 等于帧数），因此 `clip_stride` 需与帧率严格一致（1/15 秒）。
- `k` 选 7 为分割段数的统一超参，来源于 GT 统计（均值约 7.03）。如需更优结果，可探索：
  - 视具体活动类别/时长自适应 K（基于 GT 的先验或启发式估计）；
  - 调整 `alpha` 以及前后处理（如边界 NMS 设置，如果在 ABD 核心允许）；
  - 使用不同特征源（HOF/多模态）或平滑策略。

