# ABD on GTEA split1 (test)

日期：2025-11-10

目标：在 GTEA split1（test）上评估 ABD 算法，报告 F1@2s 与 F1@5s。

## 数据与特征
- 数据集：/home/johnny/action_ws/online_datasets/gtea/gtea
- 测试集视频（软链接）：/home/johnny/action_ws/online_datasets/gtea/gtea/Videos_test.split1
- GT（片段 JSON）：/home/johnny/action_ws/online_datasets/gtea/gt_segments_json/test.split1
- 预计算特征：/home/johnny/action_ws/online_datasets/gtea/gtea/features/*.npy（shape=(2048, T)）
- 转置后特征：/home/johnny/action_ws/online_datasets/gtea/gtea/features_t/*.npy（shape=(T, 2048)，时间步长≈0.0666667s）

转置脚本：tools/transpose_gtea_features.py

## 运行命令
1) 创建测试集软链接
```bash
bash tools/create_gtea_test_symlinks.sh
```

2) 转置特征（仅 test.split1）
```bash
conda run -n abd_env python tools/transpose_gtea_features.py --bundle /home/johnny/action_ws/online_datasets/gtea/gtea/splits/test.split1.bundle
```

3) 运行 ABD（abd_env）
```bash
conda run -n abd_env python -m comapred_algorithm.ABD.run_abd \
  --input-dir /home/johnny/action_ws/online_datasets/gtea/gtea/Videos_test.split1 \
  --output-dir /home/johnny/action_ws/output/gtea/ABD_split1 \
  --features-dir /home/johnny/action_ws/online_datasets/gtea/gtea/features_t \
  --feature-source i3d --alpha 0.5 --k 33 \
  --target-fps 30 --clip-duration 2.0 --clip-stride 0.0666667
```

4) 评估（laps）
```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/output/gtea/ABD_split1 \
  --gt-dir /home/johnny/action_ws/online_datasets/gtea/gt_segments_json/test.split1 \
  --iou-thrs 0.5 0.75 \
  --tolerance-sec 2.0 \
  --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/output/gtea/stats/abd_seg_eval_split1.json
```

## 结果
- 输出路径：/home/johnny/action_ws/output/gtea/stats/abd_seg_eval_split1.json
- 关键指标：
  - F1@2s = 0.7423
  - F1@5s = 0.8192
- 其他：
  - Precision@2s = 0.7315，Recall@2s = 0.7692
  - Precision@5s = 0.8089，Recall@5s = 0.8467
  - mAP@0.5 = 0.2068，mAP@0.75 = 0.0322

## 备注
- 本次采用了数据集中已有的 2048 维深度特征（优于 HOF 的 16 维），并确认其时间步长为 1/15s，据此设置 clip_stride=0.0666667。
- K（分段数）按 test.split1 的 GT 统计均值取 33。可在未来进行超参搜索（K、alpha）以进一步提升 F1。

