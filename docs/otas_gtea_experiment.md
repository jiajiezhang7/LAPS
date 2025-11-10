# OTAS 在 GTEA split1 实验记录

- 日期：2025-11-10
- 目的：在 GTEA 数据集 split1 上运行 OTAS，获得 F1@2s 与 F1@5s，并记录完整流程与配置。
- 数据集：GTEA split1（测试 7 段）

---

## 步骤 1：研究参考文档与现有实验（完成）
- 参考：docs/gtea_exp.md（LAPS 评估协议与数据布局）、comapred_algorithm/OTAS/otas_experiment.md（OTAS 既有流水线）。
- 评估脚本：tools/eval_segmentation.py（conda: laps）。
- GTEA 目录：
  - 视频：online_datasets/gtea/gtea/Videos_test.split1/*.mp4
  - GT 段标注：online_datasets/gtea/gt_segments_json/test.split1/*_segments.json
  - 划分清单：online_datasets/gtea/gtea/splits/test.split1.bundle

## 步骤 2：理解 OTAS 实现与输入/输出（完成）
- 数据接口：
  - 视频目录：{video_path}/{P}/{cam}/{P_*}.mp4（本实验拟用 P=GTEA, cam=GTEA）
  - 帧目录：{frame_path}/{P}_{cam}_{full_act}/Frame_%06d.jpg
  - video_info.pkl：key=vpath（视频绝对路径字符串），value={true_vlen, lose}
- 推理输出：mean_error/*.pkl → detect_seg/*.pkl（bdy_idx_list）
- 适配：为 GTEA 新增 tools/adapt_otas_to_segments_gtea.py（已创建），将 video_id=GTEA_GTEA_{stem} → stem，写出 LAPS 兼容 segments.json。

## 步骤 3：数据准备与格式对接（完成）
- 存储策略（按最新指示）：放弃符号链接，所有数据均直接存放在工作区真实目录 /home/johnny/action_ws/comapred_algorithm/OTAS/data/gtea。
- 执行内容：
  1) 复制 7 个测试视频至 videos/GTEA/GTEA/，重命名为 GTEA_{stem}.mp4。
     复制日志节选：
       [COPY] .../S1_Cheese_C1.mp4 -> .../GTEA_S1_Cheese_C1.mp4
       [COPY] .../S1_CofHoney_C1.mp4 -> .../GTEA_S1_CofHoney_C1.mp4
       [COPY] .../S1_Coffee_C1.mp4 -> .../GTEA_S1_Coffee_C1.mp4
       [COPY] .../S1_Hotdog_C1.mp4 -> .../GTEA_S1_Hotdog_C1.mp4
       [COPY] .../S1_Pealate_C1.mp4 -> .../GTEA_S1_Pealate_C1.mp4
       [COPY] .../S1_Peanut_C1.mp4 -> .../GTEA_S1_Peanut_C1.mp4
       [COPY] .../S1_Tea_C1.mp4 -> .../GTEA_S1_Tea_C1.mp4
  2) 在 otas 环境运行抽帧并生成 video_info.pkl。
     运行日志节选：
       Done for: ../data/gtea/videos/GTEA/GTEA/GTEA_S1_Cheese_C1.mp4
       ...（其余 6 段均已 Done）
- 完整性验证：
  - 帧目录数量：7
  - 各目录帧数（节选）：
    - GTEA_GTEA_S1_Cheese_C1: 943
    - GTEA_GTEA_S1_CofHoney_C1: 1235
    - GTEA_GTEA_S1_Coffee_C1: 1178
    - GTEA_GTEA_S1_Hotdog_C1: 718
    - GTEA_GTEA_S1_Pealate_C1: 1384
    - GTEA_GTEA_S1_Peanut_C1: 1643
    - GTEA_GTEA_S1_Tea_C1: 2009
  - video_info.pkl 条目：7（键均为 ../data/gtea/videos/GTEA/GTEA/GTEA_{stem}.mp4）

---

## 关键产物与路径（已更新）
- 适配脚本：tools/adapt_otas_to_segments_gtea.py（已创建）
- 数据根：/home/johnny/action_ws/comapred_algorithm/OTAS/data/gtea
  - videos/GTEA/GTEA/GTEA_{stem}.mp4（已复制的 MP4 文件）
  - frames/GTEA_GTEA_{stem}/Frame_%06d.jpg（抽帧输出）
  - video_info.pkl（抽帧统计信息）

## 已执行命令与日志摘要
- 复制测试视频：
```bash
WS_ROOT=/home/johnny/action_ws
DST_BASE=$WS_ROOT/comapred_algorithm/OTAS/data/gtea
SRC_DIR=$WS_ROOT/online_datasets/gtea/gtea/Videos_test.split1
BUNDLE=$WS_ROOT/online_datasets/gtea/gtea/splits/test.split1.bundle
mkdir -p "$DST_BASE/videos/GTEA/GTEA" "$DST_BASE/frames"
while IFS= read -r line; do
  stem="${line%.txt}"
  cp -f "$SRC_DIR/${stem}.mp4" "$DST_BASE/videos/GTEA/GTEA/GTEA_${stem}.mp4"
done < "$BUNDLE"
```
- 抽帧并生成 video_info.pkl（conda: otas）：
```bash
cd /home/johnny/action_ws/comapred_algorithm/OTAS/code
conda run -n otas python video_info.py \
  --video-path ../data/gtea/videos \
  --frame-path ../data/gtea/frames \
  --video-info-file ../data/gtea/video_info.pkl \
  --dataset BF
```
- 本步日志节选：
```text
[COPY] ...S1_Cheese_C1.mp4 -> .../GTEA_S1_Cheese_C1.mp4
...
[COPY] ...S1_Tea_C1.mp4 -> .../GTEA_S1_Tea_C1.mp4
Done for: ../data/gtea/videos/GTEA/GTEA/GTEA_S1_Cheese_C1.mp4
...
Done for: ../data/gtea/videos/GTEA/GTEA/GTEA_S1_Tea_C1.mp4
```
- 后续步骤所需命令（预设）：
```bash
# 结果适配（laps）
conda run -n laps python /home/johnny/action_ws/tools/adapt_otas_to_segments_gtea.py \
  --otas-pred <OTAS_BF_tf_root> \
  --raw-dir /home/johnny/action_ws/online_datasets/gtea/gtea/Videos_test.split1 \
  --output /home/johnny/action_ws/datasets/output/segmentation_outputs/GTEA_split1_OTAS

# 评估（laps）
conda run -n laps python /home/johnny/action_ws/tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/GTEA_split1_OTAS \
  --gt-dir /home/johnny/action_ws/online_datasets/gtea/gt_segments_json/test.split1 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_GTEA_split1_OTAS.json
```

---

## 备注
- conda 环境：
  - 抽帧/OTAS 推理：otas
  - 评估与适配：laps
- 命名规则：为满足 OTAS 数据装载逻辑，视频文件名为 GTEA_{stem}.mp4，帧目录为 GTEA_GTEA_{stem}/Frame_%06d.jpg。

