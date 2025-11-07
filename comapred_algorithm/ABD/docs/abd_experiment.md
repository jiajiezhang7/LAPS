# ABD 算法实验方案与进度追踪

更新时间：2025-11-07
源文档：/home/johnny/action_ws/docs/experiment_detailed_todo.md

---

## 全局约定

- 根目录：/home/johnny/action_ws
- 视角数据：
  - D01：/home/johnny/action_ws/datasets/gt_raw_videos/D01
  - D02：/home/johnny/action_ws/datasets/gt_raw_videos/D02
- GPU：1 块（ID=0）
- 输出根目录：/home/johnny/action_ws/datasets/output
- 存储迁移与符号链接：
  - /home/johnny/action_ws/comapred_algorithm/ABD/i3d_features → 符号链接 → /media/johnny/48FF-AA60/ABD/i3d_features
  - 说明：文档与脚本中的路径仍以 /home/johnny/action_ws/... 为准（符号链接透明生效）。若需直接访问外部盘，请替换为右侧实际路径。

---

## 阶段2 Q2：ABD 基线（无训练离线算法）

### 目标
复现 ABD（CVPR'22，无训练）离线算法，直接读取 D01/D02 原始视频，产出与 LAPS 兼容的分割结果，用同一评估脚本比较。

### 数据与接口约定

**输入：**
- D01/D02 原始视频：/home/johnny/action_ws/datasets/gt_raw_videos/{D01,D02}

**特征：**
- 使用 I3D 特征（通过 abd_env 批量提取）
- 不再使用 LAPS latent（prequant/quantized）

**时间映射：**
- sec ≈ clip_idx × clip_stride（默认 clip_stride=0.4s）

**输出格式：**
- 路径：/home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW}_ABD/{video_stem}/segmented_videos/{video_stem}_segments.json
- 段 JSON 元数据包含：
  - video：视频文件名
  - segments：[{start_sec, end_sec}, ...]
  - video_duration_sec：视频总时长
  - segmentation_params：{source:"i3d", alpha, k, stride, target_fps, view, clip_duration, clip_stride}
  - processed_at：处理时间戳

### 运行命令

#### D01 分割
```bash
conda run -n laps python -m comapred_algorithm.ABD.run_abd \
  --view D01 \
  --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D01 \
  --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_ABD \
  --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/i3d_features/D01 \
  --feature-source i3d --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4
```

#### D02 分割
```bash
conda run -n laps python -m comapred_algorithm.ABD.run_abd \
  --view D02 \
  --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D02 \
  --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_ABD \
  --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/i3d_features/D02 \
  --feature-source i3d --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4
```

### 评估

与 LAPS/OTAS 使用相同脚本

#### D01 评估
```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_ABD \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_ABD.json
```

#### D02 评估
```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_ABD \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D02 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_ABD.json
```

### 环境

- 推荐复用 laps 环境（ABD 仅依赖 numpy/scipy）
- 如后续引入额外特征提取器，可新建 abd_env（此处仅文档记录，暂不创建）

---

## 预期输出与检查点

- **分割输出**：segmentation_outputs/{D01_ABD,D02_ABD} 下有 segments.json 文件
- **评估结果**：stats/seg_eval/seg_eval_{D01,D02}_ABD.json 存在且指标合理
- **对比表格**：tables/table1.csv 包含 LAPS vs ABD vs Optical Flow vs OTAS 的对比

---

## 详细 ToDoList

### ABD 实验任务
- [ ] 验证 I3D 特征目录存在：/home/johnny/action_ws/comapred_algorithm/ABD/i3d_features/{D01,D02}
- [ ] 运行 D01 ABD 分割
- [ ] 运行 D02 ABD 分割
- [ ] 评估 D01 ABD 结果
- [ ] 评估 D02 ABD 结果
- [ ] 生成对比表格 (Table 1)

---

## 进度追踪

- 当前状态：待启动
- 依赖项：
  - [x] I3D 特征提取完成（符号链接已配置）
  - [x] GT 标注已准备
  - [x] 评估脚本已准备（tools/eval_segmentation.py）
  - [ ] ABD 分割运行
  - [ ] 评估与对比

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| alpha | 0.5 | 能量阈值参数 |
| k | auto | 聚类簇数（auto 自动搜索） |
| target_fps | 30 | 目标帧率 |
| clip_duration | 2.0 | 视频片段时长（秒） |
| clip_stride | 0.4 | 片段步长（秒） |

---

## 评估指标

- **F1@2s / F1@5s**：时间容差 2s/5s 下的 F1 分数
- **mAP@0.5 / mAP@0.75**：IoU 阈值 0.5/0.75 下的平均精度
- **Precision / Recall**：精准率与召回率

---

## 参考文献

- ABD: Action Boundary Detection with Deep Boundary-Aware Features (CVPR'22)
- 与 LAPS、Optical Flow Baseline、OTAS 进行无监督动作分割性能对比
