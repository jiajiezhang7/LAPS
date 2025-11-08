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
- 默认使用 HOF（Histogram of Optical Flow）片段级特征（CPU 提取，OpenCV），输出矩阵 X∈R^{N×D}
- I3D 特征作为可选方案（GPU 资源充足时）
- 不再使用 LAPS latent（prequant/quantized）

**时间映射：**
- sec ≈ clip_idx × clip_stride（默认 clip_stride=0.4s）

**输出格式：**
- 路径：/home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW}_ABD/{video_stem}/segmented_videos/{video_stem}_segments.json
- 段 JSON 元数据包含：
  - video：视频文件名
  - segments：[{start_sec, end_sec}, ...]
  - video_duration_sec：视频总时长
  - segmentation_params：{source:"hof", alpha, k, stride, target_fps, view, clip_duration, clip_stride}
  - processed_at：处理时间戳

### 运行命令
### HOF 特征提取

#### D01 HOF 提取
```bash
conda run -n abd_env python -m comapred_algorithm.ABD.batch_extract_hof \
  --view D01 --clip-duration 2.0 --clip-stride 0.4 --bins 16
```
输出目录：/home/johnny/action_ws/comapred_algorithm/ABD/hof_features/D01/{video_stem}.npy

#### D02 HOF 提取
```bash
conda run -n abd_env python -m comapred_algorithm.ABD.batch_extract_hof \
  --view D02 --clip-duration 2.0 --clip-stride 0.4 --bins 16
```
输出目录：/home/johnny/action_ws/comapred_algorithm/ABD/hof_features/D02/{video_stem}.npy


#### D01 分割
```bash
conda run -n laps python -m comapred_algorithm.ABD.run_abd \
  --view D01 \
  --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D01 \
  --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_ABD_HOF \
  --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/hof_features/D01 \
  --feature-source hof --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4
```

#### D02 分割
```bash
conda run -n laps python -m comapred_algorithm.ABD.run_abd \
  --view D02 \
  --input-dir /home/johnny/action_ws/datasets/gt_raw_videos/D02 \
  --output-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_ABD_HOF \
  --features-dir /home/johnny/action_ws/comapred_algorithm/ABD/hof_features/D02 \
  --feature-source hof --alpha 0.5 --k auto --target-fps 30 --clip-duration 2.0 --clip-stride 0.4
```

### 评估

与 LAPS/OTAS 使用相同脚本

#### D01 评估
```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_ABD_HOF \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D01 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_ABD.json
```

#### D02 评估
```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_ABD_HOF \
  --gt-dir /home/johnny/action_ws/datasets/gt_annotations/D02 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_ABD.json
```

### 环境

- 运行环境：
  - ABD 分割/评估：laps 环境
  - HOF 特征提取：abd_env 环境（已验证 OpenCV 可用）
  ```bash
  conda run -n abd_env python -c "import cv2; print(cv2.__version__)"
  ```

---

## 预期输出与检查点

- **分割输出**：segmentation_outputs/{D01_ABD,D02_ABD} 下有 segments.json 文件
- **评估结果**：stats/seg_eval/seg_eval_{D01,D02}_ABD.json 存在且指标合理
- **对比表格**：tables/table1.csv 包含 LAPS vs ABD vs Optical Flow vs OTAS 的对比

---

## 详细 ToDoList

### ABD 实验任务
- [ ] 生成 HOF 特征目录：/home/johnny/action_ws/comapred_algorithm/ABD/hof_features/{D01,D02}
- [ ] 运行 D01 ABD 分割（使用 HOF）：输出到 D01_ABD_HOF
- [ ] 运行 D02 ABD 分割（使用 HOF）：输出到 D02_ABD_HOF
- [ ] 评估 D01 ABD（HOF）结果

- [ ] 评估 D02 ABD（HOF）结果
- [ ] 生成对比表格 (Table 1)
- [x] （可选）最小代码修改：`run_abd.py` 支持 `--feature-source hof` 并将 `meta_params["source"]` 写入对应值（仅影响元数据）

---

## 进度追踪

- 当前状态：已实现 HOF 特征提取脚本并启动 D01 批量提取（后台运行）
- 依赖项：
  - [ ] HOF 特征提取（进行中：D01 已启动；输出目录 /home/johnny/action_ws/comapred_algorithm/ABD/hof_features/{D01,D02}）
  - [x] GT 标注已准备
  - [x] 评估脚本已准备（tools/eval_segmentation.py）
  - [ ] ABD 分割运行（D01_ABD_HOF / D02_ABD_HOF）
  - [ ] 评估与对比

---

## 关键参数说明

- 今日更新（2025-11-07）：
  - 新增 HOF 提取脚本：`comapred_algorithm/ABD/features_hof.py`；批处理脚本：`comapred_algorithm/ABD/batch_extract_hof.py`
  - `run_abd.py`：新增 `--feature-source hof`；元数据 `meta_params['source']` 随参数写入
  - 在 `abd_env` 验证 OpenCV 可用（4.12.0）
  - 已后台启动 D01 HOF 批量提取：`conda run -n abd_env python -m comapred_algorithm.ABD.batch_extract_hof --view D01`

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

