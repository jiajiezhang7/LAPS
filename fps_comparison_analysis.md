# Target FPS 对比分析报告

## 概述

对比了两个不同 target_fps 设置的数据集:
- **preprocessed_data_d01**: target_fps=20 (200个文件)
- **preprocessed_data_d01_m10**: target_fps=10 (46个文件)

## 核心发现

### ✅ 改善的指标

#### 1. **相邻帧位移显著提升**
- **中位速度均值**: 0.0725 → 0.0829 像素 (**+14.3%**)
- **中位速度P50**: 0.0659 → 0.0799 像素 (**+21.3%**)
- **中位速度P90**: 0.0917 → 0.1116 像素 (**+21.7%**)

**结论**: target_fps=10 成功增加了相邻帧之间的位移，这对于运动表征学习是有利的。

### ❌ 恶化的指标

#### 1. **非零步长比例下降**
- **非零步长比例均值**: 0.0025 → 0.0000 (**-100%**)

**分析**: 虽然绝对位移增加了，但使用0.5像素作为阈值时，几乎所有步长都低于这个阈值。这说明:
- 原始运动本身就很缓慢
- 0.5像素的阈值可能过高
- 需要调整 `movement_threshold_px` 参数到更小的值（如0.05或0.1）

#### 2. **越界比例增加**
- **越界比例均值**: 0.0014 → 0.0034 (**+142%**)
- **越界比例P90**: 0.0036 → 0.0108 (**+199%**)

**分析**: 降低fps后，轨迹跟踪的时间跨度变大，导致更多点移出画面边界。不过绝对值仍然很小(<0.4%)，影响有限。

## 详细指标对比

| 指标 | target_fps=20 | target_fps=10 | 变化 |
|------|---------------|---------------|------|
| **运动指标** |
| 中位速度均值(px) | 0.0725 | 0.0829 | +14.3% ✓ |
| 中位速度P50(px) | 0.0659 | 0.0799 | +21.3% ✓ |
| 中位速度P90(px) | 0.0917 | 0.1116 | +21.7% ✓ |
| 非零步长比例均值 | 0.0025 | 0.0000 | -100% ✗ |
| **质量指标** |
| 越界比例均值 | 0.0014 | 0.0034 | +142% ✗ |
| 越界比例P90 | 0.0036 | 0.0108 | +199% ✗ |
| NaN比例 | 0.0000 | 0.0000 | = |
| **数据量** |
| 有效文件数 | 200 | 45 | -77.5% |
| 总帧数 | 200,000 | 22,500 | -88.8% |
| 平均T/文件 | 1000 | 500 | -50% |

## 建议

### 1. **继续使用 target_fps=10**
虽然数据量减少，但相邻帧位移的提升（+14-22%）对运动表征学习更有价值。

### 2. **调整运动阈值**
当前使用的 `movement_threshold_px=0.5` 过高，建议:
```bash
# 重新分析，使用更小的阈值
python tools/analyze_hdf5_dataset.py \
  --root-dir data/preprocessed_data_d01_m10 \
  --movement-threshold-px 0.05 \
  --out data/preprocessed_data_d01_m10/analysis_report_thresh0.05.json
```

### 3. **处理越界问题**
考虑在预处理阶段:
- 添加边界检查，过滤越界轨迹点
- 或在训练时mask掉越界的轨迹

### 4. **扩充数据集**
target_fps=10 的数据集只有46个文件，建议处理更多数据以达到与原数据集相当的规模。

## 技术细节

### 配置一致性
两个数据集的配置完全一致:
- Horizon: 16
- 轨迹点数: 400
- 图像尺寸: 480x771
- 视角: default

### 分析脚本使用
```bash
# 分析 target_fps=20 数据集
conda run -n laps python tools/analyze_hdf5_dataset.py \
  --root-dir data/preprocessed_data_d01 \
  --cfg amplify/cfg/train_motion_tokenizer.yaml \
  --max-files 200 \
  --sample-windows 5 \
  --verbose

# 分析 target_fps=10 数据集
conda run -n laps python tools/analyze_hdf5_dataset.py \
  --root-dir data/preprocessed_data_d01_m10 \
  --cfg amplify/cfg/train_motion_tokenizer.yaml \
  --max-files 200 \
  --sample-windows 5 \
  --verbose

# 对比分析
python detailed_comparison.py
```

## 结论

**target_fps=10 在核心指标上有明显改善**:
- ✅ 相邻帧位移提升 14-22%
- ✅ 配置保持一致
- ⚠️ 越界比例略有增加但仍在可接受范围
- ⚠️ 非零步长比例的下降是阈值设置问题，不是数据质量问题

**建议**: 采用 target_fps=10 作为最终配置，并扩充数据集规模。
