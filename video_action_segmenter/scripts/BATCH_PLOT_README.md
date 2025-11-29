# 批量生成能量曲线图工具

## 功能说明

批量调用 `plot_energy_segment_from_jsonl_for_paperteaser.py` 生成多张能量曲线图,用于从中选择最佳的三个动作分割曲线。

每次运行会生成指定数量的图片,每张图片使用不同的随机种子,从而获得不同的随机片段。

## 文件说明

- **`batch_plot_energy_segments.py`**: 主批量生成脚本
- **`quick_batch_plot.sh`**: 快速调用脚本(使用默认参数)
- **`BATCH_PLOT_README.md`**: 本说明文档

## 使用方法

### 方法 1: 使用快速脚本(推荐)

```bash
# 激活 conda 环境并运行
conda activate laps
./video_action_segmenter/scripts/quick_batch_plot.sh
```

这将使用默认参数生成 20 张图片,片段长度为 100。

### 方法 2: 自定义参数

```bash
conda activate laps

python -m video_action_segmenter.scripts.batch_plot_energy_segments \
  --jsonl /path/to/stream_energy.jsonl \
  --params ./video_action_segmenter/params_d02.yaml \
  --segment-length 100 \
  --num-plots 20 \
  --output-dir ./video_action_segmenter/figures/my_batch
```

## 参数说明

### 必需参数

- `--jsonl`: energy JSONL 文件路径

### 可选参数

- `--params`: params.yaml 配置文件路径 (默认: `./video_action_segmenter/params.yaml`)
- `--segment-length`: 连续窗口片段长度,**固定值** (默认: 100)
- `--num-plots`: 生成图片的数量 (默认: 20)
- `--output-dir`: 输出目录 (默认: 自动创建带时间戳的目录)
- `--seed-start`: 起始随机种子 (默认: 1000)
- `--viz-style`: 可视化风格 (默认: "enhanced")
- `--theme`: 主题风格 (默认: "academic_blue")
- `--plot-width`: 图片宽度 (默认: 800)
- `--plot-height`: 图片高度 (默认: 240)
- `--y-min`: Y轴最小值 (可选)
- `--y-max`: Y轴最大值 (可选)

## 输出说明

### 输出目录结构

```
video_action_segmenter/figures/batch_YYYYMMDD_HHMMSS/
├── seg_len100_seed1000.png
├── seg_len100_seed1001.png
├── seg_len100_seed1002.png
├── ...
├── seg_len100_seed1019.png
└── index.txt
```

### 文件命名规则

- 格式: `seg_len{segment_length}_seed{seed:04d}.png`
- 示例: `seg_len100_seed1000.png` (片段长度=100, 种子=1000)

### 索引文件

每次批量生成会创建一个 `index.txt` 文件,记录:
- 生成参数
- 成功/失败统计
- 每个文件的状态

## 使用示例

### 示例 1: 生成 20 张图片,片段长度 100

```bash
python -m video_action_segmenter.scripts.batch_plot_energy_segments \
  --jsonl ./data/YOUR_DATA_PATH
  --params ./video_action_segmenter/params_d02.yaml \
  --segment-length 100 \
  --num-plots 20
```

### 示例 2: 生成 50 张图片,片段长度 150

```bash
python -m video_action_segmenter.scripts.batch_plot_energy_segments \
  --jsonl /path/to/energy.jsonl \
  --params ./video_action_segmenter/params_d02.yaml \
  --segment-length 150 \
  --num-plots 50 \
  --output-dir ./figures/len150_batch
```

### 示例 3: 自定义 Y 轴范围

```bash
python -m video_action_segmenter.scripts.batch_plot_energy_segments \
  --jsonl /path/to/energy.jsonl \
  --segment-length 100 \
  --num-plots 20 \
  --y-min 0.0 \
  --y-max 1.5
```

## 工作流程

1. **批量生成**: 运行脚本生成 20 张图片
2. **人工筛选**: 浏览生成的图片,选择最佳的 3 张
3. **记录种子**: 记下选中图片的种子值(从文件名中获取)
4. **重新生成**: 如需要,可以使用特定种子重新生成单张图片

## 重新生成特定种子的图片

如果你找到了理想的图片(例如 `seg_len100_seed1005.png`),可以使用原始脚本重新生成:

```bash
python -m video_action_segmenter.scripts.plot_energy_segment_from_jsonl_for_paperteaser \
  --jsonl /path/to/energy.jsonl \
  --params ./video_action_segmenter/params_d02.yaml \
  --segment-length 100 \
  --seed 1005 \
  --output ./final_figure.png
```

## 注意事项

1. **固定片段长度**: `--segment-length` 参数在所有生成的图片中保持一致
2. **随机种子递增**: 每张图片使用递增的随机种子 (seed_start, seed_start+1, ...)
3. **conda 环境**: 确保在 `laps` 环境中运行
4. **磁盘空间**: 20 张图片大约需要 1-5 MB 空间

## 故障排除

### 问题: 脚本无法找到模块

```bash
# 确保在正确的目录运行
cd /path/to/LAPS  # Change to your workspace

# 激活正确的 conda 环境
conda activate laps
```

### 问题: JSONL 文件不存在

检查文件路径是否正确:
```bash
ls -lh /path/to/your/energy.jsonl
```

### 问题: 生成的图片质量不理想

尝试调整参数:
- 增加 `--num-plots` 以获得更多选择
- 调整 `--segment-length` 改变片段长度
- 修改 `--y-min` 和 `--y-max` 调整 Y 轴范围
