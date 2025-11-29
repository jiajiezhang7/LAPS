#!/bin/bash
# 快速批量生成能量曲线图的便捷脚本

# 默认参数
JSONL="./data/YOUR_DATA_PATH"
PARAMS="./video_action_segmenter/params_d02.yaml"
SEGMENT_LENGTH=130
NUM_PLOTS=50

# 激活 conda 环境
echo "激活 conda 环境 laps..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate laps

# 运行批量生成脚本
echo "开始批量生成 ${NUM_PLOTS} 张图片 (segment_length=${SEGMENT_LENGTH})..."
python -m video_action_segmenter.scripts.batch_plot_energy_segments \
  --jsonl "${JSONL}" \
  --params "${PARAMS}" \
  --segment-length ${SEGMENT_LENGTH} \
  --num-plots ${NUM_PLOTS}

echo "完成!"
