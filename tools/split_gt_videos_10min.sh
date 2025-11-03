#!/bin/bash
# 批量切割 D01 和 D02 目录下的视频为10分钟片段

set -e

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate laps

# 项目根目录
WORKSPACE_ROOT="/home/johnny/action_ws"
SCRIPT_PATH="${WORKSPACE_ROOT}/tools/split_videos_by_10min.py"
VIDEO_ROOT="${WORKSPACE_ROOT}/datasets/gt_raw_videos"

echo "=========================================="
echo "开始切割视频（10分钟/片段）"
echo "=========================================="

# 处理 D01
echo ""
echo "处理 D01 目录..."
python "${SCRIPT_PATH}" \
    --input-dir "${VIDEO_ROOT}/D01" \
    --output-dir "${VIDEO_ROOT}/D01_segments" \
    --duration 600

# 处理 D02
echo ""
echo "处理 D02 目录..."
python "${SCRIPT_PATH}" \
    --input-dir "${VIDEO_ROOT}/D02" \
    --output-dir "${VIDEO_ROOT}/D02_segments" \
    --duration 600

echo ""
echo "=========================================="
echo "所有视频切割完成！"
echo "=========================================="
echo "D01 输出: ${VIDEO_ROOT}/D01_segments"
echo "D02 输出: ${VIDEO_ROOT}/D02_segments"
