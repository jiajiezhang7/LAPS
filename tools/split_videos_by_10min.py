#!/usr/bin/env python3
"""
将视频按10分钟为单位切割成片段
"""
import os
import argparse
import subprocess
from pathlib import Path


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        print(f"无法获取视频时长: {video_path}")
        return None


def split_video(input_path, output_dir, segment_duration=600):
    """
    将视频切割成固定时长的片段
    
    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        segment_duration: 片段时长（秒），默认600秒=10分钟
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取视频时长
    duration = get_video_duration(input_path)
    if duration is None:
        return
    
    # 计算需要切割的片段数
    num_segments = int(duration // segment_duration) + (1 if duration % segment_duration > 0 else 0)
    
    print(f"\n处理视频: {input_path.name}")
    print(f"  时长: {duration:.2f}秒 ({duration/60:.2f}分钟)")
    print(f"  将切割为 {num_segments} 个片段")
    
    # 切割视频
    for i in range(num_segments):
        start_time = i * segment_duration
        output_name = f"{input_path.stem}_seg{i+1:03d}{input_path.suffix}"
        output_path = output_dir / output_name
        
        # 先尝试使用流复制（快速）
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            '-y',
            str(output_path)
        ]
        
        print(f"  切割片段 {i+1}/{num_segments}: {output_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 如果流复制失败（通常是音频编码不兼容），则重新编码音频
        if result.returncode != 0 and 'codec not currently supported' in result.stderr:
            print(f"    流复制失败，重新编码音频...")
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(segment_duration),
                '-c:v', 'copy',  # 视频流复制
                '-c:a', 'aac',   # 音频重新编码为 AAC
                '-b:a', '128k',  # 音频比特率
                '-avoid_negative_ts', '1',
                '-y',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    错误: {result.stderr}")
        else:
            print(f"    完成")


def process_directory(input_dir, output_dir, segment_duration=600):
    """处理目录下的所有视频文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"在 {input_dir} 中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for video_file in sorted(video_files):
        split_video(video_file, output_dir, segment_duration)
    
    print(f"\n所有视频处理完成！输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='将视频按固定时长切割成片段')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入视频目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出片段目录')
    parser.add_argument('--duration', type=int, default=600,
                        help='片段时长（秒），默认600秒=10分钟')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.duration)


if __name__ == '__main__':
    main()
