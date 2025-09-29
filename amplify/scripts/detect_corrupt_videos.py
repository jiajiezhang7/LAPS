#!/usr/bin/env python3
"""
快速检测并删除损坏的MP4视频文件

该脚本递归扫描指定目录下的所有.mp4文件，使用ffmpeg快速检测是否损坏，
并可选择性地删除损坏的文件。

使用方法:
    python detect_corrupt_videos.py [--dry-run] [--timeout SECONDS] [--threads N]
    
示例:
    # 仅检测，不删除（推荐先运行）
    python detect_corrupt_videos.py --dry-run
    
    # 检测并删除损坏文件
    python detect_corrupt_videos.py
    
    # 使用128个线程并行检测，超时时间10秒
    python amplify/scripts/detect_corrupt_videos.py --threads 128 --timeout 10
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('corrupt_video_detection.log')
    ]
)
logger = logging.getLogger(__name__)

class CorruptVideoDetector:
    def __init__(self, root_dir: str, timeout: int = 5, dry_run: bool = False):
        """
        初始化损坏视频检测器
        
        Args:
            root_dir: 要扫描的根目录
            timeout: ffmpeg检测超时时间（秒）
            dry_run: 是否为试运行模式（不实际删除文件）
        """
        self.root_dir = Path(root_dir)
        self.timeout = timeout
        self.dry_run = dry_run
        self.corrupt_files = []
        self.valid_files = []
        self.error_files = []
        
        # ffmpeg错误关键词，用于识别损坏文件
        self.error_keywords = [
            'Invalid data found',
            'moov atom not found',
            'truncated',
            'corrupted',
            'damaged',
            'incomplete',
            'No such file',
            'Permission denied',
            'Header missing',
            'End of file',
            'Invalid argument'
        ]
    
    def find_mp4_files(self) -> List[Path]:
        """递归查找所有.mp4文件"""
        logger.info(f"正在扫描目录: {self.root_dir}")
        
        mp4_files = []
        for file_path in self.root_dir.rglob("*.mp4"):
            if file_path.is_file():
                mp4_files.append(file_path)
        
        logger.info(f"找到 {len(mp4_files)} 个MP4文件")
        return mp4_files
    
    def check_video_integrity(self, video_path: Path) -> Tuple[str, bool, str]:
        """
        使用ffmpeg检查单个视频文件的完整性
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Tuple[文件路径, 是否损坏, 错误信息]
        """
        try:
            # 使用ffmpeg快速检测：尝试读取第一帧
            cmd = [
                'ffmpeg',
                '-v', 'error',  # 只显示错误信息
                '-i', str(video_path),
                '-t', '0.1',  # 只处理0.1秒
                '-f', 'null',  # 输出到null
                '-'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # 检查返回码和错误输出
            if result.returncode != 0:
                error_msg = result.stderr.lower()
                
                # 检查是否包含损坏相关的错误关键词
                is_corrupt = any(keyword.lower() in error_msg for keyword in self.error_keywords)
                
                if is_corrupt:
                    return str(video_path), True, result.stderr.strip()
                else:
                    # 可能是其他类型的错误（如编码不支持等），不算损坏
                    return str(video_path), False, f"其他错误: {result.stderr.strip()}"
            
            # 返回码为0，文件正常
            return str(video_path), False, "正常"
            
        except subprocess.TimeoutExpired:
            logger.warning(f"检测超时: {video_path}")
            return str(video_path), True, "检测超时（可能损坏）"
            
        except FileNotFoundError:
            logger.error("ffmpeg未找到，请确保已安装ffmpeg")
            sys.exit(1)
            
        except Exception as e:
            logger.error(f"检测文件时出错 {video_path}: {e}")
            return str(video_path), False, f"检测异常: {str(e)}"
    
    def detect_corrupt_videos(self, max_workers: int = 4) -> None:
        """
        并行检测所有视频文件
        
        Args:
            max_workers: 最大并行线程数
        """
        mp4_files = self.find_mp4_files()
        
        if not mp4_files:
            logger.info("未找到MP4文件")
            return
        
        logger.info(f"开始检测，使用 {max_workers} 个线程")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有检测任务
            future_to_file = {
                executor.submit(self.check_video_integrity, file_path): file_path 
                for file_path in mp4_files
            }
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_file):
                file_path, is_corrupt, error_msg = future.result()
                completed += 1
                
                if is_corrupt:
                    self.corrupt_files.append((file_path, error_msg))
                    logger.warning(f"[{completed}/{len(mp4_files)}] 损坏: {file_path}")
                    logger.warning(f"  错误信息: {error_msg}")
                else:
                    if "其他错误" in error_msg:
                        self.error_files.append((file_path, error_msg))
                        logger.info(f"[{completed}/{len(mp4_files)}] 其他错误: {file_path}")
                    else:
                        self.valid_files.append(file_path)
                        if completed % 50 == 0:  # 每50个文件报告一次进度
                            logger.info(f"[{completed}/{len(mp4_files)}] 进度更新...")
        
        elapsed_time = time.time() - start_time
        logger.info(f"检测完成，耗时: {elapsed_time:.2f}秒")
        
        # 输出统计结果
        self.print_summary()
    
    def print_summary(self) -> None:
        """打印检测结果摘要"""
        total_files = len(self.corrupt_files) + len(self.valid_files) + len(self.error_files)
        
        print("\n" + "="*60)
        print("检测结果摘要")
        print("="*60)
        print(f"总文件数: {total_files}")
        print(f"正常文件: {len(self.valid_files)}")
        print(f"损坏文件: {len(self.corrupt_files)}")
        print(f"其他错误: {len(self.error_files)}")
        print("="*60)
        
        if self.corrupt_files:
            print(f"\n损坏的文件列表 ({len(self.corrupt_files)} 个):")
            for file_path, error_msg in self.corrupt_files:
                print(f"  - {file_path}")
                print(f"    错误: {error_msg}")
        
        if self.error_files:
            print(f"\n其他错误文件列表 ({len(self.error_files)} 个):")
            for file_path, error_msg in self.error_files[:10]:  # 只显示前10个
                print(f"  - {file_path}")
                print(f"    错误: {error_msg}")
            if len(self.error_files) > 10:
                print(f"  ... 还有 {len(self.error_files) - 10} 个文件")
    
    def delete_corrupt_files(self) -> None:
        """删除检测到的损坏文件"""
        if not self.corrupt_files:
            logger.info("没有发现损坏的文件")
            return
        
        if self.dry_run:
            logger.info(f"试运行模式：将删除 {len(self.corrupt_files)} 个损坏文件")
            for file_path, _ in self.corrupt_files:
                logger.info(f"  将删除: {file_path}")
            return
        
        # 确认删除
        print(f"\n即将删除 {len(self.corrupt_files)} 个损坏的文件")
        response = input("确认删除吗？(y/N): ").strip().lower()
        
        if response != 'y':
            logger.info("取消删除操作")
            return
        
        # 执行删除
        deleted_count = 0
        failed_count = 0
        
        for file_path, _ in self.corrupt_files:
            try:
                os.remove(file_path)
                logger.info(f"已删除: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"删除失败 {file_path}: {e}")
                failed_count += 1
        
        logger.info(f"删除完成：成功 {deleted_count} 个，失败 {failed_count} 个")
    
    def save_results_to_file(self, output_file: str = "corrupt_videos_report.txt") -> None:
        """将检测结果保存到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("损坏视频检测报告\n")
            f.write("="*50 + "\n")
            f.write(f"扫描目录: {self.root_dir}\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"总文件数: {len(self.corrupt_files) + len(self.valid_files) + len(self.error_files)}\n")
            f.write(f"正常文件: {len(self.valid_files)}\n")
            f.write(f"损坏文件: {len(self.corrupt_files)}\n")
            f.write(f"其他错误: {len(self.error_files)}\n\n")
            
            if self.corrupt_files:
                f.write("损坏文件列表:\n")
                f.write("-" * 30 + "\n")
                for file_path, error_msg in self.corrupt_files:
                    f.write(f"{file_path}\n")
                    f.write(f"  错误: {error_msg}\n\n")
            
            if self.error_files:
                f.write("其他错误文件列表:\n")
                f.write("-" * 30 + "\n")
                for file_path, error_msg in self.error_files:
                    f.write(f"{file_path}\n")
                    f.write(f"  错误: {error_msg}\n\n")
        
        logger.info(f"检测报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="快速检测并删除损坏的MP4视频文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --dry-run                    # 仅检测，不删除
  %(prog)s                             # 检测并删除
  %(prog)s --threads 8 --timeout 10    # 8线程，10秒超时
        """
    )
    
    parser.add_argument(
        'root_dir',
        nargs='?',
        default='/media/johnny/Data/data_motion_tokenizer/whole_d02_videos_segments_40s',
        help='要扫描的根目录 (默认: /media/johnny/Data/data_motion_tokenizer/whole_d02_videos_segments_40s)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式，仅检测不删除文件'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='ffmpeg检测超时时间（秒），默认5秒'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='并行线程数，默认4个'
    )
    
    parser.add_argument(
        '--output-report',
        default='corrupt_videos_report.txt',
        help='检测报告输出文件名'
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.root_dir):
        logger.error(f"目录不存在: {args.root_dir}")
        sys.exit(1)
    
    # 检查ffmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg未找到或无法运行，请确保已安装ffmpeg")
        sys.exit(1)
    
    # 创建检测器并运行
    detector = CorruptVideoDetector(
        root_dir=args.root_dir,
        timeout=args.timeout,
        dry_run=args.dry_run
    )
    
    try:
        # 执行检测
        detector.detect_corrupt_videos(max_workers=args.threads)
        
        # 保存报告
        detector.save_results_to_file(args.output_report)
        
        # 删除损坏文件（如果不是试运行模式）
        detector.delete_corrupt_files()
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
