from pathlib import Path
from typing import Optional
import os


def compute_per_video_energy_jsonl_path(
    base_path: Path,
    video_name: str,
    energy_source: str,
    energy_mode: str,
    seg_enable: bool,
    seg_output_dir: Path,
) -> Path:
    """根据 video_name/source/mode 生成每视频的 energy JSONL 输出路径。

    - 支持文件名占位符 {video_name}/{source}/{mode}
    - 若文件名为默认 'stream_energy.jsonl'，自动改名为 'stream_energy_{source}_{mode}.jsonl'
    - 优先输出到分割输出目录下的每视频子目录，否则输出到原目录下的 {video_name}/ 子目录
    """
    ejp = Path(base_path)
    try:
        basename = ejp.name.format(video_name=video_name, source=str(energy_source), mode=str(energy_mode))
    except Exception:
        basename = ejp.name
    if ejp.stem == "stream_energy":
        basename = f"stream_energy_{str(energy_source)}_{str(energy_mode)}.jsonl"
    if seg_enable:
        energy_dir = Path(seg_output_dir) / video_name
    else:
        energy_dir = ejp.parent / video_name
    energy_dir.mkdir(parents=True, exist_ok=True)
    out_path = energy_dir / basename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def should_skip_video_outputs(seg_output_dir: Path, video_name: str, seg_ext: str) -> bool:
    """当已存在输出（片段视频或 code_indices）时，返回 True 以跳过该视频。

    兼容扩展名：用户配置的 seg_ext 以及常见视频扩展名；codes 支持 .json/.jsonl
    """
    try:
        video_specific_dir = Path(seg_output_dir) / video_name
        seg_videos_dir = video_specific_dir / "segmented_videos"
        seg_codes_dir = video_specific_dir / "code_indices"
        exists_any = False
        if seg_videos_dir.exists() and seg_videos_dir.is_dir():
            valid_exts = {str(seg_ext).lower(), ".mp4", ".mov", ".mkv", ".avi"}
            for p in seg_videos_dir.iterdir():
                if p.is_file() and p.suffix.lower() in valid_exts:
                    exists_any = True
                    break
        if not exists_any and seg_codes_dir.exists() and seg_codes_dir.is_dir():
            for p in seg_codes_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
                    exists_any = True
                    break
        return exists_any
    except Exception:
        return False
