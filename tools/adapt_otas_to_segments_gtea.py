#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 OTAS 在 GTEA 上的 detect_bdy 输出（detect_seg/*.pkl, 含 bdy_idx_list）
适配为 LAPS 统一评估所需的 segments.json。

与通用版 adapt_otas_to_segments.py 的差异：
- OTAS 的 video_id 形如 "GTEA_GTEA_{stem}"（因我们构造了 {P}_{cam}_{act} 的帧目录），
  而 GTEA 的 GT/视频名都用 {stem}.mp4（例如 S1_Cheese_C1）。
- 本脚本直接将 video_id 去掉前两段，得到 stem，用于查找原始 mp4 和写出 JSON。

输入：
  --otas-pred 指向 OTAS 输出根（包含 detect_seg/*.pkl）
  --raw-dir   指向原始测试视频目录（online_datasets/gtea/gtea/Videos_test.split1）
  --output    输出根目录（将生成 {output}/{stem}/segmented_videos/{stem}_segments.json）

JSON 字段：
  {
    "video": "{stem}.mp4",
    "segments": [{"start_sec": float, "end_sec": float, "label": "segment_i"}],
    "video_duration_sec": float,
    "fps": float,
    "processed_at": ISO8601,
    "segmentation_params": {"from": "OTAS_GTEA", "note": "adapted by tools/adapt_otas_to_segments_gtea.py"}
  }
"""

import argparse
import os
import pickle
import json
import cv2
from typing import List, Tuple
from datetime import datetime, timezone


def to_segments(boundaries: List[int], fps: float, video_frames: int) -> List[Tuple[float, float]]:
    if video_frames <= 0 or fps <= 0:
        return []
    T = video_frames / fps
    # 边界帧索引 -> 秒，并裁剪到 [0, T]
    bsec = sorted([max(0.0, min(T, bi / fps)) for bi in boundaries])
    # 去重：小于 1 帧的间隔视为重复
    dedup: List[float] = []
    for t in bsec:
        if not dedup or abs(t - dedup[-1]) > (1.0 / fps):
            dedup.append(t)
    cuts = [0.0] + dedup + [T]
    segs: List[Tuple[float, float]] = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s > 1e-6:
            segs.append((float(s), float(e)))
    return segs


def gtea_stem_from_video_id(video_id: str) -> str:
    """将 OTAS 的 video_id（通常形如 GTEA_GTEA_{stem}）解析为 GTEA 的 stem。
    兜底策略：若分段不足 3，则返回原始 video_id。
    """
    parts = video_id.split('_')
    if len(parts) >= 3:
        # 默认直接去掉前两段（P 与 cam），余下即为 {stem}
        return '_'.join(parts[2:])
    return video_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--otas-pred', required=True, help='OTAS 输出根，内含 detect_seg/*.pkl')
    parser.add_argument('--output', required=True, help='输出根（每视频一个子目录，含 segmented_videos/*_segments.json）')
    parser.add_argument('--raw-dir', required=True, help='GTEA 测试视频目录（如 online_datasets/gtea/gtea/Videos_test.split1）')
    args = parser.parse_args()

    detect_dir = os.path.join(args.otas_pred, 'detect_seg')
    if not os.path.isdir(detect_dir):
        raise FileNotFoundError(f'detect_seg 文件夹不存在: {detect_dir}')

    pkl_files = [f for f in os.listdir(detect_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f'未在 {detect_dir} 下发现 .pkl 文件')

    os.makedirs(args.output, exist_ok=True)

    for fn in sorted(pkl_files):
        video_id = os.path.splitext(fn)[0]
        stem = gtea_stem_from_video_id(video_id)
        raw_mp4 = os.path.join(args.raw_dir, f'{stem}.mp4')
        if not os.path.exists(raw_mp4):
            print(f'[WARN] 未找到原始视频 {raw_mp4}（来自 {video_id}）；跳过')
            continue

        cap = cv2.VideoCapture(raw_mp4)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0 or frames <= 0:
            print(f'[WARN] 无效 FPS/帧数：{raw_mp4}；跳过')
            continue

        with open(os.path.join(detect_dir, fn), 'rb') as f:
            d = pickle.load(f)
        boundaries = list(d.get('bdy_idx_list', []))
        segs = to_segments(boundaries, fps=fps, video_frames=frames)

        out_dir = os.path.join(args.output, stem, 'segmented_videos')
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, f'{stem}_segments.json')
        payload = {
            'video': f'{stem}.mp4',
            'segments': [ {'start_sec': s, 'end_sec': e, 'label': f'segment_{i+1}'} for i,(s,e) in enumerate(segs) ],
            'video_duration_sec': frames / fps,
            'fps': fps,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'segmentation_params': {
                'from': 'OTAS_GTEA',
                'note': 'converted by tools/adapt_otas_to_segments_gtea.py'
            }
        }
        with open(out_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'[OK] wrote {out_json} (segments={len(segs)})')


if __name__ == '__main__':
    main()

