#!/usr/bin/env python3
"""
Quick test script for I3D feature extraction.
Run in abd_env: conda run -n abd_env python comapred_algorithm/ABD/test_i3d.py
"""
from pathlib import Path
import sys
import time


def extract_i3d_features_limited(video_path, device="cuda:0", max_duration=30.0):
    """Extract I3D features from first max_duration seconds of video."""
    import torch
    import pytorchvideo.models.hub as hub
    from pytorchvideo.data.encoded_video import EncodedVideo
    import numpy as np
    
    print(f"[1/4] Loading I3D model...")
    sys.stdout.flush()
    t0 = time.time()
    model = hub.i3d_r50(pretrained=True)
    model = model.eval()
    model = model.to(device)
    print(f"      Model loaded in {time.time() - t0:.1f}s")
    sys.stdout.flush()
    
    print(f"[2/4] Loading video...")
    sys.stdout.flush()
    t0 = time.time()
    video = EncodedVideo.from_path(str(video_path))
    video_duration = min(float(video.duration), max_duration)
    total_duration = float(video.duration)
    print(f"      Video loaded, will process {video_duration:.1f}s (total: {total_duration:.1f}s)")
    sys.stdout.flush()
    
    print(f"[3/4] Extracting features...")
    sys.stdout.flush()
    features = []
    clip_duration = 2.0
    clip_stride = 0.4
    start_sec = 0.0
    
    while start_sec < video_duration:
        end_sec = min(start_sec + clip_duration, video_duration)
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        
        if clip is not None and 'video' in clip:
            video_data = clip['video']
            if video_data is not None and video_data.shape[1] > 0:
                video_input = video_data.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(video_input)
                    feat = output.cpu().numpy().squeeze(0)
                    features.append(feat)
        
        start_sec += clip_stride
        if len(features) % 10 == 0:
            print(f"      Processed {start_sec:.1f}s / {video_duration:.1f}s ({len(features)} clips)")
            sys.stdout.flush()
    
    features = np.stack(features, axis=0).astype(np.float32)
    print(f"[4/4] Done! Extracted {features.shape[0]} feature vectors")
    sys.stdout.flush()
    return features


def main():
    # Test on first D01 video
    video_path = Path("./datasets/gt_raw_videos/D01/D01_sample_1_seg001.mp4")
    
    if not video_path.exists():
        print(f"Test video not found: {video_path}")
        return 1
    
    print(f"Testing I3D feature extraction on: {video_path.name}")
    print(f"Video size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 60)
    sys.stdout.flush()
    
    # Extract features from first 30 seconds only
    features = extract_i3d_features_limited(
        video_path=video_path,
        device="cuda:0",
        max_duration=30.0
    )
    
    if features is None:
        print("❌ Feature extraction failed!")
        return 1
    
    print("=" * 60)
    print(f"✅ Success!")
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature dtype: {features.dtype}")
    print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"   Feature mean: {features.mean():.3f}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
