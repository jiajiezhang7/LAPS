from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np


def extract_i3d_features_for_video(
    video_path: Path, 
    device: str = "cuda:0",
    clip_duration: float = 2.0,
    clip_stride: float = 0.4,
    target_fps: int = 30,
) -> Optional[np.ndarray]:
    """
    Extract I3D features from a video using pytorchvideo's pretrained I3D model.
    
    Args:
        video_path: Path to video file
        device: PyTorch device (e.g., "cuda:0" or "cpu")
        clip_duration: Duration of each video clip in seconds
        clip_stride: Stride between clips in seconds
        target_fps: Target FPS for video decoding
    
    Returns:
        Feature array of shape (N, 400) where N is the number of temporal clips.
        I3D outputs 400-dimensional features per clip.
        Returns None if extraction fails.
    """
    try:
        import torch
        import pytorchvideo.models.hub as hub
        from pytorchvideo.data.encoded_video import EncodedVideo
    except ImportError as e:
        raise RuntimeError(
            f"Required dependencies not available: {e}\n"
            "Please ensure you are running in abd_env with PyTorch and pytorchvideo installed."
        )
    
    if not video_path.exists():
        print(f"[I3D] Video not found: {video_path}")
        return None
    
    try:
        # Load pretrained I3D model (kinetics-400)
        model = hub.i3d_r50(pretrained=True)
        model = model.eval()
        model = model.to(device)
        
        # Load video
        video = EncodedVideo.from_path(str(video_path))
        video_duration = video.duration
        
        if video_duration <= 0:
            print(f"[I3D] Invalid video duration: {video_duration}")
            return None
        
        # Extract features at regular intervals
        features = []
        start_sec = 0.0
        
        while start_sec < video_duration:
            end_sec = min(start_sec + clip_duration, video_duration)
            
            # Get clip
            clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            if clip is None or 'video' not in clip:
                start_sec += clip_stride
                continue
            
            video_data = clip['video']
            if video_data.shape[1] == 0:  # No frames
                start_sec += clip_stride
                continue
            
            # Prepare input: (1, C, T, H, W)
            # pytorchvideo returns (C, T, H, W), we need to add batch dimension
            video_input = video_data.unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                output = model(video_input)
                # I3D outputs (B, 400) for kinetics-400
                feat = output.cpu().numpy().squeeze(0)  # (400,)
                features.append(feat)
            
            start_sec += clip_stride
        
        if len(features) == 0:
            print(f"[I3D] No features extracted from {video_path}")
            return None
        
        # Stack features: (N, 400)
        features = np.stack(features, axis=0)
        print(f"[I3D] Extracted {features.shape[0]} features from {video_path.name}")
        return features.astype(np.float32)
        
    except Exception as e:
        print(f"[I3D] Error extracting features from {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

