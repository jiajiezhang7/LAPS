from typing import Optional
import numpy as np
import cv2


class TimeResampler:
    """A simple wall-clock based time resampler for streaming frames.

    It emits True at approximately target_fps using system time.
    """

    def __init__(self, target_fps: float):
        self.target_fps = float(target_fps)
        self.period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self.next_emit_ts: Optional[float] = None

    def should_emit(self, now_ts: float) -> bool:
        if self.period <= 0:
            return True
        if self.next_emit_ts is None:
            self.next_emit_ts = now_ts
            return True
        if now_ts + 1e-9 >= self.next_emit_ts:
            # Accumulate to compensate jitter
            while self.next_emit_ts <= now_ts:
                self.next_emit_ts += self.period
            return True
        return False


def resize_shorter_keep_aspect(frame: np.ndarray, resize_shorter: int) -> np.ndarray:
    """Resize image keeping aspect ratio so that its shorter side equals resize_shorter.
    If resize_shorter <= 0 or matches current shorter side, returns original frame.
    """
    if resize_shorter is None or resize_shorter <= 0:
        return frame
    h, w = frame.shape[:2]
    shorter = min(h, w)
    if shorter == resize_shorter:
        return frame
    scale = resize_shorter / float(shorter)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
