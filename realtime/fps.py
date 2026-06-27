# coding: utf-8

"""
Rolling-window FPS counter with optional on-frame overlay.
"""

import time
import collections
import cv2
import numpy as np


class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window_size: int = 30):
        self._window = collections.deque(maxlen=window_size)
        self._last_tick = None

    def tick(self) -> None:
        """Record a frame event."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._window.append(now - self._last_tick)
        self._last_tick = now

    @property
    def fps(self) -> float:
        """Current rolling-average FPS."""
        if not self._window:
            return 0.0
        avg_interval = sum(self._window) / len(self._window)
        if avg_interval <= 0:
            return 0.0
        return 1.0 / avg_interval

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the FPS counter onto a BGR frame (mutates in-place)."""
        fps_str = f'FPS: {self.fps:.1f}'
        cv2.putText(
            frame, fps_str, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )
        return frame
