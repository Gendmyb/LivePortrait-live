# coding: utf-8

"""
Video-file frame source with the same interface as CameraCapture.

Useful for testing without a physical camera (e.g., WSL2)
or for driving with a pre-recorded video.
"""

import threading
import queue
import time
from typing import Optional

import cv2
import numpy as np

from .fps import FPSCounter


class VideoFrameSource:
    """
    Reads frames from a video file, looping indefinitely.

    Mimics the CameraCapture interface so it can be dropped into
    the pipeline without any other changes.
    """

    def __init__(self, video_path: str, target_fps: float = 30.0):
        self._video_path = video_path
        self._target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fps = FPSCounter(window_size=30)

    # ------------------------------------------------------------------
    # Public API (same as CameraCapture)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the video file and start the reader thread."""
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open video file: {self._video_path}')
        print(f'[videosource] Opened: {self._video_path}')
        self._running.set()
        self._thread = threading.Thread(target=self._run, name='videosource', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and release resources."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Blocking read of the latest frame (BGR). Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Thread target: read frames from video, loop, maintain pacing."""
        frame_interval = 1.0 / max(self._target_fps, 1.0)
        last_time = time.perf_counter()

        while self._running.is_set():
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Loop the video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue

            self.fps.tick()

            # Pace to target FPS
            elapsed = time.perf_counter() - last_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.perf_counter()

            # Drop oldest frame if queue is full
            try:
                self._queue.put(frame, block=False)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put(frame, block=False)
                except queue.Full:
                    pass
