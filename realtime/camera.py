# coding: utf-8

"""
USB camera capture using OpenCV, running in a background thread.

Feeds a queue.Queue(maxsize=1) so the inference thread always gets
the freshest frame (old frames are dropped when the queue is full).
"""

import threading
import queue
import time
from typing import Optional

import cv2
import numpy as np

from .config import CameraConfig
from .fps import FPSCounter


class CameraCapture:
    """USB camera capture running in a daemon thread."""

    def __init__(self, config: CameraConfig):
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)  # always freshest frame
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fps = FPSCounter(window_size=30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera and start the capture thread."""
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') if self._cfg.use_mjpg else 0
        self._cap = cv2.VideoCapture(self._cfg.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self._running.set()
        self._thread = threading.Thread(target=self._run, name='camera', daemon=True)
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
        """Blocking read of the latest frame (BGR, uint8). Returns None on timeout."""
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
        """Thread target: capture frames and push to queue."""
        while self._running.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.5)
                self._try_reconnect()
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.5)
                self._try_reconnect()
                continue

            self.fps.tick()

            # Drop oldest frame if queue is full — inference always gets the freshest
            try:
                self._queue.put(frame, block=False)
            except queue.Full:
                try:
                    self._queue.get_nowait()  # discard stale frame
                except queue.Empty:
                    pass
                try:
                    self._queue.put(frame, block=False)
                except queue.Full:
                    pass  # give up if still full (shouldn't happen)

    def _try_reconnect(self) -> None:
        """Attempt to re-open the camera after disconnection."""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._cfg.device_id)
        if self._cap.isOpened():
            print('[camera] Reconnected.')
        else:
            print('[camera] Camera not available, retrying...')
