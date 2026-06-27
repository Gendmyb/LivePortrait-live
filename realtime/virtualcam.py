# coding: utf-8

"""
OBS Virtual Camera output via pyvirtualcam.

Falls back to an OpenCV preview window if pyvirtualcam is not available
or if virtual_camera is disabled in config.  Supports MJPEG HTTP streaming
for WSL2 → Windows OBS cross-OS output.
"""

import time
from typing import Optional

import cv2
import numpy as np

from .config import OutputConfig


class VirtualCamOutput:
    """Sends rendered frames to a virtual camera (OBS), preview window, or MJPEG stream."""

    def __init__(self, config: OutputConfig):
        self._cfg = config
        self._cam: Optional[object] = None  # pyvirtualcam.Camera
        self._use_virtual = config.virtual_camera
        self._window_name = 'LivePortrait RealTime (preview)'

        # MJPEG streamer (for WSL2 → Windows OBS)
        self._mjpeg = None
        if config.mjpeg_port > 0:
            from .mjpeg_streamer import MJPEGStreamer
            self._mjpeg = MJPEGStreamer(port=config.mjpeg_port,
                                        jpeg_quality=config.mjpeg_quality)

        # Preview window: only if explicitly requested OR as last-resort fallback
        # (no virtual cam AND no MJPEG streaming → nothing would show otherwise)
        self._use_preview = config.show_preview

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the virtual camera, MJPEG server, or prepare the preview window."""
        # MJPEG streamer (starts first — available even if camera fails)
        if self._mjpeg is not None:
            self._mjpeg.start()

        if self._use_virtual:
            try:
                import pyvirtualcam
                self._cam = pyvirtualcam.Camera(
                    width=self._cfg.width,
                    height=self._cfg.height,
                    fps=self._cfg.fps,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                    print_fps=False,
                )
                print(f'[virtualcam] Opened virtual camera: '
                      f'{self._cfg.width}x{self._cfg.height} @ {self._cfg.fps}fps '
                      f'(device: {self._cam.device})')
            except ImportError:
                print('[virtualcam] pyvirtualcam not installed. '
                      'Will use MJPEG stream if configured.')
                self._use_virtual = False
            except Exception as e:
                print(f'[virtualcam] Failed to open: {e}. '
                      'Will use MJPEG stream if configured.')
                self._use_virtual = False

        # Decide whether to show the preview window
        has_any_output = self._use_virtual or (self._mjpeg is not None)
        if self._use_preview or not has_any_output:
            if not self._use_preview:
                print('[output] No output method available (virtual cam failed, '
                      'no MJPEG). Falling back to preview window.')
            self._use_preview = True
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, self._cfg.width, self._cfg.height)

    def stop(self) -> None:
        """Close the virtual camera, MJPEG server, or destroy the preview window."""
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        if self._use_preview:
            cv2.destroyWindow(self._window_name)
        if self._mjpeg is not None:
            self._mjpeg.stop()

    def send(self, frame: np.ndarray) -> None:
        """Send a BGR frame to all active outputs."""

        # MJPEG streamer (always, if enabled)
        if self._mjpeg is not None:
            self._mjpeg.send(frame)

        # pyvirtualcam
        if self._use_virtual and self._cam is not None:
            self._cam.send(frame)
            self._cam.sleep_until_next_frame()

        # Preview window (only when explicitly enabled or as last resort)
        if self._use_preview:
            cv2.imshow(self._window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                raise KeyboardInterrupt
