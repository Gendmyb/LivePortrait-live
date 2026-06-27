# coding: utf-8

"""
Lightweight MJPEG HTTP streamer.

Runs a minimal HTTP server in a background thread so that OBS
(on Windows) can pull the stream via:

    http://localhost:<port>/stream

OBS setup (Windows host, WSL2 guest):
  1. Add "Media Source" (needs VLC or ffmpeg)
     - Uncheck "Local File"
     - Input:  http://localhost:8080/stream
     - Input Format:  mpjpeg
  OR
  2. Add "Browser" source → point to http://localhost:8080/
     (serves a tiny auto-refresh page if /stream is unavailable)

The WSL2 VM's localhost is automatically forwarded to the Windows
host, so no extra port-forwarding is needed.
"""

import http.server
import threading
import time
from typing import Optional
from io import BytesIO

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MJPEG streaming handler
# ---------------------------------------------------------------------------

_BOUNDARY = b'--liveportrait-boundary\r\n'


class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
    """Serves a multi-part MJPEG stream at /stream, plus a simple viewer at /."""

    # Class-level references set by MJPEGStreamer.start()
    _latest_jpeg_buf: Optional[list] = None  # list[0] = JPEG bytes | None
    _condition: Optional[threading.Condition] = None
    _stop_event: Optional[threading.Event] = None  # signal handler to exit

    def do_GET(self):
        if self.path == '/stream':
            self._serve_stream()
        elif self.path == '/snapshot':
            self._serve_snapshot()
        elif self.path == '/':
            self._serve_viewer()
        else:
            self.send_error(404)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         f'multipart/x-mixed-replace; boundary=liveportrait-boundary')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        cond = self._condition
        buf = self._latest_jpeg_buf
        stop_ev = self._stop_event
        try:
            while not (stop_ev and stop_ev.is_set()):
                with cond:
                    # Short wait to remain responsive to stop signals
                    if not cond.wait(timeout=0.5):
                        continue
                    jpeg = buf[0]
                if jpeg is None:
                    continue
                try:
                    self.wfile.write(_BOUNDARY)
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    break
        except Exception:
            pass

    def _serve_snapshot(self):
        """Single JPEG snapshot."""
        cond = self._condition
        buf = self._latest_jpeg_buf
        with cond:
            jpeg = buf[0]
        if jpeg is None:
            self.send_error(503, 'No frame available yet')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(jpeg)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_viewer(self):
        """Tiny HTML page that auto-refreshes the MJPEG stream in an <img>."""
        html = b"""\
<!DOCTYPE html>
<html>
<head><title>LivePortrait Realtime Stream</title>
<style>
  body { margin:0; background:#000; display:flex; justify-content:center; align-items:center; min-height:100vh; }
  img { max-width:100%; max-height:100vh; }
</style>
</head>
<body>
  <img src="/stream" alt="LivePortrait stream" />
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, format, *args):
        """Suppress HTTP request logging to stderr."""
        pass


# ---------------------------------------------------------------------------
# Streamer class
# ---------------------------------------------------------------------------

class MJPEGStreamer:
    """Background HTTP server that publishes frames as an MJPEG stream."""

    def __init__(self, port: int = 8080, jpeg_quality: int = 85):
        self._port = port
        self._quality = jpeg_quality
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._frame_buf: list = [None]
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the HTTP server in a daemon thread."""
        _MJPEGHandler._latest_jpeg_buf = self._frame_buf
        _MJPEGHandler._condition = self._condition
        _MJPEGHandler._stop_event = self._stop_event

        self._server = http.server.HTTPServer(
            ('0.0.0.0', self._port), _MJPEGHandler)
        # Non-zero timeout so serve_forever() periodically checks the
        # shutdown flag instead of blocking forever in select().
        self._server.timeout = 0.5

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name='mjpeg-server',
            daemon=True,
        )
        self._thread.start()
        print(f'[mjpeg] Stream server started on http://localhost:{self._port}/stream')

    def stop(self) -> None:
        """Shut down the server — signal handlers, then close."""
        if self._server is not None:
            # 1. Tell streaming handlers to exit their loop
            self._stop_event.set()
            # 2. Wake up any handler waiting on the condition
            with self._condition:
                self._condition.notify_all()
            # 3. Shut down the accept loop
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Reset for potential reuse
        self._stop_event.clear()

    def send(self, frame_bgr: np.ndarray) -> None:
        """Encode a BGR frame as JPEG and publish to all connected clients."""
        success, jpeg = cv2.imencode('.jpg', frame_bgr,
                                     [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not success:
            return
        with self._condition:
            self._frame_buf[0] = jpeg.tobytes()
            self._condition.notify_all()
