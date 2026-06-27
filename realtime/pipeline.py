# coding: utf-8

"""
Orchestrator for the real-time LivePortrait pipeline.

Wires together three threads:
  1. Camera  — captures frames from USB camera
  2. Inference — detects face, extracts motion, runs warp+decode
  3. Output   — sends rendered frames to OBS (pyvirtualcam)

The official LivePortrait code under src/ is NOT modified.
"""

import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch

from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper

from .config import RealtimeConfig
from .camera import CameraCapture
from .video_source import VideoFrameSource
from .detector import MediaPipeDetector, crop_from_mediapipe
from .renderer import Renderer
from .virtualcam import VirtualCamOutput
from .fps import FPSCounter


def _resolve_path(relative_path: str) -> str:
    """Resolve a path relative to the project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(project_root, relative_path)


def _build_inference_config(rt_cfg) -> InferenceConfig:
    """Build an InferenceConfig pointing to existing pretrained weights."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return InferenceConfig(
        models_config=os.path.join(project_root, 'src', 'config', 'models.yaml'),
        checkpoint_F=os.path.join(project_root, 'pretrained_weights', 'liveportrait', 'base_models', 'appearance_feature_extractor.pth'),
        checkpoint_M=os.path.join(project_root, 'pretrained_weights', 'liveportrait', 'base_models', 'motion_extractor.pth'),
        checkpoint_G=os.path.join(project_root, 'pretrained_weights', 'liveportrait', 'base_models', 'spade_generator.pth'),
        checkpoint_W=os.path.join(project_root, 'pretrained_weights', 'liveportrait', 'base_models', 'warping_module.pth'),
        checkpoint_S=os.path.join(project_root, 'pretrained_weights', 'liveportrait', 'retargeting_models', 'stitching_retargeting_module.pth'),
        flag_use_half_precision=rt_cfg.flag_use_half_precision,
        flag_do_torch_compile=False,  # we compile ourselves in _optimize_models
        flag_stitching=rt_cfg.flag_stitching,
        flag_relative_motion=rt_cfg.flag_relative_motion,
        flag_pasteback=rt_cfg.flag_pasteback,
        flag_eye_retargeting=rt_cfg.flag_eye_retargeting,
        flag_lip_retargeting=rt_cfg.flag_lip_retargeting,
        animation_region=rt_cfg.animation_region,
        driving_multiplier=rt_cfg.driving_multiplier,
        driving_option=rt_cfg.driving_option,
        device_id=rt_cfg.device_id,
        flag_force_cpu=rt_cfg.flag_force_cpu,
        flag_do_crop=True,
    )


class RealtimePipeline:
    """Real-time portrait animation pipeline."""

    def __init__(self, config: RealtimeConfig):
        self._cfg = config

        # -- Build inference config with explicit paths --
        inf_cfg = _build_inference_config(config.inference)

        # -- Load LivePortrait models --
        print('[pipeline] Loading LivePortrait models...')
        self._wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
        print('[pipeline] Models loaded.')

        # -- Optimize models for real-time --
        self._optimize_models(config.inference)

        # -- Components --
        self._detector = MediaPipeDetector(config.detector)
        self._renderer = Renderer(
            self._wrapper, config.source, config.inference, self._detector)

        # Frame source: camera or video file
        if config.driving.mode == 'video':
            if not config.driving.video_path:
                raise ValueError('driving.mode is "video" but driving.video_path is empty.')
            video_path = _resolve_path(config.driving.video_path)
            self._frame_source = VideoFrameSource(video_path, target_fps=config.camera.fps)
        else:
            self._frame_source = CameraCapture(config.camera)

        self._output = VirtualCamOutput(config.output)

        # -- Queues --
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.performance.queue_size)
        self._output_queue: queue.Queue = queue.Queue(maxsize=2)

        # -- Thread control --
        self._running = threading.Event()
        self._threads: list[threading.Thread] = []

        # -- Stats --
        self._inference_fps = FPSCounter(window_size=60)
        self._face_lost_count: int = 0
        self._max_face_lost: int = 30  # warn after ~1s at 30fps

        # -- State --
        self._first_frame_cached: bool = False
        self._last_valid_output: Optional[np.ndarray] = None

        # -- Output dimensions --
        self._out_w = config.output.width
        self._out_h = config.output.height

    # ------------------------------------------------------------------
    # Model optimization
    # ------------------------------------------------------------------

    def _optimize_models(self, inf_cfg) -> None:
        """
        Convert models to FP16 and optionally torch.compile.

        Matching the approach in speed.py: permanent .half() on model weights
        avoids the overhead of per-call autocast. torch.compile is done here
        with 'reduce-overhead' mode (faster warmup than 'max-autotune').
        """
        if not inf_cfg.flag_use_half_precision:
            return

        w = self._wrapper
        if w.device == 'mps':
            return  # MPS doesn't support .half() on all ops

        print('[pipeline] Converting models to FP16...')
        # Half-precision conversion: all models except F (AppearanceFeatureExtractor).
        # F runs only once (source caching) so not worth halving.
        # M (MotionExtractor / ConvNeXtV2-Tiny) uses LayerNorm, which was
        # initially considered unsafe for FP16.  In practice, inference-only
        # FP16 works fine — the speed.py benchmark confirms M at 2.17 ms
        # with half() + max-autotune.  Without half(), M takes ~14 ms GPU
        # time (6.5× slower), making it a significant bottleneck.
        w.motion_extractor = w.motion_extractor.half()
        w.warping_module = w.warping_module.half()
        w.spade_generator = w.spade_generator.half()
        if w.stitching_retargeting_module is not None:
            for key in ('stitching', 'eye', 'lip'):
                if key in w.stitching_retargeting_module:
                    w.stitching_retargeting_module[key] = \
                        w.stitching_retargeting_module[key].half()

        # torch.compile (only on the two heaviest modules)
        if inf_cfg.flag_do_torch_compile:
            # `max-autotune` delivers the best throughput (matching speed.py
            # benchmark of ~72 ms for M+W+G on WSL2), but uses CUDA graphs
            # internally to eliminate per-call Python overhead.  CUDA graph
            # tree managers live in thread-local storage → the first call to
            # the compiled function determines which thread "owns" the graphs.
            #
            # We register torch.compile HERE (main thread) so that the wrapper
            # attribute is set before threads start, but defer the actual
            # warmup / first-call to _inference_loop().  That way the JIT
            # compilation + CUDA graph capture happen in the inference thread's
            # TLS, avoiding:
            #   AssertionError: torch._C._is_key_in_tls(attr_name)
            print('[pipeline] Registering torch.compile (mode=reduce-overhead)...')
            print('[pipeline] JIT compilation will run on the '
                  'first inference call (warmup: ~30-60 s).')
            torch._dynamo.config.suppress_errors = True
            # Compile M, W, G — the three per-frame models.
            # The FIRST call to each compiled function MUST happen in the
            # inference thread (see _inference_loop warmup section).
            # F (AppearanceFeatureExtractor) runs once — not worth compile.
            #
            # Mode: 'reduce-overhead' (not 'max-autotune').
            # On this GPU (RTX 3060 Laptop, 30 SMs), max-autotune's GEMM
            # autotuner is skipped entirely ("Not enough SMs"), but the
            # failed-autotuning attempt adds compilation overhead.
            # reduce-overhead still captures CUDA graphs for Python-overhead
            # reduction but skips GEMM autotuning — faster warmup and
            # equivalent (or better) runtime on this hardware.
            w.motion_extractor = torch.compile(
                w.motion_extractor, mode='reduce-overhead')
            w.warping_module = torch.compile(
                w.warping_module, mode='reduce-overhead')
            w.spade_generator = torch.compile(
                w.spade_generator, mode='reduce-overhead')
            # Let the wrapper call cudagraph_mark_step_begin() before each
            # invocation — required when inductor CUDA graphs are active.
            w.compile = True
            # Warmup is deferred to _inference_loop() — see above.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all threads."""
        self._running.set()

        # Init source (load image, crop, extract F feature, cache x_s/R_s)
        source_path = _resolve_path(self._cfg.source.image_path)
        self._renderer.initialize(source_path)

        # Warm up detector
        self._detector.warmup()

        # Start output
        self._output.start()

        # Start threads
        source_name = 'camera' if self._cfg.driving.mode == 'camera' else 'video'
        self._threads = [
            threading.Thread(target=self._frame_source_loop, name=source_name, daemon=True),
            threading.Thread(target=self._inference_loop, name='inference', daemon=True),
            threading.Thread(target=self._output_loop, name='output', daemon=True),
        ]
        for t in self._threads:
            t.start()

        print('[pipeline] All threads started. Press Ctrl+C to stop.')

    def stop(self) -> None:
        """Graceful shutdown: signal threads, join, release resources."""
        print('[pipeline] Shutting down...')
        self._running.clear()

        for t in self._threads:
            t.join(timeout=2.0)

        self._frame_source.stop()
        self._detector.close()
        self._output.stop()
        print('[pipeline] Stopped.')

    def wait(self) -> None:
        """Block until KeyboardInterrupt, then stop."""
        try:
            while self._running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Thread loops
    # ------------------------------------------------------------------

    def _frame_source_loop(self) -> None:
        """Thread 1: Capture frames from camera or video file."""
        self._frame_source.start()
        while self._running.is_set():
            frame = self._frame_source.read(timeout=0.1)
            if frame is None:
                continue
            self._push_freshest(self._frame_queue, frame)

    def _inference_loop(self) -> None:
        """Thread 2: Detect face → extract motion → warp+decode."""
        import time as _time
        frame_count = 0
        t_detect_total = 0.0
        t_infer_total = 0.0
        t_other_total = 0.0

        # Enable cuDNN auto-tuner for convolution algorithm selection.
        # In single-frame inference the default heuristics often pick
        # sub-optimal algorithms; the auto-tuner benchmarks each algo at
        # first call and caches the result.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Also allow TF32 on Ampere+ GPUs (RTX 3060) — faster matmul
            # with negligible accuracy loss.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # --- Source keypoints & torch.compile warmup -----------------------
        # Both MUST run in the inference thread because M (MotionExtractor)
        # is torch.compile'd and its first call captures CUDA graphs.
        # Calling these in the main thread would create graphs in the wrong
        # thread-local storage, causing:
        #   AssertionError: torch._C._is_key_in_tls(attr_name)
        # ------------------------------------------------------------------
        # Phase A: extract source keypoints (first call to compiled M)
        if not self._renderer.initialized:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # flush any pending work
            self._renderer.init_source_keypoints()

        # Phase B: warmup compiled M, W, G with dummy data
        if (self._cfg.inference.flag_do_torch_compile and
                self._cfg.inference.flag_use_half_precision):
            w = self._wrapper
            if w.device != 'mps' and torch.cuda.is_available():
                print('[inference] Warming up compiled kernels '
                      '(reduce-overhead: ~30-60 s)...')
                dummy_source = torch.randn(1, 3, 256, 256, device=w.device).half()
                dummy_f3d = torch.randn(1, 32, 16, 64, 64, device=w.device).half()
                dummy_kp = torch.randn(1, 21, 3, device=w.device).half()
                with torch.no_grad():
                    for i in range(5):
                        _ = w.get_kp_info(dummy_source)
                        _ = w.warp_decode(dummy_f3d, dummy_kp, dummy_kp)
                        torch.cuda.synchronize()
                print('[inference] torch.compile warmup complete.')
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------

        while self._running.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            _t0 = _time.perf_counter()

            # -- Detect face --
            det = self._detector.detect(frame)
            _t1 = _time.perf_counter()
            if det is None:
                self._face_lost_count += 1
                if self._face_lost_count >= self._max_face_lost:
                    print(f'[inference] No face detected for '
                          f'{self._face_lost_count} frames — '
                          f'will reset driving reference on re-detection.')
                    self._face_lost_count = 0
                    # Reset driving reference so the animation doesn't jump
                    # when face re-appears at a different position/expression.
                    self._first_frame_cached = False
                    self._renderer.reset_driving_reference()
                # Push last valid output to keep stream alive
                if self._last_valid_output is not None:
                    self._push_freshest(self._output_queue,
                                        self._last_valid_output.copy())
                continue
            self._face_lost_count = 0

            # -- Crop driving face to 256×256 --
            crop = crop_from_mediapipe(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                det['landmarks_px'],
                dsize=256,
                scale=self._cfg.source.scale,
                vy_ratio=self._cfg.source.vy_ratio,
                flag_do_rot=self._cfg.source.flag_do_rot,
            )
            img_crop_256 = crop['img_crop']  # RGB, 256×256

            # -- Prepare driving tensor --
            # prepare_source expects BGR or RGB? Actually it just normalizes,
            # it's color-agnostic. But the model was trained on RGB.
            I_d = self._wrapper.prepare_source(img_crop_256)

            # -- Cache first driving frame reference --
            if not self._first_frame_cached:
                self._renderer.cache_first_driving_frame(I_d)
                self._first_frame_cached = True
                print('[inference] First driving frame cached. '
                      'Starting animation...')
                continue  # skip first frame (no output since delta=0)

            # -- Run inference --
            do_profile = (frame_count > 1 and
                          frame_count % self._cfg.performance.log_interval == 0)
            try:
                result_256 = self._renderer.infer_frame(
                    I_d, profile=do_profile)  # BGR, 256×256
            except torch.cuda.OutOfMemoryError as e:
                print(f'[inference] CUDA OOM: {e}. Skipping frame.')
                torch.cuda.empty_cache()
                continue
            _t2 = _time.perf_counter()

            self._inference_fps.tick()

            # -- Resize to output dimensions --
            result_out = cv2.resize(result_256, (self._out_w, self._out_h))

            self._last_valid_output = result_out
            self._push_freshest(self._output_queue, result_out)

            # Accumulate timing
            t_detect_total += (_t1 - _t0)
            t_infer_total += (_t2 - _t1)
            t_other_total += (_time.perf_counter() - _t2)

            frame_count += 1
            log_interval = self._cfg.performance.log_interval
            if frame_count > 1 and frame_count % log_interval == 0:
                n = log_interval
                print(f'[inference] Frame {frame_count}, '
                      f'Inference FPS: {self._inference_fps.fps:.1f}  '
                      f'(detect: {t_detect_total/n*1000:.0f}ms  '
                      f'infer: {t_infer_total/n*1000:.0f}ms  '
                      f'other: {t_other_total/n*1000:.0f}ms)')
                t_detect_total = 0.0
                t_infer_total = 0.0
                t_other_total = 0.0

    def _output_loop(self) -> None:
        """Thread 3: Send rendered frames to virtual camera / preview."""
        while self._running.is_set():
            try:
                frame = self._output_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._output.send(frame)
            except KeyboardInterrupt:
                self._running.clear()
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _push_freshest(q: queue.Queue, item) -> None:
        """Push to a maxsize=1 queue, dropping the oldest item if full."""
        try:
            q.put(item, block=False)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put(item, block=False)
            except queue.Full:
                pass
