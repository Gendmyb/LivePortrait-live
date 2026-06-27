# coding: utf-8

"""
Source caching and per-frame renderer for real-time LivePortrait.

The key performance insight: the Appearance Feature Extractor (F) runs
ONCE for the source image, and every subsequent frame only needs
M (motion) + W (warp) + G (generate).

This module:
1. Caches source features (f_s, x_s, R_s) at init
2. Caches first-frame driving reference (x_d_0) for relative motion
3. Composes relative motion per frame and calls warp_decode()
"""

from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch

from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.camera import get_rotation_matrix
from src.utils.helper import calc_motion_multiplier
from src.utils.io import load_image_rgb, resize_to_limit

from .config import SourceConfig, InferenceRTConfig
from .detector import MediaPipeDetector, crop_source_for_realtime


class Renderer:
    """
    Manages source feature caching and per-frame warp+decode.

    Usage
    -----
    renderer = Renderer(wrapper, source_cfg, inference_cfg, detector)
    renderer.initialize()                         # one-time source setup

    # on first valid camera frame:
    renderer.cache_first_driving_frame(x_driving_tensor)

    # every subsequent frame:
    result_bgr_256 = renderer.infer_frame(x_driving_tensor)
    """

    def __init__(self, wrapper: LivePortraitWrapper,
                 source_cfg: SourceConfig,
                 inference_cfg: InferenceRTConfig,
                 detector: MediaPipeDetector):
        self._wrapper = wrapper
        self._src_cfg = source_cfg
        self._inf_cfg = inference_cfg
        self._detector = detector
        self._device = wrapper.device

        # -- Source cache (set in initialize() + init_source_keypoints()) --
        self._I_s: Optional[torch.Tensor] = None         # source tensor (1×3×256×256)
        self.f_s: Optional[torch.Tensor] = None          # F output (feature_3d)
        self.x_s_info: Optional[Dict[str, Any]] = None   # source kp_info dict
        self.x_s: Optional[torch.Tensor] = None           # transformed source kp
        self.x_c_s: Optional[torch.Tensor] = None         # canonical source kp
        self.R_s: Optional[torch.Tensor] = None           # source rotation matrix

        # -- First driving frame reference (set in cache_first_driving_frame()) --
        self.x_d_0_info: Optional[Dict[str, Any]] = None
        self.R_d_0: Optional[torch.Tensor] = None
        self.x_d_0_new: Optional[torch.Tensor] = None    # for expression-friendly
        self._motion_multiplier: Optional[torch.Tensor] = None

        self.initialized: bool = False

    # ------------------------------------------------------------------
    # One-time source setup
    # ------------------------------------------------------------------

    def initialize(self, source_image_path: str) -> None:
        """
        Load source image, detect face, crop, extract F feature.

        Runs in the MAIN thread — safe: F is not compiled, no CUDA graphs.
        Source keypoint extraction (M) is deferred to init_source_keypoints()
        which MUST be called from the inference thread (first compiled-M call
        captures CUDA graphs in the correct TLS).
        """
        # 1. Load & resize source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, max_dim=1280, division=2)
        print(f'[renderer] Loaded source: {img_rgb.shape[1]}×{img_rgb.shape[0]}')

        # 2. Detect face & crop to 256×256 using MediaPipe
        crop_info = crop_source_for_realtime(img_rgb, self._detector, self._src_cfg)
        if crop_info is None:
            raise RuntimeError(
                f'No face detected in source image: {source_image_path}. '
                f'Try a different image or lower min_detection_confidence.')

        img_crop_256 = crop_info['img_crop_256x256']
        print(f'[renderer] Source face cropped to 256×256')

        # 3. Prepare source tensor (normalize, permute to 1×3×256×256)
        self._I_s = self._wrapper.prepare_source(img_crop_256)

        # 4. Extract and cache F output (feature_3d)
        self.f_s = self._wrapper.extract_feature_3d(self._I_s)
        print(f'[renderer] Source appearance feature cached (F done).')

        # M extraction deferred — see init_source_keypoints()

    def init_source_keypoints(self) -> None:
        """
        Extract source keypoints (M) — MUST be called from the inference
        thread when M is torch.compile'd, so that CUDA graph capture happens
        in the correct thread-local storage.
        """
        # 5. Extract and cache source keypoints (M)
        self.x_s_info = self._wrapper.get_kp_info(self._I_s)
        self.x_c_s = self.x_s_info['kp']
        self.x_s = self._wrapper.transform_keypoint(self.x_s_info)
        self.R_s = get_rotation_matrix(
            self.x_s_info['pitch'], self.x_s_info['yaw'], self.x_s_info['roll'])

        self.initialized = True
        print(f'[renderer] Source keypoints cached. Ready.')

    # ------------------------------------------------------------------
    # First-frame driving reference
    # ------------------------------------------------------------------

    def cache_first_driving_frame(self, x_driving: torch.Tensor) -> None:
        """
        Cache the first valid driving frame as the motion reference (x_d_0).

        Must be called once before infer_frame() — otherwise relative motion
        has no baseline.
        """
        self.x_d_0_info = self._wrapper.get_kp_info(x_driving)
        self.R_d_0 = get_rotation_matrix(
            self.x_d_0_info['pitch'],
            self.x_d_0_info['yaw'],
            self.x_d_0_info['roll'],
        )

        # Pre-compute x_d_0_new for expression-friendly mode
        if self._inf_cfg.flag_relative_motion:
            delta_new = self.x_s_info['exp'] + \
                (self.x_d_0_info['exp'] - self.x_d_0_info['exp'])  # = x_s_info['exp']
            R_new = (self.R_d_0 @ self.R_d_0.permute(0, 2, 1)) @ self.R_s
            scale_new = self.x_s_info['scale']
            t_new = self.x_s_info['t']
            t_new[..., 2].fill_(0)
            self.x_d_0_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new

            if self._inf_cfg.driving_option == 'expression-friendly':
                self._motion_multiplier = calc_motion_multiplier(self.x_s, self.x_d_0_new)
            else:
                self._motion_multiplier = None
        else:
            self.x_d_0_new = None
            self._motion_multiplier = None

        print('[renderer] First driving frame cached as reference.')

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    # Keypoint indices for per-region animation (mirrors live_portrait_pipeline.py)
    _LIP_IDX = [6, 12, 14, 17, 19, 20]
    _EYES_IDX = [11, 13, 15, 16, 18]
    _EXP_IDX = [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    _EXP_SPECIAL_IDX = [(3, 5, 1), (5, 2), (8, 2), (9, slice(1, None))]

    def infer_frame(self, x_driving: torch.Tensor,
                    profile: bool = False) -> np.ndarray:
        """
        Run one frame of inference: M → motion composition → W+G.

        Parameters
        ----------
        x_driving : torch.Tensor  1×3×256×256, normalized to 0~1
        profile : bool  If True, print per-stage GPU timing (CUDA events).

        Returns
        -------
        np.ndarray  uint8 BGR, shape (256, 256, 3)
        """
        use_cuda = torch.cuda.is_available()
        ev = None
        if profile and use_cuda:
            _mk = lambda: torch.cuda.Event(enable_timing=True)
            ev = {'start': _mk(), 'after_M': _mk(), 'after_compose': _mk(),
                  'after_WG': _mk(), 'after_parse': _mk()}
            ev['start'].record()

        if not self.initialized:
            raise RuntimeError('Renderer not initialized. Call initialize() first.')
        if self.x_d_0_info is None:
            raise RuntimeError(
                'Driving reference not cached. Call cache_first_driving_frame() first.')

        # 1. Extract driving keypoints (M)
        x_d_i_info = self._wrapper.get_kp_info(x_driving)
        R_d_i = get_rotation_matrix(
            x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

        if ev: ev['after_M'].record()

        # 2. Compose relative motion (mostly CPU-bound small tensor ops)
        region = self._inf_cfg.animation_region
        if self._inf_cfg.flag_relative_motion:
            delta_new = self.x_s_info['exp'].clone()
            if region == 'all' or region == 'exp':
                delta_new = self.x_s_info['exp'] + \
                    (x_d_i_info['exp'] - self.x_d_0_info['exp'])
            elif region == 'lip':
                for idx in self._LIP_IDX:
                    delta_new[:, idx, :] = (self.x_s_info['exp'] +
                        (x_d_i_info['exp'] - self.x_d_0_info['exp']))[:, idx, :]
            elif region == 'eyes':
                for idx in self._EYES_IDX:
                    delta_new[:, idx, :] = (self.x_s_info['exp'] +
                        (x_d_i_info['exp'] - self.x_d_0_info['exp']))[:, idx, :]

            if region in ('all', 'pose'):
                R_new = (R_d_i @ self.R_d_0.permute(0, 2, 1)) @ self.R_s
            else:
                R_new = self.R_s

            if region == 'all':
                scale_new = self.x_s_info['scale'] * \
                    (x_d_i_info['scale'] / self.x_d_0_info['scale'])
            else:
                scale_new = self.x_s_info['scale']

            if region in ('all', 'pose'):
                t_new = self.x_s_info['t'] + \
                    (x_d_i_info['t'] - self.x_d_0_info['t'])
            else:
                t_new = self.x_s_info['t']
        else:
            delta_new = x_d_i_info['exp']
            R_new = R_d_i
            scale_new = x_d_i_info['scale']
            t_new = x_d_i_info['t']

        t_new[..., 2].fill_(0)
        x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new

        if (self._inf_cfg.flag_relative_motion and
                self._inf_cfg.driving_option == 'expression-friendly' and
                self._motion_multiplier is not None and
                self.x_d_0_new is not None):
            x_d_diff = (x_d_i_new - self.x_d_0_new) * self._motion_multiplier
            x_d_i_new = x_d_diff + self.x_s

        x_d_i_new = self.x_s + (x_d_i_new - self.x_s) * self._inf_cfg.driving_multiplier

        if self._inf_cfg.flag_stitching:
            x_d_i_new = self._wrapper.stitching(self.x_s, x_d_i_new)

        if ev: ev['after_compose'].record()

        # 6. Warp + Decode (W + G)
        out = self._wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)

        if ev: ev['after_WG'].record()

        result_rgb = self._wrapper.parse_output(out['out'])[0]

        if ev: ev['after_parse'].record()

        # Convert to BGR for downstream consumers
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        if ev:
            torch.cuda.synchronize()
            print(f'[profile] M:{ev["start"].elapsed_time(ev["after_M"]):.1f}ms  '
                  f'compose:{ev["after_M"].elapsed_time(ev["after_compose"]):.1f}ms  '
                  f'W:{ev["after_compose"].elapsed_time(ev["after_WG"]):.1f}ms  '
                  f'G+parse:{ev["after_WG"].elapsed_time(ev["after_parse"]):.1f}ms')

        return result_bgr

    def reset_driving_reference(self) -> None:
        """Clear the driving reference so it can be re-captured (e.g. after face loss)."""
        self.x_d_0_info = None
        self.R_d_0 = None
        self.x_d_0_new = None
        self._motion_multiplier = None
