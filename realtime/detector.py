# coding: utf-8

"""
Lightweight face detection and cropping using MediaPipe Face Landmarker.

Replaces InsightFace's RetinaFace + ONNX 106-point pipeline for real-time use.
MediaPipe provides 478 3D landmarks in a single pass, and its tracking mode
gives temporal consistency across frames.
"""

import numpy as np
from math import cos, sin, acos
from typing import Optional, Dict, Any

import cv2

from .config import DetectorConfig, SourceConfig

DTYPE = np.float32

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (canonical topology)
# ---------------------------------------------------------------------------
# Eye corners for computing pt2 (eye-center, lip-center)
_MP_LEFT_EYE_OUTER = 33
_MP_LEFT_EYE_INNER = 133
_MP_RIGHT_EYE_INNER = 362
_MP_RIGHT_EYE_OUTER = 263
_MP_LIP_UPPER = 13
_MP_LIP_LOWER = 14

# Additional eye contour points for a more robust eye-center
_MP_LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
_MP_RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]


def _mediapipe_landmarks_to_pixel(landmarks_norm, width, height):
    """Convert normalized MediaPipe landmarks (Wx3) to pixel coordinates (Wx2)."""
    px = landmarks_norm[:, 0] * width
    py = landmarks_norm[:, 1] * height
    return np.stack([px, py], axis=1).astype(DTYPE)


def _mediapipe_pt2(landmarks_px, use_lip=True):
    """
    Extract the canonical 2-point representation (eye-center, lip-center)
    from MediaPipe's 478 face-mesh landmarks.

    Uses multiple contour points for robust eye-center estimation, matching
    the robustness approach of parse_pt2_from_pt101/106 in src/utils/crop.py.
    """
    # Robust eye centers: mean of the full eye contour
    left_eye = np.mean(landmarks_px[_MP_LEFT_EYE_CONTOUR], axis=0)
    right_eye = np.mean(landmarks_px[_MP_RIGHT_EYE_CONTOUR], axis=0)

    if use_lip:
        pt_center_eye = (left_eye + right_eye) / 2.0
        pt_center_lip = (landmarks_px[_MP_LIP_UPPER] + landmarks_px[_MP_LIP_LOWER]) / 2.0
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        return np.stack([left_eye, right_eye], axis=0)


def _estimate_similar_transform_from_pts(pts_px, dsize=256, scale=2.3,
                                          vx_ratio=0.0, vy_ratio=-0.125,
                                          flag_do_rot=True, use_lip=True):
    """
    Compute the affine warp matrix from original image → cropped image.

    Same algorithm as src/utils/crop.py:_estimate_similar_transform_from_pts
    but works with arbitrary landmark arrays (MediaPipe 478-point).
    """
    pt2 = _mediapipe_pt2(pts_px, use_lip=use_lip)

    # -- Rotation from eye → lip vector --
    uy = pt2[1] - pt2[0]
    length = np.linalg.norm(uy)
    if length <= 1e-3:
        uy = np.array([0.0, 1.0], dtype=DTYPE)
    else:
        uy = uy / length
    ux = np.array([uy[1], -uy[0]], dtype=DTYPE)

    angle = acos(float(ux[0]))
    if ux[1] < 0:
        angle = -angle

    # -- Center & size from all landmarks --
    rot_mat = np.array([ux, uy], dtype=DTYPE)  # 2×2
    center0 = np.mean(pts_px, axis=0)
    rpts = (pts_px - center0) @ rot_mat.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    size_vec = rb_pt - lt_pt
    m = max(size_vec[0], size_vec[1])
    size_vec = np.array([m, m], dtype=DTYPE)
    center1 = (lt_pt + rb_pt) / 2.0

    size = size_vec * scale
    center = center0 + ux * center1[0] + uy * center1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)

    # -- Build affine matrix M_INV: original → crop --
    s = dsize / float(size[0])
    tgt_center = np.array([dsize / 2.0, dsize / 2.0], dtype=DTYPE)

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]
        tcx, tcy = tgt_center[0], tgt_center[1]
        M_INV = np.array([
            [s * costheta,  s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
            [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)],
        ], dtype=DTYPE)
    else:
        M_INV = np.array([
            [s, 0.0, tgt_center[0] - s * center[0]],
            [0.0, s, tgt_center[1] - s * center[1]],
        ], dtype=DTYPE)

    return M_INV


def crop_from_mediapipe(img_rgb: np.ndarray, landmarks_px: np.ndarray,
                        dsize: int = 256, scale: float = 2.3,
                        vy_ratio: float = -0.125, flag_do_rot: bool = True
                        ) -> Dict[str, Any]:
    """
    Crop a face region using MediaPipe 478-point landmarks.

    Parameters
    ----------
    img_rgb : np.ndarray  (H, W, 3) uint8 RGB
    landmarks_px : np.ndarray  (478, 2) pixel coordinates
    dsize : int  Output crop size (square).
    scale : float  Face crop scale (larger = smaller face).
    vy_ratio : float  Vertical offset (- = more forehead).
    flag_do_rot : bool  Whether to apply rotation correction.

    Returns
    -------
    dict with keys 'img_crop', 'M_o2c', 'M_c2o' — same interface as
    src/utils/crop.py:crop_image().
    """
    M_INV = _estimate_similar_transform_from_pts(
        landmarks_px, dsize=dsize, scale=scale,
        vy_ratio=vy_ratio, flag_do_rot=flag_do_rot,
    )

    img_crop = cv2.warpAffine(img_rgb, M_INV, (dsize, dsize), flags=cv2.INTER_LINEAR)
    M_o2c = np.vstack([M_INV, np.array([0.0, 0.0, 1.0], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    return {
        'img_crop': img_crop,
        'M_o2c': M_o2c,
        'M_c2o': M_c2o,
    }


class MediaPipeDetector:
    """Lightweight face detection using MediaPipe Face Landmarker."""

    def __init__(self, config: DetectorConfig):
        self._cfg = config
        self._landmarker = None
        self._mp_face_landmarker = None

    def warmup(self, dummy_shape=(480, 640, 3)) -> None:
        """Run a dummy inference to JIT-compile MediaPipe's model."""
        dummy = np.zeros(dummy_shape, dtype=np.uint8)
        self.detect(dummy)
        print('[detector] MediaPipe Face Landmarker warmed up.')

    def detect(self, img_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect a face in a BGR image and return landmarks + crop info.

        Returns None if no face is found.

        Returns dict with:
            'landmarks_px' : (478, 2) float32 pixel coordinates
            'bbox' : (x, y, w, h) int bounding box
        """
        if self._landmarker is None:
            self._init_landmarker()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # MediaPipe expects a mp.Image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=img_rgb)
        result = self._landmarker.detect(mp_image)

        if result.face_landmarks is None or len(result.face_landmarks) == 0:
            return None

        # Take the first (most prominent) face
        face_landmarks = result.face_landmarks[0]
        landmarks_arr = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=DTYPE)
        landmarks_px = _mediapipe_landmarks_to_pixel(landmarks_arr, w, h)

        # Bounding box from landmarks
        xs = landmarks_px[:, 0]
        ys = landmarks_px[:, 1]
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min()), int(ys.max() - ys.min()))

        return {'landmarks_px': landmarks_px, 'bbox': bbox}

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _init_landmarker(self) -> None:
        """Lazy-init MediaPipe Face Landmarker (auto-downloads model)."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._mp_face_landmarker = mp_vision

        model_path = self._get_model_path()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self._cfg.min_detection_confidence,
            min_tracking_confidence=self._cfg.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _get_model_path() -> str:
        """Return path to the MediaPipe face landmarker model, downloading if needed."""
        import os
        import urllib.request

        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'liveportrait')
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, 'face_landmarker_v2_with_blendshapes.task')

        if not os.path.exists(model_path):
            url = ('https://storage.googleapis.com/mediapipe-models/'
                   'face_landmarker/face_landmarker/float16/latest/'
                   'face_landmarker.task')
            print(f'[detector] Downloading MediaPipe model to {model_path}...')
            try:
                urllib.request.urlretrieve(url, model_path)
                print('[detector] Download complete.')
            except Exception as e:
                raise RuntimeError(
                    f'Failed to download MediaPipe model from {url}.\n'
                    f'Error: {e}\n'
                    f'Please download it manually and place it at:\n'
                    f'  {model_path}\n'
                    f'Or set the environment variable MEDIAPIPE_MODEL_PATH '
                    f'to the model location.'
                ) from e

        return model_path


# ---------------------------------------------------------------------------
# Backward-compatible helper for source image setup
# ---------------------------------------------------------------------------

def crop_source_for_realtime(img_rgb: np.ndarray, detector: MediaPipeDetector,
                              src_cfg: SourceConfig) -> Optional[Dict[str, Any]]:
    """
    Full source-image crop pipeline using MediaPipe instead of Cropper.

    1. Detect face with MediaPipe
    2. Crop to 256×256 with the same parameters as the original pipeline
    3. Return dict with 'img_crop_256x256', 'M_c2o', 'M_o2c'

    Returns None if no face detected.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    det = detector.detect(img_bgr)
    if det is None:
        return None

    crop_result = crop_from_mediapipe(
        img_rgb,
        det['landmarks_px'],
        dsize=256,
        scale=src_cfg.scale,
        vy_ratio=src_cfg.vy_ratio,
        flag_do_rot=src_cfg.flag_do_rot,
    )

    return {
        'img_crop_256x256': crop_result['img_crop'],
        'M_c2o': crop_result['M_c2o'],
        'M_o2c': crop_result['M_o2c'],
    }
