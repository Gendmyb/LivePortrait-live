# coding: utf-8

"""
Configuration dataclasses for LivePortrait real-time inference.
Follows the same PrintableConfig pattern as src/config/.
"""

import os
import os.path as osp
import yaml
from dataclasses import dataclass, field
from typing import Optional


class PrintableConfig:
    """Printable Config defining str function (same as src/config/base_config.py)."""

    def __repr__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            lines += f'{key}: {str(val)}'.split('\n')
        return '\n    '.join(lines)


@dataclass(repr=False)
class CameraConfig(PrintableConfig):
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    use_mjpg: bool = True


@dataclass(repr=False)
class DrivingConfig(PrintableConfig):
    mode: str = 'camera'           # 'camera' or 'video'
    video_path: str = ''           # path to video file (when mode='video')


@dataclass(repr=False)
class SourceConfig(PrintableConfig):
    image_path: str = 'assets/examples/source/s0.jpg'
    scale: float = 2.3
    vy_ratio: float = -0.125
    flag_do_rot: bool = True


@dataclass(repr=False)
class DetectorConfig(PrintableConfig):
    model_complexity: int = 1  # 0=fastest, 1=balanced, 2=best
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False  # False = tracking mode for video


@dataclass(repr=False)
class InferenceRTConfig(PrintableConfig):
    device_id: int = 0
    flag_force_cpu: bool = False
    flag_use_half_precision: bool = True
    flag_do_torch_compile: bool = False  # off by default (warmup cost)
    flag_stitching: bool = False
    flag_relative_motion: bool = True
    flag_pasteback: bool = False
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    animation_region: str = 'all'
    driving_multiplier: float = 1.0
    driving_option: str = 'expression-friendly'


@dataclass(repr=False)
class OutputConfig(PrintableConfig):
    fps: int = 30
    width: int = 640
    height: int = 480
    virtual_camera: bool = True
    show_preview: bool = False
    mjpeg_port: int = 8080          # HTTP MJPEG streaming port (0 = disabled)
    mjpeg_quality: int = 85         # JPEG quality for MJPEG stream


@dataclass(repr=False)
class PerformanceConfig(PrintableConfig):
    queue_size: int = 1
    log_interval: int = 100


@dataclass(repr=False)
class RealtimeConfig(PrintableConfig):
    driving: DrivingConfig = field(default_factory=DrivingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    inference: InferenceRTConfig = field(default_factory=InferenceRTConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


def load_config(config_path: str) -> RealtimeConfig:
    """Load a YAML config file into a RealtimeConfig dataclass."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    return RealtimeConfig(
        driving=DrivingConfig(**data.get('driving', {})),
        camera=CameraConfig(**data.get('camera', {})),
        source=SourceConfig(**data.get('source', {})),
        detector=DetectorConfig(**data.get('detector', {})),
        inference=InferenceRTConfig(**data.get('inference', {})),
        output=OutputConfig(**data.get('output', {})),
        performance=PerformanceConfig(**data.get('performance', {})),
    )
