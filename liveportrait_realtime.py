#!/usr/bin/env python
# coding: utf-8

"""
LivePortrait Real-Time Inference

Camera or video → LivePortrait → preview window (or OBS Virtual Camera).

Usage:
    python liveportrait_realtime.py                                    # USB camera
    python liveportrait_realtime.py --driving-video assets/examples/driving/d0.mp4  # video file
    python liveportrait_realtime.py --source my_face.jpg --camera 1
    python liveportrait_realtime.py --config path/to/config.yaml
"""

import os
import sys

# Ensure the project root is on sys.path so that imports from
# src/ work correctly (e.g. ``from src.live_portrait_wrapper import ...``).
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tyro
from dataclasses import dataclass
from typing import Optional

from realtime.config import load_config
from realtime.pipeline import RealtimePipeline


@dataclass
class CLIArgs:
    config: Optional[str] = None          # path to YAML config file
    source: Optional[str] = None          # override source image path
    camera: Optional[int] = None          # override camera device ID
    driving_video: Optional[str] = None   # use video file as driving source
    preview: bool = False                 # force-show OpenCV preview window


def main():
    args = tyro.cli(CLIArgs)

    # Load config
    config_path = args.config or os.path.join(_project_root, 'config_realtime.yaml')
    if not os.path.exists(config_path):
        print(f'Config file not found: {config_path}')
        sys.exit(1)

    print(f'Loading config from: {config_path}')
    cfg = load_config(config_path)

    # CLI overrides
    if args.source:
        cfg.source.image_path = args.source
    if args.camera is not None:
        cfg.camera.device_id = args.camera
    if args.driving_video:
        cfg.driving.mode = 'video'
        cfg.driving.video_path = args.driving_video
    if args.preview:
        cfg.output.show_preview = True

    print(cfg)

    # Run
    pipeline = RealtimePipeline(cfg)
    pipeline.start()
    pipeline.wait()


if __name__ == '__main__':
    main()
