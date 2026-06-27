# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LivePortrait is an efficient portrait animation system by Kuaishou Technology that animates a static portrait (human or animal) using motion from a driving video/image. It transfers facial expressions and head pose from a driving source to a target portrait, with stitching to paste the animated face back into the original image, and retargeting controls for eyes and lips.

Paper: [arXiv 2407.03168](https://arxiv.org/pdf/2407.03168)

## Setup

PyTorch must be installed separately (via conda or pip) — it is NOT in the requirements files. The README recommends conda.

```bash
# Core dependencies (PyTorch must already be installed)
pip install -r requirements_base.txt

# For GPU ONNX runtime
pip install -r requirements.txt

# For macOS Apple Silicon
pip install -r requirements_macOS.txt
```

Pretrained weights are **not** in the repo (gitignored). They must be downloaded separately into `pretrained_weights/` — see the README for download links. The expected structure:

```
pretrained_weights/
  insightface/models/buffalo_l/   # Face detection + landmark ONNX models
  liveportrait/                   # Human models (5 .pth files + landmark.onnx)
  liveportrait_animals/           # Animal models (5 .pth files + xpose.pth)
```

## Running Inference

### CLI — Humans
```bash
python inference.py -s assets/examples/source/s0.jpg -d assets/examples/driving/d0.mp4
```

### CLI — Animals
```bash
python inference_animals.py -s assets/examples/source/s39.jpg -d assets/examples/driving/d18.mp4
```

### Gradio Web UI — Humans
```bash
python app.py                    # starts on port 8890 by default
python app.py --share            # public share link
python app.py -p 7860            # custom port
```

### Gradio Web UI — Animals
```bash
python app_animals.py
```

### Key CLI flags (all in `src/config/argument_config.py`)
- `-s` / `-d` — source and driving paths (image, video, or `.pkl` template for driving)
- `-o` — output directory (default: `animations/`)
- `--flag_stitching` — stitch face back to original (default: True; disable for large head motion or animals)
- `--flag_relative_motion` — use relative motion transfer (default: True)
- `--flag_pasteback` — paste animated crop back to original image space (default: True)
- `--flag_do_crop` — crop source to face space (default: True)
- `--animation_region` — one of `"exp"`, `"pose"`, `"lip"`, `"eyes"`, `"all"`
- `--driving_option` — `"expression-friendly"` or `"pose-friendly"`
- `--flag_eye_retargeting` / `--flag_lip_retargeting` — eye/lip ratio transfer (WIP, not recommended)
- `--flag_use_half_precision` — FP16 inference (default: True; set False if black boxes appear)
- `--flag_force_cpu` — CPU-only inference (WIP)
- `--scale` — face crop scale (default: 2.3; larger = smaller face area)
- `--det_thresh` — face detection threshold (default: 0.15)

Driving videos can be pre-processed into `.pkl` motion templates for faster reuse and privacy.

## Architecture

### The 5-Model Pipeline (F → M → W → G → S)

The core neural pipeline mirrors the paper. All model architectures are defined in `src/config/models.yaml` and instantiated by `src/utils/helper.py:load_model()`.

| Component | Module | Paper Role | Purpose |
|-----------|--------|------------|---------|
| **F** | `AppearanceFeatureExtractor` | Encoder | Downsamples 3×256×256 source face → 3D feature volume (32×16×64×64) |
| **M** | `MotionExtractor` | Keypoint Detector | ConvNeXtV2-Tiny backbone → 21 implicit keypoints + head pose (pitch/yaw/roll) + expression deformation + scale + translation |
| **W** | `WarpingNetwork` | Warping Module | Computes dense optical flow from sparse keypoints via `DenseMotionNetwork` (hourglass), warps the source feature volume |
| **G** | `SPADEDecoder` | Generator | SPADE-based decoder; generates the final animated face (256×256, or 512×512 with optional upscale) from warped features |
| **S** | `StitchingRetargetingNetwork` | Stitching + Retargeting | Three small MLPs: shoulder stitching (keypoint alignment), eye retargeting, lip retargeting |

### Pipeline Classes

- **`LivePortraitWrapper`** (`src/live_portrait_wrapper.py`) — Loads and owns all 5 models. Provides low-level inference methods: `prepare_source()`, `extract_feature_3d()`, `get_kp_info()`, `transform_keypoint()`, `stitching()`, `retarget_eye()`, `retarget_lip()`, `warp_decode()`. Has FP16 context manager (`inference_ctx()`) and `torch.compile()` support. `LivePortraitWrapperAnimal` subclass loads animal-specific checkpoints.

- **`LivePortraitPipeline`** (`src/live_portrait_pipeline.py`) — Human inference orchestrator. Full `execute()` flow: load source → crop face → extract appearance feature (F) → extract keypoints (M) → for each driving frame: compose motion, optionally stitch (S), optionally retarget (R), warp+decode (W+G), paste-back to original image.

- **`LivePortraitPipelineAnimal`** (`src/live_portrait_pipeline_animal.py`) — Animal inference. Simpler: no eye/lip retargeting, source must be an image (no source video), uses XPose for animal landmark detection.

- **`GradioPipeline`** (`src/gradio_pipeline.py`) — Extends `LivePortraitPipeline` with real-time expression editing controls for the Gradio UI (eyeball direction, smile, wink, eyebrow, lip, head movement sliders).

### Inference Flow (Human)

1. **Load source**: Image or video, resize to limit max dimension
2. **Crop face**: RetinaFace detection → affine transform → 256×256 crop
3. **Extract source features** (cached per source):
   - F: `extract_feature_3d(I_s)` → 3D appearance volume
   - M: `get_kp_info(I_s)` → keypoints, then `transform_keypoint()` (Eqn. 2 from paper)
   - Compute rotation matrix: `R_s = get_rotation_matrix(pitch, yaw, roll)`
4. **Process driving video**: Detect landmarks per frame → compute eye/lip ratios → extract keypoints via M → build motion template dict → optionally cache as `.pkl`
5. **Per-frame animation loop**:
   - Compose motion (relative: `delta = x_s_exp + (x_d_exp - x_d0_exp)`)
   - Apply region mask (`exp`/`pose`/`lip`/`eyes`/`all`)
   - Stitch (S): correct keypoint misalignment
   - Retarget (R): transfer eye/lip closure ratios
   - Warp + decode (W + G): deform features → generate image
   - Paste-back: inverse affine transform to original image coordinates
6. **Output**: Concatenated video (driving | source | generated), MP4 with optional audio

### Configuration System

Layered design using Python dataclasses:

- **`ArgumentConfig`** (`src/config/argument_config.py`) — All user-facing CLI + Gradio arguments. Parsed by `tyro` (typed argument parser). Has aliases like `-s`, `-d`, `-o`.
- **`InferenceConfig`** (`src/config/inference_config.py`) — Internal inference parameters: checkpoint paths, model YAML path, output format/fps/CRF, defaults for boolean flags.
- **`CropConfig`** (`src/config/crop_config.py`) — Face detection and cropping parameters.
- **`models.yaml`** (`src/config/models.yaml`) — All neural network architecture hyperparameters.

`partial_fields()` bridges user args to internal configs by filtering kwargs to only attributes the target config class has.

### Key Utility Modules

- **`src/utils/cropper.py`** — `Cropper` class: face/animal detection via InsightFace (RetinaFace for humans) and XPose (for animals), landmark detection, video cropping with trajectory tracking
- **`src/utils/crop.py`** — Affine transforms for face cropping and paste-back
- **`src/utils/camera.py`** — Head pose predictions → Euler angles, rotation matrices
- **`src/utils/video.py`** — Video I/O via imageio + ffmpeg (concatenation, audio handling, GIF output)
- **`src/utils/helper.py`** — Model factory (`load_model()`), file path helpers
- **`src/utils/filter.py`** — Kalman filter smoothing for source video frames

### Bundled Dependencies

- `src/utils/dependencies/insightface/` — Modified InsightFace for face detection
- `src/utils/dependencies/XPose/` — XPose animal pose estimation (UniPose with Swin-T backbone, CUDA kernels)

## Model Loading

`load_model()` in `helper.py` is the factory: reads model type from `models.yaml`, instantiates the correct class, loads the `.pth` checkpoint via `torch.load()`, sets eval mode. The stitching/retargeting module is special — a single `.pth` contains three separate state dicts (`retarget_shoulder`, `retarget_mouth`, `retarget_eye`) that become three `StitchingRetargetingNetwork` instances.

## Platform Notes

- **GPU**: CUDA with FP16 (default). Set `--flag_use_half_precision False` if black boxes appear (GPU incompatibility).
- **macOS Apple Silicon**: Use `requirements_macOS.txt` (CPU PyTorch + onnxruntime-silicon). MPS device selection is in `LivePortraitWrapper`.
- **CPU**: WIP — `--flag_force_cpu` exists but may not work fully.

## Testing

There is no formal test suite — this is a research/demo codebase.

## Real-time Subproject (`realtime/`)

Implements `update.md`'s design — a real-time inference framework wrapping official code without modifying `src/`. Full user-facing docs: **[README_REALTIME.md](README_REALTIME.md)**.

### Quick Start

```bash
# Install extra deps
pip install -r requirements_realtime.txt

# Video-file driving (default on WSL2 — no camera needed)
python liveportrait_realtime.py \
    --driving-video assets/examples/driving/d0.mp4

# Then open OBS → Media Source → http://localhost:8080/stream → format: mpjpeg
# Or browser: http://localhost:8080/

# USB camera driving (requires hardware access)
python liveportrait_realtime.py --camera 0

# CLI overrides
python liveportrait_realtime.py --source my_face.jpg --camera 1
python liveportrait_realtime.py --config /path/to/custom.yaml
```

### Package Structure

```
realtime/
  __init__.py          # Package marker, exports RealtimePipeline
  config.py            # YAML config loader + PrintableConfig dataclasses
  camera.py            # USB camera capture (OpenCV, queue.Queue maxsize=1)
  video_source.py      # Video-file frame source (loops, same interface as camera)
  detector.py          # MediaPipe Face Landmarker + crop_from_mediapipe()
  renderer.py          # Source feature caching (F once) + per-frame warp_decode (M+W+G)
  virtualcam.py        # Output: MJPEG HTTP / pyvirtualcam / cv2.imshow preview
  mjpeg_streamer.py    # Lightweight HTTP MJPEG streaming server
  fps.py               # Rolling-window FPS counter + overlay
  pipeline.py          # Orchestrator: 3 threads, queues, lifecycle
liveportrait_realtime.py  # CLI entry point (tyro)
config_realtime.yaml       # Default YAML configuration
requirements_realtime.txt  # mediapipe, pyvirtualcam, pyyaml, tyro
README_REALTIME.md         # User documentation
```

### Architecture

3-thread pipeline, zero modifications to `src/`:

```
[Camera/Video] ──(queue maxsize=1)──→ [Inference] ──(queue maxsize=2)──→ [Output]
     │                                      │                              │
  OpenCV /                           1. MediaPipe detect             MJPEG HTTP
  VideoSource                        2. Crop to 256×256               → OBS
                                     3. M: get_kp_info()              or
                                     4. Relative motion compose    pyvirtualcam
                                     5. W+G: warp_decode()         or cv2.imshow
```

### Runtime Behavior & Log Flow

Normal startup sequence:
```
[pipeline] Loading LivePortrait models...
[pipeline] Models loaded.
[pipeline] Converting models to FP16...       # M+W+G permanently half()'d
[pipeline] Registering torch.compile...        # M+W+G compiled (reduce-overhead)
[mjpeg] Stream server started on http://localhost:8080/stream
[pipeline] All threads started.
[inference] Warming up compiled kernels...     # ~30-60s, in inference thread
[renderer] Source keypoints cached. Ready.
[inference] torch.compile warmup complete.
[renderer] First driving frame cached as reference.
[inference] Frame 100, FPS: 11.5  (detect: 13ms  infer: 74ms)
[profile] M:3.6ms  compose:0.4ms  W+G:66.1ms  G+parse:2.0ms
```

### Key Optimizations (in `pipeline._optimize_models()`)

1. **M + W + G permanently `.half()`'d** (FP16 weights) — avoids per-call autocast overhead. F is not half()'d (runs once for source caching).
2. **torch.compile on M + W + G** — `mode='reduce-overhead'` with CUDA graphs. Compilation registered in main thread; **warmup + first-call deferred to inference thread** so CUDA graph tree managers live in the correct thread-local storage (avoids cross-thread `AssertionError`).
3. **Source keypoints init also deferred to inference thread** — `renderer.init_source_keypoints()` called in inference thread to ensure first M call captures CUDA graphs correctly.
4. **Source F cached** — `extract_feature_3d()` runs once at init, never per-frame.
5. **Driving first-frame cached** — `x_d_0_info` cached as relative-motion baseline.
6. **cuDNN benchmark + TF32** — `torch.backends.cudnn.benchmark = True`, TF32 matmul enabled for Ampere GPU.

### Config Key Points

- `driving.mode: "camera"` or `"video"` — switch between USB camera and video file
- `driving.video_path` — path to video when mode=video
- `inference.flag_do_torch_compile: true` — controls pipeline's own compile (NOT wrapper's)
- `inference.flag_use_half_precision: true` — controls pipeline's half() conversion
- `output.mjpeg_port: 8080` — HTTP MJPEG streaming port (0 = disabled; primary output for WSL2)
- `output.virtual_camera: true` — pyvirtualcam (only works on native Linux, not WSL2)
- `output.show_preview: false` — OpenCV preview window (WSLg often shows black)

### Performance Benchmarks (WSL2 + RTX 3060 Laptop)

speed.py baseline (FP16 + torch.compile max-autotune):
| Module | Time |
|--------|------|
| F (Appearance) | 4.55 ms |
| M (Motion) | 2.17 ms |
| W (Warp) | 27.75 ms |
| G (SPADE) | 41.82 ms |
| **M+W+G total** | **~72 ms → ~14 FPS** |

Realtime pipeline (FP16 + reduce-overhead compile, measured):
| Stage | Time |
|-------|------|
| Detector (MediaPipe CPU) | 13 ms |
| M (MotionExtractor GPU) | 3.6 ms |
| W+G (Warp + SPADE GPU) | 66 ms |
| **Total** | **~87 ms → 11.5 FPS** |

**WSL2 GPU overhead**: ~2x slower than native. To reach 20-30 FPS, run natively on Windows or bare-metal Linux, or explore TensorRT (FasterLivePortrait).

### Known Issues / Limitations

- **WSL2 → OBS**: pyvirtualcam/v4l2loopback doesn't work cross-OS. Use built-in MJPEG HTTP streaming (`http://localhost:8080/stream`) instead.
- **Camera requires usbipd on WSL2**: `usbipd bind --busid X-X; usbipd attach --wsl --busid X-X`. Or use `--driving-video` for video file driving.
- **Output is face-only** (256×256 upscaled to output res): paste-back to original frame is not implemented in real-time path.
- **Animal mode not supported**: human only.
- **No eye/lip retargeting**: disabled by default for speed; stitching also off.
- **Preview window black on WSLg**: OpenCV window may show black on WSL2's X server. MJPEG streaming is unaffected.

### Error Recovery

- **No face detected**: skips frame, pushes last valid output, resets driving reference after ~30 consecutive misses (~1s)
- **CUDA OOM**: caught per-frame, logs warning, clears cache, continues
- **Camera disconnect**: auto-retries every 0.5s
- **MediaPipe model**: auto-downloads to `~/.cache/liveportrait/` on first run; falls back with clear error if offline
- **MJPEG shutdown**: `stop_event` signals all handler threads → `condition.notify_all()` wakes them → `server.shutdown()` completes cleanly
