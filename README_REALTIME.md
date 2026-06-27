# LivePortrait Real-Time Inference

USB 摄像头 / 视频文件 → LivePortrait 动画 → OBS 推流，**基于官方代码零修改**的外部封装。

![](https://img.shields.io/badge/python-3.10+-blue) ![](https://img.shields.io/badge/pytorch-2.6+-red) ![](https://img.shields.io/badge/GPU-CUDA-green)

---

## 快速开始

### 1. 安装依赖

```bash
# PyTorch 需预先安装（推荐 conda）
pip install -r requirements_realtime.txt
```

依赖项：`mediapipe` `pyvirtualcam` `opencv-python` `pyyaml` `tyro`

### 2. 准备权重

确保 `pretrained_weights/` 下有标准 LivePortrait 模型文件（参见主 README）。

### 3. 运行

```bash
# 视频文件驱动（无需摄像头，适合 WSL2 / 测试）
python liveportrait_realtime.py \
    --driving-video assets/examples/driving/d0.mp4

# USB 摄像头驱动（需要硬件访问）
python liveportrait_realtime.py --camera 0

# 指定源图片 + 自定义配置
python liveportrait_realtime.py \
    --source my_face.jpg \
    --driving-video my_video.mp4 \
    --config /path/to/custom.yaml
```

### 4. OBS 接收推流

WSL2 环境下自动启动 MJPEG HTTP 推流（默认端口 8080）：

1. OBS → 添加 **"媒体源"** (Media Source)
2. 取消勾选 "本地文件"
3. 输入：`http://localhost:8080/stream`
4. 输入格式：`mpjpeg`
5. 或在浏览器打开 `http://localhost:8080/` 预览

原生 Linux 环境下可使用 pyvirtualcam（OBS 直接识别为虚拟摄像头）。

---

## 配置说明

编辑 `config_realtime.yaml`：

```yaml
# --- 驱动源 ---
driving:
  mode: "video"            # "camera" 或 "video"
  video_path: "assets/examples/driving/d0.mp4"

# --- 源肖像 ---
source:
  image_path: "assets/examples/source/s0.jpg"   # 改成你的照片
  scale: 2.3              # 裁剪缩放 (越大脸越小)
  vy_ratio: -0.125        # 垂直偏移 (负值=更多额头)

# --- 推理 ---
inference:
  flag_use_half_precision: true     # FP16 (显著提速)
  flag_do_torch_compile: true       # torch.compile (~30s 预热, ~2x 提速)
  flag_stitching: false             # 关闭以提速
  animation_region: "all"           # "all" / "exp" / "pose" / "lip" / "eyes"
  driving_option: "expression-friendly"

# --- 输出 ---
output:
  mjpeg_port: 8080        # MJPEG 推流端口 (0=禁用)
  mjpeg_quality: 85       # JPEG 质量
  show_preview: false      # OpenCV 预览窗口 (WSL2 上可能黑屏)
```

完整选项见 `config_realtime.yaml`。

---

## 架构

```
 Camera/Video ──(queue maxsize=1)──→  Inference  ──(queue maxsize=2)──→  Output
      │                                   │                               │
   OpenCV /                         1. MediaPipe 人脸检测           MJPEG HTTP
   VideoSource                      2. 裁剪 256×256                   → OBS
                                    3. M: 运动关键点                 或 pyvirtualcam
                                    4. 相对运动合成                 或 cv2.imshow
                                    5. W+G: 扭曲+生成
```

**三线程管线**：
- **帧源线程** — 摄像头/视频读取，queue maxsize=1（始终最新帧）
- **推理线程** — 人脸检测 → 裁剪 → MotionExtractor → Warp → SPADE Decoder
- **输出线程** — MJPEG 编码 → HTTP 推流 / 虚拟摄像头

**核心优化**：
- F (AppearanceFeatureExtractor) 仅运行一次，缓存 source feature
- M/W/G 永久转换为 FP16
- M/W/G 使用 `torch.compile(mode='reduce-overhead')` 编译
- CUDA graph 捕获在推理线程中完成（避免跨线程 TLS 冲突）
- 首帧驱动缓存作为相对运动基准

---

## 性能

| 环境 | 人脸检测 | M | W+G | 总帧率 |
|------|---------|---|-----|--------|
| WSL2 + RTX 3060 Laptop | 13ms | 3.6ms | 66ms | **~11.5 FPS** |
| 原生 Linux + RTX 3060 (预估) | 13ms | ~2ms | ~35ms | **~20-25 FPS** |
| 原生 Linux + RTX 4090 | - | - | - | **~78 FPS** (论文数据) |

WSL2 下 GPU 虚拟化带来约 2x 性能损耗。要获得更高帧率请使用原生 Linux/Windows 或 TensorRT。

社区项目 [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) 使用 TensorRT 可达 30+ FPS (RTX 3090)。

---

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--config <path>` | YAML 配置文件路径 (默认: `config_realtime.yaml`) |
| `--source <path>` | 源肖像图片 (覆盖配置文件) |
| `--camera <id>` | USB 摄像头设备 ID (覆盖配置文件) |
| `--driving-video <path>` | 视频文件驱动 (覆盖配置文件) |
| `--preview` | 强制显示 OpenCV 预览窗口 |

---

## 常见问题

**Q: WSL2 上摄像头无法使用？**
A: 需要 USB/IP 透传。在 Windows 宿主机 PowerShell (管理员) 中：
```powershell
usbipd list                       # 查看摄像头 BUSID
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```
或使用 `--driving-video` 模式，无需摄像头。

**Q: MJPEG 推流没有画面？**
A: 确认 OBS 媒体源地址是 `http://localhost:8080/stream`，格式 `mpjpeg`。可先在浏览器打开 `http://localhost:8080/` 确认服务器正常。

**Q: 预览窗口黑屏？**
A: WSLg 的 OpenCV 渲染兼容性问题，不影响 MJPEG 推流。使用 OBS 拉流即可。

**Q: 如何换自己的照片？**
A: 编辑 `config_realtime.yaml` 中 `source.image_path`，或加 `--source my_face.jpg`。照片要求：正面、清晰、光照均匀。

**Q: FPS 太低？**
A: WSL2 有约 2x GPU 虚拟化开销。在原生 Linux 或 Windows 上运行可提升至 20-25 FPS。也可研究 TensorRT 方案。

**Q: torch.compile 预热太久？**
A: 首次运行需 30-60 秒编译内核。后续运行受益于编译缓存，启动更快。

---

## 项目结构

```
realtime/
  __init__.py          # 包入口, 导出 RealtimePipeline
  config.py            # YAML 配置加载 + dataclass 定义
  camera.py            # USB 摄像头采集 (OpenCV)
  video_source.py      # 视频文件帧源 (循环播放)
  detector.py          # MediaPipe 人脸检测 + 裁剪
  renderer.py          # 源特征缓存 + 逐帧推理
  virtualcam.py        # 输出: MJPEG / pyvirtualcam / 预览窗口
  mjpeg_streamer.py    # 轻量 HTTP MJPEG 推流服务
  fps.py               # 滚动窗口 FPS 计数器 + 叠加层
  pipeline.py          # 三线程管线编排器
liveportrait_realtime.py  # CLI 入口 (tyro)
config_realtime.yaml      # 默认配置
requirements_realtime.txt # 额外依赖
```
