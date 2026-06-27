# LivePortrait Realtime

LivePortrait 实时推理加装包 —— 将 [LivePortrait](https://github.com/KlingTeam/LivePortrait) 的静态人像动画 pipeline 改造为实时运行，支持摄像头和视频文件输入，输出 MJPEG 流供 OBS 推流。

## 背景

原版 LivePortrait 是离线推理框架，处理一段视频需要数分钟。本项目的目标是将 5 模型 pipeline（F→M→W→G→S）优化为实时帧循环，同时不改动 `src/` 一行代码。

基于 [update.md](https://github.com/KlingTeam/LivePortrait/blob/main/update.md) 的设计思路，将三者（Camera/Video → Inference → Output）解耦为三线程流水线。

## 功能

- **摄像头实时驱动** — 支持 USB 摄像头输入（`--camera 0`）
- **视频文件驱动** — 支持视频文件循环输入（`--driving-video`），WSL2 首选模式
- **MJPEG HTTP 流** — 输出 `http://localhost:8080/stream`，OBS 添加"媒体源"即可推流
- **pyvirtualcam** — 原生 Linux 虚拟摄像头输出（OBS Virtual Camera）
- **人脸检测** — MediaPipe Face Landmarker，自动下载模型
- **Source 特征缓存** — F（AppearanceFeatureExtractor）仅在初始化时运行一次
- **FP16 + torch.compile** — M、W、G 三模型半精度 + CUDA graph 加速

## 系统要求

- Python 3.9+
- CUDA GPU（推荐 RTX 3060 及以上），或 Apple Silicon（MPS，实验性）
- 原版 LivePortrait 已部署并下载预训练权重

## 安装

```bash
# 1. 克隆原版 LivePortrait
git clone https://github.com/KlingTeam/LivePortrait
cd LivePortrait
pip install -r requirements_base.txt

# 2. 下载预训练权重（参见原项目 README）
#    pretrained_weights/
#      insightface/models/buffalo_l/
#      liveportrait/   (5 个 .pth + landmark.onnx)

# 3. 拉取本加装包
git init
git remote add origin https://github.com/Gendmyb/LivePortrait-live.git
git fetch origin realtime
git reset --hard origin/realtime

# 4. 安装额外依赖
pip install -r requirements_realtime.txt
```

## 使用方法

### 视频文件驱动（WSL2 推荐）

```bash
python liveportrait_realtime.py --driving-video assets/examples/driving/d0.mp4
```

### USB 摄像头驱动

```bash
python liveportrait_realtime.py --camera 0
```

### 自定义源图像

```bash
python liveportrait_realtime.py \
    --source my_face.jpg \
    --driving-video assets/examples/driving/d0.mp4
```

### 配置文件

```bash
python liveportrait_realtime.py --config /path/to/custom.yaml
```

## OBS 接收推流

### WSL2 → Windows OBS

1. 启动 realtime pipeline（MJPEG 默认开在 8080 端口）
2. OBS 添加"媒体源"，取消勾选"本地文件"
3. 输入：`http://localhost:8080/stream`
4. 输入格式：`mpjpeg`

WSL2 的 localhost 自动转发到 Windows，无需额外配置。

### 原生 Linux

可直接使用 pyvirtualcam（`output.virtual_camera: true`），在 OBS 中添加"视频捕获设备"。

或者浏览器打开 `http://localhost:8080/` 查看实时画面。

## 架构

```
[Camera / Video] ──(queue maxsize=1)──→ [Inference] ──(queue maxsize=2)──→ [Output]
       │                                       │                               │
   OpenCV /                              1. MediaPipe 人脸检测           MJPEG HTTP :8080
   VideoSource                           2. Crop → 256×256                → OBS 媒体源
                                         3. M: get_kp_info()             或 pyvirtualcam
                                         4. 相对运动合成                 或 cv2.imshow
                                         5. W+G: warp_decode()
```

三线程独立运行，队列限制最大长度以保证延迟最低（始终取最新帧）。

## 性能

WSL2 + RTX 3060 Laptop 实测：

| 阶段 | 耗时 |
|------|------|
| MediaPipe 人脸检测 (CPU) | ~13ms |
| M (MotionExtractor, GPU) | ~3.6ms |
| W+G (Warp + SPADE, GPU) | ~66ms |
| **总计** | **~11.5 FPS** |

> WSL2 GPU-PV 虚拟化带来约 2x 开销。在原生 Windows/Linux 上预期可达 20-25 FPS，RTX 4090 可达 70+ FPS。

## 配置说明

默认配置文件 `config_realtime.yaml`：

| 配置项 | 说明 |
|--------|------|
| `driving.mode` | `"camera"` 或 `"video"` |
| `driving.video_path` | 视频文件路径（mode=video 时） |
| `camera.device_id` | USB 摄像头设备号 |
| `source.image_path` | 源人像图片路径 |
| `source.scale` | 裁剪缩放（越大脸部越小） |
| `inference.flag_do_torch_compile` | 启用 torch.compile（约 1 分钟预热） |
| `inference.flag_use_half_precision` | FP16 推理 |
| `output.mjpeg_port` | MJPEG 流端口（0 禁用） |
| `output.virtual_camera` | pyvirtualcam 输出（仅原生 Linux） |

## 限制

- **人脸模式** — 仅支持人类，不支持动物
- **脸部输出** — 输出为 256×256 脸部，未实现 paste-back 拼回原图
- **无眼/唇重定向** — 为速度关闭
- **WSL2 摄像头** — 需 `usbipd` 工具桥接，建议用 `--driving-video`
- **pyvirtualcam** — WSL2 不可用，使用 MJPEG 替代

## 许可

本加装包基于原版 LivePortrait（MIT License）。详见 [LICENSE](https://github.com/KlingTeam/LivePortrait/blob/main/LICENSE)。
