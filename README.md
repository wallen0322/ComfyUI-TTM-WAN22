# ComfyUI-TTM

TTM (Time-to-Move) node for Wan2.2 in ComfyUI. Dual-clock denoising for motion-controlled video generation.

为ComfyUI实现的TTM节点，用于Wan2.2模型的运动控制视频生成，采用双时钟去噪技术。

Based on: https://github.com/time-to-move/TTM

## Installation / 安装

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/wallen0322/ComfyUI-TTM-WAN22.git
```

Restart ComfyUI. The node will appear under `conditioning/video_models`.

重启ComfyUI后，节点会出现在 `conditioning/video_models` 分类下。

## Node: WanTTMImageToVideo

### Inputs / 输入参数

| Input | Type | Description |
|-------|------|-------------|
| start_image | IMAGE | First frame / 起始图像（第一帧） |
| motion_signal_video | IMAGE | Motion trajectory video / 运动轨迹视频 |
| motion_signal_mask | MASK | Motion region mask / 运动区域掩码 |
| tweak_index | INT | Denoising start for background / 背景去噪起始步数 (default: 3) |
| tstrong_index | INT | Denoising start for masked region / 物体去噪起始步数 (default: 7) |
| positive/negative | CONDITIONING | Text prompts / 文本提示词 |
| vae | VAE | VAE model / VAE模型 |
| width/height/length | INT | Video dimensions / 视频尺寸 |

## Dual-Clock Parameters / 双时钟参数详解

### What are tweak_index and tstrong_index? / 参数含义

TTM uses **dual-clock denoising** to separately control background and motion regions:

TTM使用**双时钟去噪**技术，分别控制背景和运动区域：

#### `tweak_index` - Background Control / 背景控制
- **Controls**: Areas **outside** the motion mask (background)
- **控制对象**：运动掩码**外部**区域（背景）
- **Effect**: When background starts denoising
- **作用**：背景从第几步开始去噪
- Lower value = earlier start, may cause scene deformation
- 数值越低 = 开始越早，可能导致场景变形
- Higher value = later start, background may appear too static
- 数值越高 = 开始越晚，背景可能过于静止

#### `tstrong_index` - Motion Control / 运动控制
- **Controls**: Areas **inside** the motion mask (moving objects)
- **控制对象**：运动掩码**内部**区域（运动物体）
- **Effect**: When objects start following motion trajectory
- **作用**：物体从第几步开始按轨迹运动
- Lower value = less controlled motion, may drift from path
- 数值越低 = 运动控制弱，可能偏离轨迹
- Higher value = rigid motion, may appear unnatural
- 数值越高 = 运动僵硬，可能不自然

### How it Works / 工作原理

For example, with 50 inference steps and `tweak_index=3, tstrong_index=7`:

例如，50步推理，参数设为 `tweak_index=3, tstrong_index=7`：

```
Steps:    0 ──────── 3 ───── 7 ──────────── 50
          │          │       │              │
Background: [noise]→[denoise]──────────→[done]
背景:      【噪声】→【去噪】──────────→【完成】
          │          │       │              │
Object:   [noise]→[wait]→[motion denoise]→[done]
物体:      【噪声】→【等待】→【按轨迹去噪】→【完成】
```

**Key**: The gap between two parameters allows background to stabilize before object motion begins.

**关键**：两个参数的间隔让背景先稳定，再让物体开始运动。

### Recommended Parameters / 推荐参数

**Cut-and-Drag** (Object moving from A to B / 物体从A移动到B):
- `tweak_index=3, tstrong_index=7`

**Camera Control** (Camera movement / 相机运动):
- `tweak_index=2, tstrong_index=5`

**Constraint / 约束条件**: `0 ≤ tweak_index ≤ tstrong_index ≤ num_inference_steps`

### Adjustment Tips / 调参建议

**If background deforms / 如果背景变形**:
- Increase `tweak_index` (try 4-5)
- 增加 `tweak_index` (尝试4-5)

**If motion is too rigid / 如果运动太僵硬**:
- Decrease `tstrong_index` (try 5-6)
- 降低 `tstrong_index` (尝试5-6)

**If object drifts from path / 如果物体偏离轨迹**:
- Increase `tstrong_index` (try 8-9)
- 增加 `tstrong_index` (尝试8-9)

**If background too static / 如果背景过于静止**:
- Decrease `tweak_index` (try 2)
- 降低 `tweak_index` (尝试2)

## Workflow

```
LoadImage ──┐
LoadVideo ──┼──> WanTTMImageToVideo ──> KSampler ──> VAEDecode ──> SaveVideo
LoadMask ───┘
```

## Requirements

- ComfyUI
- Wan2.2-I2V model from HuggingFace
- ComfyUI-VideoHelperSuite (for video loading)

## License

Based on TTM project. Follow original license terms.
