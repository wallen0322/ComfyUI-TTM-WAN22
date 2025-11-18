# ComfyUI-TTM

TTM (Time-to-Move) node for Wan2.2 in ComfyUI. Dual-clock denoising for motion-controlled video generation.

Based on: https://github.com/time-to-move/TTM

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone [your-repo-url] comfyui-TTM
```

Restart ComfyUI. The node will appear under `conditioning/video_models`.

## Node: WanTTMImageToVideo

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| start_image | IMAGE | First frame |
| motion_signal_video | IMAGE | Motion trajectory video |
| motion_signal_mask | MASK | Motion region mask |
| tweak_index | INT | Denoising start for background (default: 3) |
| tstrong_index | INT | Denoising start for masked region (default: 7) |
| positive/negative | CONDITIONING | Text prompts |
| vae | VAE | VAE model |
| width/height/length | INT | Video dimensions |

### Recommended Parameters

**Cut-and-Drag**: `tweak_index=3`, `tstrong_index=7`  
**Camera Control**: `tweak_index=2`, `tstrong_index=5`

Constraint: `0 ≤ tweak_index ≤ tstrong_index ≤ num_inference_steps`

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
