# TTM 节点使用说明

## 节点清单

### 1. WanTTMConditioning
准备 TTM 运动控制条件

**关键参数**:
- `tweak_index` (0-15): 背景去噪起始步 **[滑块优化]**
  - 推荐: 2-5
  - 小值: 背景早去噪
  - 大值: 背景晚去噪
  
- `tstrong_index` (0-20): 运动控制起始步 **[滑块优化]**
  - 推荐: 5-10
  - 必须 ≥ tweak_index
  - 控制运动轨迹精度

### 2. WanTTMSamplerComplete ⭐
完整 TTM 采样器（双模型 MoE + 双时钟）

**基础参数**:
- `steps` (10-150): 采样步数 **[优化范围]**
  - 推荐: 30-80
  - 默认: 50
  
- `cfg` (0-15): CFG 强度 **[滑块优化]**
  - 推荐: 3.0-7.0
  - 默认: 5.0

**MoE 参数** (高级):
- `boundary` (0.5-1.0): 模型切换点 **[精确滑块]**
  - 推荐: 0.85-0.92
  - 默认: 0.875 (87.5%)
  
- `sigma_shift` (1-20): Flow matching 偏移 **[滑块优化]**
  - 推荐: 6.0-10.0
  - 默认: 8.0

- `denoise` (0-1): 去噪强度 **[滑块]**
  - 默认: 1.0 (完全去噪)

## 工作流示例

```
[Load Checkpoint (Wan2.2)]
├─ model → model_high ──┐
└─ model → model_low ───┤
                        │
[VAE/CLIP] ─────────────┤
[Images] ───────────────┤
                        │
    WanTTMConditioning
    ├─ tweak: 3
    └─ tstrong: 7
         │
    positive ──────┐
    negative ──────┤
    latent ────────┤
                   │
    WanTTMSamplerComplete
    ├─ steps: 50
    ├─ cfg: 5.0
    ├─ boundary: 0.875
    └─ sigma_shift: 8.0
         │
      [latent]
         │
    [VAEDecode] → [Save]
```

## UI 优化说明

### 滑块参数 (0-1 或小范围)
- `cfg`: 0-15，step 0.1
- `boundary`: 0.5-1.0，step 0.005
- `denoise`: 0-1，step 0.05
- `tweak_index`: 0-15，step 1
- `tstrong_index`: 0-20，step 1

### 标准输入
- `seed`: 大范围整数
- `steps`: 10-150 (优化范围)
- `width/height`: 64-2048 (实用范围)

### 分组布局
1. **Core**: models, conditioning, latent
2. **Basic**: seed, steps, cfg
3. **Sampler**: sampler_name, scheduler
4. **MoE**: boundary, sigma_shift
5. **Advanced**: denoise

## 参数调优建议

### 物体移动 (Cut-and-Drag)
```
tweak: 3
tstrong: 7
boundary: 0.875
steps: 50
cfg: 5.0
```

### 相机运动
```
tweak: 2
tstrong: 5
boundary: 0.85
steps: 50
cfg: 6.0
```

### 快速预览
```
steps: 30
cfg: 4.0
denoise: 0.8
```

### 高质量
```
steps: 80
cfg: 5.5
sigma_shift: 9.0
```

## 调试输出

运行时会打印：
```
[TTM-MoE] MoE Switch: 44/50 (boundary=0.875)
[TTM-MoE] Dual-Clock: tweak=3, tstrong=7
[TTM-MoE] Checkpoints: [0, 3, 7, 44, 50]
[TTM-MoE] Segment 0->3: high model
[TTM-MoE] Segment 3->7: high model
[TTM-MoE] Applying TTM blend at step 7
[TTM-MoE] Segment 7->44: high model
[TTM-MoE] Segment 44->50: low model
```

这些信息帮助验证 TTM 逻辑是否正确执行。
