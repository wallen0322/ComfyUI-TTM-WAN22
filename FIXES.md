# 修复说明

## 问题 1: 步数限制太高 ❌
**原因**: `steps` 参数设置 min=20，不支持蒸馏模型的 1-8 步
**修复**: 改为 min=1, max=10000，支持所有模型

## 问题 2: 输出全是噪声 ❌
**原因**: 降噪逻辑错误，`disable_noise` 和 `force_full_denoise` 参数使用不当

### 错误逻辑（之前）
```python
# 每个小段都独立采样，force_full_denoise 设置混乱
for segment in segments:
    sample(model, start, end, force_full_denoise=(end==steps))
```

### 正确逻辑（参考 WanMoeKSampler）
```python
# 两段式采样，正确设置 disable_noise 和 force_full_denoise

# Phase 1: 高噪阶段
if start_with_high:
    sample(
        model_high, 
        start=0, 
        end=switching_step,
        disable_noise=end_with_low,      # 如果还有低噪阶段，不添加噪声
        force_full_denoise=end_with_low  # 如果还有低噪阶段，不强制完全去噪
    )

# Phase 2: 低噪阶段  
if end_with_low:
    sample(
        model_low,
        start=switching_step,
        end=steps,
        disable_noise=False,             # 低噪阶段不添加新噪声
        force_full_denoise=True          # 最终阶段强制完全去噪
    )
```

## 问题 3: MoE 切换计算错误 ⚠️
**原因**: 时间步计算方式不对

### 错误方式（之前）
```python
timesteps = [model_sampling.timestep(sigma) for sigma in sigmas]
for i, t in enumerate(timesteps):
    if t / 1000.0 < boundary:
        switching_step = i
```

### 正确方式（WanMoeKSampler）
```python
# 注意：除以 1000，并从 timesteps[1:] 开始检查
timesteps = [sampling.timestep(sigma) / 1000.0 for sigma in sigmas.tolist()]
for i, t in enumerate(timesteps[1:]):
    if t < boundary:
        switching_step = i
        break
```

## 问题 4: CFG 参数不够灵活 ⚠️
**原因**: 高噪和低噪模型使用相同的 CFG
**修复**: 分离为 `cfg_high` 和 `cfg_low`，典型值 5.0 和 4.0

## 关键修复点

### 1. 参数更新
```python
# 之前 ❌
io.Int.Input("steps", min=20, max=100)
io.Float.Input("cfg", default=5.0)

# 现在 ✅
io.Int.Input("steps", min=1, max=10000)  # 支持蒸馏模型
io.Float.Input("cfg_high", default=5.0)
io.Float.Input("cfg_low", default=4.0)
```

### 2. 采样逻辑简化
- ❌ 之前：复杂的多段式采样，每段都要判断
- ✅ 现在：简单的两段式，清晰易懂

### 3. 正确的 disable_noise 和 force_full_denoise
| 阶段 | disable_noise | force_full_denoise | 说明 |
|------|---------------|-------------------|------|
| 高噪（有低噪后续） | True | True | 不添加噪声，不强制去噪 |
| 高噪（无低噪后续） | False | True | 添加噪声，强制去噪 |
| 低噪（最终） | False | True | 不添加噪声，强制去噪 |

### 4. TTM blending 时机
- 在高噪阶段结束时：如果 `tweak <= switching_step <= tstrong`
- 在低噪阶段结束时：如果 `switching_step < tstrong <= steps`

## 测试建议

### 蒸馏模型测试
```
steps: 8
cfg_high: 5.0
cfg_low: 4.0
boundary: 0.875
sigma_shift: 8.0
```

### 标准模型测试
```
steps: 50
cfg_high: 5.0
cfg_low: 4.0
boundary: 0.875
sigma_shift: 8.0
```

### I2V 测试
```
boundary: 0.9  # I2V 推荐值
```

## 参考
- WanMoeKSampler: https://github.com/stduhpf/ComfyUI-WanMoeKSampler
- 关键学习点：
  1. 简单两段式采样
  2. 正确的 disable_noise/force_full_denoise 逻辑
  3. 时间步从 timesteps[1:] 开始检查
  4. 分离 cfg_high 和 cfg_low
