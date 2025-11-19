# TTM 调试检查清单

## ✅ 代码已备份
- Commit: `34e1e0a`
- 分支: `main`
- 远程: `origin/main`

## 🔧 调试环境

### 网络代理
- HTTP Proxy: `http://127.0.0.1:7897`
- HTTPS Proxy: `http://127.0.0.1:7897`
- Git 已配置代理

### 当前文件
```
ComfyUI-TTM-WAN22/
├── __init__.py                  ✓
├── nodes_ttm.py                 ✓
├── ttm_conditioning.py          ✓ (UI优化)
├── ttm_sampler_complete.py      ✓ (完整实现)
├── ttm_sampler.py               ✓ (Legacy)
├── requirements.txt             ✓
├── test_check.py                ✓
└── USAGE.md                     ✓
```

## 📋 调试步骤

### 1. 节点加载检查
- [ ] 重启 ComfyUI
- [ ] 检查节点是否出现在菜单
  - `conditioning` → **WanTTMConditioning**
  - `sampling` → **WanTTMSamplerComplete**
  - `sampling` → **WanTTMSampler** (legacy)

### 2. 参数界面检查
#### WanTTMConditioning
- [ ] tweak_index: 滑块 0-15，默认 3
- [ ] tstrong_index: 滑块 0-20，默认 7
- [ ] width/height: 合理范围
- [ ] 所有输入连接正常

#### WanTTMSamplerComplete
- [ ] cfg: 滑块 0-15，默认 5.0
- [ ] boundary: 滑块 0.5-1.0，默认 0.875
- [ ] sigma_shift: 滑块 1-20，默认 8.0
- [ ] denoise: 滑块 0-1，默认 1.0
- [ ] steps: 10-150，默认 50

### 3. 基础功能测试
- [ ] WanTTMConditioning 执行
  - 检查 motion latent 编码
  - 检查 mask 处理
  - 检查 conditioning 输出
  
- [ ] WanTTMSamplerComplete 执行
  - 检查 MoE 切换逻辑
  - 检查 TTM blending
  - 查看控制台输出

### 4. 控制台输出检查
期望看到：
```
[TTM-MoE] MoE Switch: 44/50 (boundary=0.875)
[TTM-MoE] Dual-Clock: tweak=3, tstrong=7
[TTM-MoE] Checkpoints: [0, 3, 7, 44, 50]
[TTM-MoE] Segment 0->3: high model, final=False
[TTM-MoE] Segment 3->7: high model, final=False
[TTM-MoE] Applying TTM blend at step 7
[TTM-MoE] Segment 7->44: high model, final=False
[TTM-MoE] Segment 44->50: low model, final=True
```

### 5. 常见问题排查

#### 节点未加载
- 检查 `__init__.py` 导出
- 检查语法错误: `python -m py_compile *.py`
- 查看 ComfyUI 启动日志

#### 参数不显示/异常
- 检查 `io.*.Input` 定义
- 检查 min/max/step 范围
- 检查 tooltip 字符串

#### 执行报错
- 检查输入类型匹配
- 检查 tensor 维度
- 查看完整 traceback

#### MoE 切换异常
- 检查 boundary 计算逻辑
- 确认 timestep 映射正确
- 验证模型切换时机

#### TTM blending 无效果
- 检查 tweak/tstrong 范围
- 确认 motion_latent 存在
- 验证 mask 维度匹配

## 🐛 调试命令

### 快速语法检查
```powershell
cd d:\ComfyUINeo\custom_nodes\ComfyUI-TTM-WAN22
python test_check.py
```

### 查看 Git 状态
```powershell
git status
git log --oneline -5
```

### 测试导入
```powershell
python -c "from ttm_conditioning import WanTTMConditioning; print('OK')"
python -c "from ttm_sampler_complete import WanTTMSamplerComplete; print('OK')"
```

## 📊 测试用例

### 最小测试
- 1 张 start_image
- 1 个简单 motion video (3-5帧)
- 1 个对应 mask
- steps=30 (快速测试)
- cfg=5.0

### 标准测试
- 832x480
- 81 frames
- tweak=3, tstrong=7
- boundary=0.875
- steps=50

### 压力测试
- 1024x576
- 81+ frames
- 复杂运动轨迹
- steps=80

## 🔍 需要关注的点

1. **MoE 切换是否正确**
   - 观察 "MoE Switch" 输出
   - 确认 boundary 时间点

2. **TTM blending 是否生效**
   - 观察 "Applying TTM blend" 输出
   - 检查输出次数 (应该在 tweak~tstrong 区间)

3. **内存使用**
   - 注意显存占用
   - 检查是否有内存泄漏

4. **采样速度**
   - 记录每步耗时
   - 对比标准 KSampler

5. **输出质量**
   - 背景是否稳定
   - 运动是否符合轨迹
   - 是否有明显伪影

## ✅ 成功标准

- [ ] 节点正常加载
- [ ] UI 参数显示正确
- [ ] 能完成完整采样
- [ ] MoE 切换时机正确
- [ ] TTM blending 生效
- [ ] 控制台输出符合预期
- [ ] 没有报错或警告
- [ ] 输出视频质量可接受

## 📝 调试日志

记录测试过程：
- 时间：
- 配置：
- 结果：
- 问题：
- 解决：

---

准备开始调试！
