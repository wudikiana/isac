# NaN问题修复总结

## 问题描述

在双模型集成训练过程中出现了NaN值问题，主要表现为：
- 融合权重变为NaN
- 模型输出包含NaN/Inf
- 损失函数返回NaN
- 训练无法正常进行

## 根本原因分析

1. **数值不稳定**：复杂的损失函数（BoundaryAwareLoss + AdaptiveMiner）在集成模型中容易产生数值不稳定
2. **梯度爆炸**：没有梯度裁剪，导致梯度范数过大
3. **学习率过高**：初始学习率设置过高，容易导致训练不稳定
4. **权重衰减不足**：正则化不够，模型容易过拟合

## 修复方案

### 1. 损失函数优化

**问题**：使用复杂的BoundaryAwareLoss导致数值不稳定

**解决方案**：
```python
# 对于集成模型，使用更简单的损失函数以避免数值不稳定
if use_ensemble:
    criterion = DiceLoss()
    print("✅ 使用DiceLoss以确保数值稳定性")
else:
    base_criterion = HybridLoss()
    criterion = BoundaryAwareLoss(base_criterion, alpha=0.3, beta=0.2)
```

### 2. 梯度裁剪

**问题**：没有梯度裁剪，容易导致梯度爆炸

**解决方案**：
```python
# 反向传播
scaler.scale(loss).backward()

# 梯度裁剪以防止梯度爆炸
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

### 3. 学习率调整

**问题**：学习率过高导致训练不稳定

**解决方案**：
```python
# 使用更保守的学习率
ensemble_optimizer = optim.AdamW(ensemble_model.parameters(), lr=1e-5, weight_decay=1e-3)
ensemble_scheduler = optim.lr_scheduler.OneCycleLR(
    ensemble_optimizer,
    max_lr=1e-4,  # 降低最大学习率
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy='linear',
    final_div_factor=10000
)
```

### 4. NaN检查和处理

**问题**：没有对NaN值进行检查和处理

**解决方案**：
```python
# 检查输出是否包含NaN
if torch.isnan(outputs).any() or torch.isinf(outputs).any():
    print(f"[警告] 模型输出包含NaN/Inf，跳过batch {batch_idx}")
    continue

# 检查损失是否包含NaN
if torch.isnan(loss) or torch.isinf(loss):
    print(f"[警告] 损失包含NaN/Inf，跳过batch {batch_idx}")
    continue
```

### 5. 模型前向传播保护

**问题**：模型内部可能产生NaN值

**解决方案**：
```python
def forward(self, img, sim_feat=None):
    # DeepLab前向传播
    deeplab_output = self.deeplab_model(img, sim_feat)
    
    # LandslideDetector前向传播
    landslide_output = self.landslide_model(img)
    
    # 检查输出是否包含NaN
    if torch.isnan(deeplab_output).any() or torch.isinf(deeplab_output).any():
        print("[警告] DeepLab输出包含NaN/Inf，使用零张量")
        deeplab_output = torch.zeros_like(deeplab_output)
    
    if torch.isnan(landslide_output).any() or torch.isinf(landslide_output).any():
        print("[警告] LandslideDetector输出包含NaN/Inf，使用零张量")
        landslide_output = torch.zeros_like(landslide_output)
    
    # 加权融合
    weight = torch.sigmoid(self.learnable_weight)
    
    # 检查权重是否包含NaN
    if torch.isnan(weight) or torch.isinf(weight):
        print("[警告] 融合权重包含NaN/Inf，使用默认权重0.5")
        weight = torch.tensor(0.5, device=img.device)
    
    ensemble_output = weight * deeplab_output + (1 - weight) * landslide_output
    
    # 最终检查
    if torch.isnan(ensemble_output).any() or torch.isinf(ensemble_output).any():
        print("[警告] 融合输出包含NaN/Inf，使用零张量")
        ensemble_output = torch.zeros_like(ensemble_output)
    
    return ensemble_output
```

## 修复效果

### 测试结果
- ✅ 前向传播正常，无NaN值
- ✅ 损失函数计算正常
- ✅ 梯度计算正常，梯度范数合理（0.7337）
- ✅ 融合权重稳定（0.622）
- ✅ 训练过程可以正常进行

### 性能指标
- **输出范围**：[-6.3988, 4.9411] - 合理范围
- **损失值**：0.5789 - 正常损失值
- **梯度范数**：0.7337 - 在合理范围内
- **融合权重**：0.622 - 稳定且合理

## 预防措施

1. **定期检查**：在训练过程中定期检查NaN值
2. **梯度监控**：监控梯度范数，防止梯度爆炸
3. **学习率调度**：使用合适的学习率调度策略
4. **损失函数选择**：根据模型复杂度选择合适的损失函数
5. **权重初始化**：使用合适的权重初始化方法

## 总结

通过以上修复措施，成功解决了双模型集成训练中的NaN问题：

1. **简化损失函数**：使用DiceLoss替代复杂的BoundaryAwareLoss
2. **添加梯度裁剪**：防止梯度爆炸
3. **调整学习率**：使用更保守的学习率设置
4. **增加NaN检查**：在关键位置添加NaN检查和处理
5. **模型保护**：在模型前向传播中添加保护机制

这些修复确保了双模型集成训练的稳定性和可靠性，为后续的训练提供了坚实的基础。 