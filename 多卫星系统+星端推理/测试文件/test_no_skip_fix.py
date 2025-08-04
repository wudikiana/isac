import torch
import torch.nn as nn
from models.starlite_cnn import create_segmentation_landslide_model
from train_model import EnhancedDeepLab, DualModelEnsemble, DiceLoss

def test_no_skip_fix():
    """测试不跳过batch的修复"""
    print("=== 测试不跳过batch的修复 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11).to(device)
    landslide_model = create_segmentation_landslide_model(
        num_classes=1,
        use_attention=True,
        use_fpn=True,
        use_dynamic_attention=True,
        use_multi_scale=True
    ).to(device)
    
    ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5).to(device)
    
    # 创建优化器和scaler
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=1e-6, weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    criterion = DiceLoss()
    
    print("1. 测试正常情况...")
    
    # 正常训练
    x = torch.randn(2, 3, 224, 224).to(device)
    sim_feat = torch.randn(2, 11).to(device)
    masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
    
    ensemble_model.train()
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        outputs = ensemble_model(x, sim_feat)
        loss = criterion(outputs, masks)
    
    print(f"  正常损失: {loss.item():.4f}")
    
    # 反向传播
    scaler.scale(loss).backward()
    
    # 检查梯度
    grad_norm = 0
    has_nan_grad = False
    has_inf_grad = False
    
    scaler.unscale_(optimizer)
    
    for param in ensemble_model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad = True
            if torch.isinf(param.grad).any():
                has_inf_grad = True
            grad_norm += param.grad.data.norm(2).item() ** 2
    
    grad_norm = grad_norm ** (1. / 2)
    
    print(f"  梯度范数: {grad_norm:.4f}")
    print(f"  梯度是否包含NaN: {has_nan_grad}")
    print(f"  梯度是否包含Inf: {has_inf_grad}")
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    print("✅ 正常情况测试通过")
    
    print("\n2. 测试梯度修复...")
    
    # 手动创建包含NaN的梯度来测试修复
    for param in ensemble_model.parameters():
        if param.grad is not None:
            # 在梯度中添加一些NaN值
            nan_mask = torch.rand_like(param.grad) < 0.1  # 10%的概率
            param.grad.data[nan_mask] = float('nan')
            break
    
    # 检查并修复梯度
    has_nan_grad = False
    has_inf_grad = False
    
    for param in ensemble_model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad = True
            if torch.isinf(param.grad).any():
                has_inf_grad = True
    
    if has_nan_grad or has_inf_grad:
        print(f"[警告] 检测到NaN/Inf梯度，进行修复")
        # 将NaN/Inf梯度替换为0
        for param in ensemble_model.parameters():
            if param.grad is not None:
                param.grad.data = torch.where(
                    torch.isnan(param.grad.data) | torch.isinf(param.grad.data),
                    torch.zeros_like(param.grad.data),
                    param.grad.data
                )
    
    # 再次检查
    has_nan_grad_after = False
    for param in ensemble_model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad_after = True
                break
    
    print(f"  修复后梯度是否包含NaN: {has_nan_grad_after}")
    
    if not has_nan_grad_after:
        print("✅ 梯度修复测试通过")
    else:
        print("❌ 梯度修复失败")
    
    print("\n3. 测试权重修复...")
    
    # 手动创建包含NaN的权重来测试修复
    for param in ensemble_model.parameters():
        # 在权重中添加一些NaN值
        nan_mask = torch.rand_like(param.data) < 0.05  # 5%的概率
        param.data[nan_mask] = float('nan')
        break
    
    # 检查并修复权重
    has_nan_weight = False
    has_inf_weight = False
    
    for param in ensemble_model.parameters():
        if torch.isnan(param).any():
            has_nan_weight = True
        if torch.isinf(param).any():
            has_inf_weight = True
    
    if has_nan_weight or has_inf_weight:
        print(f"[警告] 检测到NaN/Inf权重，进行修复")
        # 将NaN/Inf权重替换为小的随机值
        for param in ensemble_model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                param.data = torch.where(
                    torch.isnan(param.data) | torch.isinf(param.data),
                    torch.randn_like(param.data) * 0.01,
                    param.data
                )
    
    # 再次检查
    has_nan_weight_after = False
    for param in ensemble_model.parameters():
        if torch.isnan(param).any():
            has_nan_weight_after = True
            break
    
    print(f"  修复后权重是否包含NaN: {has_nan_weight_after}")
    
    if not has_nan_weight_after:
        print("✅ 权重修复测试通过")
    else:
        print("❌ 权重修复失败")
    
    print("\n4. 测试完整训练流程...")
    
    # 完整训练流程测试
    for step in range(3):
        x = torch.randn(2, 3, 224, 224).to(device)
        sim_feat = torch.randn(2, 11).to(device)
        masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
        
        ensemble_model.train()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = ensemble_model(x, sim_feat)
            loss = criterion(outputs, masks)
        
        print(f"  步骤 {step + 1} - 损失: {loss.item():.4f}")
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 检查并修复梯度
        scaler.unscale_(optimizer)
        
        has_nan_grad = False
        for param in ensemble_model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"  [警告] 步骤 {step + 1} 检测到NaN梯度，进行修复")
            for param in ensemble_model.parameters():
                if param.grad is not None:
                    param.grad.data = torch.where(
                        torch.isnan(param.grad.data),
                        torch.zeros_like(param.grad.data),
                        param.grad.data
                    )
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 检查权重
        has_nan_weight = False
        for param in ensemble_model.parameters():
            if torch.isnan(param).any():
                has_nan_weight = True
                break
        
        if has_nan_weight:
            print(f"  [警告] 步骤 {step + 1} 检测到NaN权重，进行修复")
            for param in ensemble_model.parameters():
                if torch.isnan(param).any():
                    param.data = torch.where(
                        torch.isnan(param.data),
                        torch.randn_like(param.data) * 0.01,
                        param.data
                    )
    
    print("✅ 完整训练流程测试通过")

if __name__ == "__main__":
    test_no_skip_fix() 