import torch
import torch.nn as nn
from models.starlite_cnn import create_segmentation_landslide_model
from train_model import EnhancedDeepLab, DualModelEnsemble, DiceLoss

def test_gradient_stability():
    """测试梯度稳定性修复"""
    print("=== 测试梯度稳定性修复 ===")
    
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
    
    # 创建优化器（使用修复后的设置）
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=1e-6, weight_decay=1e-3)
    criterion = DiceLoss()
    
    print("1. 测试正常训练...")
    
    # 正常训练测试
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        x = torch.randn(2, 3, 224, 224).to(device)
        sim_feat = torch.randn(2, 11).to(device)
        masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
        
        ensemble_model.train()
        optimizer.zero_grad()
        
        outputs = ensemble_model(x, sim_feat)
        loss = criterion(outputs, masks)
        
        print(f"  损失: {loss.item():.4f}")
        print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        grad_norm = 0
        has_nan_grad = False
        has_inf_grad = False
        
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
        
        if has_nan_grad or has_inf_grad:
            print("  ❌ 梯度包含NaN/Inf!")
            continue
        
        # 梯度裁剪
        if grad_norm > 10:
            print(f"  ⚠️ 梯度范数过大 ({grad_norm:.2f})，进行梯度裁剪并降低学习率")
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=5.0)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        elif grad_norm > 5:
            print(f"  ⚠️ 梯度范数较大 ({grad_norm:.2f})，进行梯度裁剪")
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 检查权重
        has_nan_weight = False
        has_inf_weight = False
        for param in ensemble_model.parameters():
            if torch.isnan(param).any():
                has_nan_weight = True
            if torch.isinf(param).any():
                has_inf_weight = True
        
        print(f"  权重是否包含NaN: {has_nan_weight}")
        print(f"  权重是否包含Inf: {has_inf_weight}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
    
    print("\n2. 测试极端情况...")
    
    # 创建可能导致大梯度的数据
    x = torch.randn(2, 3, 224, 224).to(device) * 10  # 大值输入
    sim_feat = torch.randn(2, 11).to(device) * 10
    masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
    
    ensemble_model.train()
    optimizer.zero_grad()
    
    outputs = ensemble_model(x, sim_feat)
    loss = criterion(outputs, masks)
    
    print(f"  极端数据损失: {loss.item():.4f}")
    
    loss.backward()
    
    # 检查梯度
    grad_norm = 0
    for param in ensemble_model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** (1. / 2)
    
    print(f"  极端数据梯度范数: {grad_norm:.4f}")
    
    # 应用梯度裁剪
    if grad_norm > 10:
        print(f"  ⚠️ 梯度范数过大，进行裁剪")
        torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=5.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
    
    optimizer.step()
    
    print("✅ 梯度稳定性测试完成")

if __name__ == "__main__":
    test_gradient_stability() 