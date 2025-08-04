import torch
import torch.nn as nn
import torch.nn.functional as F
from models.starlite_cnn import create_segmentation_landslide_model
from train_model import EnhancedDeepLab, DualModelEnsemble, DiceLoss
import numpy as np

def debug_training_nan():
    """诊断训练过程中的NaN问题"""
    print("=== 诊断训练过程中的NaN问题 ===")
    
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
    
    # 创建优化器
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=1e-5, weight_decay=1e-3)
    
    # 创建损失函数
    criterion = DiceLoss()
    
    print("1. 模拟正常训练...")
    
    # 正常训练
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        # 创建正常数据
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        sim_feat = torch.randn(batch_size, 11).to(device)
        masks = torch.randint(0, 2, (batch_size, 1, 224, 224)).float().to(device)
        
        ensemble_model.train()
        optimizer.zero_grad()
        
        outputs = ensemble_model(x, sim_feat)
        loss = criterion(outputs, masks)
        
        print(f"  正常训练 - 损失: {loss.item():.4f}")
        print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        print(f"  输出是否包含NaN: {torch.isnan(outputs).any()}")
        
        loss.backward()
        optimizer.step()
    
    print("\n2. 模拟极端数据情况...")
    
    # 测试极端数据
    extreme_cases = [
        ("全零输入", torch.zeros(2, 3, 224, 224).to(device)),
        ("全一输入", torch.ones(2, 3, 224, 224).to(device)),
        ("极大值输入", torch.full((2, 3, 224, 224), 1000.0).to(device)),
        ("极小值输入", torch.full((2, 3, 224, 224), -1000.0).to(device)),
        ("包含NaN的输入", torch.tensor([[[[float('nan')]]]], device=device).expand(2, 3, 224, 224)),
        ("包含Inf的输入", torch.tensor([[[[float('inf')]]]], device=device).expand(2, 3, 224, 224)),
    ]
    
    for case_name, x in extreme_cases:
        print(f"\n测试: {case_name}")
        
        sim_feat = torch.randn(2, 11).to(device)
        masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
        
        try:
            ensemble_model.eval()
            with torch.no_grad():
                outputs = ensemble_model(x, sim_feat)
                print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"  输出是否包含NaN: {torch.isnan(outputs).any()}")
                print(f"  输出是否包含Inf: {torch.isinf(outputs).any()}")
        except Exception as e:
            print(f"  ❌ 前向传播失败: {e}")
    
    print("\n3. 模拟梯度爆炸情况...")
    
    # 测试梯度爆炸
    x = torch.randn(2, 3, 224, 224).to(device)
    sim_feat = torch.randn(2, 11).to(device)
    masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
    
    ensemble_model.train()
    optimizer.zero_grad()
    
    outputs = ensemble_model(x, sim_feat)
    loss = criterion(outputs, masks)
    
    # 手动设置极大梯度
    loss.backward()
    
    # 检查梯度
    max_grad = 0
    for name, param in ensemble_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
            if grad_norm > 100:
                print(f"  ⚠️ 大梯度: {name} = {grad_norm:.2f}")
    
    print(f"  最大梯度范数: {max_grad:.2f}")
    
    # 尝试梯度裁剪
    torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
    
    # 再次检查梯度
    max_grad_after = 0
    for name, param in ensemble_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_grad_after = max(max_grad_after, grad_norm)
    
    print(f"  裁剪后最大梯度范数: {max_grad_after:.2f}")
    
    print("\n4. 模拟学习率过高的情况...")
    
    # 使用高学习率
    high_lr_optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=1e-2, weight_decay=1e-3)
    
    for step in range(5):
        x = torch.randn(2, 3, 224, 224).to(device)
        sim_feat = torch.randn(2, 11).to(device)
        masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
        
        ensemble_model.train()
        high_lr_optimizer.zero_grad()
        
        outputs = ensemble_model(x, sim_feat)
        loss = criterion(outputs, masks)
        
        print(f"  步骤 {step + 1} - 损失: {loss.item():.4f}")
        print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ❌ 损失变为NaN/Inf!")
            break
        
        loss.backward()
        high_lr_optimizer.step()
    
    print("\n5. 检查模型权重变化...")
    
    # 检查权重是否变为NaN
    nan_weight_count = 0
    inf_weight_count = 0
    
    for name, param in ensemble_model.named_parameters():
        if torch.isnan(param).any():
            print(f"  ❌ 权重包含NaN: {name}")
            nan_weight_count += 1
        if torch.isinf(param).any():
            print(f"  ❌ 权重包含Inf: {name}")
            inf_weight_count += 1
    
    print(f"  包含NaN的权重数量: {nan_weight_count}")
    print(f"  包含Inf的权重数量: {inf_weight_count}")
    
    print("\n6. 模拟数据加载问题...")
    
    # 测试数据加载中的问题
    problematic_data = [
        ("空掩码", torch.zeros(2, 1, 224, 224).to(device)),
        ("全一掩码", torch.ones(2, 1, 224, 224).to(device)),
        ("包含NaN的掩码", torch.tensor([[[[float('nan')]]]], device=device).expand(2, 1, 224, 224)),
    ]
    
    x = torch.randn(2, 3, 224, 224).to(device)
    sim_feat = torch.randn(2, 11).to(device)
    
    for mask_name, masks in problematic_data:
        print(f"\n测试掩码: {mask_name}")
        
        try:
            ensemble_model.eval()
            with torch.no_grad():
                outputs = ensemble_model(x, sim_feat)
                loss = criterion(outputs, masks)
                print(f"  损失: {loss.item():.4f}")
                print(f"  损失是否包含NaN: {torch.isnan(loss)}")
                print(f"  损失是否包含Inf: {torch.isinf(loss)}")
        except Exception as e:
            print(f"  ❌ 计算失败: {e}")

if __name__ == "__main__":
    debug_training_nan() 