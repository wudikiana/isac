import torch
import torch.nn as nn
from models.starlite_cnn import create_segmentation_landslide_model
from train_model import EnhancedDeepLab, DualModelEnsemble, DiceLoss

def test_nan_fix():
    """测试NaN修复"""
    print("=== 测试NaN修复 ===")
    
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
    
    # 创建测试数据
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    sim_feat = torch.randn(batch_size, 11).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 224, 224)).float().to(device)
    
    # 创建损失函数
    criterion = DiceLoss()
    
    # 创建优化器
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=1e-5, weight_decay=1e-3)
    
    print("开始前向传播测试...")
    
    # 前向传播
    with torch.no_grad():
        outputs = ensemble_model(x, sim_feat)
        print(f"输出形状: {outputs.shape}")
        print(f"输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        print(f"输出是否包含NaN: {torch.isnan(outputs).any()}")
        print(f"输出是否包含Inf: {torch.isinf(outputs).any()}")
    
    print("\n开始训练测试...")
    
    # 训练测试
    ensemble_model.train()
    optimizer.zero_grad()
    
    outputs = ensemble_model(x, sim_feat)
    loss = criterion(outputs, masks)
    
    print(f"损失值: {loss.item():.4f}")
    print(f"损失是否包含NaN: {torch.isnan(loss)}")
    print(f"损失是否包含Inf: {torch.isinf(loss)}")
    
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        for p in ensemble_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        print(f"梯度范数: {total_norm:.4f}")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        print("✅ 训练测试通过")
    else:
        print("❌ 损失包含NaN/Inf")
    
    print(f"融合权重: {ensemble_model.get_fusion_weight():.3f}")

if __name__ == "__main__":
    test_nan_fix() 