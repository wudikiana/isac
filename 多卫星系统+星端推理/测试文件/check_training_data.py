import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import get_multimodal_patch_dataloaders, DualModelEnsemble, DiceLoss
from models.starlite_cnn import create_segmentation_landslide_model
import numpy as np

def check_training_data():
    """检查训练数据中可能导致NaN的问题"""
    print("=== 检查训练数据 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据加载器
    try:
        train_loader, val_loader = get_multimodal_patch_dataloaders(
            data_root="data/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=4,
            num_workers=0,  # 减少worker数量以便调试
            damage_boost=5,
            normal_ratio=0.05
        )
        print("✅ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return
    
    # 创建模型
    try:
        from train_model import EnhancedDeepLab
        deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11).to(device)
        landslide_model = create_segmentation_landslide_model(
            num_classes=1,
            use_attention=True,
            use_fpn=True,
            use_dynamic_attention=True,
            use_multi_scale=True
        ).to(device)
        
        ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5).to(device)
        print("✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 创建损失函数和优化器
    criterion = DiceLoss()
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=1e-5, weight_decay=1e-3)
    
    print("\n1. 检查训练数据...")
    
    nan_batch_count = 0
    inf_batch_count = 0
    total_batches = 0
    
    for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
        if batch_idx >= 10:  # 只检查前10个batch
            break
            
        total_batches += 1
        
        # 检查数据
        if torch.isnan(images).any():
            print(f"❌ Batch {batch_idx}: 图像包含NaN")
            nan_batch_count += 1
        if torch.isinf(images).any():
            print(f"❌ Batch {batch_idx}: 图像包含Inf")
            inf_batch_count += 1
            
        if torch.isnan(masks).any():
            print(f"❌ Batch {batch_idx}: 掩码包含NaN")
            nan_batch_count += 1
        if torch.isinf(masks).any():
            print(f"❌ Batch {batch_idx}: 掩码包含Inf")
            inf_batch_count += 1
            
        if torch.isnan(sim_feats).any():
            print(f"❌ Batch {batch_idx}: 仿真特征包含NaN")
            nan_batch_count += 1
        if torch.isinf(sim_feats).any():
            print(f"❌ Batch {batch_idx}: 仿真特征包含Inf")
            inf_batch_count += 1
        
        # 检查数据范围
        print(f"Batch {batch_idx}:")
        print(f"  图像范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
        print(f"  掩码范围: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
        print(f"  仿真特征范围: [{sim_feats.min().item():.4f}, {sim_feats.max().item():.4f}]")
        
        # 移动到设备
        images = images.to(device)
        masks = masks.to(device)
        sim_feats = sim_feats.to(device)
        
        # 尝试前向传播
        try:
            ensemble_model.train()
            optimizer.zero_grad()
            
            outputs = ensemble_model(images, sim_feats)
            loss = criterion(outputs, masks)
            
            print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"  损失: {loss.item():.4f}")
            print(f"  输出是否包含NaN: {torch.isnan(outputs).any()}")
            print(f"  损失是否包含NaN: {torch.isnan(loss)}")
            
            if torch.isnan(outputs).any() or torch.isnan(loss):
                print(f"  ❌ Batch {batch_idx} 产生NaN!")
                continue
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            grad_norm = 0
            for param in ensemble_model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** (1. / 2)
            
            print(f"  梯度范数: {grad_norm:.4f}")
            
            if grad_norm > 10:
                print(f"  ⚠️ 大梯度: {grad_norm:.2f}")
            
            optimizer.step()
            
        except Exception as e:
            print(f"  ❌ Batch {batch_idx} 训练失败: {e}")
    
    print(f"\n统计结果:")
    print(f"  总batch数: {total_batches}")
    print(f"  包含NaN的batch数: {nan_batch_count}")
    print(f"  包含Inf的batch数: {inf_batch_count}")
    
    print("\n2. 检查验证数据...")
    
    val_nan_count = 0
    val_inf_count = 0
    val_total = 0
    
    for batch_idx, (images, masks, sim_feats) in enumerate(val_loader):
        if batch_idx >= 5:  # 只检查前5个batch
            break
            
        val_total += 1
        
        # 检查数据
        if torch.isnan(images).any() or torch.isnan(masks).any() or torch.isnan(sim_feats).any():
            val_nan_count += 1
        if torch.isinf(images).any() or torch.isinf(masks).any() or torch.isinf(sim_feats).any():
            val_inf_count += 1
    
    print(f"验证数据统计:")
    print(f"  总batch数: {val_total}")
    print(f"  包含NaN的batch数: {val_nan_count}")
    print(f"  包含Inf的batch数: {val_inf_count}")

if __name__ == "__main__":
    check_training_data() 