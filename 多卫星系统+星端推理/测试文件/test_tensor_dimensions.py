#!/usr/bin/env python3
"""
测试tensor维度问题修复
"""

import torch
import numpy as np
from train_model import get_multimodal_patch_dataloaders, custom_collate_fn

def test_tensor_dimensions():
    """测试tensor维度是否一致"""
    print("="*60)
    print("🔧 测试tensor维度修复")
    print("="*60)
    
    try:
        # 获取数据加载器
        print("正在加载数据...")
        train_loader, val_loader = get_multimodal_patch_dataloaders(
            data_root="D:/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=4,  # 小batch size用于测试
            num_workers=0,  # 单进程便于调试
            damage_boost=1,  # 减少数据量
            normal_ratio=0.05,
            preload_to_memory=False,
            preload_to_gpu=False
        )
        
        print("✅ 数据加载器创建成功")
        
        # 测试第一个batch
        print("\n测试第一个batch...")
        for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  图像形状: {images.shape}")
            print(f"  掩码形状: {masks.shape}")
            print(f"  特征形状: {sim_feats.shape}")
            
            # 检查维度一致性
            assert images.dim() == 4, f"图像应该是4D tensor，实际是{images.dim()}D"
            assert masks.dim() == 4, f"掩码应该是4D tensor，实际是{masks.dim()}D"
            assert sim_feats.dim() == 2, f"特征应该是2D tensor，实际是{sim_feats.dim()}D"
            
            print(f"  ✅ 维度检查通过")
            
            # 只测试第一个batch
            break
        
        print("\n✅ 所有测试通过！tensor维度问题已修复")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_custom_collate():
    """测试自定义collate函数"""
    print("\n" + "="*60)
    print("🔧 测试自定义collate函数")
    print("="*60)
    
    # 创建测试数据
    test_batch = []
    for i in range(4):
        # 创建不同维度的测试数据
        if i % 2 == 0:
            img = torch.randn(3, 64, 64)  # 正常3D
            mask = torch.randn(1, 64, 64)  # 正常3D
        else:
            img = torch.randn(64, 64, 3)  # 需要转置
            mask = torch.randn(64, 64)    # 需要增加维度
        sim_feat = torch.randn(11)
        
        test_batch.append((img, mask, sim_feat))
    
    print("原始数据形状:")
    for i, (img, mask, sim_feat) in enumerate(test_batch):
        print(f"  样本{i}: img={img.shape}, mask={mask.shape}, sim_feat={sim_feat.shape}")
    
    # 测试collate函数
    try:
        images, masks, sim_feats = custom_collate_fn(test_batch)
        print(f"\nCollate后形状:")
        print(f"  图像: {images.shape}")
        print(f"  掩码: {masks.shape}")
        print(f"  特征: {sim_feats.shape}")
        print("✅ 自定义collate函数测试通过")
    except Exception as e:
        print(f"❌ Collate函数测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_collate()
    test_tensor_dimensions() 