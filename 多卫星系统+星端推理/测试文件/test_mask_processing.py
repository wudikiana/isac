#!/usr/bin/env python3
"""
测试增强的掩码处理功能
"""

import torch
import numpy as np
from train_model import process_xview2_mask
import os

def test_mask_processing():
    """测试掩码处理功能"""
    print("=== 测试增强的掩码处理功能 ===")
    
    # 创建测试掩码（模拟xView2掩码值：0=背景, 1=未损坏, 2=轻微损坏, 3=中等损坏, 4=严重损坏）
    test_mask = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3]
    ], dtype=torch.float32)
    
    print("原始掩码:")
    print(test_mask)
    print(f"原始掩码唯一值: {torch.unique(test_mask).tolist()}")
    
    # 测试不同的处理方式
    damage_levels = ['all', 'light', 'binary', 'multi']
    
    for level in damage_levels:
        print(f"\n--- 测试 damage_level='{level}' ---")
        processed = process_xview2_mask(test_mask, level)
        print(f"处理后掩码:")
        print(processed)
        print(f"处理后唯一值: {torch.unique(processed).tolist()}")
        print(f"损坏像素数量: {torch.sum(processed > 0).item()}")
        
        # 计算统计信息
        if level == 'light':
            light_damage = torch.sum(processed == 0.3).item()
            medium_damage = torch.sum(processed == 0.6).item()
            severe_damage = torch.sum(processed == 1.0).item()
            print(f"轻微损坏像素: {light_damage}")
            print(f"中等损坏像素: {medium_damage}")
            print(f"严重损坏像素: {severe_damage}")
        elif level == 'multi':
            light_damage = torch.sum(processed == 1.0).item()
            medium_damage = torch.sum(processed == 2.0).item()
            severe_damage = torch.sum(processed == 3.0).item()
            print(f"轻微损坏像素: {light_damage}")
            print(f"中等损坏像素: {medium_damage}")
            print(f"严重损坏像素: {severe_damage}")
    
    print("\n=== 测试完成 ===")

def test_dataset_integration():
    """测试数据集集成"""
    print("\n=== 测试数据集集成 ===")
    
    try:
        from data_utils.data_loader import XView2SegmentationDataset
        
        # 检查数据目录
        data_root = "data/combined_dataset"
        train_images_dir = os.path.join(data_root, "images", "train2017")
        train_masks_dir = os.path.join(data_root, "masks", "train2017")
        
        if os.path.exists(train_images_dir) and os.path.exists(train_masks_dir):
            print("创建数据集实例...")
            
            # 测试不同的damage_level
            for damage_level in ['all', 'light', 'binary', 'multi']:
                print(f"\n测试 damage_level='{damage_level}':")
                dataset = XView2SegmentationDataset(
                    train_images_dir, 
                    train_masks_dir, 
                    damage_level=damage_level
                )
                
                if len(dataset) > 0:
                    # 加载第一个样本
                    img, mask, sim_feat, str_feats = dataset[0]
                    unique_values = torch.unique(mask)
                    print(f"  掩码形状: {mask.shape}")
                    print(f"  掩码唯一值: {unique_values.tolist()}")
                    print(f"  损坏像素数: {torch.sum(mask > 0).item()}")
                else:
                    print("  数据集为空")
        else:
            print("数据目录不存在，跳过数据集测试")
            
    except Exception as e:
        print(f"数据集测试失败: {e}")

if __name__ == "__main__":
    test_mask_processing()
    test_dataset_integration() 