#!/usr/bin/env python3
"""
测试警告控制功能
"""

import os
from data_utils.data_loader import XView2SegmentationDataset

def test_warning_control():
    """测试不同的警告控制选项"""
    print("=== 测试警告控制功能 ===")
    
    data_root = "data/combined_dataset"
    train_images_dir = os.path.join(data_root, "images", "train2017")
    train_masks_dir = os.path.join(data_root, "masks", "train2017")
    
    if not os.path.exists(train_images_dir) or not os.path.exists(train_masks_dir):
        print("数据目录不存在，跳过测试")
        return
    
    # 测试1: 显示警告（默认行为）
    print("\n--- 测试1: 显示警告（默认行为） ---")
    dataset1 = XView2SegmentationDataset(
        train_images_dir, 
        train_masks_dir, 
        show_warnings=True,
        skip_problematic_samples=False
    )
    print(f"数据集大小: {len(dataset1)}")
    
    # 测试2: 隐藏警告
    print("\n--- 测试2: 隐藏警告 ---")
    dataset2 = XView2SegmentationDataset(
        train_images_dir, 
        train_masks_dir, 
        show_warnings=False,
        skip_problematic_samples=False
    )
    print(f"数据集大小: {len(dataset2)}")
    
    # 测试3: 跳过有问题的样本
    print("\n--- 测试3: 跳过有问题的样本 ---")
    dataset3 = XView2SegmentationDataset(
        train_images_dir, 
        train_masks_dir, 
        show_warnings=True,
        skip_problematic_samples=True
    )
    print(f"数据集大小: {len(dataset3)}")
    
    # 测试4: 隐藏警告并跳过有问题的样本
    print("\n--- 测试4: 隐藏警告并跳过有问题的样本 ---")
    dataset4 = XView2SegmentationDataset(
        train_images_dir, 
        train_masks_dir, 
        show_warnings=False,
        skip_problematic_samples=True
    )
    print(f"数据集大小: {len(dataset4)}")
    
    print(f"\n=== 总结 ===")
    print(f"原始数据集大小: {len(dataset1)}")
    print(f"跳过问题样本后大小: {len(dataset3)}")
    print(f"跳过的样本数量: {len(dataset1) - len(dataset3)}")

if __name__ == "__main__":
    test_warning_control() 