#!/usr/bin/env python3
"""
简单测试选项1：隐藏警告但保留所有数据
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_option1():
    """测试选项1：隐藏警告但保留所有数据"""
    print("=== 测试选项1：隐藏警告但保留所有数据 ===")
    
    try:
        # 导入数据加载器
        from data_utils.data_loader import get_segmentation_dataloaders
        
        print("✓ 成功导入数据加载器")
        
        # 测试选项1：隐藏警告但保留所有数据
        print("\n--- 应用选项1 ---")
        train_loader, val_loader, test_loader = get_segmentation_dataloaders(
            data_root="data/combined_dataset",
            batch_size=2,
            num_workers=0,  # 使用0避免多进程问题
            show_warnings=False,  # 隐藏警告
            skip_problematic_samples=False  # 保留所有数据
        )
        
        print(f"✓ 成功创建数据加载器")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")
        
        # 测试加载一个批次
        print("\n--- 测试数据加载 ---")
        batch = next(iter(train_loader))
        if len(batch) == 4:  # 图像, 掩码, 仿真特征, 字符串特征
            images, masks, sim_feats, str_feats = batch
            print(f"✓ 批次图像形状: {images.shape}")
            print(f"✓ 批次掩码形状: {masks.shape}")
            print(f"✓ 批次仿真特征形状: {sim_feats.shape}")
            print("✓ 数据加载成功！")
        else:
            print(f"✗ 批次格式不正确，期望4个元素，实际{len(batch)}个")
        
        print("\n=== 测试完成 ===")
        print("✓ 选项1已成功应用：隐藏了警告信息但保留了所有数据")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_option1() 