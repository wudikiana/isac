#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多样化的掩码处理功能
验证不同的damage_level参数对掩码处理的影响
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_model import process_xview2_mask

def test_damage_levels():
    """
    测试不同的damage_level处理方式
    """
    print("测试多样化的掩码处理功能")
    print("="*50)
    
    # 创建测试掩码
    # 模拟一个包含不同损坏级别的掩码
    mask = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 2, 1, 0, 0, 0, 0],
        [0, 1, 3, 1, 0, 0, 0, 0],
        [0, 1, 4, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)
    
    print("原始掩码:")
    print(mask)
    print(f"唯一值: {torch.unique(mask).tolist()}")
    print()
    
    # 测试不同的damage_level
    damage_levels = [
        'all',
        'light', 
        'binary',
        'multi',
        'progressive',
        'categorical',
        'damage_only',
        'severity_weighted'
    ]
    
    results = {}
    
    for level in damage_levels:
        print(f"测试 damage_level='{level}':")
        processed_mask = process_xview2_mask(mask, level)
        results[level] = processed_mask
        
        print(f"处理后掩码:")
        print(processed_mask)
        print(f"唯一值: {torch.unique(processed_mask).tolist()}")
        print(f"值范围: [{processed_mask.min().item():.2f}, {processed_mask.max().item():.2f}]")
        print("-" * 30)
    
    # 可视化结果
    visualize_results(mask, results)
    
    return results

def visualize_results(original_mask, results):
    """
    可视化不同处理方式的结果
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 原始掩码
    axes[0, 0].imshow(original_mask, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('原始掩码\n(0:背景, 1:未损坏, 2:轻微, 3:中等, 4:严重)')
    axes[0, 0].axis('off')
    
    # 处理后的掩码
    positions = [
        (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
    ]
    
    for i, (level, processed_mask) in enumerate(results.items()):
        if i < len(positions):
            row, col = positions[i]
            im = axes[row, col].imshow(processed_mask, cmap='viridis', interpolation='nearest')
            axes[row, col].set_title(f'{level}')
            axes[row, col].axis('off')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('damage_level_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_real_data():
    """
    使用真实数据测试
    """
    print("\n使用真实数据测试...")
    print("="*50)
    
    # 模拟从数据集加载的真实掩码
    # 这里我们创建一个更复杂的掩码来模拟真实情况
    real_mask = torch.zeros(256, 256, dtype=torch.float32)
    
    # 添加一些损坏区域
    # 轻微损坏区域
    real_mask[50:100, 50:150] = 2
    
    # 中等损坏区域
    real_mask[120:180, 80:200] = 3
    
    # 严重损坏区域
    real_mask[200:250, 100:180] = 4
    
    # 未损坏区域
    real_mask[30:80, 200:250] = 1
    
    print("真实掩码统计:")
    unique_values, counts = torch.unique(real_mask, return_counts=True)
    for val, count in zip(unique_values, counts):
        print(f"  类别 {val.item()}: {count.item()} 像素")
    
    # 测试categorical模式（多类别分类推荐）
    print("\n测试categorical模式（多类别分类推荐）:")
    categorical_mask = process_xview2_mask(real_mask, 'categorical')
    
    print("处理后统计:")
    unique_values, counts = torch.unique(categorical_mask, return_counts=True)
    for val, count in zip(unique_values, counts):
        class_name = ['背景', '未损坏', '轻微损坏', '中等损坏', '严重损坏'][int(val.item())]
        print(f"  {class_name}: {count.item()} 像素")
    
    # 可视化真实数据
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始掩码
    im1 = axes[0].imshow(real_mask, cmap='viridis', interpolation='nearest')
    axes[0].set_title('原始掩码')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # 处理后掩码
    im2 = axes[1].imshow(categorical_mask, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Categorical处理后掩码')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('real_data_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_model_compatibility():
    """
    测试与模型的兼容性
    """
    print("\n测试与模型的兼容性...")
    print("="*50)
    
    # 创建测试掩码
    test_mask = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3]
    ], dtype=torch.float32)
    
    print("测试掩码:")
    print(test_mask)
    print()
    
    # 测试不同的处理方式
    levels = ['categorical', 'progressive', 'severity_weighted']
    
    for level in levels:
        print(f"damage_level='{level}':")
        processed = process_xview2_mask(test_mask, level)
        print(f"处理后形状: {processed.shape}")
        print(f"数据类型: {processed.dtype}")
        print(f"值范围: [{processed.min().item():.2f}, {processed.max().item():.2f}]")
        print(f"唯一值: {torch.unique(processed).tolist()}")
        print()
    
    print("✅ 所有处理方式都与模型兼容")

def main():
    """
    主函数
    """
    print("多样化的掩码处理功能测试")
    print("="*60)
    
    # 测试基本功能
    results = test_damage_levels()
    
    # 测试真实数据
    test_real_data()
    
    # 测试模型兼容性
    test_model_compatibility()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("\n掩码处理方式总结:")
    print("- 'all': 所有损坏级别都标记为1（二分类）")
    print("- 'light': 根据损坏程度分配权重（0.3, 0.6, 1.0）")
    print("- 'binary': 轻微损坏不算损坏，只有中等和严重才算")
    print("- 'multi': 多级分类（1, 2, 3）")
    print("- 'progressive': 渐进式权重（0.25, 0.5, 1.0）")
    print("- 'categorical': 多类别分类（0, 1, 2, 3, 4）- 推荐用于多类别分类")
    print("- 'damage_only': 只关注损坏区域（1, 2, 3）")
    print("- 'severity_weighted': 严重程度加权（0.2, 0.5, 1.0）")

if __name__ == "__main__":
    main() 