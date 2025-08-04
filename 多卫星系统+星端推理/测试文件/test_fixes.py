#!/usr/bin/env python3
"""
测试修复后的代码功能
"""
import torch
import numpy as np
from train_model import process_xview2_mask, postprocess

def test_mask_processing():
    """测试掩码处理功能"""
    print("="*50)
    print("测试掩码处理功能")
    print("="*50)
    
    # 创建测试掩码
    test_mask = torch.tensor([
        [0, 0, 1, 1],
        [0, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 1, 1, 1]
    ], dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    
    print(f"原始掩码: {test_mask.squeeze()}")
    print(f"掩码形状: {test_mask.shape}")
    
    # 测试不同的掩码处理方式
    damage_levels = ['all', 'light', 'binary', 'multi']
    
    for level in damage_levels:
        processed = process_xview2_mask(test_mask, level)
        print(f"\n{level} 处理结果:")
        print(f"  形状: {processed.shape}")
        print(f"  唯一值: {torch.unique(processed)}")
        print(f"  内容:\n{processed.squeeze()}")

def test_postprocessing():
    """测试后处理功能"""
    print("\n" + "="*50)
    print("测试后处理功能")
    print("="*50)
    
    # 创建测试输出
    test_output = torch.randn(1, 1, 64, 64)  # 模拟模型输出
    print(f"测试输出形状: {test_output.shape}")
    print(f"输出范围: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    # 测试后处理
    try:
        processed = postprocess(test_output, min_area=10, merge_distance=5)
        print(f"后处理成功!")
        print(f"处理后形状: {processed.shape}")
        print(f"处理后范围: [{processed.min():.4f}, {processed.max():.4f}]")
    except Exception as e:
        print(f"后处理失败: {e}")

def test_configuration():
    """测试配置参数"""
    print("\n" + "="*50)
    print("测试配置参数")
    print("="*50)
    
    # 模拟配置参数
    config = {
        'damage_level': 'all',
        'enable_postprocess': True,
        'postprocess_min_area': 100,
        'postprocess_merge_distance': 10,
        'debug_output_frequency': 200,
        'epoch_debug_frequency': 10
    }
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    print("开始测试修复后的代码...")
    
    test_mask_processing()
    test_postprocessing()
    test_configuration()
    
    print("\n" + "="*50)
    print("测试完成!")
    print("="*50) 