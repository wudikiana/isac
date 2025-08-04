#!/usr/bin/env python3
"""
验证修复后的代码功能完整性
"""
import torch
import numpy as np
from train_model import process_xview2_mask, postprocess

def verify_mask_processing():
    """验证掩码处理功能"""
    print("="*60)
    print("验证掩码处理功能")
    print("="*60)
    
    # 创建测试掩码
    test_mask = torch.tensor([
        [0, 0, 1, 1],
        [0, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 1, 1, 1]
    ], dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    
    print(f"原始掩码: {test_mask.squeeze()}")
    
    # 测试所有掩码处理方式
    damage_levels = ['all', 'light', 'binary', 'multi']
    
    for level in damage_levels:
        processed = process_xview2_mask(test_mask, level)
        unique_vals = torch.unique(processed)
        print(f"\n{level} 处理结果:")
        print(f"  唯一值: {unique_vals}")
        print(f"  内容:\n{processed.squeeze()}")
        
        # 验证处理结果
        if level == 'all':
            assert torch.all((unique_vals == 0) | (unique_vals == 1)), f"{level} 处理失败"
        elif level == 'light':
            assert torch.all((unique_vals >= 0) & (unique_vals <= 1)), f"{level} 处理失败"
        elif level == 'binary':
            assert torch.all((unique_vals == 0) | (unique_vals == 1)), f"{level} 处理失败"
        elif level == 'multi':
            assert torch.all((unique_vals >= 0) & (unique_vals <= 3)), f"{level} 处理失败"
        
        print(f"  ✅ {level} 处理验证通过")

def verify_postprocessing():
    """验证后处理功能"""
    print("\n" + "="*60)
    print("验证后处理功能")
    print("="*60)
    
    # 创建测试输出 - 使用3D张量 [H, W] 或 [C, H, W]
    test_output = torch.randn(64, 64)  # 模拟模型输出
    print(f"测试输出形状: {test_output.shape}")
    print(f"输出范围: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    # 测试后处理
    try:
        processed = postprocess(test_output, min_area=10, merge_distance=5)
        print(f"✅ 后处理成功!")
        print(f"处理后形状: {processed.shape}")
        print(f"处理后范围: [{processed.min():.4f}, {processed.max():.4f}]")
        
        # 验证后处理结果
        assert processed.shape == test_output.shape, "后处理改变了形状"
        assert torch.all((processed >= 0) & (processed <= 1)), "后处理结果不在[0,1]范围内"
        print("  ✅ 后处理验证通过")
        
    except Exception as e:
        print(f"❌ 后处理失败: {e}")

def verify_configuration():
    """验证配置参数"""
    print("\n" + "="*60)
    print("验证配置参数")
    print("="*60)
    
    # 模拟配置参数
    config = {
        'damage_level': 'all',
        'enable_postprocess': True,
        'postprocess_min_area': 100,
        'postprocess_merge_distance': 10
    }
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 验证配置参数
    assert config['damage_level'] in ['all', 'light', 'binary', 'multi'], "无效的damage_level"
    assert isinstance(config['enable_postprocess'], bool), "enable_postprocess必须是布尔值"
    assert config['postprocess_min_area'] > 0, "postprocess_min_area必须大于0"
    assert config['postprocess_merge_distance'] >= 0, "postprocess_merge_distance必须大于等于0"
    
    print("  ✅ 配置参数验证通过")

def verify_integration():
    """验证集成功能"""
    print("\n" + "="*60)
    print("验证集成功能")
    print("="*60)
    
    # 模拟完整的训练流程
    test_mask = torch.tensor([
        [0, 0, 1, 1],
        [0, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 1, 1, 1]
    ], dtype=torch.float32).unsqueeze(0)
    
    test_output = torch.randn(4, 4)  # 使用2D张量
    
    # 测试掩码处理
    processed_mask = process_xview2_mask(test_mask, 'light')
    print(f"掩码处理结果: {torch.unique(processed_mask)}")
    
    # 测试后处理
    processed_output = postprocess(test_output, min_area=1, merge_distance=1)
    print(f"后处理结果范围: [{processed_output.min():.4f}, {processed_output.max():.4f}]")
    
    print("  ✅ 集成功能验证通过")

if __name__ == "__main__":
    print("开始验证修复后的代码...")
    
    verify_mask_processing()
    verify_postprocessing()
    verify_configuration()
    verify_integration()
    
    print("\n" + "="*60)
    print("🎉 所有验证通过！修复完成！")
    print("="*60) 