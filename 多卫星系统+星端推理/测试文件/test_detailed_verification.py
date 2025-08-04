#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细验证测试 - 检查样本是否真的被处理
"""

import torch
import numpy as np
import sys
import os
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入修复后的函数
from train_model import postprocess, simple_postprocess

def test_sample_processing_verification():
    """详细验证样本处理过程"""
    print("=== 详细验证样本处理过程 ===")
    
    # 创建测试数据
    batch_size = 4
    height, width = 64, 64
    
    # 创建不同特征的测试数据
    test_cases = [
        ("随机数据", torch.randn(batch_size, 1, height, width)),
        ("全零数据", torch.zeros(batch_size, 1, height, width)),
        ("全一数据", torch.ones(batch_size, 1, height, width)),
        ("低方差数据", torch.randn(batch_size, 1, height, width) * 0.01 + 0.5),
        ("高方差数据", torch.randn(batch_size, 1, height, width) * 3.0),
    ]
    
    for case_name, batch_data in test_cases:
        print(f"\n--- 测试用例: {case_name} ---")
        
        # 检查原始数据特征
        print(f"原始数据范围: [{batch_data.min():.4f}, {batch_data.max():.4f}]")
        print(f"原始数据均值: {batch_data.mean():.4f}")
        print(f"原始数据标准差: {batch_data.std():.4f}")
        
        # 测试简化后处理
        print("\n简化后处理结果:")
        simple_results = []
        simple_times = []
        
        for i in range(batch_size):
            start_time = time.time()
            result = simple_postprocess(batch_data[i])
            end_time = time.time()
            
            simple_results.append(result)
            simple_times.append(end_time - start_time)
            
            # 检查每个样本的处理结果
            prob = torch.sigmoid(batch_data[i]).cpu().numpy()
            if prob.ndim > 2:
                prob = prob.squeeze()
            
            print(f"  样本 {i}:")
            print(f"    概率范围: [{prob.min():.4f}, {prob.max():.4f}]")
            print(f"    处理时间: {simple_times[i]*1000:.4f}毫秒")
            print(f"    结果形状: {result.shape}")
            print(f"    结果范围: [{result.min():.4f}, {result.max():.4f}]")
            print(f"    结果均值: {result.mean():.4f}")
            print(f"    非零像素数: {(result > 0).sum().item()}")
        
        # 统计信息
        total_time = sum(simple_times)
        avg_time = total_time / batch_size
        print(f"\n  总处理时间: {total_time*1000:.4f}毫秒")
        print(f"  平均处理时间: {avg_time*1000:.4f}毫秒")
        
        # 检查是否有样本被过滤
        all_zero_count = sum(1 for r in simple_results if r.sum() == 0)
        all_one_count = sum(1 for r in simple_results if r.sum() == r.numel())
        
        print(f"  全零结果样本数: {all_zero_count}")
        print(f"  全一结果样本数: {all_one_count}")
        
        if all_zero_count == batch_size:
            print("  ⚠️  警告: 所有样本都被过滤为全零!")
        elif all_one_count == batch_size:
            print("  ⚠️  警告: 所有样本都被过滤为全一!")
        else:
            print("  ✓ 样本处理正常")

if __name__ == "__main__":
    test_sample_processing_verification()
    print("\n=== 所有测试完成 ===") 