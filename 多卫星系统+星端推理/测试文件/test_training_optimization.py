#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练优化效果
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

def test_performance_comparison():
    """测试后处理函数的性能对比"""
    print("开始性能对比测试...")
    
    # 创建测试数据
    batch_size = 8
    height, width = 256, 256
    
    # 模拟模型输出 - 不同概率分布
    outputs = []
    
    # 正常分布
    outputs.append(torch.randn(batch_size, 1, height, width))
    
    # 低方差分布（模拟概率范围过小的情况）
    outputs.append(torch.randn(batch_size, 1, height, width) * 0.1 + 0.5)
    
    # 高方差分布
    outputs.append(torch.randn(batch_size, 1, height, width) * 2.0)
    
    test_cases = ["正常分布", "低方差分布", "高方差分布"]
    
    for i, (output, case_name) in enumerate(zip(outputs, test_cases)):
        print(f"\n测试用例 {i+1}: {case_name}")
        
        # 测试完整后处理函数
        start_time = time.time()
        results_full = []
        for j in range(batch_size):
            result = postprocess(output[j], debug_mode=False)
            results_full.append(result)
        full_time = time.time() - start_time
        
        # 测试简化后处理函数
        start_time = time.time()
        results_simple = []
        for j in range(batch_size):
            result = simple_postprocess(output[j])
            results_simple.append(result)
        simple_time = time.time() - start_time
        
        print(f"  完整后处理: {full_time:.4f}秒")
        print(f"  简化后处理: {simple_time:.4f}秒")
        if simple_time > 0:
            print(f"  性能提升: {full_time/simple_time:.2f}倍")
        else:
            print(f"  性能提升: 无法计算（简化处理时间过短）")
        
        # 检查结果一致性
        results_match = True
        for full, simple in zip(results_full, results_simple):
            if not torch.allclose(full, simple, atol=1e-3):
                results_match = False
                break
        
        print(f"  结果一致性: {'✓' if results_match else '✗'}")

def test_probability_range_handling():
    """测试概率范围处理"""
    print("\n\n测试概率范围处理...")
    
    # 测试极低概率范围的情况
    low_range_output = torch.randn(1, 1, 64, 64) * 0.001 + 0.5  # 范围约0.002
    
    print("测试低概率范围输入...")
    try:
        # 不显示警告
        result1 = postprocess(low_range_output, debug_mode=False)
        print("✓ 静默模式处理成功")
        
        # 显示警告
        result2 = postprocess(low_range_output, debug_mode=True)
        print("✓ 调试模式处理成功")
        
    except Exception as e:
        print(f"✗ 处理失败: {e}")

def test_batch_processing():
    """测试批处理性能"""
    print("\n\n测试批处理性能...")
    
    # 模拟一个batch的数据
    batch_size = 16
    height, width = 128, 128
    batch_output = torch.randn(batch_size, 1, height, width)
    
    # 测试批处理时间
    start_time = time.time()
    results = []
    for i in range(batch_size):
        result = simple_postprocess(batch_output[i])
        results.append(result)
    batch_time = time.time() - start_time
    
    print(f"批处理 {batch_size} 个样本耗时: {batch_time:.4f}秒")
    print(f"平均每个样本: {batch_time/batch_size*1000:.2f}毫秒")
    
    # 检查结果形状
    for i, result in enumerate(results):
        expected_shape = (height, width)
        if result.shape != expected_shape:
            print(f"✗ 样本 {i} 形状错误: {result.shape} != {expected_shape}")
            break
    else:
        print("✓ 所有样本形状正确")

if __name__ == "__main__":
    test_performance_comparison()
    test_probability_range_handling()
    test_batch_processing()
    print("\n所有测试完成！") 