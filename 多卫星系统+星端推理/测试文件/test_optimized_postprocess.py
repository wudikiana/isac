#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的后处理函数
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

def test_adaptive_threshold():
    """测试自适应阈值功能"""
    print("=== 测试自适应阈值功能 ===")
    
    # 创建低方差但高值的测试数据
    test_cases = [
        ("低方差高值", torch.randn(1, 1, 32, 32) * 0.01 + 0.6),  # 范围约[0.59, 0.61]
        ("低方差低值", torch.randn(1, 1, 32, 32) * 0.01 + 0.4),  # 范围约[0.39, 0.41]
        ("正常方差", torch.randn(1, 1, 32, 32)),  # 正常分布
    ]
    
    for case_name, data in test_cases:
        print(f"\n--- {case_name} ---")
        
        # 检查原始数据
        prob = torch.sigmoid(data).cpu().numpy().squeeze()
        print(f"概率范围: [{prob.min():.4f}, {prob.max():.4f}]")
        print(f"概率均值: {prob.mean():.4f}")
        
        # 测试固定阈值 vs 自适应阈值
        print("\n固定阈值 (0.5):")
        fixed_result = simple_postprocess(data, threshold=0.5, adaptive=False)
        fixed_binary = fixed_result.numpy()
        print(f"  结果均值: {fixed_binary.mean():.4f}")
        print(f"  非零像素数: {fixed_binary.sum()}")
        
        print("\n自适应阈值:")
        adaptive_result = simple_postprocess(data, threshold=0.5, adaptive=True)
        adaptive_binary = adaptive_result.numpy()
        print(f"  结果均值: {adaptive_binary.mean():.4f}")
        print(f"  非零像素数: {adaptive_binary.sum()}")
        
        # 比较结果
        if np.array_equal(fixed_binary, adaptive_binary):
            print("  ✓ 结果相同")
        else:
            print("  ✗ 结果不同")
            print(f"  差异像素数: {np.sum(fixed_binary != adaptive_binary)}")

def test_conservative_filtering():
    """测试保守过滤功能"""
    print("\n\n=== 测试保守过滤功能 ===")
    
    # 创建高置信度测试数据
    test_cases = [
        ("全高值", torch.ones(1, 1, 32, 32)),  # 全1
        ("高值范围", torch.randn(1, 1, 32, 32) * 0.1 + 0.9),  # 高值范围
        ("混合值", torch.randn(1, 1, 32, 32) * 2.0),  # 混合值
    ]
    
    for case_name, data in test_cases:
        print(f"\n--- {case_name} ---")
        
        # 检查原始数据
        prob = torch.sigmoid(data).cpu().numpy().squeeze()
        print(f"概率范围: [{prob.min():.4f}, {prob.max():.4f}]")
        print(f"概率均值: {prob.mean():.4f}")
        
        # 测试保守过滤
        result = simple_postprocess(data, adaptive=True)
        binary = result.numpy()
        
        print(f"后处理结果:")
        print(f"  结果均值: {binary.mean():.4f}")
        print(f"  非零像素数: {binary.sum()}")
        print(f"  过滤比例: {(1 - binary.mean())*100:.1f}%")

def test_precision_timing():
    """测试高精度计时"""
    print("\n\n=== 测试高精度计时 ===")
    
    # 创建测试数据
    data = torch.randn(1, 1, 64, 64)
    
    # 使用纳秒级精度计时
    start_ns = time.perf_counter_ns()
    result = simple_postprocess(data, adaptive=True)
    elapsed_ns = time.perf_counter_ns() - start_ns
    
    print(f"处理时间: {elapsed_ns / 1e6:.4f}毫秒")
    print(f"处理时间: {elapsed_ns / 1e3:.4f}微秒")
    print(f"处理时间: {elapsed_ns:.0f}纳秒")
    
    # 批量测试
    batch_size = 16
    start_ns = time.perf_counter_ns()
    for i in range(batch_size):
        result = simple_postprocess(data, adaptive=True)
    elapsed_ns = time.perf_counter_ns() - start_ns
    
    print(f"\n批量处理 {batch_size} 个样本:")
    print(f"总时间: {elapsed_ns / 1e6:.4f}毫秒")
    print(f"平均时间: {elapsed_ns / batch_size / 1e6:.4f}毫秒")

def test_warning_system():
    """测试警告系统"""
    print("\n\n=== 测试警告系统 ===")
    
    # 测试全零输出
    print("测试全零输出:")
    data_zero = torch.randn(1, 1, 32, 32) * -10  # 极低值
    result = postprocess(data_zero, debug_mode=True)
    
    # 测试全一输出
    print("\n测试全一输出:")
    data_one = torch.randn(1, 1, 32, 32) * 10  # 极高值
    result = postprocess(data_one, debug_mode=True)
    
    # 测试正常输出
    print("\n测试正常输出:")
    data_normal = torch.randn(1, 1, 32, 32)
    result = postprocess(data_normal, debug_mode=True)

def test_edge_cases():
    """测试边界情况"""
    print("\n\n=== 测试边界情况 ===")
    
    edge_cases = [
        ("极小尺寸", torch.randn(1, 1, 8, 8)),
        ("极大数值", torch.randn(1, 1, 32, 32) * 100),
        ("极小数值", torch.randn(1, 1, 32, 32) * 0.001),
        ("边界概率", torch.randn(1, 1, 32, 32) * 0.01 + 0.5),  # 接近0.5
    ]
    
    for case_name, data in edge_cases:
        print(f"\n--- {case_name} ---")
        try:
            result = simple_postprocess(data, adaptive=True)
            binary = result.numpy()
            print(f"  处理成功: {result.shape}")
            print(f"  结果均值: {binary.mean():.4f}")
            print(f"  非零像素数: {binary.sum()}")
        except Exception as e:
            print(f"  处理失败: {e}")

if __name__ == "__main__":
    test_adaptive_threshold()
    test_conservative_filtering()
    test_precision_timing()
    test_warning_system()
    test_edge_cases()
    print("\n=== 所有测试完成 ===") 