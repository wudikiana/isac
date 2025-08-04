#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试概率计算
"""

import torch
import numpy as np

def debug_probability():
    """调试概率计算"""
    print("=== 调试概率计算 ===")
    
    # 创建测试数据
    test_cases = [
        ("低方差高值", torch.randn(1, 1, 32, 32) * 0.01 + 0.6),
        ("低方差低值", torch.randn(1, 1, 32, 32) * 0.01 + 0.4),
    ]
    
    for case_name, data in test_cases:
        print(f"\n--- {case_name} ---")
        
        # 检查原始数据
        print(f"原始数据范围: [{data.min():.4f}, {data.max():.4f}]")
        print(f"原始数据均值: {data.mean():.4f}")
        
        # 检查sigmoid后的概率
        prob = torch.sigmoid(data).cpu().numpy().squeeze()
        print(f"Sigmoid后概率范围: [{prob.min():.4f}, {prob.max():.4f}]")
        print(f"Sigmoid后概率均值: {prob.mean():.4f}")
        print(f"Sigmoid后概率标准差: {prob.std():.4f}")
        
        # 检查阈值化结果
        binary_05 = (prob > 0.5).astype(np.uint8)
        print(f"阈值0.5结果: 均值={binary_05.mean():.4f}, 非零像素={binary_05.sum()}")
        
        # 检查不同阈值
        for thresh in [0.4, 0.45, 0.5, 0.55, 0.6]:
            binary = (prob > thresh).astype(np.uint8)
            print(f"阈值{thresh}: 均值={binary.mean():.4f}, 非零像素={binary.sum()}")

if __name__ == "__main__":
    debug_probability() 