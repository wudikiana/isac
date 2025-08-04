 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试refined变量
"""

import torch
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_refined_variable():
    """调试refined变量"""
    print("=== 调试refined变量 ===")
    
    # 创建测试数据
    data = torch.randn(1, 1, 32, 32) * -10  # 极低值
    
    print("原始数据:")
    print(f"  范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  均值: {data.mean():.4f}")
    
    # 手动执行postprocess的逻辑
    prob = torch.sigmoid(data).cpu().numpy().squeeze()
    print(f"\nSigmoid后概率:")
    print(f"  范围: [{prob.min():.4f}, {prob.max():.4f}]")
    print(f"  均值: {prob.mean():.4f}")
    
    # 检查概率范围
    prob_range = prob.max() - prob.min()
    print(f"\n概率范围: {prob_range:.4f}")
    
    if prob_range < 0.01:
        print("使用自适应阈值")
        adaptive_thresh = prob.mean()
        binary = (prob > adaptive_thresh).astype(np.uint8)
    else:
        print("使用Otsu阈值")
        binary = (prob > 0.5).astype(np.uint8)  # 简化处理
    
    print(f"\n二值化结果:")
    print(f"  均值: {binary.mean():.4f}")
    print(f"  非零像素: {binary.sum()}")
    
    # 检查是否需要保守过滤
    if binary.mean() > 0.95:
        print("需要保守过滤")
        high_conf_thresh = max(0.8, np.percentile(prob, 90))
        binary = (prob > high_conf_thresh).astype(np.uint8)
        print(f"保守过滤后: 均值={binary.mean():.4f}, 非零像素={binary.sum()}")
    
    # 模拟refined变量（简化版）
    refined = binary.copy()
    
    print(f"\nrefined变量:")
    print(f"  形状: {refined.shape}")
    print(f"  均值: {refined.mean():.4f}")
    print(f"  非零像素: {refined.sum()}")
    print(f"  总像素: {refined.size}")
    print(f"  sum() == 0: {refined.sum() == 0}")
    print(f"  sum() == size: {refined.sum() == refined.size}")
    print(f"  0 < sum() < size: {0 < refined.sum() < refined.size}")
    
    # 手动执行警告逻辑
    print(f"\n警告逻辑检查:")
    if refined.sum() == 0:
        print("  refined.sum() == 0")
        if prob.max() < 0.3:
            print("  应该输出: 信息：后处理输出全零（低概率区域）")
        else:
            print("  应该输出: 警告：后处理输出全零（高概率区域被过滤）")
    elif refined.sum() == refined.size:
        print("  refined.sum() == refined.size")
        if prob.min() > 0.7:
            print("  应该输出: 信息：后处理输出全一（高概率区域）")
        else:
            print("  应该输出: 警告：后处理输出全一（低概率区域被保留）")
    elif refined.sum() > 0 and refined.sum() < refined.size:
        print("  refined.sum() > 0 and refined.sum() < refined.size")
        prob_mean = prob.mean()
        prob_std = prob.std()
        if prob_std > 0.1:
            print(f"  应该输出: 信息：后处理输出正常 (均值={prob_mean:.2f}, 标准差={prob_std:.2f})")
        else:
            print(f"  应该输出: 信息：后处理输出正常 (低方差, 均值={prob_mean:.2f})")

if __name__ == "__main__":
    debug_refined_variable()