#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试警告系统
"""

import torch
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入修复后的函数
from train_model import postprocess, simple_postprocess

def test_warning_system():
    """测试警告系统"""
    print("=== 测试警告系统 ===")
    
    # 测试全零输出
    print("\n1. 测试全零输出:")
    data_zero = torch.randn(1, 1, 32, 32) * -10  # 极低值
    result = postprocess(data_zero, debug_mode=True)
    
    # 测试全一输出
    print("\n2. 测试全一输出:")
    data_one = torch.randn(1, 1, 32, 32) * 10  # 极高值
    result = postprocess(data_one, debug_mode=True)
    
    # 测试正常输出
    print("\n3. 测试正常输出:")
    data_normal = torch.randn(1, 1, 32, 32)
    result = postprocess(data_normal, debug_mode=True)
    
    # 测试低方差高值
    print("\n4. 测试低方差高值:")
    data_low_var_high = torch.randn(1, 1, 32, 32) * 0.1 + 0.5
    result = postprocess(data_low_var_high, debug_mode=True)

def test_simple_postprocess_warnings():
    """测试simple_postprocess的警告系统"""
    print("\n\n=== 测试simple_postprocess警告系统 ===")
    
    # 设置调试模式
    simple_postprocess._debug_mode = True
    
    # 测试全零输出
    print("\n1. 测试全零输出:")
    data_zero = torch.randn(1, 1, 32, 32) * -10  # 极低值
    result = simple_postprocess(data_zero, adaptive=True)
    
    # 测试全一输出
    print("\n2. 测试全一输出:")
    data_one = torch.randn(1, 1, 32, 32) * 10  # 极高值
    result = simple_postprocess(data_one, adaptive=True)
    
    # 测试正常输出
    print("\n3. 测试正常输出:")
    data_normal = torch.randn(1, 1, 32, 32)
    result = simple_postprocess(data_normal, adaptive=True)
    
    # 测试低方差高值
    print("\n4. 测试低方差高值:")
    data_low_var_high = torch.randn(1, 1, 32, 32) * 0.1 + 0.5
    result = simple_postprocess(data_low_var_high, adaptive=True)

if __name__ == "__main__":
    test_warning_system()
    test_simple_postprocess_warnings()
    print("\n=== 所有测试完成 ===") 