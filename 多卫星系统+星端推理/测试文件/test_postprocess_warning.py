import numpy as np
import torch
from train_model import postprocess

def test_postprocess_warnings():
    """测试后处理警告逻辑"""
    print("=== 测试后处理警告逻辑 ===")
    
    # 测试场景1：全零输入
    print("\n1. 测试全零输入:")
    zero_output = torch.zeros(1, 1, 64, 64)
    result1 = postprocess(zero_output, debug_mode=True)
    print(f"   输入形状: {zero_output.shape}")
    print(f"   输出形状: {result1.shape}")
    print(f"   输出统计: min={result1.min():.4f}, max={result1.max():.4f}, mean={result1.mean():.4f}")
    
    # 测试场景2：全一输入
    print("\n2. 测试全一输入:")
    ones_output = torch.ones(1, 1, 64, 64)
    result2 = postprocess(ones_output, debug_mode=True)
    print(f"   输入形状: {ones_output.shape}")
    print(f"   输出形状: {result2.shape}")
    print(f"   输出统计: min={result2.min():.4f}, max={result2.max():.4f}, mean={result2.mean():.4f}")
    
    # 测试场景3：正常输入
    print("\n3. 测试正常输入:")
    normal_output = torch.randn(1, 1, 64, 64)
    result3 = postprocess(normal_output, debug_mode=True)
    print(f"   输入形状: {normal_output.shape}")
    print(f"   输出形状: {result3.shape}")
    print(f"   输出统计: min={result3.min():.4f}, max={result3.max():.4f}, mean={result3.mean():.4f}")
    
    # 测试场景4：低方差高值输入
    print("\n4. 测试低方差高值输入:")
    high_value = torch.ones(1, 1, 64, 64) * 0.8
    result4 = postprocess(high_value, debug_mode=True)
    print(f"   输入形状: {high_value.shape}")
    print(f"   输出形状: {result4.shape}")
    print(f"   输出统计: min={result4.min():.4f}, max={result4.max():.4f}, mean={result4.mean():.4f}")

if __name__ == "__main__":
    test_postprocess_warnings() 