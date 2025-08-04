import numpy as np
import torch
from train_model import postprocess

def test_postprocess_detailed():
    """详细测试后处理警告逻辑"""
    print("=== 详细测试后处理警告逻辑 ===")
    
    # 测试场景1：全零输入（应该输出全零）
    print("\n1. 测试全零输入:")
    zero_output = torch.zeros(1, 1, 64, 64)
    result1 = postprocess(zero_output, debug_mode=True)
    print(f"   输出统计: min={result1.min():.4f}, max={result1.max():.4f}, mean={result1.mean():.4f}")
    print(f"   输出全零: {result1.sum() == 0}")
    print(f"   输出全一: {result1.sum() == result1.numel()}")
    
    # 测试场景2：全一输入（应该输出全零，因为被保守过滤）
    print("\n2. 测试全一输入:")
    ones_output = torch.ones(1, 1, 64, 64)
    result2 = postprocess(ones_output, debug_mode=True)
    print(f"   输出统计: min={result2.min():.4f}, max={result2.max():.4f}, mean={result2.mean():.4f}")
    print(f"   输出全零: {result2.sum() == 0}")
    print(f"   输出全一: {result2.sum() == result2.numel()}")
    
    # 测试场景3：正常输入
    print("\n3. 测试正常输入:")
    normal_output = torch.randn(1, 1, 64, 64)
    result3 = postprocess(normal_output, debug_mode=True)
    print(f"   输出统计: min={result3.min():.4f}, max={result3.max():.4f}, mean={result3.mean():.4f}")
    print(f"   输出全零: {result3.sum() == 0}")
    print(f"   输出全一: {result3.sum() == result3.numel()}")
    
    # 测试场景4：低方差高值输入
    print("\n4. 测试低方差高值输入:")
    high_value = torch.ones(1, 1, 64, 64) * 0.8
    result4 = postprocess(high_value, debug_mode=True)
    print(f"   输出统计: min={result4.min():.4f}, max={result4.max():.4f}, mean={result4.mean():.4f}")
    print(f"   输出全零: {result4.sum() == 0}")
    print(f"   输出全一: {result4.sum() == result4.numel()}")
    
    # 测试场景5：中等概率输入
    print("\n5. 测试中等概率输入:")
    medium_output = torch.ones(1, 1, 64, 64) * 0.6
    result5 = postprocess(medium_output, debug_mode=True)
    print(f"   输出统计: min={result5.min():.4f}, max={result5.max():.4f}, mean={result5.mean():.4f}")
    print(f"   输出全零: {result5.sum() == 0}")
    print(f"   输出全一: {result5.sum() == result5.numel()}")
    
    # 测试场景6：低概率输入
    print("\n6. 测试低概率输入:")
    low_output = torch.ones(1, 1, 64, 64) * 0.2
    result6 = postprocess(low_output, debug_mode=True)
    print(f"   输出统计: min={result6.min():.4f}, max={result6.max():.4f}, mean={result6.mean():.4f}")
    print(f"   输出全零: {result6.sum() == 0}")
    print(f"   输出全一: {result6.sum() == result6.numel()}")

if __name__ == "__main__":
    test_postprocess_detailed() 