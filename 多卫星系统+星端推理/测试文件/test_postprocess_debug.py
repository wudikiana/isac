import numpy as np
import torch
from train_model import postprocess

def test_postprocess_debug():
    """调试后处理函数的问题"""
    print("=== 调试后处理函数 ===")
    
    # 测试场景：全一输入
    print("\n测试全一输入:")
    ones_output = torch.ones(1, 1, 64, 64)
    
    # 手动执行后处理步骤
    print("1. 原始输入:")
    print(f"   形状: {ones_output.shape}")
    print(f"   值范围: {ones_output.min():.4f} - {ones_output.max():.4f}")
    
    # 应用sigmoid
    prob = torch.sigmoid(ones_output).cpu().numpy()
    print("2. 应用sigmoid后:")
    print(f"   形状: {prob.shape}")
    print(f"   值范围: {prob.min():.4f} - {prob.max():.4f}")
    print(f"   均值: {prob.mean():.4f}")
    
    # 提取2D数组
    if prob.ndim == 4:
        prob_2d = prob[0, 0] if prob.shape[1] == 1 else prob[0, :, :, 0]
    print("3. 提取2D数组后:")
    print(f"   形状: {prob_2d.shape}")
    print(f"   值范围: {prob_2d.min():.4f} - {prob_2d.max():.4f}")
    print(f"   均值: {prob_2d.mean():.4f}")
    
    # 检查概率范围
    prob_range = prob_2d.max() - prob_2d.min()
    print(f"4. 概率范围: {prob_range:.4f}")
    
    # 二值化
    if prob_range < 0.01:
        adaptive_thresh = prob_2d.mean()
        binary = (prob_2d > adaptive_thresh).astype(np.uint8)
        print(f"5. 使用自适应阈值 {adaptive_thresh:.4f}")
    else:
        binary = (prob_2d > 0.5).astype(np.uint8)
        print("5. 使用固定阈值 0.5")
    
    print(f"   二值化结果: min={binary.min()}, max={binary.max()}, mean={binary.mean():.4f}")
    
    # 检查是否需要保守过滤
    if binary.mean() > 0.95:
        high_conf_thresh = max(0.8, np.percentile(prob_2d, 90))
        binary = (prob_2d > high_conf_thresh).astype(np.uint8)
        print(f"6. 保守过滤后: min={binary.min()}, max={binary.max()}, mean={binary.mean():.4f}")
    
    # 运行完整的后处理函数
    print("\n7. 运行完整后处理函数:")
    result = postprocess(ones_output, debug_mode=True)
    print(f"   最终结果: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")

if __name__ == "__main__":
    test_postprocess_debug() 