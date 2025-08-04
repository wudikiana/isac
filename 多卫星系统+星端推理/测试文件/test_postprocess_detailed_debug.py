import numpy as np
import torch
from train_model import postprocess

def test_postprocess_detailed_debug():
    """详细调试后处理函数"""
    print("=== 详细调试后处理函数 ===")
    
    # 测试场景1：全零输入
    print("\n1. 测试全零输入:")
    zero_output = torch.zeros(1, 1, 64, 64)
    
    # 手动执行步骤
    prob = torch.sigmoid(zero_output).cpu().numpy()
    prob_2d = prob[0, 0] if prob.shape[1] == 1 else prob[0, :, :, 0]
    prob_range = prob_2d.max() - prob_2d.min()
    prob_mean = prob_2d.mean()
    
    print(f"   sigmoid后: mean={prob_mean:.4f}, range={prob_range:.4f}")
    
    if prob_range < 0.01:
        if prob_mean > 0.7:
            binary = np.ones_like(prob_2d, dtype=np.uint8)
            print(f"   -> 高概率区域，输出全1")
        elif prob_mean < 0.3:
            binary = np.zeros_like(prob_2d, dtype=np.uint8)
            print(f"   -> 低概率区域，输出全0")
        else:
            binary = (prob_2d > 0.5).astype(np.uint8)
            print(f"   -> 中等概率区域，使用0.5阈值")
    else:
        binary = (prob_2d > 0.5).astype(np.uint8)
        print(f"   -> 正常范围，使用0.5阈值")
    
    print(f"   二值化结果: mean={binary.mean():.4f}")
    
    # 保守过滤
    if binary.mean() > 0.95:
        high_conf_thresh = max(0.8, np.percentile(prob_2d, 90))
        binary = (prob_2d > high_conf_thresh).astype(np.uint8)
        print(f"   保守过滤后: mean={binary.mean():.4f}")
    
    # 测试场景2：全一输入
    print("\n2. 测试全一输入:")
    ones_output = torch.ones(1, 1, 64, 64)
    
    prob = torch.sigmoid(ones_output).cpu().numpy()
    prob_2d = prob[0, 0] if prob.shape[1] == 1 else prob[0, :, :, 0]
    prob_range = prob_2d.max() - prob_2d.min()
    prob_mean = prob_2d.mean()
    
    print(f"   sigmoid后: mean={prob_mean:.4f}, range={prob_range:.4f}")
    
    if prob_range < 0.01:
        if prob_mean > 0.7:
            binary = np.ones_like(prob_2d, dtype=np.uint8)
            print(f"   -> 高概率区域，输出全1")
        elif prob_mean < 0.3:
            binary = np.zeros_like(prob_2d, dtype=np.uint8)
            print(f"   -> 低概率区域，输出全0")
        else:
            binary = (prob_2d > 0.5).astype(np.uint8)
            print(f"   -> 中等概率区域，使用0.5阈值")
    else:
        binary = (prob_2d > 0.5).astype(np.uint8)
        print(f"   -> 正常范围，使用0.5阈值")
    
    print(f"   二值化结果: mean={binary.mean():.4f}")
    
    # 保守过滤
    if binary.mean() > 0.95:
        high_conf_thresh = max(0.8, np.percentile(prob_2d, 90))
        binary = (prob_2d > high_conf_thresh).astype(np.uint8)
        print(f"   保守过滤后: mean={binary.mean():.4f}")
    
    # 测试场景3：高概率输入
    print("\n3. 测试高概率输入:")
    high_output = torch.ones(1, 1, 64, 64) * 0.9
    
    prob = torch.sigmoid(high_output).cpu().numpy()
    prob_2d = prob[0, 0] if prob.shape[1] == 1 else prob[0, :, :, 0]
    prob_range = prob_2d.max() - prob_2d.min()
    prob_mean = prob_2d.mean()
    
    print(f"   sigmoid后: mean={prob_mean:.4f}, range={prob_range:.4f}")
    
    if prob_range < 0.01:
        if prob_mean > 0.7:
            binary = np.ones_like(prob_2d, dtype=np.uint8)
            print(f"   -> 高概率区域，输出全1")
        elif prob_mean < 0.3:
            binary = np.zeros_like(prob_2d, dtype=np.uint8)
            print(f"   -> 低概率区域，输出全0")
        else:
            binary = (prob_2d > 0.5).astype(np.uint8)
            print(f"   -> 中等概率区域，使用0.5阈值")
    else:
        binary = (prob_2d > 0.5).astype(np.uint8)
        print(f"   -> 正常范围，使用0.5阈值")
    
    print(f"   二值化结果: mean={binary.mean():.4f}")
    
    # 保守过滤
    if binary.mean() > 0.95:
        high_conf_thresh = max(0.8, np.percentile(prob_2d, 90))
        binary = (prob_2d > high_conf_thresh).astype(np.uint8)
        print(f"   保守过滤后: mean={binary.mean():.4f}")

if __name__ == "__main__":
    test_postprocess_detailed_debug() 