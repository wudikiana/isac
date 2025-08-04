"""
逐步调试后处理函数
"""
import torch
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入修复后的函数
from train_model import simple_postprocess

def debug_step_by_step():
    """逐步调试后处理函数"""
    print("=== 逐步调试后处理函数 ===")
    
    # 创建测试数据
    data = torch.randn(1, 1, 32, 32) * 0.1 + 0.5  # 低方差高值
    
    print("原始数据:")
    print(f"  范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  均值: {data.mean():.4f}")
    
    # 步骤1: sigmoid转换
    prob = torch.sigmoid(data).cpu().numpy().squeeze()
    print(f"\n步骤1 - Sigmoid转换:")
    print(f"  概率范围: [{prob.min():.4f}, {prob.max():.4f}]")
    print(f"  概率均值: {prob.mean():.4f}")
    print(f"  概率标准差: {prob.std():.4f}")
    
    # 步骤2: 检查自适应逻辑
    prob_mean = prob.mean()
    prob_std = prob.std()
    
    print(f"\n步骤2 - 自适应逻辑检查:")
    print(f"  标准差 < 0.05: {prob_std < 0.05}")
    print(f"  均值 > 0.7: {prob_mean > 0.7}")
    print(f"  均值 < 0.3: {prob_mean < 0.3}")
    print(f"  均值 > 0.55: {prob_mean > 0.55}")
    print(f"  均值 < 0.45: {prob_mean < 0.45}")
    
    # 步骤3: 手动执行逻辑
    print(f"\n步骤3 - 手动执行逻辑:")
    if prob_std < 0.05:
        print("  进入低方差分支")
        if prob_mean > 0.7:
            print("  高均值区域 -> 全1")
            binary = np.ones_like(prob, dtype=np.uint8)
        elif prob_mean < 0.3:
            print("  低均值区域 -> 全0")
            binary = np.zeros_like(prob, dtype=np.uint8)
        elif prob_mean > 0.55:
            print("  中等高值区域 -> 使用0.5阈值")
            binary = (prob > 0.5).astype(np.uint8)
            print(f"    阈值0.5结果: 均值={binary.mean():.4f}, 非零像素={binary.sum()}")
        elif prob_mean < 0.45:
            print("  中等低值区域 -> 使用0.5阈值")
            binary = (prob > 0.5).astype(np.uint8)
            print(f"    阈值0.5结果: 均值={binary.mean():.4f}, 非零像素={binary.sum()}")
        else:
            print("  边界值区域 -> 使用均值阈值")
            binary = (prob > prob_mean).astype(np.uint8)
            print(f"    均值阈值{prob_mean:.4f}结果: 均值={binary.mean():.4f}, 非零像素={binary.sum()}")
    else:
        print("  进入正常方差分支")
        binary = (prob > 0.5).astype(np.uint8)
        print(f"  固定阈值0.5结果: 均值={binary.mean():.4f}, 非零像素={binary.sum()}")
    
    # 步骤4: 检查保守过滤
    print(f"\n步骤4 - 保守过滤检查:")
    print(f"  当前结果均值: {binary.mean():.4f}")
    print(f"  是否需要保守过滤: {binary.mean() > 0.95}")
    
    if binary.mean() > 0.95:
        high_conf_thresh = max(0.8, np.percentile(prob, 90))
        print(f"  保守过滤阈值: {high_conf_thresh:.4f}")
        binary = (prob > high_conf_thresh).astype(np.uint8)
        print(f"  保守过滤后: 均值={binary.mean():.4f}, 非零像素={binary.sum()}")
    
    # 步骤5: 最终结果
    print(f"\n步骤5 - 最终结果:")
    print(f"  最终均值: {binary.mean():.4f}")
    print(f"  最终非零像素: {binary.sum()}")
    
    # 步骤6: 对比函数调用
    print(f"\n步骤6 - 函数调用对比:")
    result = simple_postprocess(data, adaptive=True)
    func_binary = result.numpy()
    print(f"  函数调用结果: 均值={func_binary.mean():.4f}, 非零像素={func_binary.sum()}")
    
    # 检查是否一致
    if np.array_equal(binary, func_binary):
        print("  ✓ 手动计算与函数调用结果一致")
    else:
        print("  ✗ 手动计算与函数调用结果不一致")
        print(f"  差异像素数: {np.sum(binary != func_binary)}")

if __name__ == "__main__":
    debug_step_by_step()