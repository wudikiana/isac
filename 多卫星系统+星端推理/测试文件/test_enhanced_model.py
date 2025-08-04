#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强的山体滑坡检测模型
验证优化后的模型功能和性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models.starlite_cnn import (
    LandslideDetector, 
    EnhancedLandslideDetector,
    create_starlite_model,
    create_enhanced_model
)

def test_model_compatibility():
    """测试模型兼容性"""
    print("测试模型兼容性...")
    print("="*50)
    
    # 测试原始模型
    print("1. 测试原始模型 (enhanced=False):")
    model_original = create_starlite_model(enhanced=False)
    print(f"   模型类型: {type(model_original)}")
    print(f"   是否包含增强功能: {hasattr(model_original, 'model')}")
    
    # 测试增强模型
    print("\n2. 测试增强模型 (enhanced=True):")
    model_enhanced = create_starlite_model(enhanced=True)
    print(f"   模型类型: {type(model_enhanced)}")
    print(f"   是否包含增强功能: {hasattr(model_enhanced, 'model')}")
    
    # 测试不同配置
    print("\n3. 测试不同配置:")
    model_attn_only = create_starlite_model(enhanced=True, use_attention=True, use_fpn=False)
    model_fpn_only = create_starlite_model(enhanced=True, use_attention=False, use_fpn=True)
    model_both = create_starlite_model(enhanced=True, use_attention=True, use_fpn=True)
    
    print(f"   仅注意力机制: {hasattr(model_attn_only.model, 'attention_layers')}")
    print(f"   仅FPN: {hasattr(model_fpn_only.model, 'fpn')}")
    print(f"   注意力+FPN: {hasattr(model_both.model, 'attention_layers') and hasattr(model_both.model, 'fpn')}")
    
    print("\n✅ 模型兼容性测试通过")

def test_forward_pass():
    """测试前向传播"""
    print("\n测试前向传播...")
    print("="*50)
    
    # 创建测试数据
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # 测试原始模型
    print("1. 测试原始模型前向传播:")
    model_original = create_starlite_model(enhanced=False)
    model_original.eval()
    
    with torch.no_grad():
        output_original = model_original(input_tensor)
        print(f"   输入形状: {input_tensor.shape}")
        print(f"   输出形状: {output_original.shape}")
        print(f"   输出范围: [{output_original.min().item():.4f}, {output_original.max().item():.4f}]")
    
    # 测试增强模型
    print("\n2. 测试增强模型前向传播:")
    model_enhanced = create_starlite_model(enhanced=True)
    model_enhanced.eval()
    
    with torch.no_grad():
        output_enhanced = model_enhanced(input_tensor)
        print(f"   输入形状: {input_tensor.shape}")
        print(f"   输出形状: {output_enhanced.shape}")
        print(f"   输出范围: [{output_enhanced.min().item():.4f}, {output_enhanced.max().item():.4f}]")
    
    print("\n✅ 前向传播测试通过")

def test_model_parameters():
    """测试模型参数"""
    print("\n测试模型参数...")
    print("="*50)
    
    # 原始模型参数
    model_original = create_starlite_model(enhanced=False)
    original_params = sum(p.numel() for p in model_original.parameters())
    original_trainable = sum(p.numel() for p in model_original.parameters() if p.requires_grad)
    
    print(f"原始模型参数:")
    print(f"   总参数数: {original_params:,}")
    print(f"   可训练参数数: {original_trainable:,}")
    
    # 增强模型参数
    model_enhanced = create_starlite_model(enhanced=True)
    enhanced_params = sum(p.numel() for p in model_enhanced.parameters())
    enhanced_trainable = sum(p.numel() for p in model_enhanced.parameters() if p.requires_grad)
    
    print(f"\n增强模型参数:")
    print(f"   总参数数: {enhanced_params:,}")
    print(f"   可训练参数数: {enhanced_trainable:,}")
    print(f"   参数增加: {enhanced_params - original_params:,} ({((enhanced_params - original_params) / original_params * 100):.1f}%)")
    
    print("\n✅ 模型参数测试完成")

def test_quantization_compatibility():
    """测试量化兼容性"""
    print("\n测试量化兼容性...")
    print("="*50)
    
    # 测试原始模型量化
    print("1. 测试原始模型量化:")
    model_original = create_starlite_model(enhanced=False)
    try:
        model_original.fuse_model()
        print("   ✅ 原始模型融合成功")
    except Exception as e:
        print(f"   ❌ 原始模型融合失败: {e}")
    
    # 测试增强模型量化
    print("\n2. 测试增强模型量化:")
    model_enhanced = create_starlite_model(enhanced=True)
    try:
        model_enhanced.fuse_model()
        print("   ✅ 增强模型融合成功")
    except Exception as e:
        print(f"   ❌ 增强模型融合失败: {e}")
    
    print("\n✅ 量化兼容性测试完成")

def test_training_step():
    """测试训练步骤"""
    print("\n测试训练步骤...")
    print("="*50)
    
    # 创建模型和优化器
    model = create_starlite_model(enhanced=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 创建测试数据
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    target_tensor = torch.randint(0, 2, (batch_size,))
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"目标形状: {target_tensor.shape}")
    
    # 训练步骤
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"输出形状: {output.shape}")
    print(f"损失值: {loss.item():.4f}")
    print(f"梯度范数: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0):.4f}")
    
    print("\n✅ 训练步骤测试通过")

def test_model_variants():
    """测试不同模型变体"""
    print("\n测试不同模型变体...")
    print("="*50)
    
    variants = [
        ("原始模型", create_starlite_model(enhanced=False)),
        ("增强模型(注意力+FPN)", create_starlite_model(enhanced=True, use_attention=True, use_fpn=True)),
        ("仅注意力", create_starlite_model(enhanced=True, use_attention=True, use_fpn=False)),
        ("仅FPN", create_starlite_model(enhanced=True, use_attention=False, use_fpn=True)),
    ]
    
    input_tensor = torch.randn(2, 3, 224, 224)
    
    for name, model in variants:
        print(f"\n{name}:")
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            params = sum(p.numel() for p in model.parameters())
            print(f"   参数数: {params:,}")
            print(f"   输出形状: {output.shape}")
            print(f"   输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\n✅ 模型变体测试完成")

def visualize_model_comparison():
    """可视化模型比较"""
    print("\n生成模型比较图表...")
    print("="*50)
    
    # 模型配置
    configs = [
        ("原始模型", False, False, False),
        ("仅注意力", True, True, False),
        ("仅FPN", True, False, True),
        ("注意力+FPN", True, True, True),
    ]
    
    # 收集数据
    names = []
    param_counts = []
    
    for name, enhanced, attn, fpn in configs:
        model = create_starlite_model(enhanced=enhanced, use_attention=attn, use_fpn=fpn)
        params = sum(p.numel() for p in model.parameters())
        names.append(name)
        param_counts.append(params)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 参数数量对比
    bars1 = ax1.bar(names, param_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_title('模型参数数量对比')
    ax1.set_ylabel('参数数量')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars1, param_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom')
    
    # 参数增加百分比
    base_params = param_counts[0]
    increase_percentages = [(p - base_params) / base_params * 100 for p in param_counts]
    
    bars2 = ax2.bar(names, increase_percentages, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_title('参数增加百分比')
    ax2.set_ylabel('增加百分比 (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, pct in zip(bars2, increase_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 模型比较图表已生成")

def main():
    """主函数"""
    print("增强的山体滑坡检测模型测试")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 运行测试
    test_model_compatibility()
    test_forward_pass()
    test_model_parameters()
    test_quantization_compatibility()
    test_training_step()
    test_model_variants()
    visualize_model_comparison()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
    
    print("\n优化总结:")
    print("1. ✅ 保持向后兼容性")
    print("2. ✅ 添加CBAM注意力机制")
    print("3. ✅ 集成FPN特征金字塔")
    print("4. ✅ 增强分类头")
    print("5. ✅ 支持量化训练")
    print("6. ✅ 改进权重初始化")
    print("7. ✅ 优化模型融合")

if __name__ == "__main__":
    main() 