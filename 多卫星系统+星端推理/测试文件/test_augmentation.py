#!/usr/bin/env python3
"""
测试数据增强和模型修改是否正确
"""

import torch
import torch.nn as nn
from train_model import DeepLabWithSimFeature, AdvancedAugmentation
import numpy as np
from PIL import Image

def test_model_dropout():
    """测试模型中的Dropout是否正确启用"""
    print("=== 测试模型Dropout ===")
    
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    
    # 检查sim_fc中是否有dropout
    has_dropout = False
    for module in model.sim_fc.modules():
        if isinstance(module, nn.Dropout):
            has_dropout = True
            print(f"✅ 发现Dropout层: {module}")
            break
            
    if not has_dropout:
        print("❌ sim_fc中没有发现Dropout层")
    
    # 检查是否有dropout2d
    if hasattr(model, 'dropout2d'):
        print(f"✅ 发现Dropout2d层: {model.dropout2d}")
    else:
        print("❌ 没有发现Dropout2d层")
    
    # 测试前向传播
    dummy_img = torch.randn(2, 3, 64, 64)
    dummy_sim_feat = torch.randn(2, 11)
    
    model.train()  # 确保dropout启用
    output1 = model(dummy_img, dummy_sim_feat)
    
    model.train()  # 再次运行，结果应该不同（因为dropout）
    output2 = model(dummy_img, dummy_sim_feat)
    
    if not torch.allclose(output1, output2, atol=1e-6):
        print("✅ Dropout正常工作，两次前向传播结果不同")
    else:
        print("❌ Dropout可能没有正常工作")
    
    print(f"输出形状: {output1.shape}")
    print()

def test_augmentation():
    """测试数据增强是否正确工作"""
    print("=== 测试数据增强 ===")
    
    # 创建测试图像和掩码
    test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
    
    # 创建增强器
    aug = AdvancedAugmentation(is_training=True)
    
    # 应用增强
    aug_img, aug_mask = aug(test_img, test_mask)
    
    print(f"原始图像形状: {test_img.shape}, 类型: {test_img.dtype}")
    print(f"增强后图像形状: {aug_img.shape}, 类型: {aug_img.dtype}")
    print(f"原始掩码形状: {test_mask.shape}, 类型: {test_mask.dtype}")
    print(f"增强后掩码形状: {aug_mask.shape}, 类型: {aug_mask.dtype}")
    
    # 检查图像是否被归一化
    if aug_img.dtype == torch.float32:
        print("✅ 图像已正确转换为float32")
    else:
        print("❌ 图像类型转换有问题")
    
    # 检查数值范围
    if torch.min(aug_img) >= -3 and torch.max(aug_img) <= 3:
        print("✅ 图像数值范围正常（归一化后）")
    else:
        print("❌ 图像数值范围异常")
    
    print()

def test_optimizer():
    """测试优化器设置"""
    print("=== 测试优化器设置 ===")
    
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    
    # 检查优化器类型和参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    print(f"优化器类型: {type(optimizer).__name__}")
    print(f"学习率: {optimizer.param_groups[0]['lr']}")
    print(f"权重衰减: {optimizer.param_groups[0]['weight_decay']}")
    
    # 检查学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    print(f"调度器类型: {type(scheduler).__name__}")
    print(f"调度器模式: {scheduler.mode}")
    print(f"调度器因子: {scheduler.factor}")
    print(f"调度器耐心值: {scheduler.patience}")
    
    print()

if __name__ == "__main__":
    print("开始测试模型和数据增强修改...")
    print("=" * 50)
    
    test_model_dropout()
    test_augmentation()
    test_optimizer()
    
    print("=" * 50)
    print("测试完成！") 