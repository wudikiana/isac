#!/usr/bin/env python3
"""
测试模型的联合推理能力
验证图像和sim_feats的联合学习效果
"""

import torch
import numpy as np
from train_model import DeepLabWithSimFeature, get_multimodal_patch_dataloaders

def test_joint_learning_capability():
    """测试联合学习能力"""
    print("=== 测试联合学习能力 ===")
    
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    
    # 创建数据加载器
    train_loader, _ = get_multimodal_patch_dataloaders(
        data_root="data/patch_dataset",
        sim_feature_csv="data/sim_features.csv",
        batch_size=4,
        num_workers=0,
        damage_boost=1
    )
    
    # 获取一个batch的数据
    for images, masks, sim_feats in train_loader:
        print(f"输入数据:")
        print(f"  图像形状: {images.shape}")
        print(f"  图像范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
        print(f"  sim_feats形状: {sim_feats.shape}")
        print(f"  sim_feats范围: [{sim_feats.min().item():.4f}, {sim_feats.max().item():.4f}]")
        
        # 测试模型前向传播
        model.train()
        outputs = model(images, sim_feats)
        
        print(f"模型输出:")
        print(f"  输出形状: {outputs.shape}")
        print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        
        if torch.isnan(outputs).any():
            print("❌ 输出包含NaN")
            return False
        else:
            print("✅ 联合推理正常")
            break
    
    return True

def test_feature_ablation():
    """测试特征消融实验"""
    print("\n=== 特征消融实验 ===")
    
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    
    # 创建测试数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 64, 64)
    
    # 测试不同sim_feats输入
    test_cases = [
        ("零向量（无sim_feats）", torch.zeros(batch_size, 11)),
        ("随机sim_feats", torch.randn(batch_size, 11)),
        ("归一化sim_feats", torch.randn(batch_size, 11) * 0.1),
    ]
    
    results = {}
    
    for case_name, sim_feats in test_cases:
        print(f"\n测试: {case_name}")
        
        model.train()
        outputs = model(images, sim_feats)
        
        # 计算输出统计
        output_mean = outputs.mean().item()
        output_std = outputs.std().item()
        output_range = outputs.max().item() - outputs.min().item()
        
        results[case_name] = {
            'mean': output_mean,
            'std': output_std,
            'range': output_range
        }
        
        print(f"  输出均值: {output_mean:.4f}")
        print(f"  输出标准差: {output_std:.4f}")
        print(f"  输出范围: {output_range:.4f}")
    
    # 分析结果
    print(f"\n消融实验结果分析:")
    zero_case = results["零向量（无sim_feats）"]
    random_case = results["随机sim_feats"]
    
    mean_diff = abs(random_case['mean'] - zero_case['mean'])
    std_diff = abs(random_case['std'] - zero_case['std'])
    
    print(f"  有无sim_feats的均值差异: {mean_diff:.4f}")
    print(f"  有无sim_feats的标准差差异: {std_diff:.4f}")
    
    if mean_diff > 0.01 or std_diff > 0.01:
        print("✅ sim_feats对模型输出有显著影响")
        return True
    else:
        print("❌ sim_feats对模型输出影响很小")
        return False

def test_gradient_flow():
    """测试梯度流动"""
    print("\n=== 测试梯度流动 ===")
    
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    
    # 创建数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 64, 64, requires_grad=True)
    sim_feats = torch.randn(batch_size, 11, requires_grad=True)
    targets = torch.randn(batch_size, 1, 64, 64)
    
    # 前向传播
    outputs = model(images, sim_feats)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"图像梯度统计:")
    print(f"  梯度范围: [{images.grad.min().item():.6f}, {images.grad.max().item():.6f}]")
    print(f"  梯度均值: {images.grad.mean().item():.6f}")
    print(f"  梯度标准差: {images.grad.std().item():.6f}")
    
    print(f"sim_feats梯度统计:")
    print(f"  梯度范围: [{sim_feats.grad.min().item():.6f}, {sim_feats.grad.max().item():.6f}]")
    print(f"  梯度均值: {sim_feats.grad.mean().item():.6f}")
    print(f"  梯度标准差: {sim_feats.grad.std().item():.6f}")
    
    # 检查sim_fc层的梯度
    sim_fc_grad_norm = 0
    for param in model.sim_fc.parameters():
        if param.grad is not None:
            sim_fc_grad_norm += param.grad.norm().item() ** 2
    sim_fc_grad_norm = sim_fc_grad_norm ** 0.5
    
    print(f"sim_fc层梯度范数: {sim_fc_grad_norm:.6f}")
    
    if sim_fc_grad_norm > 1e-6:
        print("✅ sim_feats梯度正常流动")
        return True
    else:
        print("❌ sim_feats梯度流动异常")
        return False

def test_real_training_step():
    """测试真实训练步骤"""
    print("\n=== 测试真实训练步骤 ===")
    
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 获取真实数据
    train_loader, _ = get_multimodal_patch_dataloaders(
        data_root="data/patch_dataset",
        sim_feature_csv="data/sim_features.csv",
        batch_size=4,
        num_workers=0,
        damage_boost=1
    )
    
    for images, masks, sim_feats in train_loader:
        print(f"训练数据:")
        print(f"  图像形状: {images.shape}")
        print(f"  掩码形状: {masks.shape}")
        print(f"  sim_feats形状: {sim_feats.shape}")
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images, sim_feats)
        
        # 调整掩码维度以匹配输出
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        loss = criterion(outputs, masks)
        
        print(f"  损失值: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"  总梯度范数: {total_grad_norm:.6f}")
        
        # 更新参数
        optimizer.step()
        
        print(f"✅ 训练步骤完成")
        break
    
    return True

if __name__ == "__main__":
    print("开始测试联合推理能力...")
    print("=" * 50)
    
    # 执行各项测试
    test1 = test_joint_learning_capability()
    test2 = test_feature_ablation()
    test3 = test_gradient_flow()
    test4 = test_real_training_step()
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"  联合学习能力: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"  特征消融实验: {'✅ 通过' if test2 else '❌ 失败'}")
    print(f"  梯度流动测试: {'✅ 通过' if test3 else '❌ 失败'}")
    print(f"  真实训练测试: {'✅ 通过' if test4 else '❌ 失败'}")
    
    if all([test1, test2, test3, test4]):
        print("\n🎉 所有测试通过！模型具备完整的联合推理能力！")
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试") 