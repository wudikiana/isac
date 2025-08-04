#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多类别分类训练示例脚本
演示如何使用多样化的掩码处理和多类别分类模型

掩码含义：
0: 背景
1: 未损坏
2: 轻微损坏
3: 中等损坏
4: 严重损坏
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from train_model import (
    get_multi_class_dataloaders,
    MultiClassDeepLab,
    MultiClassLoss,
    train_multi_class_epoch,
    val_multi_class_epoch,
    evaluate_multi_class_performance
)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载器配置
    data_root = "data/combined_dataset"
    batch_size = 16
    num_workers = 4
    
    # 获取多类别数据加载器
    print("正在加载多类别数据...")
    train_loader, val_loader, test_loader = get_multi_class_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        damage_level='categorical',  # 使用多类别分类模式
        show_warnings=False,
        skip_problematic_samples=True
    )
    
    # 创建多类别模型
    print("创建多类别分类模型...")
    model = MultiClassDeepLab(
        in_channels=3,
        num_classes=5,  # 5个类别：背景、未损坏、轻微损坏、中等损坏、严重损坏
        sim_feat_dim=11
    ).to(device)
    
    # 创建多类别损失函数
    print("创建多类别损失函数...")
    criterion = MultiClassLoss(
        alpha=0.4,  # 交叉熵损失权重
        beta=0.3,   # Dice损失权重
        gamma=0.2,  # Focal损失权重
        delta=0.1   # 边界损失权重
    ).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练参数
    num_epochs = 30
    best_damage_iou = 0.0
    
    # 训练历史记录
    train_losses = []
    train_accuracies = []
    train_damage_ious = []
    val_losses = []
    val_accuracies = []
    val_damage_ious = []
    val_severe_ious = []
    
    print("开始多类别分类训练...")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 训练阶段
        train_loss, train_acc, train_damage_iou = train_multi_class_epoch(
            model, train_loader, optimizer, criterion, device, epoch+1, scaler
        )
        
        # 验证阶段
        val_loss, val_acc, val_damage_iou, val_severe_iou, class_ious = val_multi_class_epoch(
            model, val_loader, criterion, device, scaler
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_damage_ious.append(train_damage_iou)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_damage_ious.append(val_damage_iou)
        val_severe_ious.append(val_severe_iou)
        
        # 打印结果
        print(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Damage IoU: {train_damage_iou:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Damage IoU: {val_damage_iou:.4f}")
        print(f"验证 - Severe Damage IoU: {val_severe_iou:.4f}")
        print(f"学习率: {current_lr:.2e}")
        
        # 打印每个类别的IoU
        class_names = ['背景', '未损坏', '轻微损坏', '中等损坏', '严重损坏']
        print("各类别IoU:")
        for i, (name, iou) in enumerate(zip(class_names, class_ious)):
            print(f"  {name}: {iou:.4f}")
        
        # 保存最佳模型
        if val_damage_iou > best_damage_iou:
            best_damage_iou = val_damage_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_damage_iou': best_damage_iou,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_severe_iou': val_severe_iou,
                'class_ious': class_ious
            }, 'models/best_multi_class_model.pth')
            print(f"✅ 保存最佳模型 (Damage IoU: {best_damage_iou:.4f})")
        
        print("-" * 40)
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='训练损失')
    axes[0, 0].plot(val_losses, label='验证损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(train_accuracies, label='训练准确率')
    axes[0, 1].plot(val_accuracies, label='验证准确率')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 损坏IoU曲线
    axes[1, 0].plot(train_damage_ious, label='训练损坏IoU')
    axes[1, 0].plot(val_damage_ious, label='验证损坏IoU')
    axes[1, 0].set_title('损坏区域IoU曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Damage IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 严重损坏IoU曲线
    axes[1, 1].plot(val_severe_ious, label='验证严重损坏IoU', color='red')
    axes[1, 1].set_title('严重损坏IoU曲线')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Severe Damage IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('multi_class_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 最终评估
    print("\n" + "="*60)
    print("训练完成！最终评估结果：")
    print("="*60)
    
    # 加载最佳模型进行测试
    checkpoint = torch.load('models/best_multi_class_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 在测试集上评估
    test_loss, test_acc, test_damage_iou, test_severe_iou, test_class_ious = val_multi_class_epoch(
        model, test_loader, criterion, device, scaler
    )
    
    print(f"测试集结果:")
    print(f"  损失: {test_loss:.4f}")
    print(f"  准确率: {test_acc:.4f}")
    print(f"  损坏IoU: {test_damage_iou:.4f}")
    print(f"  严重损坏IoU: {test_severe_iou:.4f}")
    
    print("\n各类别IoU:")
    class_names = ['背景', '未损坏', '轻微损坏', '中等损坏', '严重损坏']
    for i, (name, iou) in enumerate(zip(class_names, test_class_ious)):
        print(f"  {name}: {iou:.4f}")
    
    print("\n" + "="*60)
    print("多类别分类训练示例完成！")
    print("="*60)

if __name__ == "__main__":
    # 创建模型目录
    os.makedirs('models', exist_ok=True)
    
    # 运行训练
    main() 