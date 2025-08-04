#!/usr/bin/env python3
"""
测试sim_feats的加载是否正确
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from train_model import get_multimodal_patch_dataloaders, load_sim_features

def test_sim_features_loading():
    """测试sim_feats的加载"""
    print("=== 测试sim_feats加载 ===")
    
    # 测试sim_feature_dict的加载
    try:
        sim_feature_dict = load_sim_features('data/sim_features.csv')
        print(f"✅ sim_feature_dict加载成功，包含 {len(sim_feature_dict)} 个条目")
        
        # 检查几个样本的结构
        sample_count = 0
        for key, value in sim_feature_dict.items():
            if sample_count < 3:
                print(f"样本 {sample_count + 1}:")
                print(f"  键: {key}")
                print(f"  值类型: {type(value)}")
                if isinstance(value, tuple):
                    print(f"  数值特征: {len(value[0])} 个")
                    print(f"  字符串特征: {len(value[1])} 个")
                    print(f"  数值特征示例: {value[0][:3]}...")
                else:
                    print(f"  特征: {value}")
                sample_count += 1
            else:
                break
    except Exception as e:
        print(f"❌ sim_feature_dict加载失败: {e}")
        return
    
    # 测试数据加载器
    try:
        train_loader, val_loader = get_multimodal_patch_dataloaders(
            data_root="data/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=4,
            num_workers=0,  # 使用单进程避免pickle问题
            damage_boost=1
        )
        print(f"✅ 数据加载器创建成功")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        # 测试一个batch
        for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  图像形状: {images.shape}")
            print(f"  掩码形状: {masks.shape}")
            print(f"  sim_feats形状: {sim_feats.shape}")
            print(f"  sim_feats类型: {sim_feats.dtype}")
            print(f"  sim_feats范围: [{sim_feats.min().item():.4f}, {sim_feats.max().item():.4f}]")
            print(f"  sim_feats均值: {sim_feats.mean().item():.4f}")
            print(f"  sim_feats标准差: {sim_feats.std().item():.4f}")
            
            # 检查是否有非零值
            non_zero_count = (sim_feats != 0).sum().item()
            total_count = sim_feats.numel()
            non_zero_ratio = non_zero_count / total_count
            print(f"  非零值比例: {non_zero_ratio:.4f} ({non_zero_count}/{total_count})")
            
            if non_zero_ratio > 0:
                print("✅ sim_feats包含非零值，加载正常")
            else:
                print("❌ sim_feats全为零，可能存在加载问题")
            
            break  # 只测试第一个batch
            
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_model_with_sim_features():
    """测试模型是否能正确处理sim_feats"""
    print("\n=== 测试模型处理sim_feats ===")
    
    try:
        from train_model import DeepLabWithSimFeature
        
        model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
        
        # 创建测试数据
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 64, 64)
        dummy_sim_feats = torch.randn(batch_size, 11)  # 非零的sim_feats
        
        print(f"测试输入:")
        print(f"  图像形状: {dummy_images.shape}")
        print(f"  sim_feats形状: {dummy_sim_feats.shape}")
        print(f"  sim_feats范围: [{dummy_sim_feats.min().item():.4f}, {dummy_sim_feats.max().item():.4f}]")
        
        # 前向传播
        model.train()
        outputs = model(dummy_images, dummy_sim_feats)
        
        print(f"模型输出:")
        print(f"  输出形状: {outputs.shape}")
        print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
        print(f"  输出均值: {outputs.mean().item():.4f}")
        
        if torch.isnan(outputs).any():
            print("❌ 模型输出包含NaN值")
        elif torch.isinf(outputs).any():
            print("❌ 模型输出包含Inf值")
        else:
            print("✅ 模型输出正常")
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.abspath('.'))

def test_sim_features():
    print("测试仿真特征加载...")
    
    try:
        from run_inference import load_sim_features
        sim_features = load_sim_features()
        
        print(f"✅ 成功加载 {len(sim_features)} 个仿真特征")
        
        if sim_features:
            # 显示第一个特征
            first_key = list(sim_features.keys())[0]
            first_features = sim_features[first_key]
            print(f"第一个文件名: {first_key}")
            print(f"特征维度: {first_features.shape}")
            print(f"特征值: {first_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ 仿真特征加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试sim_feats加载...")
    print("=" * 50)
    
    test_sim_features_loading()
    test_model_with_sim_features()
    
    print("=" * 50)
    print("测试完成！") 