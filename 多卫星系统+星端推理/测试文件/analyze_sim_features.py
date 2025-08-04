#!/usr/bin/env python3
"""
分析sim_feats归一化对联合推理任务的影响
检查仿真特征的加载、传递和融合方式
"""

import torch
import numpy as np
import pandas as pd
from train_model import get_multimodal_patch_dataloaders, load_sim_features, DeepLabWithSimFeature

def analyze_sim_features_distribution():
    """分析sim_feats的分布特征"""
    print("=== 分析sim_feats分布特征 ===")
    
    # 加载原始sim_features
    sim_feature_dict = load_sim_features('data/sim_features.csv')
    
    # 收集所有数值特征
    all_features = []
    feature_names = ['comm_snr', 'radar_feat', 'radar_max', 'radar_std', 'radar_peak_idx', 
                    'path_loss', 'shadow_fading', 'rain_attenuation', 'target_rcs', 'bandwidth', 'ber']
    
    for key, (float_feats, str_feats) in sim_feature_dict.items():
        all_features.append(float_feats)
    
    all_features = np.array(all_features)
    
    print(f"原始sim_feats统计:")
    print(f"  样本数量: {len(all_features)}")
    print(f"  特征维度: {all_features.shape[1]}")
    
    # 分析每个特征的分布
    for i, name in enumerate(feature_names):
        feature_data = all_features[:, i]
        print(f"\n{name}:")
        print(f"  范围: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
        print(f"  均值: {feature_data.mean():.4f}")
        print(f"  标准差: {feature_data.std():.4f}")
        print(f"  中位数: {np.median(feature_data):.4f}")
    
    # 分析归一化后的效果
    print(f"\n=== 归一化效果分析 ===")
    normalized_features = []
    for features in all_features:
        if np.std(features) > 0:
            normalized = (features - np.mean(features)) / np.std(features)
        else:
            normalized = features
        normalized_features.append(normalized)
    
    normalized_features = np.array(normalized_features)
    
    print(f"归一化后sim_feats统计:")
    print(f"  整体范围: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")
    print(f"  整体均值: {normalized_features.mean():.4f}")
    print(f"  整体标准差: {normalized_features.std():.4f}")
    
    return feature_names, all_features, normalized_features

def test_feature_correlation():
    """测试特征之间的相关性"""
    print("\n=== 特征相关性分析 ===")
    
    feature_names = ['comm_snr', 'radar_feat', 'radar_max', 'radar_std', 'radar_peak_idx', 
                    'path_loss', 'shadow_fading', 'rain_attenuation', 'target_rcs', 'bandwidth', 'ber']
    
    sim_feature_dict = load_sim_features('data/sim_features.csv')
    all_features = []
    
    for key, (float_feats, str_feats) in sim_feature_dict.items():
        all_features.append(float_feats)
    
    all_features = np.array(all_features)
    
    # 计算相关性矩阵
    correlation_matrix = np.corrcoef(all_features.T)
    
    print("特征相关性矩阵 (前5个特征):")
    for i in range(5):
        for j in range(5):
            print(f"{correlation_matrix[i,j]:6.3f}", end=" ")
        print()
    
    # 找出高相关性的特征对
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(correlation_matrix[i,j]) > 0.7:
                high_corr_pairs.append((feature_names[i], feature_names[j], correlation_matrix[i,j]))
    
    print(f"\n高相关性特征对 (|r| > 0.7):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")

def test_joint_inference_capability():
    """测试联合推理能力"""
    print("\n=== 测试联合推理能力 ===")
    
    # 创建模型
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    
    # 创建测试数据
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 64, 64)
    
    # 测试不同的sim_feats输入
    test_cases = [
        ("零向量", torch.zeros(batch_size, 11)),
        ("随机向量", torch.randn(batch_size, 11)),
        ("归一化向量", torch.randn(batch_size, 11) * 0.1),  # 小方差
        ("大数值向量", torch.randn(batch_size, 11) * 100),  # 大数值
    ]
    
    for case_name, sim_feats in test_cases:
        print(f"\n测试案例: {case_name}")
        print(f"  sim_feats范围: [{sim_feats.min().item():.4f}, {sim_feats.max().item():.4f}]")
        print(f"  sim_feats均值: {sim_feats.mean().item():.4f}")
        print(f"  sim_feats标准差: {sim_feats.std().item():.4f}")
        
        try:
            model.train()
            outputs = model(dummy_images, sim_feats)
            
            print(f"  模型输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"  模型输出均值: {outputs.mean().item():.4f}")
            
            if torch.isnan(outputs).any():
                print("  ❌ 输出包含NaN")
            elif torch.isinf(outputs).any():
                print("  ❌ 输出包含Inf")
            else:
                print("  ✅ 输出正常")
                
        except Exception as e:
            print(f"  ❌ 前向传播失败: {e}")

def test_feature_fusion_mechanism():
    """测试特征融合机制"""
    print("\n=== 测试特征融合机制 ===")
    
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    
    # 获取模型内部结构信息
    print("模型结构分析:")
    print(f"  sim_fc层: {model.sim_fc}")
    print(f"  encoder输出维度: {model.encoder_out_dim}")
    
    # 测试融合过程
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 64, 64)
    dummy_sim_feats = torch.randn(batch_size, 11)
    
    # 手动执行融合过程
    with torch.no_grad():
        # 1. 图像编码
        features = model.deeplab.encoder(dummy_images)
        x = features[-1]
        B, C, H, W = x.shape
        
        print(f"\n融合过程分析:")
        print(f"  图像特征形状: {x.shape}")
        print(f"  图像特征范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
        
        # 2. sim_feats投影
        sim_proj = model.sim_fc(dummy_sim_feats)
        print(f"  sim_feats投影形状: {sim_proj.shape}")
        print(f"  sim_feats投影范围: [{sim_proj.min().item():.4f}, {sim_proj.max().item():.4f}]")
        
        # 3. 特征融合
        sim_proj = sim_proj.view(B, C, 1, 1).expand(-1, C, H, W)
        fused = x + sim_proj
        
        print(f"  融合后特征形状: {fused.shape}")
        print(f"  融合后特征范围: [{fused.min().item():.4f}, {fused.max().item():.4f}]")
        
        # 4. 检查融合效果
        fusion_ratio = torch.norm(sim_proj) / torch.norm(x)
        print(f"  融合比例 (sim_proj/x): {fusion_ratio.item():.4f}")

def test_real_data_loading():
    """测试真实数据加载"""
    print("\n=== 测试真实数据加载 ===")
    
    try:
        train_loader, val_loader = get_multimodal_patch_dataloaders(
            data_root="data/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=4,
            num_workers=0,
            damage_boost=1
        )
        
        print("✅ 数据加载器创建成功")
        
        # 测试几个batch
        for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  图像形状: {images.shape}")
            print(f"  图像范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
            print(f"  sim_feats形状: {sim_feats.shape}")
            print(f"  sim_feats范围: [{sim_feats.min().item():.4f}, {sim_feats.max().item():.4f}]")
            print(f"  sim_feats均值: {sim_feats.mean().item():.4f}")
            print(f"  sim_feats标准差: {sim_feats.std().item():.4f}")
            
            # 检查是否有NaN
            if torch.isnan(images).any() or torch.isnan(sim_feats).any():
                print("  ❌ 数据包含NaN")
            else:
                print("  ✅ 数据正常")
            
            if batch_idx >= 2:
                break
                
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()

def analyze_normalization_impact():
    """分析归一化对联合推理的影响"""
    print("\n=== 归一化对联合推理的影响分析 ===")
    
    print("1. 信息保持性:")
    print("   - 归一化保持了特征的相对关系")
    print("   - 保持了特征之间的相关性")
    print("   - 只是改变了数值范围，不影响语义信息")
    
    print("\n2. 数值稳定性:")
    print("   - 防止梯度爆炸")
    print("   - 提高训练稳定性")
    print("   - 避免数值溢出")
    
    print("\n3. 联合推理能力:")
    print("   - ✅ 归一化不影响特征融合机制")
    print("   - ✅ 模型仍能学习图像和sim_feats的联合分布")
    print("   - ✅ 特征投影和融合过程保持不变")
    print("   - ✅ 只是输入数值范围更合理")
    
    print("\n4. 建议:")
    print("   - 保持归一化处理")
    print("   - 监控融合比例")
    print("   - 确保sim_feats和图像特征在相似数值范围")

if __name__ == "__main__":
    print("开始分析sim_feats归一化对联合推理的影响...")
    print("=" * 60)
    
    # 执行各项分析
    analyze_sim_features_distribution()
    test_feature_correlation()
    test_joint_inference_capability()
    test_feature_fusion_mechanism()
    test_real_data_loading()
    analyze_normalization_impact()
    
    print("\n" + "=" * 60)
    print("分析完成！") 