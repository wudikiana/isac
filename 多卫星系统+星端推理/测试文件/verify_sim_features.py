#!/usr/bin/env python3
"""
验证仿真特征数据质量
"""

import pandas as pd
import numpy as np

def verify_sim_features(file_path='data/sim_features.csv'):
    """验证仿真特征数据"""
    print("=== 仿真特征数据验证 ===")
    
    # 读取数据
    df = pd.read_csv(file_path)
    print(f"总行数: {len(df)}")
    
    # 检查完整特征
    complete_features = (df != -1).all(axis=1).sum()
    missing_features = (df == -1).any(axis=1).sum()
    
    print(f"完整特征数: {complete_features}")
    print(f"缺失特征数: {missing_features}")
    print(f"完整率: {complete_features/len(df)*100:.2f}%")
    
    # 检查路径格式
    print("\n=== 路径格式检查 ===")
    sample_paths = df['img_path'].head(5).tolist()
    for i, path in enumerate(sample_paths, 1):
        print(f"{i}. {path}")
    
    # 检查数值特征范围
    print("\n=== 数值特征范围检查 ===")
    numeric_cols = ['comm_snr', 'radar_feat', 'radar_max', 'radar_std', 
                   'radar_peak_idx', 'path_loss', 'shadow_fading', 
                   'rain_attenuation', 'target_rcs', 'bandwidth', 'ber']
    
    for col in numeric_cols:
        if col in df.columns:
            values = df[col]
            print(f"{col}:")
            print(f"  范围: [{values.min():.4f}, {values.max():.4f}]")
            print(f"  均值: {values.mean():.4f}")
            print(f"  标准差: {values.std():.4f}")
    
    # 检查字符串特征
    print("\n=== 字符串特征检查 ===")
    string_cols = ['channel_type', 'modulation']
    for col in string_cols:
        if col in df.columns:
            unique_values = df[col].unique()
            print(f"{col}: {unique_values}")
    
    # 检查tier3目录
    tier3_count = df['img_path'].str.contains('tier3').sum()
    print(f"\n=== 目录分布 ===")
    print(f"tier3目录文件数: {tier3_count}")
    
    # 检查各目录文件数
    for split in ['train2017', 'val2017', 'test2017', 'tier3']:
        count = df['img_path'].str.contains(split).sum()
        print(f"{split}: {count} 个文件")

if __name__ == "__main__":
    verify_sim_features() 