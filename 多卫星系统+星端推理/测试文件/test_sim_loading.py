#!/usr/bin/env python3
"""
测试仿真特征加载功能
"""

from data_utils.data_loader import load_sim_features

def test_sim_loading():
    """测试仿真特征加载"""
    print("=== 测试仿真特征加载 ===")
    
    # 加载仿真特征
    features = load_sim_features('data/sim_features.csv')
    print(f"加载了 {len(features)} 个仿真特征")
    
    if features:
        # 显示第一个特征
        first_key = list(features.keys())[0]
        first_feature = features[first_key]
        print(f"第一个特征文件: {first_key}")
        print(f"特征维度: {len(first_feature)}")
        print(f"特征值: {first_feature}")
        
        # 检查是否有-1值
        has_negative_ones = any(-1 in feature for feature in features.values())
        print(f"包含-1值: {has_negative_ones}")
        
        # 统计特征质量
        complete_features = sum(1 for feature in features.values() if -1 not in feature)
        print(f"完整特征数: {complete_features}")
        print(f"完整率: {complete_features/len(features)*100:.2f}%")
    else:
        print("未加载到仿真特征")

if __name__ == "__main__":
    test_sim_loading() 