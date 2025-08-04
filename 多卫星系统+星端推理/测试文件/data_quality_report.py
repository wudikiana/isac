#!/usr/bin/env python3
"""
数据质量报告生成器
分析数据集中的标注不一致问题并提供解决方案
"""

import os
import numpy as np
from PIL import Image
from collections import defaultdict

def analyze_dataset_quality(data_root="data/combined_dataset"):
    """分析数据集质量"""
    print("=== 数据集质量分析报告 ===")
    
    # 检查各个数据集分割
    splits = ['train2017', 'val2017', 'test2017']
    total_stats = {
        'total_images': 0,
        'problematic_images': 0,
        'disaster_types': defaultdict(int),
        'problematic_by_disaster': defaultdict(int)
    }
    
    for split in splits:
        images_dir = os.path.join(data_root, "images", split)
        masks_dir = os.path.join(data_root, "masks", split)
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"跳过 {split} - 目录不存在")
            continue
            
        print(f"\n--- {split} 数据集分析 ---")
        
        split_stats = analyze_split(images_dir, masks_dir)
        
        # 更新总统计
        total_stats['total_images'] += split_stats['total_images']
        total_stats['problematic_images'] += split_stats['problematic_images']
        
        for disaster_type, count in split_stats['disaster_types'].items():
            total_stats['disaster_types'][disaster_type] += count
            
        for disaster_type, count in split_stats['problematic_by_disaster'].items():
            total_stats['problematic_by_disaster'][disaster_type] += count
    
    # 生成总报告
    print(f"\n=== 总体统计 ===")
    print(f"总图像数量: {total_stats['total_images']}")
    print(f"问题图像数量: {total_stats['problematic_images']}")
    print(f"问题比例: {total_stats['problematic_images']/total_stats['total_images']*100:.2f}%")
    
    print(f"\n--- 按灾害类型统计 ---")
    for disaster_type, count in sorted(total_stats['disaster_types'].items()):
        problematic_count = total_stats['problematic_by_disaster'][disaster_type]
        problem_ratio = problematic_count / count * 100 if count > 0 else 0
        print(f"{disaster_type}: {count} 张图像, {problematic_count} 张有问题 ({problem_ratio:.1f}%)")
    
    # 提供建议
    print(f"\n=== 建议 ===")
    if total_stats['problematic_images'] > 0:
        print("1. 使用 show_warnings=False 隐藏警告信息")
        print("2. 使用 skip_problematic_samples=True 跳过问题样本")
        print("3. 考虑手动检查并修复标注不一致的数据")
        print("4. 在训练时使用数据增强来平衡数据集")
    else:
        print("数据集质量良好，无需特殊处理")

def analyze_split(images_dir, masks_dir):
    """分析单个数据集分割"""
    stats = {
        'total_images': 0,
        'problematic_images': 0,
        'disaster_types': defaultdict(int),
        'problematic_by_disaster': defaultdict(int)
    }
    
    problematic_files = []
    
    for f in os.listdir(images_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            mask_name = os.path.splitext(f)[0] + "_target.png"
            mask_path = os.path.join(masks_dir, mask_name)
            
            if os.path.exists(mask_path):
                stats['total_images'] += 1
                
                # 提取灾害类型
                disaster_type = f.split('_')[0]
                stats['disaster_types'][disaster_type] += 1
                
                # 检查掩码内容
                try:
                    mask = Image.open(mask_path)
                    if mask.mode != 'L':
                        mask = mask.convert('L')
                    mask_np = np.array(mask)
                    has_damage = (mask_np >= 2).sum() > 0
                    
                    # 如果是灾后图像但没有损坏区域
                    if 'post_disaster' in f and not has_damage:
                        stats['problematic_images'] += 1
                        stats['problematic_by_disaster'][disaster_type] += 1
                        problematic_files.append(f)
                        
                except Exception as e:
                    print(f"警告：无法读取掩码 {mask_path}: {e}")
                    continue
    
    print(f"总图像数量: {stats['total_images']}")
    print(f"问题图像数量: {stats['problematic_images']}")
    if stats['total_images'] > 0:
        print(f"问题比例: {stats['problematic_images']/stats['total_images']*100:.2f}%")
    
    if problematic_files:
        print(f"前5个问题文件示例: {problematic_files[:5]}")
    
    return stats

def generate_clean_dataset_config():
    """生成清理数据集的配置"""
    print(f"\n=== 推荐的数据加载配置 ===")
    print("""
# 推荐的数据加载配置
from data_utils.data_loader import XView2SegmentationDataset

# 选项1: 隐藏警告但保留所有数据
dataset = XView2SegmentationDataset(
    images_dir="data/combined_dataset/images/train2017",
    masks_dir="data/combined_dataset/masks/train2017",
    show_warnings=False,  # 隐藏警告
    skip_problematic_samples=False  # 保留所有数据
)

# 选项2: 跳过有问题的样本
dataset = XView2SegmentationDataset(
    images_dir="data/combined_dataset/images/train2017", 
    masks_dir="data/combined_dataset/masks/train2017",
    show_warnings=True,  # 显示警告
    skip_problematic_samples=True  # 跳过问题样本
)

# 选项3: 完全静默模式
dataset = XView2SegmentationDataset(
    images_dir="data/combined_dataset/images/train2017",
    masks_dir="data/combined_dataset/masks/train2017", 
    show_warnings=False,  # 隐藏警告
    skip_problematic_samples=True  # 跳过问题样本
)
""")

if __name__ == "__main__":
    analyze_dataset_quality()
    generate_clean_dataset_config() 