#!/usr/bin/env python3
"""
检查被警告的掩码文件内容
"""

import os
import numpy as np
from PIL import Image

def check_mask_content(mask_path):
    """检查掩码文件的内容"""
    try:
        mask = Image.open(mask_path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        mask_np = np.array(mask)
        unique_values = np.unique(mask_np)
        
        print(f"掩码文件: {os.path.basename(mask_path)}")
        print(f"  形状: {mask_np.shape}")
        print(f"  唯一值: {unique_values.tolist()}")
        print(f"  损坏像素数 (>=2): {(mask_np >= 2).sum()}")
        print(f"  损坏像素数 (>=3): {(mask_np >= 3).sum()}")
        print(f"  损坏像素数 (>=4): {(mask_np >= 4).sum()}")
        print()
        
        return (mask_np >= 2).sum() == 0
    except Exception as e:
        print(f"读取掩码文件 {mask_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("=== 检查被警告的掩码文件内容 ===")
    
    # 被警告的文件列表
    warned_files = [
        "socal-fire_00001383_post_disaster.png",
        "socal-fire_00001387_post_disaster.png", 
        "socal-fire_00001390_post_disaster.png",
        "socal-fire_00001391_post_disaster.png",
        "socal-fire_00001394_post_disaster.png",
        "socal-fire_00001396_post_disaster.png",
        "socal-fire_00001398_post_disaster.png",
        "socal-fire_00001399_post_disaster.png",
        "socal-fire_00001402_post_disaster.png"
    ]
    
    masks_dir = "data/combined_dataset/masks/train2017"
    
    no_damage_count = 0
    total_count = 0
    
    for filename in warned_files:
        mask_name = filename.replace('.png', '_target.png')
        mask_path = os.path.join(masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            total_count += 1
            has_no_damage = check_mask_content(mask_path)
            if has_no_damage:
                no_damage_count += 1
        else:
            print(f"掩码文件不存在: {mask_path}")
    
    print(f"=== 统计结果 ===")
    print(f"总检查文件数: {total_count}")
    print(f"无损坏区域文件数: {no_damage_count}")
    print(f"有损坏区域文件数: {total_count - no_damage_count}")
    
    if no_damage_count > 0:
        print(f"\n结论: 确实存在 {no_damage_count} 个灾后图像没有损坏区域")
        print("这些警告是合理的，表明数据集中存在标注不一致的情况")
    else:
        print("\n结论: 所有被警告的文件都有损坏区域，可能是检查逻辑有问题")

if __name__ == "__main__":
    main() 