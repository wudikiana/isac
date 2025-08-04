import os
import numpy as np
from PIL import Image
import cv2

def check_mask_dilation():
    """检查掩码膨胀操作的影响"""
    
    # 检查原始掩码
    print("=== 检查原始掩码 ===")
    train_masks_dir = "data/combined_dataset/masks/train2017"
    mask_files = [f for f in os.listdir(train_masks_dir) if f.endswith('_target.png')][:5]
    
    for mask_file in mask_files:
        mask_path = os.path.join(train_masks_dir, mask_file)
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        
        print(f"\n{mask_file}:")
        print(f"  原始掩码形状: {mask_np.shape}")
        print(f"  原始掩码唯一值: {np.unique(mask_np)}")
        print(f"  损坏像素数量 (值>=2): {(mask_np >= 2).sum()}")
        print(f"  损坏像素比例: {(mask_np >= 2).sum() / mask_np.size * 100:.2f}%")
        
        # 检查patch级别的损坏情况
        patch_size = 64
        stride = 64
        damage_patches = 0
        total_patches = 0
        
        for i in range(0, mask_np.shape[0] - patch_size + 1, stride):
            for j in range(0, mask_np.shape[1] - patch_size + 1, stride):
                patch = mask_np[i:i+patch_size, j:j+patch_size]
                if (patch >= 2).sum() > 0:
                    damage_patches += 1
                total_patches += 1
        
        print(f"  Patch级别损坏比例: {damage_patches/total_patches*100:.1f}% ({damage_patches}/{total_patches})")
        
        # 检查膨胀操作的影响
        print(f"\n  掩码膨胀影响分析:")
        mask_bin = (mask_np >= 2).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        
        for dilation_iter in [0, 1, 2]:
            if dilation_iter == 0:
                mask_dilated = mask_bin
            else:
                mask_dilated = cv2.dilate(mask_bin, kernel, iterations=dilation_iter)
            
            # 检查patch级别的膨胀影响
            damage_patches_after_dilation = 0
            for i in range(0, mask_dilated.shape[0] - patch_size + 1, stride):
                for j in range(0, mask_dilated.shape[1] - patch_size + 1, stride):
                    patch = mask_dilated[i:i+patch_size, j:j+patch_size]
                    if patch.sum() > 0:
                        damage_patches_after_dilation += 1
            
            print(f"    膨胀{dilation_iter}次: {damage_patches_after_dilation}/{total_patches} patches ({damage_patches_after_dilation/total_patches*100:.1f}%)")

def check_patch_generation_logic():
    """检查patch生成逻辑"""
    print("\n=== Patch生成逻辑分析 ===")
    
    # 模拟patch生成过程
    patch_size = 64
    stride = 64
    dilation_iter = 2
    
    # 创建一个简单的测试掩码
    test_mask = np.zeros((1024, 1024), dtype=np.uint8)
    # 在中心区域添加一些损坏像素
    test_mask[400:600, 400:600] = 2
    
    print(f"测试掩码损坏像素数量: {(test_mask >= 2).sum()}")
    print(f"测试掩码损坏像素比例: {(test_mask >= 2).sum() / test_mask.size * 100:.2f}%")
    
    # 模拟patch生成
    damage_patches = 0
    normal_patches = 0
    
    for i in range(0, test_mask.shape[0] - patch_size + 1, stride):
        for j in range(0, test_mask.shape[1] - patch_size + 1, stride):
            patch = test_mask[i:i+patch_size, j:j+patch_size]
            
            # 掩码膨胀
            mask_bin = (patch >= 2).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(mask_bin, kernel, iterations=dilation_iter)
            
            if mask_dilated.sum() > 0:
                damage_patches += 1
            else:
                normal_patches += 1
    
    print(f"Patch分类结果:")
    print(f"  损坏patch: {damage_patches}")
    print(f"  正常patch: {normal_patches}")
    print(f"  损坏比例: {damage_patches/(damage_patches+normal_patches)*100:.1f}%")

if __name__ == "__main__":
    check_mask_dilation()
    check_patch_generation_logic() 