import os
import numpy as np

def check_patch_dataset():
    # 检查训练集
    train_images = os.listdir('data/patch_dataset/train/images')
    train_masks = os.listdir('data/patch_dataset/train/masks')
    
    # 检查验证集
    val_images = os.listdir('data/patch_dataset/val/images')
    val_masks = os.listdir('data/patch_dataset/val/masks')
    
    print(f'训练集: {len(train_images)} 图像, {len(train_masks)} 掩码')
    print(f'验证集: {len(val_images)} 图像, {len(val_masks)} 掩码')
    
    # 详细检查前几个样本
    print("\n=== 详细检查前5个训练样本 ===")
    for i in range(min(5, len(train_masks))):
        mask = np.load(os.path.join('data/patch_dataset/train/masks', train_masks[i]))
        print(f'样本 {i}: mask形状={mask.shape}, 唯一值={np.unique(mask)}, 总和={mask.sum()}')
    
    print("\n=== 详细检查前5个验证样本 ===")
    for i in range(min(5, len(val_masks))):
        mask = np.load(os.path.join('data/patch_dataset/val/masks', val_masks[i]))
        print(f'样本 {i}: mask形状={mask.shape}, 唯一值={np.unique(mask)}, 总和={mask.sum()}')
    
    # 检查正常样本比例
    train_normal_count = 0
    val_normal_count = 0
    
    # 检查训练集正常样本
    for f in train_masks[:100]:  # 只检查前100个样本
        mask = np.load(os.path.join('data/patch_dataset/train/masks', f))
        if mask.sum() == 0:
            train_normal_count += 1
    
    # 检查验证集正常样本
    for f in val_masks[:100]:  # 只检查前100个样本
        mask = np.load(os.path.join('data/patch_dataset/val/masks', f))
        if mask.sum() == 0:
            val_normal_count += 1
    
    print(f'\n训练集正常样本比例 (前100个): {train_normal_count/100*100:.1f}%')
    print(f'验证集正常样本比例 (前100个): {val_normal_count/100*100:.1f}%')
    
    # 检查patch索引文件
    if os.path.exists('data/patch_dataset/patch_index_train.csv'):
        with open('data/patch_dataset/patch_index_train.csv', 'r') as f:
            lines = f.readlines()
        print(f'训练集索引文件: {len(lines)-1} 行 (不包括标题)')
    
    if os.path.exists('data/patch_dataset/patch_index_val.csv'):
        with open('data/patch_dataset/patch_index_val.csv', 'r') as f:
            lines = f.readlines()
        print(f'验证集索引文件: {len(lines)-1} 行 (不包括标题)')

if __name__ == "__main__":
    check_patch_dataset() 