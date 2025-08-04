import os
import numpy as np

mask_dir = 'data/patch_dataset/train/masks'
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

if not mask_files:
    print("未找到任何mask patch文件！")
    exit(1)

for i, fname in enumerate(mask_files[:10]):  # 检查前10个
    path = os.path.join(mask_dir, fname)
    mask = np.load(path)
    print(f"{fname}: unique={np.unique(mask)}, min={mask.min()}, max={mask.max()}, mean={mask.mean():.4f}, sum={mask.sum()}")