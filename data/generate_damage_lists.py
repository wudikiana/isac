import os
from PIL import Image
import numpy as np

# 支持自动遍历所有子集
masks_root = 'data/combined_dataset/masks'
subsets = [d for d in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, d))]

for subset in subsets:
    masks_dir = os.path.join(masks_root, subset)
    has_damage = []
    no_damage = []
    for fname in os.listdir(masks_dir):
        if not fname.endswith('.png'):
            continue
        mask_path = os.path.join(masks_dir, fname)
        mask = Image.open(mask_path).convert('L').resize((256, 256), resample=Image.Resampling.NEAREST)
        mask_np = np.array(mask)
        if (mask_np == 2).sum() > 0:
            has_damage.append(fname)
        else:
            no_damage.append(fname)
    out_has = f'data/combined_dataset/{subset}_has_damage.txt'
    out_no = f'data/combined_dataset/{subset}_no_damage.txt'
    with open(out_has, 'w') as f:
        for fname in has_damage:
            f.write(fname + '\n')
    with open(out_no, 'w') as f:
        for fname in no_damage:
            f.write(fname + '\n')
    print(f'{subset}: 有损坏区域: {len(has_damage)}，无损坏区域: {len(no_damage)}') 