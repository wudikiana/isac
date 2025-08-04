#from data_utils.data_loader import save_patches_to_disk

if __name__ == "__main__":
    import os
    import numpy as np
    from PIL import Image
    import random  # 添加random导入

    def save_patches_with_normal_samples(images_dir, masks_dir, out_dir, patch_size=64, stride=32, dilation_iter=2, patch_index_file=None, normal_ratio=0.05):
        try:
            import cv2
            from tqdm import tqdm
        except ImportError as e:
            print(f"导入错误: {e}")
            return
        
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)
        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and os.path.exists(os.path.join(masks_dir, os.path.splitext(f)[0] + "_target.png"))
        ]
        patch_idx = 0
        # 新增：打开patch索引文件
        patch_index_f = None
        if patch_index_file is not None:
            patch_index_f = open(patch_index_file, 'a')
        damage_patches = []
        normal_patches = []
        
        for img_name in tqdm(image_files, desc='离线生成patch'):
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, os.path.splitext(img_name)[0] + "_target.png")
            image = Image.open(img_path).convert('RGB').resize((1024, 1024))
            mask = Image.open(mask_path).convert('L').resize((1024, 1024), resample=Image.Resampling.NEAREST)
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            for i in range(0, 1024 - patch_size + 1, stride):
                for j in range(0, 1024 - patch_size + 1, stride):
                    img_patch = image_np[i:i+patch_size, j:j+patch_size, :]
                    mask_patch = mask_np[i:i+patch_size, j:j+patch_size]
                    
                    # 1. 创建原始二值掩码（不膨胀）
                    mask_bin = (mask_patch == 2).astype(np.uint8)  # 假设2表示缺陷

                    # 2. 判断原始小块是否正常
                    is_originally_normal = (mask_bin > 0).sum() == 0

                    # 3. 分类处理
                    if is_originally_normal:
                        # 0.2%概率保留为正常样本
                        if random.random() < 0.002:
                            # 直接使用原始掩码（不需要膨胀）
                            normal_patches.append((img_patch, mask_bin, img_name))
                        # 3%概率膨胀为损坏样本
                        elif random.random() < 0.006:
                            kernel = np.ones((3, 3), np.uint8)
                            mask_dilated = cv2.dilate(mask_bin, kernel, iterations=dilation_iter)
                            damage_patches.append((img_patch, mask_dilated, img_name))
                        # 96.8%概率直接跳过，不进行任何操作
                    else:
                        # 原本就是损坏样本 - 需要膨胀处理
                        kernel = np.ones((3, 3), np.uint8)
                        mask_dilated = cv2.dilate(mask_bin, kernel, iterations=dilation_iter)
                        damage_patches.append((img_patch, mask_dilated, img_name))
        
        # 保存损坏patch
        for img_patch, mask_dilated, img_name in damage_patches:
            patch_name = f'{patch_idx:07d}.npy'
            np.save(os.path.join(out_dir, 'images', patch_name), img_patch)
            np.save(os.path.join(out_dir, 'masks', patch_name), mask_dilated)
            if patch_index_f is not None:
                patch_index_f.write(f'{patch_name},{img_name}\n')
            patch_idx += 1
        
        # 保存所有正常patch（已经通过10%概率控制数量）
        
        for img_patch, mask_dilated, img_name in normal_patches:
            patch_name = f'{patch_idx:07d}.npy'
            np.save(os.path.join(out_dir, 'images', patch_name), img_patch)
            np.save(os.path.join(out_dir, 'masks', patch_name), mask_dilated)
            if patch_index_f is not None:
                patch_index_f.write(f'{patch_name},{img_name}\n')
            patch_idx += 1
        if patch_index_f is not None:
            patch_index_f.close()
        print(f'共生成patch数: {patch_idx}')
        print(f'  损坏patch: {len(damage_patches)}')
        print(f'  正常patch: {len(normal_patches)}')
        print(f'  正常样本比例: {len(normal_patches)/(len(damage_patches)+len(normal_patches))*100:.1f}%')

    # 生成训练集patch
    patch_index_train = "D:/patch_dataset/patch_index_train.csv"
    with open(patch_index_train, 'w') as f:
        f.write('patch_name,origin_image_name\n')
    save_patches_with_normal_samples(
        images_dir="data/combined_dataset/images/train2017",
        masks_dir="data/combined_dataset/masks/train2017",
        out_dir="D:/patch_dataset/train",
        patch_size=64,   # 可根据需要调整
        stride=32,       # 设置stride为64
        dilation_iter=2, # 保持膨胀操作以确保足够的损坏样本
        patch_index_file=patch_index_train,
        normal_ratio=0.05  # 5%的正常样本
    )
    # 生成验证集patch
    patch_index_val = "D:/patch_dataset/patch_index_val.csv"
    with open(patch_index_val, 'w') as f:
        f.write('patch_name,origin_image_name\n')
    save_patches_with_normal_samples(
        images_dir="data/combined_dataset/images/val2017",
        masks_dir="data/combined_dataset/masks/val2017",
        out_dir="D:/patch_dataset/val",
        patch_size=64,
        stride=32,       # 设置stride为64
        dilation_iter=2, # 保持膨胀操作以确保足够的损坏样本
        patch_index_file=patch_index_val,
        normal_ratio=0.05  # 5%的正常样本
    )
    print("所有patch已生成完毕！")
