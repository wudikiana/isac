import os
import tarfile
import shutil
import json
from PIL import Image
from tqdm import tqdm
import random
import glob

# 配置路径
DATA_ROOT = "data"
XVIEW2_DIR = os.path.join(DATA_ROOT, "xview2")
COMBINED_DIR = os.path.join(DATA_ROOT, "combined_dataset")

# 确保目录存在
os.makedirs(XVIEW2_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)


def find_dir(root, name):
    """递归查找第一个名为 name（不区分大小写）的目录"""
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            if d.lower() == name.lower():
                return os.path.join(dirpath, d)
    return None


def process_xview2_subset(subset_path, out_img_dir, out_mask_dir):
    images_dir = find_dir(subset_path, "images")
    targets_dir = find_dir(subset_path, "targets")
    labels_dir = find_dir(subset_path, "labels")  # JSON文件通常在labels目录
    if not images_dir or not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return 0, 0
    if out_mask_dir and (not targets_dir or not os.path.exists(targets_dir)):
        print(f"Targets directory not found: {targets_dir}")
        return 0, 0
    img_files = glob.glob(os.path.join(images_dir, '**', '*.*'), recursive=True)
    img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    img_count, mask_count, json_count = 0, 0, 0
    for img_file in tqdm(img_files, desc=f"Processing {out_img_dir} images"):
        rel_path = os.path.relpath(img_file, images_dir).replace(os.sep, '_')
        new_img_name = rel_path
        dst_img_path = os.path.join(out_img_dir, new_img_name)
        try:
            img = Image.open(img_file).convert("RGB")
            img.save(dst_img_path, "JPEG", quality=90)
            img_count += 1
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
        # 复制JSON元数据文件
        if labels_dir:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            json_file = os.path.join(labels_dir, base_name + ".json")
            dst_json_path = os.path.join(out_img_dir, os.path.splitext(new_img_name)[0] + ".json")
            if os.path.exists(json_file):
                shutil.copy(json_file, dst_json_path)
                json_count += 1
        if out_mask_dir and targets_dir:
            mask_file = os.path.splitext(os.path.basename(img_file))[0] + "_target.png"
            mask_path = os.path.join(targets_dir, mask_file)
            dst_mask_path = os.path.join(out_mask_dir, f"{os.path.splitext(new_img_name)[0]}_target.png")
            if os.path.exists(mask_path):
                shutil.copy(mask_path, dst_mask_path)
                mask_count += 1
    print(f"共复制图片: {img_count}, 掩码: {mask_count}, JSON: {json_count}")
    return img_count, mask_count

def process_xview2_dataset():
    # train
    train_img_dir = os.path.join(COMBINED_DIR, "images", "train2017")
    train_mask_dir = os.path.join(COMBINED_DIR, "masks", "train2017")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    process_xview2_subset(os.path.join(XVIEW2_DIR, "train_images_labels_targets"), train_img_dir, train_mask_dir)
    # val
    val_img_dir = os.path.join(COMBINED_DIR, "images", "val2017")
    val_mask_dir = os.path.join(COMBINED_DIR, "masks", "val2017")
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    process_xview2_subset(os.path.join(XVIEW2_DIR, "hold_images_labels_targets"), val_img_dir, val_mask_dir)
    # test
    test_img_dir = os.path.join(COMBINED_DIR, "images", "test2017")
    test_mask_dir = os.path.join(COMBINED_DIR, "masks", "test2017")
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    process_xview2_subset(os.path.join(XVIEW2_DIR, "test_images_labels_targets"), test_img_dir, test_mask_dir)
    # tier3 只做推理
    tier3_img_dir = os.path.join(COMBINED_DIR, "images", "tier3")
    os.makedirs(tier3_img_dir, exist_ok=True)
    process_xview2_subset(os.path.join(XVIEW2_DIR, "tier3"), tier3_img_dir, None)
    print("xView2 dataset processed. All splits ready.")

def main():
    print("Step 1: Process xView2 data...")
    process_xview2_dataset()
    print("All done! Combined dataset ready at data/combined_dataset/")

if __name__ == "__main__":
    main()
