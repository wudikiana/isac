import os
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2
from tqdm import tqdm
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ====== 修复后的AdvancedAugmentation类 ======
class AdvancedAugmentation:
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.train_transform = A.Compose([
            # 基础几何变换
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.4),
            # 颜色/光照
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.CLAHE(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.Equalize(p=0.2),
            ], p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            # 噪声/模糊
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=5, p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.4),
            # 天气/环境增强
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.1),
                A.RandomRain(blur_value=2, p=0.1),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.1),
                A.RandomSunFlare(src_radius=80, p=0.1),
            ], p=0.2),
            # 格式转换
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    def __call__(self, image, mask):
        if self.is_training:
            augmented = self.train_transform(image=image, mask=mask)
        else:
            augmented = self.val_transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

# ====== 修复后的DamageAwareDataset类 ======
class DamageAwareDataset(Dataset):
    def __init__(self, base_dataset, damage_boost=2, damage_threshold=1, normal_ratio=0.05):
        self.base_dataset = base_dataset
        self.damage_indices = []
        self.normal_indices = []
        self.damage_threshold = damage_threshold
        self.normal_ratio = normal_ratio  # 正常样本比例，默认5%
        
        for i in tqdm(range(len(base_dataset)), desc="Scanning damage samples"):
            _, mask, *_ = base_dataset[i]
            mask_sum = mask.sum().item()
            if mask_sum > self.damage_threshold:
                self.damage_indices.append(i)
            else:
                self.normal_indices.append(i)
        
        # 如果没有找到正常样本，创建一些正常样本
        if len(self.normal_indices) == 0:
            print("警告: 没有找到正常样本，将创建合成正常样本")
            # 从损坏样本中随机选择一些，将其掩码清零作为正常样本
            num_normal_needed = max(1, int(len(self.damage_indices) * self.normal_ratio))
            selected_indices = random.sample(self.damage_indices, min(num_normal_needed, len(self.damage_indices)))
            self.normal_indices = selected_indices
            print(f"创建了 {len(self.normal_indices)} 个合成正常样本")
        
        # 确保正常样本数量符合比例要求
        target_normal_count = int(len(self.damage_indices) * self.normal_ratio)
        if len(self.normal_indices) > target_normal_count:
            # 如果正常样本太多，随机选择
            self.normal_indices = random.sample(self.normal_indices, target_normal_count)
        elif len(self.normal_indices) < target_normal_count:
            # 如果正常样本太少，从损坏样本中复制一些并清零掩码
            additional_needed = target_normal_count - len(self.normal_indices)
            additional_indices = random.sample(self.damage_indices, min(additional_needed, len(self.damage_indices)))
            self.normal_indices.extend(additional_indices)
        
        self.damage_boost = damage_boost
        print(f"Found {len(self.damage_indices)} damage samples and {len(self.normal_indices)} normal samples")
        print(f"Normal sample ratio: {len(self.normal_indices) / (len(self.damage_indices) + len(self.normal_indices)) * 100:.1f}%")
        print(f"Total samples after boosting: {len(self)}")

    def __len__(self):
        return len(self.normal_indices) + len(self.damage_indices) * self.damage_boost

    def __getitem__(self, idx):
        if idx < len(self.damage_indices) * self.damage_boost:
            # 损坏样本
            damage_idx = idx % len(self.damage_indices)
            return self.base_dataset[self.damage_indices[damage_idx]]
        else:
            # 正常样本
            normal_idx = (idx - len(self.damage_indices) * self.damage_boost) % len(self.normal_indices)
            img, mask, sim_feat = self.base_dataset[self.normal_indices[normal_idx]]
            
            # 对于正常样本，将掩码清零（确保没有损坏区域）
            if mask.sum() > 0:
                mask = torch.zeros_like(mask)
            
            return img, mask, sim_feat

class YOLOLandslideDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, disaster_class_ids=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.disaster_class_ids = disaster_class_ids or [
            0, 1, 2, 3, 5, 6, 7, 8, 10, 24, 25, 27, 28, 29, 33, 
            44, 56, 57, 58, 59, 60, 62, 63, 67, 73
        ]
        self.image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        if not self.image_files:
            self.image_files = glob.glob(os.path.join(images_dir, "*.jpeg")) + \
                             glob.glob(os.path.join(images_dir, "*.png"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), 
                           color=(random.randint(0, 255), 
                                  random.randint(0, 255), 
                                  random.randint(0, 255)))
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(parts[0])
                            if class_id in self.disaster_class_ids:
                                label = 1
                                break
                        except ValueError:
                            continue
        if self.transform:
            img = self.transform(img)
        return img, label

class CombinedLandslideDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(images_dir, '*.*'))
        self.image_files = [f for f in self.image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)
        label = 0
        if "xview2" in base_name.lower() and "post" in base_name.lower():
            label = 1
        elif "xview2" in base_name.lower() and "pre" in base_name.lower():
            label = 0
        else:
            label_path = os.path.join(self.labels_dir, os.path.splitext(base_name)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            label = 1
                            break
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        if self.transform:
            img = self.transform(img)
        return img, label

def get_segmentation_dataloaders(data_root="data/combined_dataset", batch_size=4, num_workers=8, show_warnings=False, skip_problematic_samples=False):
    """
    获取分割任务的数据加载器
    返回: (train_loader, val_loader, test_loader)
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        show_warnings: 是否显示数据质量警告
        skip_problematic_samples: 是否跳过有问题的样本
    """
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集 - 应用选项1：隐藏警告但保留所有数据
    train_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "train2017"),
        os.path.join(data_root, "masks", "train2017"),
        transform=train_transform,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )
    val_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "val2017"),
        os.path.join(data_root, "masks", "val2017"), 
        transform=val_transform,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )
    test_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "test2017"),
        os.path.join(data_root, "masks", "test2017"),
        transform=val_transform,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

def get_landslide_dataloaders(data_root="data/combined_dataset", batch_size=4):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def make_loader(split):
        images_dir = os.path.join(data_root, "images", split)
        labels_dir = os.path.join(data_root, "labels", split)
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"警告: {split} 目录不存在: {images_dir} 或 {labels_dir}")
            return None
        dataset = CombinedLandslideDataset(images_dir, labels_dir, transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train2017"), num_workers=8, pin_memory=True)
    return make_loader("train2017"), make_loader("val2017"), make_loader("test2017")

def get_calibration_loader(data_root="data/combined_dataset", batch_size=32, num_samples=100):
    train_images_dir = os.path.join(data_root, "images", "train2017")
    train_labels_dir = os.path.join(data_root, "labels", "train2017")
    dataset = CombinedLandslideDataset(train_images_dir, train_labels_dir, transform=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    indices = random.sample(range(len(dataset)), num_samples)
    calib_subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

def load_sim_features(sim_feature_csv='data/sim_features.csv', normalize=True):
    """
    加载并归一化仿真特征
    Args:
        sim_feature_csv: 仿真特征CSV文件路径
        normalize: 是否进行归一化处理
    """
    sim_dict = {}
    all_features = []  # 用于计算归一化参数
    
    try:
        with open(sim_feature_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 数值型特征
                float_feats = []
                for col in ['comm_snr', 'radar_feat', 'radar_max', 'radar_std', 
                           'radar_peak_idx', 'path_loss', 'shadow_fading', 
                           'rain_attenuation', 'target_rcs', 'bandwidth', 'ber']:
                    try:
                        value = float(row[col])
                        # 处理异常值
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0
                        float_feats.append(value)
                    except (ValueError, KeyError):
                        float_feats.append(0.0)
                
                # 字符串型特征
                str_feats = [row.get('channel_type', ''), row.get('modulation', '')]
                sim_dict[row['img_path']] = (float_feats, str_feats)
                all_features.append(float_feats)
        
        # 归一化处理
        if normalize and all_features:
            all_features = np.array(all_features)
            # 计算每个特征的统计信息
            feature_means = np.mean(all_features, axis=0)
            feature_stds = np.std(all_features, axis=0)
            
            # 避免除零
            feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
            
            # 归一化所有特征
            for img_path, (float_feats, str_feats) in sim_dict.items():
                float_feats = np.array(float_feats)
                normalized_feats = (float_feats - feature_means) / feature_stds
                sim_dict[img_path] = (normalized_feats.tolist(), str_feats)
            
            print(f"仿真特征归一化完成，均值: {feature_means}, 标准差: {feature_stds}")
        
        print(f"成功加载 {len(sim_dict)} 个仿真特征")
        
    except FileNotFoundError:
        print(f"警告: 仿真特征文件 {sim_feature_csv} 不存在")
    except Exception as e:
        print(f"加载仿真特征时出错: {e}")
    
    return sim_dict

def process_xview2_mask(mask_tensor, damage_level='all'):
    """
    增强的xView2掩码处理函数
    Args:
        mask_tensor: 输入掩码张量
        damage_level: 损坏级别处理方式
            'all': 所有损坏级别(2,3,4)都标记为1
            'light': 轻微损坏(2)标记为0.3，中等(3)标记为0.6，严重(4)标记为1.0
            'binary': 轻微(2)标记为0，中等和严重(3,4)标记为1
            'multi': 轻微(2)标记为1，中等(3)标记为2，严重(4)标记为3
    """
    if damage_level == 'all':
        # 原始行为：所有损坏级别都标记为1
        return (mask_tensor >= 2).float()
    elif damage_level == 'light':
        # 根据损坏程度分配不同权重
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 2] = 0.3  # 轻微损坏
        result[mask_tensor == 3] = 0.6  # 中等损坏
        result[mask_tensor == 4] = 1.0  # 严重损坏
        return result
    elif damage_level == 'binary':
        # 轻微损坏不算损坏，只有中等和严重损坏才算
        return (mask_tensor >= 3).float()
    elif damage_level == 'multi':
        # 多级分类：轻微=1，中等=2，严重=3
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 2] = 1.0  # 轻微损坏
        result[mask_tensor == 3] = 2.0  # 中等损坏
        result[mask_tensor == 4] = 3.0  # 严重损坏
        return result
    else:
        # 默认行为
        return (mask_tensor >= 2).float()

class XView2SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, sim_feature_dict=None, transform=None, mask_transform=None, 
                 damage_sample_txts=None, damage_prob=0.7, is_training=True, damage_level='all',
                 show_warnings=True, skip_problematic_samples=False):
        """
        增强的xView2数据集加载器
        Args:
            damage_level: 损坏级别处理方式，与process_xview2_mask函数一致
            show_warnings: 是否显示数据质量警告
            skip_problematic_samples: 是否跳过有问题的样本（灾后图像无损坏区域）
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.sim_feature_dict = sim_feature_dict
        self.transform = transform
        self.mask_transform = mask_transform
        self.is_training = is_training
        self.damage_level = damage_level
        self.show_warnings = show_warnings
        self.skip_problematic_samples = skip_problematic_samples
        
        # 智能文件筛选：根据文件名判断是否为灾后图像
        self.image_files = []
        problematic_files = []
        
        for f in os.listdir(images_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_name = os.path.splitext(f)[0] + "_target.png"
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(mask_path):
                    # 检查掩码内容，确保灾后图像确实包含损坏区域
                    try:
                        mask = Image.open(mask_path)
                        if mask.mode != 'L':
                            mask = mask.convert('L')
                        mask_np = np.array(mask)
                        has_damage = (mask_np >= 2).sum() > 0
                        
                        # 如果是灾后图像但没有损坏区域，记录警告
                        if 'post_disaster' in f and not has_damage:
                            if self.show_warnings:
                                print(f"警告：灾后图像 {f} 未检测到损坏区域")
                            problematic_files.append(f)
                            # 如果选择跳过有问题的样本，则不添加到文件列表
                            if not self.skip_problematic_samples:
                                self.image_files.append(f)
                        else:
                            self.image_files.append(f)
                    except Exception as e:
                        if self.show_warnings:
                            print(f"警告：无法读取掩码 {mask_path}: {e}")
                        continue
        
        if self.skip_problematic_samples and problematic_files:
            print(f"已跳过 {len(problematic_files)} 个有问题的样本")
        
        self.use_weighted_sampling = False
        if damage_sample_txts is not None:
            has_damage_txt, no_damage_txt = damage_sample_txts
            with open(has_damage_txt) as f:
                self.has_damage = [line.strip() for line in f if line.strip()]
            with open(no_damage_txt) as f:
                self.no_damage = [line.strip() for line in f if line.strip()]
            self.use_weighted_sampling = True
            self.damage_prob = damage_prob
            self.length = len(self.has_damage) + len(self.no_damage)
        else:
            self.length = len(self.image_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            if self.use_weighted_sampling:
                found_damage = False
                for _ in range(10):
                    if random.random() < 0.85 and len(self.has_damage) > 0:
                        mask_name = random.choice(self.has_damage)
                    else:
                        mask_name = random.choice(self.no_damage)
                    img_name = mask_name.replace('_target', '')
                    img_path = os.path.join(self.images_dir, img_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        mask = Image.open(mask_path)
                        if mask.mode != 'L':
                            mask = mask.convert('L')
                        mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                        mask_np = np.array(mask)
                        if (mask_np >= 2).sum() > 0:
                            found_damage = True
                            break
                if not found_damage:
                    mask_name = random.choice(self.no_damage)
                    img_name = mask_name.replace('_target', '')
                    img_path = os.path.join(self.images_dir, img_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    mask = Image.open(mask_path)
                    if mask.mode != 'L':
                        mask = mask.convert('L')
                    mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                    mask_np = np.array(mask)
            else:
                img_name = self.image_files[idx]
                mask_name = os.path.splitext(img_name)[0] + "_target.png"
                img_path = os.path.join(self.images_dir, img_name)
                mask_path = os.path.join(self.masks_dir, mask_name)
                mask = Image.open(mask_path)
                if mask.mode != 'L':
                    mask = mask.convert('L')
                mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                mask_np = np.array(mask)
            image = Image.open(img_path).convert('RGB')
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_tensor = T.ToTensor()(mask)
            mask_tensor = process_xview2_mask(mask_tensor, self.damage_level)
            if self.transform:
                if 'albumentations' in str(type(self.transform)):
                    augmented = self.transform(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']
                else:
                    image = self.transform(image)
            sim_feat_tensor = torch.zeros(11)
            str_feats = ["", ""]
            if hasattr(self, 'sim_feature_dict') and self.sim_feature_dict is not None:
                key = os.path.basename(img_path)
                if key in self.sim_feature_dict:
                    sim_feats = self.sim_feature_dict[key]
                    sim_feat_tensor = torch.tensor(sim_feats[:11], dtype=torch.float32)
                    str_feats = sim_feats[11:]
            return image, mask_tensor, sim_feat_tensor, str_feats
        except Exception as e:
            print(f"[警告] 加载样本 {idx} 时出错: {e}, 自动跳过，尝试下一个样本。")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

class PatchSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不一致"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        image = np.load(img_path)
        mask = np.load(mask_path)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if image.ndim == 3 and image.shape[0] != 1 and image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, mask_tensor

def get_patch_dataloaders(data_root="data/patch_dataset", batch_size=4, num_workers=8):
    train_images_dir = os.path.join(data_root, "train/images")
    train_masks_dir = os.path.join(data_root, "train/masks")
    val_images_dir = os.path.join(data_root, "val/images")
    val_masks_dir = os.path.join(data_root, "val/masks")
    train_dataset = PatchSegmentationDataset(train_images_dir, train_masks_dir)
    val_dataset = PatchSegmentationDataset(val_images_dir, val_masks_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

# 修复MultiModalPatchSegmentationDataset，保证image和sim_feats归一化
class MultiModalPatchSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, patch_index_csv, sim_feature_dict, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.sim_feature_dict = sim_feature_dict
        self.patch2img = {}
        with open(patch_index_csv, 'r') as f:
            next(f)
            for line in f:
                patch, img = line.strip().split(',')
                self.patch2img[patch] = img
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不一致"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        try:
            img_patch_name = self.image_files[idx]
            mask_patch_name = self.mask_files[idx]
            img_patch_path = os.path.join(self.images_dir, img_patch_name)
            mask_patch_path = os.path.join(self.masks_dir, mask_patch_name)
            image = np.load(img_patch_path)
            mask = np.load(mask_patch_path)
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if mask.dtype != np.float32:
                mask = mask.astype(np.float32)
            if image.ndim == 3 and image.shape[0] != 1 and image.shape[0] != 3:
                image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.from_numpy(image)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            origin_img = self.patch2img.get(img_patch_name, None)
            if origin_img is not None and self.sim_feature_dict is not None:
                sim_feats = self.sim_feature_dict.get(origin_img, np.zeros(11, dtype=np.float32))
            else:
                sim_feats = np.zeros(11, dtype=np.float32)
            sim_feats = np.array(sim_feats, dtype=np.float32)
            if np.std(sim_feats) > 0:
                sim_feats = (sim_feats - np.mean(sim_feats)) / np.std(sim_feats)
            sim_feats_tensor = torch.tensor(sim_feats, dtype=torch.float32)
            # 统一归一化处理
            if self.transform is not None:
                image_tensor = self.transform(image_tensor)
            else:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = (image_tensor - mean) / std
            return image_tensor, mask_tensor, sim_feats_tensor
        except Exception as e:
            print(f"[警告] 加载patch样本 {idx} 时出错: {e}, 自动跳过，尝试下一个样本。")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

class AugmentedDataset(Dataset):
    def __init__(self, dataset, augment_fn):
        self.dataset = dataset
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, sim_feat = self.dataset[idx]
        
        # 反归一化到0-255范围，因为albumentations期望uint8输入
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img * 255.0
        img = torch.clamp(img, 0, 255)
        
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        mask = mask.squeeze().numpy()

        # 增强前检查
        if np.isnan(img).any() or np.isinf(img).any():
            print(f"[调试] 增强前img异常 idx={idx} nan={np.isnan(img).sum()} inf={np.isinf(img).sum()} uniq={np.unique(img)}")
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(mask).any() or np.isinf(mask).any():
            print(f"[调试] 增强前mask异常 idx={idx} nan={np.isnan(mask).sum()} inf={np.isinf(mask).sum()} uniq={np.unique(mask)}")
            mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        img, mask = self.augment_fn(img, mask)

        # 增强后检查
        if np.isnan(img).any() or np.isinf(img).any():
            print(f"[调试] 增强后img异常 idx={idx} nan={np.isnan(img).sum()} inf={np.isinf(img).sum()} uniq={np.unique(img)}")
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(mask).any() or np.isinf(mask).any():
            print(f"[调试] 增强后mask异常 idx={idx} nan={np.isnan(mask).sum()} inf={np.isinf(mask).sum()} uniq={np.unique(mask)}")
            mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        mask = np.clip(mask, 0, 1)
        return img, mask, sim_feat

# ====== 优化的collate函数 ======
def optimized_collate(batch):
    """优化的collate函数，确保tensor维度一致"""
    images, masks, feats = zip(*batch)
    
    # 确保所有图像tensor维度一致 [B, C, H, W]
    images = list(images)
    for i, img in enumerate(images):
        if img.dim() == 2:
            images[i] = img.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif img.dim() == 3 and img.shape[0] not in [1, 3]:
            images[i] = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    # 确保所有mask tensor维度一致 [B, 1, H, W]
    masks = list(masks)
    for i, mask in enumerate(masks):
        if mask.dim() == 2:
            masks[i] = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif mask.dim() == 1:
            masks[i] = mask.unsqueeze(0).unsqueeze(0)  # [H] -> [1, 1, H]
    
    # 确保所有sim_feats tensor维度一致 [B, 11]
    feats = list(feats)
    for i, sim_feat in enumerate(feats):
        if sim_feat.dim() == 0:
            feats[i] = sim_feat.unsqueeze(0)  # [] -> [1]
    
    # 使用torch.stack进行批处理
    try:
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        feats = torch.stack(feats, dim=0)
    except Exception as e:
        print(f"Collate错误: {e}")
        print(f"图像形状: {[img.shape for img in images]}")
        print(f"掩码形状: {[mask.shape for mask in masks]}")
        print(f"特征形状: {[sim_feat.shape for sim_feat in feats]}")
        raise e
    
    return images, masks, feats

def custom_collate_fn(batch):
    images = torch.stack([x[0] for x in batch])
    masks = torch.stack([x[1] for x in batch])
    feats = torch.stack([x[2] for x in batch])
    return images, masks, feats

def get_multimodal_patch_dataloaders(data_root="data/patch_dataset", 
                                    sim_feature_csv="data/sim_features.csv", 
                                    batch_size=4, 
                                    num_workers=12,
                                    damage_boost=2,
                                    normal_ratio=0.05):
    sim_feature_dict = load_sim_features(sim_feature_csv)
    # 定义ImageNet归一化transform
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def val_transform(img_tensor):
        return (img_tensor - mean) / std
    train_dataset = MultiModalPatchSegmentationDataset(
        os.path.join(data_root, "train/images"),
        os.path.join(data_root, "train/masks"),
        os.path.join(data_root, "patch_index_train.csv"),
        sim_feature_dict,
        transform=None
    )
    val_dataset = MultiModalPatchSegmentationDataset(
        os.path.join(data_root, "val/images"),
        os.path.join(data_root, "val/masks"),
        os.path.join(data_root, "patch_index_val.csv"),
        sim_feature_dict,
        transform=val_transform
    )
    aug = AdvancedAugmentation(is_training=True)
    train_dataset = DamageAwareDataset(train_dataset, damage_boost=damage_boost, damage_threshold=1, normal_ratio=normal_ratio)
    train_dataset = AugmentedDataset(train_dataset, aug)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=optimized_collate # 使用优化的collate函数
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn # 使用自定义的collate函数
    )
    return train_loader, val_loader

def analyze_damage_distribution(data_root="data/combined_dataset"):
    """
    分析数据集中的损坏分布，为模型训练提供建议
    """
    import os
    from collections import Counter
    
    print("=== 数据集损坏分布分析 ===")
    
    # 分析训练集
    train_masks_dir = os.path.join(data_root, "masks", "train2017")
    if os.path.exists(train_masks_dir):
        damage_counts = {'no_damage': 0, 'light': 0, 'medium': 0, 'severe': 0, 'total': 0}
        
        for mask_file in os.listdir(train_masks_dir):
            if mask_file.endswith('_target.png'):
                mask_path = os.path.join(train_masks_dir, mask_file)
                try:
                    mask = Image.open(mask_path)
                    if mask.mode != 'L':
                        mask = mask.convert('L')
                    mask_np = np.array(mask)
                    
                    # 统计不同损坏级别
                    light_damage = np.sum(mask_np == 2)
                    medium_damage = np.sum(mask_np == 3)
                    severe_damage = np.sum(mask_np == 4)
                    total_damage = light_damage + medium_damage + severe_damage
                    
                    if total_damage == 0:
                        damage_counts['no_damage'] += 1
                    elif severe_damage > 0:
                        damage_counts['severe'] += 1
                    elif medium_damage > 0:
                        damage_counts['medium'] += 1
                    else:
                        damage_counts['light'] += 1
                    
                    damage_counts['total'] += 1
                    
                except Exception as e:
                    print(f"处理掩码 {mask_file} 时出错: {e}")
        
        print(f"训练集统计:")
        print(f"  总样本数: {damage_counts['total']}")
        print(f"  无损坏样本: {damage_counts['no_damage']} ({damage_counts['no_damage']/damage_counts['total']*100:.1f}%)")
        print(f"  轻微损坏样本: {damage_counts['light']} ({damage_counts['light']/damage_counts['total']*100:.1f}%)")
        print(f"  中等损坏样本: {damage_counts['medium']} ({damage_counts['medium']/damage_counts['total']*100:.1f}%)")
        print(f"  严重损坏样本: {damage_counts['severe']} ({damage_counts['severe']/damage_counts['total']*100:.1f}%)")
        
        # 建议
        damage_ratio = (damage_counts['light'] + damage_counts['medium'] + damage_counts['severe']) / damage_counts['total']
        print(f"\n=== 训练建议 ===")
        print(f"当前损坏样本比例: {damage_ratio*100:.1f}%")
        
        if damage_ratio < 0.1:
            print("建议: 损坏样本过少，建议使用 damage_boost=3-5, normal_ratio=0.1")
        elif damage_ratio < 0.3:
            print("建议: 损坏样本较少，建议使用 damage_boost=2-3, normal_ratio=0.15")
        elif damage_ratio < 0.5:
            print("建议: 损坏样本适中，建议使用 damage_boost=1-2, normal_ratio=0.2")
        else:
            print("建议: 损坏样本充足，建议使用 damage_boost=1, normal_ratio=0.25")
        
        # 根据损坏级别建议掩码处理方式
        if damage_counts['light'] > damage_counts['medium'] + damage_counts['severe']:
            print("建议掩码处理方式: 'light' (轻微损坏占主导)")
        elif damage_counts['severe'] > damage_counts['light'] + damage_counts['medium']:
            print("建议掩码处理方式: 'binary' (严重损坏占主导)")
        else:
            print("建议掩码处理方式: 'all' (损坏级别分布均匀)")
    
    return damage_counts

def get_optimal_training_params(data_root="data/combined_dataset"):
    """
    获取最优的训练参数建议
    """
    damage_counts = analyze_damage_distribution(data_root)
    
    # 根据损坏分布计算最优参数
    damage_ratio = (damage_counts['light'] + damage_counts['medium'] + damage_counts['severe']) / damage_counts['total']
    
    if damage_ratio < 0.1:
        return {
            'damage_boost': 4,
            'normal_ratio': 0.1,
            'damage_level': 'all',
            'batch_size': 8,
            'learning_rate': 1e-4
        }
    elif damage_ratio < 0.3:
        return {
            'damage_boost': 3,
            'normal_ratio': 0.15,
            'damage_level': 'all',
            'batch_size': 12,
            'learning_rate': 1e-4
        }
    elif damage_ratio < 0.5:
        return {
            'damage_boost': 2,
            'normal_ratio': 0.2,
            'damage_level': 'light',
            'batch_size': 16,
            'learning_rate': 2e-4
        }
    else:
        return {
            'damage_boost': 1,
            'normal_ratio': 0.25,
            'damage_level': 'light',
            'batch_size': 20,
            'learning_rate': 3e-4
        }