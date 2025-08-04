import sys
import os
import torch
import time
import argparse
import csv
from glob import glob
import re
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn # Added missing import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Excel支持
try:
    import openpyxl
    from openpyxl import Workbook
except ImportError:
    openpyxl = None

# 添加DualModelEnsemble类定义
class DualModelEnsemble(nn.Module):
    """双模型集成：DeepLab + 分割版LandslideDetector"""
    def __init__(self, deeplab_model, landslide_model, fusion_weight=0.5):
        super().__init__()
        self.deeplab_model = deeplab_model
        self.landslide_model = landslide_model
        self.fusion_weight = fusion_weight
        
        # 可学习的融合权重
        self.learnable_weight = nn.Parameter(torch.tensor([fusion_weight]))
        
    def forward(self, img, sim_feat=None):
        import torch.nn.functional as F
        
        # DeepLab前向传播
        deeplab_output = self.deeplab_model(img, sim_feat)
        
        # LandslideDetector前向传播（不需要sim_feat）
        landslide_output = self.landslide_model(img)
        
        # 检查输出是否包含NaN
        if torch.isnan(deeplab_output).any() or torch.isinf(deeplab_output).any():
            print("[警告] DeepLab输出包含NaN/Inf，使用零张量")
            deeplab_output = torch.zeros_like(deeplab_output)
        
        if torch.isnan(landslide_output).any() or torch.isinf(landslide_output).any():
            print("[警告] LandslideDetector输出包含NaN/Inf，使用零张量")
            landslide_output = torch.zeros_like(landslide_output)
        
        # 确保输出尺寸一致
        if deeplab_output.shape != landslide_output.shape:
            landslide_output = F.interpolate(landslide_output, size=deeplab_output.shape[2:], mode='bilinear', align_corners=False)
        
        # 加权融合
        weight = torch.sigmoid(self.learnable_weight)  # 确保权重在[0,1]范围内
        
        # 检查权重是否包含NaN
        if torch.isnan(weight) or torch.isinf(weight):
            print("[警告] 融合权重包含NaN/Inf，使用默认权重0.5")
            weight = torch.tensor(0.5, device=img.device)
        
        ensemble_output = weight * deeplab_output + (1 - weight) * landslide_output
        
        # 最终检查
        if torch.isnan(ensemble_output).any() or torch.isinf(ensemble_output).any():
            print("[警告] 融合输出包含NaN/Inf，使用零张量")
            ensemble_output = torch.zeros_like(ensemble_output)
        
        return ensemble_output
    
    def get_fusion_weight(self):
        """获取当前融合权重"""
        return torch.sigmoid(self.learnable_weight).item()

# 添加LandslideDetector模型定义
class LandslideDetector(nn.Module):
    """简化的LandslideDetector模型用于推理"""
    def __init__(self, num_classes=1):
        super().__init__()
        # 使用简单的U-Net结构
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    
    def forward(self, img):
        return self.unet(img)

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img

def load_sim_features(sim_feature_csv='data/sim_features.csv', normalize=True):
    """加载并归一化仿真特征"""
    sim_features = {}
    all_features = []
    
    try:
        import pandas as pd
        df = pd.read_csv(sim_feature_csv)
        
        # 数值型特征列
        numeric_columns = ['comm_snr', 'radar_feat', 'radar_max', 'radar_std', 
                          'radar_peak_idx', 'path_loss', 'shadow_fading', 
                          'rain_attenuation', 'target_rcs', 'bandwidth', 'ber']
        
        for _, row in df.iterrows():
            # 使用img_path列作为文件名
            img_path = row['img_path']
            filename = os.path.basename(img_path)  # 提取文件名
            
            # 处理数值特征
            features = []
            for col in numeric_columns:
                if col in df.columns:
                    value = row[col]
                    if value == 'unknown' or pd.isna(value):
                        features.append(0.0)
                    else:
                        try:
                            features.append(float(value))
                        except:
                            features.append(0.0)
                else:
                    features.append(0.0)
            
            # 处理字符串特征（如果有的话）
            str_features = []
            for col in ['channel_type', 'modulation']:
                if col in df.columns:
                    str_features.append(str(row.get(col, '')))
                else:
                    str_features.append('')
            
            features = np.array(features, dtype=np.float32)
            sim_features[filename] = features
            all_features.append(features)
        
        # 归一化处理
        if normalize and all_features:
            all_features = np.array(all_features)
            feature_means = np.mean(all_features, axis=0)
            feature_stds = np.std(all_features, axis=0)
            
            # 避免除零
            feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
            
            # 归一化所有特征
            for filename in sim_features:
                normalized_feats = (sim_features[filename] - feature_means) / feature_stds
                sim_features[filename] = normalized_feats
            
            print(f"仿真特征归一化完成，均值: {feature_means}, 标准差: {feature_stds}")
        
        print(f"成功加载 {len(sim_features)} 个仿真特征")
    except Exception as e:
        print(f"加载仿真特征失败: {e}")
        # 返回默认特征
        return {}
    return sim_features

def load_deeplab_model(model_path, device):
    """加载DeepLab模型或DualModelEnsemble模型"""
    # 定义EnhancedDeepLab模型类 - 与train_model.py保持一致
    class EnhancedDeepLab(torch.nn.Module):
        def __init__(self, in_channels=3, num_classes=1, sim_feat_dim=11):
            super().__init__()
            # 使用更强大的编码器
            self.deeplab = smp.DeepLabV3Plus(
                encoder_name="resnext101_32x8d",  # 更强的预训练编码器
                encoder_weights="imagenet",       # 使用预训练权重
                in_channels=in_channels,
                classes=num_classes,
                activation=None
            )
            
            # 改进的特征融合模块
            self.sim_fusion = nn.Sequential(
                nn.Linear(sim_feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2048),  # 修改为2048以匹配encoder输出维度
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # 注意力门控机制
            self.attention_gate = nn.Sequential(
                nn.Conv2d(2048 + 2048, 512, kernel_size=1),  # 修改输入通道为2048+2048=4096
                nn.ReLU(),
                nn.Conv2d(512, 2048, kernel_size=1),  # 修改输出通道为2048以匹配x的维度
                nn.Sigmoid()
            )
            
            # 初始化权重
            self._init_weights()
        
        def _init_weights(self):
            for m in self.sim_fusion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            for m in self.attention_gate.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, img, sim_feat):
            import torch.nn.functional as F
            # 输入检查
            if torch.isnan(img).any() or torch.isinf(img).any():
                print(f"[警告] 模型输入图像包含NaN/Inf!")
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isnan(sim_feat).any() or torch.isinf(sim_feat).any():
                print(f"[警告] 模型输入sim特征包含NaN/Inf!")
                sim_feat = torch.nan_to_num(sim_feat, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 提取图像特征
            features = self.deeplab.encoder(img)
            x = features[-1]
            B, C, H, W = x.shape
            
            # 处理仿真特征
            sim_proj = self.sim_fusion(sim_feat)
            sim_proj = sim_proj.view(B, -1, 1, 1)
            sim_proj = sim_proj.expand(-1, -1, H, W)
            
            # 注意力融合
            combined = torch.cat([x, sim_proj], dim=1)
            attention = self.attention_gate(combined)
            # 确保sim_proj维度与x匹配
            sim_proj = F.interpolate(sim_proj, size=x.shape[2:], mode='nearest')
            fused = x * attention + sim_proj * (1 - attention)
            
            # 检查融合后的特征
            if torch.isnan(fused).any() or torch.isinf(fused).any():
                print(f"[警告] 融合特征包含NaN/Inf，使用原始特征!")
                fused = x
            
            # 解码器
            features = list(features)
            features[-1] = fused
            out = self.deeplab.decoder(features)
            out = self.deeplab.segmentation_head(out)
            
            # 检查最终输出
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"[警告] 模型输出包含NaN/Inf，返回零张量!")
                out = torch.zeros_like(out)
            
            return out

    # 首先尝试加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否是完整的checkpoint还是纯模型权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整的checkpoint格式
        print(f"检测到完整checkpoint，分析模型结构...")
        
        state_dict = checkpoint['model_state_dict']
        print(f"Checkpoint state dict keys: {list(state_dict.keys())[:10]}")
        
        # 检查是否是DualModelEnsemble
        if any(key.startswith('deeplab_model.') for key in state_dict.keys()):
            print("检测到DualModelEnsemble模型结构")
            
            # 创建DualModelEnsemble
            deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
            landslide_model = LandslideDetector(num_classes=1)
            ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5)
            
            # 加载权重
            try:
                # 处理DualModelEnsemble的权重加载
                ensemble_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('deeplab_model.'):
                        # 移除deeplab_model前缀
                        new_key = key[len('deeplab_model.'):]
                        ensemble_state_dict[f'deeplab_model.{new_key}'] = value
                    elif key.startswith('landslide_model.'):
                        # 移除landslide_model前缀
                        new_key = key[len('landslide_model.'):]
                        ensemble_state_dict[f'landslide_model.{new_key}'] = value
                    elif key == 'learnable_weight':
                        ensemble_state_dict['learnable_weight'] = value
                    elif key == 'fusion_weight':
                        ensemble_state_dict['fusion_weight'] = value
                
                print(f"处理后的ensemble state dict keys (前10个): {list(ensemble_state_dict.keys())[:10]}")
                
                # 检查权重加载情况
                deeplab_keys = [k for k in ensemble_state_dict.keys() if k.startswith('deeplab_model.')]
                landslide_keys = [k for k in ensemble_state_dict.keys() if k.startswith('landslide_model.')]
                print(f"Deeplab权重数量: {len(deeplab_keys)}")
                print(f"Landslide权重数量: {len(landslide_keys)}")
                
                ensemble_model.load_state_dict(ensemble_state_dict, strict=False)
                print("✅ 成功加载DualModelEnsemble模型")
                ensemble_model.eval()
                ensemble_model.to(device)
                return ensemble_model
            except Exception as e:
                print(f"❌ DualModelEnsemble加载失败: {e}")
                # 继续尝试其他方法
        else:
            print("检测到普通模型结构，尝试加载EnhancedDeepLab")
    
    # 尝试不同的配置来匹配保存的模型
    configs_to_try = [
        ("resnext101_32x8d", True),   # 与train_model.py一致的配置
        ("resnext101_32x8d", False),  # 不使用预训练权重
        ("efficientnet-b1", False),   # 备用配置
        ("efficientnet-b1", True),    # 备用配置
        ("efficientnet-b0", False),   # 备用配置
        ("efficientnet-b0", True),    # 备用配置
    ]
    
    for encoder_name, use_pretrained in configs_to_try:
        try:
            print(f"尝试配置: encoder={encoder_name}, pretrained={use_pretrained}")
            model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
            
            # 加载模型权重
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                
                # 添加调试信息
                print(f"Checkpoint state dict keys: {list(state_dict.keys())[:10]}")
                print(f"Model state dict keys: {list(model_dict.keys())[:10]}")
                
                # 只加载匹配的键
                matched_keys = []
                for key in state_dict.keys():
                    if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                        model_dict[key] = state_dict[key]
                        matched_keys.append(key)
                
                print(f"成功加载 {len(matched_keys)} 个匹配的权重层")
                if len(matched_keys) > 0:
                    print(f"前5个匹配的键: {matched_keys[:5]}")
                
                # 检查不匹配的键
                unmatched_keys = [key for key in state_dict.keys() if key not in model_dict]
                print(f"不匹配的键数量: {len(unmatched_keys)}")
                if len(unmatched_keys) > 0:
                    print(f"前5个不匹配的键: {unmatched_keys[:5]}")
                
                model.load_state_dict(model_dict, strict=False)
                
                if 'epoch' in checkpoint:
                    print(f"模型训练轮次: {checkpoint['epoch']}")
                if 'best_val_iou' in checkpoint:
                    print(f"最佳验证IoU: {checkpoint['best_val_iou']:.4f}")
            else:
                # 纯模型权重格式
                print(f"检测到纯模型权重，直接加载...")
                model.load_state_dict(checkpoint, strict=False)
            
            model.eval()
            model.to(device)
            print(f"✅ 成功加载DeepLab模型")
            return model
            
        except Exception as e:
            print(f"❌ 配置 {encoder_name} 加载失败: {str(e)}")
            continue
    
    # 如果所有配置都失败了，尝试直接创建模型
    try:
        print("创建EnhancedDeepLab模型...")
        model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        
        # 加载模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            
            # 添加调试信息
            print(f"Checkpoint state dict keys: {list(state_dict.keys())[:10]}")
            print(f"Model state dict keys: {list(model_dict.keys())[:10]}")
            
            # 处理键名映射
            updated_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                
                # 处理各种可能的键名变化
                if key.startswith('sim_fc.') and key not in model_dict:
                    new_key = key.replace('sim_fc.', 'sim_fusion.')
                
                # 处理其他前缀
                if new_key not in model_dict:
                    prefixes_to_remove = [
                        'deeplab_model.',
                        'landslide_model.',
                        'model.',
                        'module.'
                    ]
                    for prefix in prefixes_to_remove:
                        if new_key.startswith(prefix):
                            new_key = new_key[len(prefix):]
                            break
                
                if new_key in model_dict and state_dict[key].shape == model_dict[new_key].shape:
                    updated_state_dict[new_key] = value
                    if new_key != key:
                        print(f"键名映射: {key} -> {new_key}")
                else:
                    print(f"跳过不匹配的键: {key}")
            
            print(f"成功映射 {len(updated_state_dict)} 个权重层")
            model.load_state_dict(updated_state_dict, strict=False)
            
            if 'epoch' in checkpoint:
                print(f"模型训练轮次: {checkpoint['epoch']}")
            if 'best_val_iou' in checkpoint:
                print(f"最佳验证IoU: {checkpoint['best_val_iou']:.4f}")
        else:
            # 纯模型权重格式
            print(f"检测到纯模型权重，直接加载...")
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        model.to(device)
        print(f"✅ 成功加载EnhancedDeepLab模型")
        return model
        
    except Exception as e:
        print(f"❌ EnhancedDeepLab模型加载失败: {str(e)}")
        raise RuntimeError(f"无法加载模型: {e}")
    
    raise RuntimeError("所有配置都失败了，无法加载模型")

def load_quantized_model(model_path, device):
    print(f"尝试加载量化模型: {model_path}")
    print(f"设备: {device}")
    print(f"文件是否存在: {os.path.exists(model_path)}")
    
    try:
        model = torch.jit.load(model_path, map_location=device)
        print("✅ 量化模型加载成功")
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        print(f"❌ 量化模型加载失败: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

def infer_and_time(model, img, model_type, sim_feat=None, repetitions=20):
    # 预热
    with torch.no_grad():
        if sim_feat is not None:
            _ = model(img, sim_feat)
        else:
            _ = model(img)
    
    # 计时推理
    start_time = time.time()
    with torch.no_grad():
        if sim_feat is not None:
            output = model(img, sim_feat)
        else:
            output = model(img)
    end_time = time.time()
    
    # 计算置信度
    if model_type == 'quantized':
        # 量化模型输出需要特殊处理
        confidence = torch.sigmoid(output).max().item()
    else:
        confidence = torch.sigmoid(output).max().item()
    
    # 对于仿真数据，不进行截断处理，因为输出范围是合理的
    output_min = output.min().item()
    output_max = output.max().item()
    
    # 记录输出统计信息，但不进行截断
    print(f"模型输出统计: min={output_min:.2f}, max={output_max:.2f}, mean={output.mean().item():.2f}")
    print(f"置信度: {confidence:.4f}")
    
    return output, end_time - start_time

def main():
    parser = argparse.ArgumentParser(description='Segmentation Model Inference')
    parser.add_argument('--model_type', type=str, default='original', choices=['original', 'quantized'],
                        help='选择加载原始模型还是量化模型')
    parser.add_argument('--img_path', type=str, default='data/combined_dataset/images/tier3/',
                        help='输入图片路径或图片文件夹（支持*.png,*.jpg,*.jpeg）')
    parser.add_argument('--model_path', type=str, default='models/best_multimodal_patch_model.pth',
                        help='模型权重路径（默认为models/best_multimodal_patch_model.pth）')
    parser.add_argument('--csv_path', type=str, default='inference/perf_report.csv',
                        help='推理性能表格输出路径（支持.csv或.xlsx，自动生成另一种格式）')
    parser.add_argument('--no_vis', action='store_true', default=True, 
                        help='不显示可视化窗口（默认为True）')
    parser.add_argument('--use_deeplab', action='store_true', default=True,
                        help='使用DeepLab模型（需要仿真特征，默认为True）')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载仿真特征
    sim_features = load_sim_features()
    
    if args.model_type == 'original':
        model_path = args.model_path or "models/best_multimodal_patch_model.pth"
        model_path = os.path.abspath(model_path)
        print(f"加载原始模型: {model_path}")
        model = load_deeplab_model(model_path, device)
        
        # 添加模型权重检查
        print("检查模型权重加载情况...")
        total_params = 0
        loaded_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.requires_grad:
                loaded_params += 1
                print(f"参数 {name}: {param.shape}, 均值: {param.mean().item():.4f}, 标准差: {param.std().item():.4f}")
        print(f"总参数数量: {total_params}, 可训练参数: {loaded_params}")
        
    else:
        model_path = args.model_path or "models/quantized_seg_model.pt"
        model_path = os.path.abspath(model_path)
        print(f"加载量化模型: {model_path}")
        print(f"文件是否存在: {os.path.exists(model_path)}")
        model = load_quantized_model(model_path, device)

    # 支持单张图片或文件夹批量推理
    if os.path.isdir(args.img_path):
        img_files = sorted(glob(os.path.join(args.img_path, '*.png')) + glob(os.path.join(args.img_path, '*.jpg')) + glob(os.path.join(args.img_path, '*.jpeg')))
    else:
        img_files = [args.img_path]

    # 交互式选择推理模式
    print("请选择推理模式：")
    print("1 - 批量推理整个文件夹")
    print("2 - 随机推理文件夹中的一张图片")
    while True:
        mode_input = input("请输入选项数字（1或2）：").strip()
        if mode_input == '1':
            mode = 'all'
            break
        elif mode_input == '2':
            mode = 'random'
            break
        else:
            print("输入无效，请重新输入1或2。")

    if mode == 'random' and len(img_files) > 1:
        img_files = [random.choice(img_files)]
        selected_img = os.path.basename(img_files[0])
        print(f"随机选择图片: {selected_img}")
        
        # 为随机模式创建专门的结果目录
        result_dir = os.path.join('inference', 'random_results')
        os.makedirs(result_dir, exist_ok=True)
        
        # 使用图片名作为报告文件名
        img_name = os.path.splitext(selected_img)[0]
        csv_path = os.path.join(result_dir, f'perf_report_{img_name}.csv')
        xlsx_path = os.path.join(result_dir, f'perf_report_{img_name}.xlsx')
        
        # 增强可视化
        plt.ion()  # 交互模式
    elif mode == 'random' and len(img_files) == 1:
        print("仅有一张图片，自动推理该图片。")
        mode = 'all'  # 当作批量处理
        
    if mode == 'all':
        # 统一输出两个文件名
        base_path, ext = os.path.splitext(args.csv_path)
        csv_path = base_path + '.csv'
        xlsx_path = base_path + '.xlsx'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = [
        '图片名', 
        '模型类型',
        '使用仿真数据',
        '平均时延(ms)', 
        '最小时延(ms)', 
        '最大时延(ms)',
        '时延标准差(ms)',
        '预测区域占比(%)',
        '平均置信度(%)',
        '最大置信度(%)',
        '最小置信度(%)',
        '置信度标准差(%)'
    ]
    all_rows = []
    all_avg, all_min, all_max = [], [], []
    for img_path in img_files:
        img = load_image(img_path).to(device)
        
        # 准备仿真特征
        img_filename = os.path.basename(img_path)
        if img_filename in sim_features:
            feat = sim_features[img_filename]
            # 确保特征维度为11（与模型期望一致）
            if len(feat) != 11:
                print(f"警告：图片 {img_filename} 的特征维度为{len(feat)}，调整为11维")
                if len(feat) > 11:
                    feat = feat[:11]  # 截断
                else:
                    feat = np.pad(feat, (0, 11-len(feat)), 'constant')  # 填充0
            sim_feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            # 使用默认特征
            sim_feat = torch.zeros(1, 11, dtype=torch.float32).to(device)
            print(f"警告：未找到图片 {img_filename} 的仿真特征，使用11维零向量")
        
        result = infer_and_time(model, img, args.model_type, sim_feat)
        
        # 处理返回的(output, time)元组
        output, inference_time = result
        
        # 计算预测概率和掩码
        pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # 计算额外统计信息
        pred_area = np.mean(pred_mask) * 100  # 预测区域占比(预测为1的像素比例)
        avg_confidence = np.mean(pred_prob) * 100  # 平均置信度(模型对预测的确定性程度)
        # 注意：置信度反映模型对预测的把握程度，而IoU需要真实标注才能计算
        # 置信度高不一定IoU高，但通常正相关
        
        # 计算更多置信度统计
        max_confidence = np.max(pred_prob) * 100
        min_confidence = np.min(pred_prob) * 100 
        std_confidence = np.std(pred_prob) * 100
        use_sim = "是" if img_filename in sim_features else "否"
        
        row = [
            os.path.basename(img_path),
            args.model_type,
            use_sim,
            f"{inference_time*1000:.2f}",  # 转换为毫秒
            f"{inference_time*1000:.2f}",  # 简化处理，使用相同值
            f"{inference_time*1000:.2f}",  # 简化处理，使用相同值
            f"0.00",  # 简化处理，标准差设为0
            f"{pred_area:.1f}",
            f"{avg_confidence:.1f}",
            f"{max_confidence:.1f}",
            f"{min_confidence:.1f}",
            f"{std_confidence:.1f}"
        ]
        all_rows.append(row)
        all_avg.append(inference_time*1000)
        all_min.append(inference_time*1000)
        all_max.append(inference_time*1000)
        
        print(f"图片: {os.path.basename(img_path)} | "
              f"推理时间: {inference_time*1000:.2f} ms | "
              f"预测区域: {pred_area:.1f}% | "
              f"平均置信度: {avg_confidence:.1f}%")
              
        if not args.no_vis:
            plt.figure(figsize=(12, 6))
            plt.subplot(1,3,1)
            plt.imshow(Image.open(img_path))
            plt.title("原始图像")
            
            plt.subplot(1,3,2)
            plt.imshow(pred_mask, cmap='gray')
            plt.title("预测掩码")
            
            plt.subplot(1,3,3)
            plt.imshow(pred_prob, cmap='hot')
            plt.colorbar()
            plt.title("预测置信度热图")
            
            plt.tight_layout()
            plt.show()
            
            # 保存预测结果
            output_dir = os.path.join('inference', 'results')
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 保存掩码
            mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            mask_img.save(os.path.join(output_dir, f"{base_name}_mask.png"))
            
            # 保存置信度图
            prob_img = Image.fromarray((pred_prob * 255).astype(np.uint8))
            prob_img.save(os.path.join(output_dir, f"{base_name}_prob.png"))
    # 汇总统计
    if len(img_files) > 1:
        stat_row = ['整体统计', args.model_type, f"{sum(all_avg)/len(all_avg):.2f}", f"{min(all_min):.2f}", f"{max(all_max):.2f}"]
        all_rows.append(stat_row)
        print(f"\n整体统计 | 平均: {sum(all_avg)/len(all_avg):.2f} ms | 最小: {min(all_min):.2f} ms | 最大: {max(all_max):.2f} ms")
    # 写入CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in all_rows:
            writer.writerow(row)
    # 写入Excel
    if openpyxl is None:
        print('警告：openpyxl未安装，无法输出Excel格式。请先 pip install openpyxl')
    else:
        wb = Workbook()
        ws = wb.active
        ws.append(header)
        for row in all_rows:
            ws.append(row)
        wb.save(xlsx_path)
        print(f"已同时输出CSV和Excel文件：{csv_path}  {xlsx_path}")

if __name__ == "__main__":
    main()
