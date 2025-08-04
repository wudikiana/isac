2
import sys
import os
import torch
import time
import argparse
import csv
from glob import glob
import re
import random
import json
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from train_model import DeepLabWithSimFeature  # 导入我们的自定义模型
# Excel支持
try:
    import openpyxl
    from openpyxl import Workbook
except ImportError:
    openpyxl = None

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    # 应用与训练时相同的归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    
    return img

def load_sim_features(sim_feature_csv="data/sim_features.csv"):
    """加载仿真特征数据"""
    try:
        sim_features_df = pd.read_csv(sim_feature_csv)
        print(f"CSV文件形状: {sim_features_df.shape}")
        print(f"CSV列名: {sim_features_df.columns.tolist()}")
        
        # 查找文件名列
        filename_col = None
        possible_filename_cols = ['img_path', 'filename', 'Filename', 'file_name', 'File', 'image', 'Image', 'name', 'Name']
        
        for col in possible_filename_cols:
            if col in sim_features_df.columns:
                filename_col = col
                break
        
        if filename_col is None:
            # 如果没有找到文件名列，使用第一列作为文件名
            filename_col = sim_features_df.columns[0]
            print(f"未找到标准文件名列，使用第一列 '{filename_col}' 作为文件名")
        
        # 创建文件名到特征的映射
        sim_feature_dict = {}
        # 只选择数值型特征列，排除字符串列
        numeric_cols = []
        for col in sim_features_df.columns:
            if col != filename_col:
                # 检查列是否为数值型
                try:
                    pd.to_numeric(sim_features_df[col], errors='raise')
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    print(f"跳过非数值列: {col} (包含字符串值)")
        
        print(f"使用数值特征列: {numeric_cols}")
        
        for _, row in sim_features_df.iterrows():
            filename = str(row[filename_col])
            try:
                features = row[numeric_cols].values.astype(np.float32)
                sim_feature_dict[filename] = features
            except Exception as e:
                print(f"跳过行 {filename}: {e}")
                continue
        
        print(f"✅ 成功加载 {len(sim_feature_dict)} 个仿真特征")
        print(f"📊 特征维度: {len(numeric_cols)}")
        print(f"📋 特征列: {numeric_cols}")
        print(f"🔍 特征统计:")
        print(f"   - 数值列数: {len(numeric_cols)}")
        print(f"   - 字符串列数: {len(sim_features_df.columns) - len(numeric_cols) - 1}")
        print(f"   - 总样本数: {len(sim_feature_dict)}")
        return sim_feature_dict
        
    except Exception as e:
        print(f"加载仿真特征失败: {e}")
        print("请检查CSV文件格式是否正确")
        return {}

def get_sim_features_for_image(image_path, sim_feature_dict):
    """根据图像路径获取对应的仿真特征"""
    # 提取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 尝试多种可能的匹配方式
    possible_keys = [
        filename,  # 完整文件名
        filename.replace('_post_disaster', '').replace('_pre_disaster', ''),  # 基础文件名
        filename + '_post_disaster',
        filename + '_pre_disaster'
    ]
    
    # 首先尝试直接匹配
    for key in possible_keys:
        if key in sim_feature_dict:
            features = sim_feature_dict[key]
            # 归一化特征
            features = (features - np.mean(features)) / np.std(features)
            return torch.tensor(features, dtype=torch.float32)
    
    # 如果直接匹配失败，尝试在路径中查找
    for csv_key in sim_feature_dict.keys():
        # 检查CSV键是否包含我们的文件名
        if filename in csv_key or filename.replace('_post_disaster', '').replace('_pre_disaster', '') in csv_key:
            features = sim_feature_dict[csv_key]
            # 归一化特征
            features = (features - np.mean(features)) / np.std(features)
            print(f"找到匹配: {filename} -> {csv_key}")
            return torch.tensor(features, dtype=torch.float32)
    
    # 如果还是找不到，尝试模糊匹配
    for csv_key in sim_feature_dict.keys():
        # 提取CSV键中的文件名部分
        csv_filename = os.path.basename(csv_key)
        csv_filename = os.path.splitext(csv_filename)[0]
        
        if filename == csv_filename or filename.replace('_post_disaster', '').replace('_pre_disaster', '') == csv_filename.replace('_post_disaster', '').replace('_pre_disaster', ''):
            features = sim_feature_dict[csv_key]
            # 归一化特征
            features = (features - np.mean(features)) / np.std(features)
            print(f"模糊匹配成功: {filename} -> {csv_key}")
            return torch.tensor(features, dtype=torch.float32)
    
    # 如果找不到对应特征，返回零向量
    print(f"警告: 未找到图像 {filename} 对应的仿真特征，使用零向量")
    print(f"可用的CSV键示例: {list(sim_feature_dict.keys())[:3]}")  # 显示前3个键作为示例
    return torch.zeros(11, dtype=torch.float32)

def load_deeplab_model(model_path, device):
    # 使用我们的自定义多模态模型
    model = DeepLabWithSimFeature(
        in_channels=3,
        num_classes=1,
        sim_feat_dim=11
    )
    
    # 智能加载模型 - 支持检查点格式和纯权重格式
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 检查点格式
        print("检测到检查点格式，加载模型状态字典...")
        state_dict = checkpoint['model_state_dict']
    else:
        # 纯权重格式
        print("检测到纯权重格式，直接加载...")
        state_dict = checkpoint
    
    # 处理模型结构兼容性问题
    print("检查模型结构兼容性...")
    model_state_dict = model.state_dict()
    
    # 处理sim_fc层的键名变化（从sim_fc.2到sim_fc.3）
    updated_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('sim_fc.2.') and key not in model_state_dict:
            # 将sim_fc.2.weight -> sim_fc.3.weight
            new_key = key.replace('sim_fc.2.', 'sim_fc.3.')
            if new_key in model_state_dict:
                updated_state_dict[new_key] = value
                print(f"键名映射: {key} -> {new_key}")
            else:
                print(f"警告: 无法映射键 {key}")
        else:
            updated_state_dict[key] = value
    
    # 加载兼容后的状态字典
    try:
        model.load_state_dict(updated_state_dict, strict=False)
        print("模型加载成功（使用非严格模式）")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    model.eval()
    model.to(device)
    return model

def load_quantized_model(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model

def infer_and_time(model, img, model_type, sim_feats=None, repetitions=20):
    # 如果没有提供sim_feats，使用零向量作为默认值
    if sim_feats is None:
        batch_size = img.shape[0]
        sim_feats = torch.zeros(batch_size, 11, device=img.device)
    else:
        # 确保sim_feats在正确的设备上
        sim_feats = sim_feats.to(img.device)
        if sim_feats.dim() == 1:
            sim_feats = sim_feats.unsqueeze(0)  # [11] -> [1, 11]
    
    # 预热
    with torch.no_grad():
        _ = model(img, sim_feats)
    
    times = []
    with torch.no_grad():
        for _ in range(repetitions):
            start = time.perf_counter()
            pred = model(img, sim_feats)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 处理预测输出
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    
    # 检查原始输出范围
    raw_pred = pred.detach().cpu()
    print(f"🔍 模型原始输出:")
    print(f"  原始范围: [{raw_pred.min().item():.3f}, {raw_pred.max().item():.3f}]")
    print(f"  原始均值: {raw_pred.mean().item():.3f}")
    print(f"  原始标准差: {raw_pred.std().item():.3f}")
    
    # 确保输出维度正确并应用sigmoid
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
    else:
        pred_mask = torch.sigmoid(pred).cpu().numpy()[0]
    
    return avg_time, min_time, max_time, pred_mask, sim_feats

def main():
    parser = argparse.ArgumentParser(description='Segmentation Model Inference')
    parser.add_argument('--model_type', type=str, default='original', choices=['original', 'quantized'],
                        help='选择加载原始模型还是量化模型')
    parser.add_argument('--img_path', type=str, default='data/combined_dataset/images/tier3/',
                        help='输入图片路径或图片文件夹（支持*.png,*.jpg,*.jpeg）')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径（可选）')
    parser.add_argument('--csv_path', type=str, default='inference/perf_report.csv',
                        help='推理性能表格输出路径（支持.csv或.xlsx，自动生成另一种格式）')
    parser.add_argument('--no_vis', action='store_true', help='不显示可视化窗口')
    parser.add_argument('--sim_feature_csv', type=str, default='data/sim_features.csv',
                        help='仿真特征CSV文件路径')
    parser.add_argument('--joint_inference', action='store_true', 
                        help='启用联合推理（使用真实的仿真特征）')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 用户手动选择推理模式
    print("\n" + "="*50)
    print("推理类型选择")
    print("="*50)
    print("1 - 联合推理（图像 + 仿真特征）")
    print("2 - 标准推理（仅图像，零向量特征）")
    print("="*50)
    
    while True:
        try:
            inference_type = input("请选择推理类型（1或2）：").strip()
            if inference_type == '1':
                use_joint_inference = True
                print("✅ 选择联合推理模式（图像 + 仿真特征）")
                break
            elif inference_type == '2':
                use_joint_inference = False
                print("✅ 选择标准推理模式（仅图像，零向量特征）")
                break
            else:
                print("输入无效，请输入1或2")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return
        except Exception as e:
            print(f"输入错误: {e}，请重新输入")
    
    # 加载仿真特征（如果选择联合推理）
    sim_feature_dict = {}
    if use_joint_inference:
        print("\n" + "="*50)
        print("仿真特征类型选择")
        print("="*50)
        print("1 - 使用真实仿真特征（从CSV文件加载）")
        print("2 - 使用零向量特征（默认）")
        print("="*50)
        
        while True:
            try:
                feature_type = input("请选择特征类型（1或2）：").strip()
                if feature_type == '1':
                    use_real_features = True
                    print("✅ 选择真实仿真特征（从CSV文件加载）")
                    break
                elif feature_type == '2':
                    use_real_features = False
                    print("✅ 选择零向量特征（默认）")
                    break
                else:
                    print("输入无效，请输入1或2")
            except KeyboardInterrupt:
                print("\n用户取消操作")
                return
            except Exception as e:
                print(f"输入错误: {e}，请重新输入")
        
        if use_real_features:
            print("🔄 正在加载仿真特征...")
            sim_feature_dict = load_sim_features(args.sim_feature_csv)
            if not sim_feature_dict:
                print("⚠️ 警告: 仿真特征加载失败，将使用零向量进行推理")
                use_real_features = False
        else:
            print("📊 使用零向量特征进行联合推理")
    else:
        use_real_features = False
        print("📊 标准推理模式，使用零向量特征")
    if args.model_type == 'original':
        model_path = args.model_path or "models/best_multimodal_patch_model.pth"
        model = load_deeplab_model(model_path, device)
        if model is None:
            print("模型加载失败，退出程序")
            return
    else:
        model_path = args.model_path or "models/quantized_seg_model_stage3.pt"
        model = load_quantized_model(model_path, device)

    # 支持单张图片或文件夹批量推理
    if os.path.isdir(args.img_path):
        img_files = sorted(glob(os.path.join(args.img_path, '*.png')) + glob(os.path.join(args.img_path, '*.jpg')) + glob(os.path.join(args.img_path, '*.jpeg')))
    else:
        img_files = [args.img_path]

    # 用户手动选择推理模式
    print("\n" + "="*50)
    print("推理模式选择")
    print("="*50)
    print("1 - 批量推理整个文件夹")
    print("2 - 随机推理文件夹中的一张图片")
    print("3 - 推理单张指定图片")
    print("="*50)
    
    while True:
        try:
            mode_input = input("请输入选项数字（1、2或3）：").strip()
            if mode_input == '1':
                mode = 'all'
                print(f"✅ 选择批量推理模式，将处理 {len(img_files)} 张图片")
                break
            elif mode_input == '2':
                if len(img_files) > 1:
                    mode = 'random'
                    selected_img = random.choice(img_files)
                    img_files = [selected_img]
                    print(f"✅ 选择随机推理模式，随机选择图片: {os.path.basename(selected_img)}")
                else:
                    mode = 'single'
                    print("✅ 仅有一张图片，自动选择单张推理模式")
                break
            elif mode_input == '3':
                mode = 'single'
                print("✅ 选择单张推理模式")
                break
            else:
                print("输入无效，请输入1、2或3")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return
        except Exception as e:
            print(f"输入错误: {e}，请重新输入")

    # 统一输出两个文件名
    base_path, ext = os.path.splitext(args.csv_path)
    csv_path = base_path + '.csv'
    xlsx_path = base_path + '.xlsx'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = ['图片名', '模型类型', '推理模式', '平均时延(ms)', '最小时延(ms)', '最大时延(ms)', '仿真特征状态']
    all_rows = []
    all_avg, all_min, all_max = [], [], []
    for img_path in img_files:
        img = load_image(img_path).to(device)
        
        # 获取对应的仿真特征
        sim_feats = None
        sim_feat_status = "零向量"
        
        if use_joint_inference:
            if use_real_features and sim_feature_dict:
                sim_feats = get_sim_features_for_image(img_path, sim_feature_dict)
                if sim_feats.sum() != 0:
                    sim_feat_status = "真实特征"
                else:
                    sim_feat_status = "零向量(未找到)"
            else:
                # 联合推理但使用零向量特征
                sim_feat_status = "零向量(联合推理)"
        else:
            # 标准推理，使用零向量特征
            sim_feat_status = "零向量(标准推理)"
        
        avg_time, min_time, max_time, pred_mask, used_sim_feats = infer_and_time(
            model, img, args.model_type, sim_feats
        )
        
        inference_mode = "联合推理" if use_joint_inference else "标准推理"
        row = [
            os.path.basename(img_path), 
            args.model_type, 
            inference_mode,
            f"{avg_time:.2f}", 
            f"{min_time:.2f}", 
            f"{max_time:.2f}",
            sim_feat_status
        ]
        all_rows.append(row)
        all_avg.append(avg_time)
        all_min.append(min_time)
        all_max.append(max_time)
        
        print(f"\n{'='*60}")
        print(f"📸 图像信息:")
        print(f"  文件名: {os.path.basename(img_path)}")
        print(f"  完整路径: {img_path}")
        print(f"  图像尺寸: {img.shape[2]}x{img.shape[3]} (CxHxW)")
        
        print(f"\n🤖 推理配置:")
        print(f"  模型类型: {args.model_type}")
        print(f"  推理模式: {inference_mode}")
        print(f"  特征状态: {sim_feat_status}")
        print(f"  设备: {device}")
        
        print(f"\n⚡ 性能指标:")
        print(f"  平均延迟: {avg_time:.2f} ms")
        print(f"  最小延迟: {min_time:.2f} ms")
        print(f"  最大延迟: {max_time:.2f} ms")
        print(f"  吞吐量: {1000/avg_time:.1f} FPS")
        
        if sim_feats is not None and sim_feats.sum() != 0:
            print(f"\n📊 仿真特征详情:")
            print(f"  特征维度: {used_sim_feats.shape}")
            print(f"  特征范围: [{used_sim_feats.min().item():.3f}, {used_sim_feats.max().item():.3f}]")
            print(f"  特征均值: {used_sim_feats.mean().item():.3f}")
            print(f"  特征标准差: {used_sim_feats.std().item():.3f}")
            print(f"  非零特征数: {(used_sim_feats != 0).sum().item()}/{used_sim_feats.numel()}")
        elif use_joint_inference and not use_real_features:
            print(f"\n📊 仿真特征详情:")
            print(f"  使用零向量特征进行联合推理")
            print(f"  特征维度: {used_sim_feats.shape}")
        
        print(f"\n🎯 预测结果:")
        print(f"  预测掩码形状: {pred_mask.shape}")
        print(f"  预测范围: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
        print(f"  预测均值: {pred_mask.mean():.3f}")
        print(f"  预测标准差: {pred_mask.std():.3f}")
        
        # 多阈值分析
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        print(f"\n📊 多阈值分析:")
        for threshold in thresholds:
            pred_pixels = (pred_mask > threshold).sum()
            pred_ratio = pred_pixels / pred_mask.size
            print(f"  阈值 {threshold}: {pred_pixels} 像素 ({pred_ratio:.2%})")
        
        # 置信度分布分析
        print(f"\n📈 置信度分布:")
        low_conf = (pred_mask < 0.1).sum() / pred_mask.size
        mid_conf = ((pred_mask >= 0.1) & (pred_mask < 0.5)).sum() / pred_mask.size
        high_conf = (pred_mask >= 0.5).sum() / pred_mask.size
        print(f"  低置信度 (<0.1): {low_conf:.2%}")
        print(f"  中置信度 (0.1-0.5): {mid_conf:.2%}")
        print(f"  高置信度 (≥0.5): {high_conf:.2%}")
        
        # 建议阈值
        if pred_mask.max() < 0.5:
            suggested_threshold = pred_mask.max() * 0.8
            print(f"\n⚠️ 建议:")
            print(f"  预测最大值较低 ({pred_mask.max():.3f})，建议使用更低阈值: {suggested_threshold:.3f}")
            print(f"  使用建议阈值: {(pred_mask > suggested_threshold).sum()} 像素 ({(pred_mask > suggested_threshold).sum() / pred_mask.size:.2%})")
        
        print(f"{'='*60}")
        
        # 可视化（仅对最后一张图片）
        if not args.no_vis and img_path == img_files[-1]:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(Image.open(img_path))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(pred_mask, cmap='gray')
            plt.title("Predicted Mask (Raw)")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask > 0.5, cmap='gray')
            plt.title("Predicted Mask (Binary)")
            plt.axis('off')
            
            plt.tight_layout()
    plt.show()
    
    # 汇总统计
    if len(img_files) > 1:
        inference_mode = "联合推理" if use_joint_inference else "标准推理"
        stat_row = ['整体统计', args.model_type, inference_mode, f"{sum(all_avg)/len(all_avg):.2f}", f"{min(all_min):.2f}", f"{max(all_max):.2f}", ""]
        all_rows.append(stat_row)
        
        print(f"\n{'='*80}")
        print(f"📈 整体性能统计")
        print(f"{'='*80}")
        print(f"📊 处理统计:")
        print(f"  总图像数: {len(img_files)}")
        print(f"  推理模式: {inference_mode}")
        print(f"  模型类型: {args.model_type}")
        
        print(f"\n⚡ 性能统计:")
        print(f"  平均延迟: {sum(all_avg)/len(all_avg):.2f} ms")
        print(f"  最小延迟: {min(all_min):.2f} ms")
        print(f"  最大延迟: {max(all_max):.2f} ms")
        print(f"  延迟标准差: {((sum((x - sum(all_avg)/len(all_avg))**2 for x in all_avg) / len(all_avg))**0.5):.2f} ms")
        print(f"  平均吞吐量: {1000/(sum(all_avg)/len(all_avg)):.1f} FPS")
        
        print(f"\n💾 输出文件:")
        print(f"  CSV报告: {csv_path}")
        print(f"  Excel报告: {xlsx_path}")
        print(f"{'='*80}")
    
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