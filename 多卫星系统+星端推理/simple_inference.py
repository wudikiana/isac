import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import os
import time
import argparse
import matplotlib.pyplot as plt

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img

def load_sim_features(sim_feature_csv='data/sim_features.csv'):
    """加载仿真特征"""
    sim_features = {}
    try:
        import pandas as pd
        df = pd.read_csv(sim_feature_csv)
        for _, row in df.iterrows():
            filename = row['filename']
            # 提取特征列（假设前11列是特征）
            features = row.iloc[1:12].values.astype(np.float32)
            sim_features[filename] = features
        print(f"成功加载 {len(sim_features)} 个仿真特征")
    except Exception as e:
        print(f"加载仿真特征失败: {e}")
        # 返回默认特征
        return {}
    return sim_features

def load_deeplab_model(model_path, device):
    """加载DeepLab模型"""
    # 定义与保存的模型完全匹配的EnhancedDeepLab模型类
    class EnhancedDeepLab(torch.nn.Module):
        def __init__(self, in_channels=3, num_classes=1, sim_feat_dim=11):
            super().__init__()
            # 使用EfficientNet-B0编码器，与保存的模型匹配
            self.deeplab = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b0",  # 与保存的模型匹配
                encoder_weights=None,  # 不使用预训练权重，避免网络问题
                in_channels=in_channels,
                classes=num_classes,
                activation=None
            )
            
            # 仿真特征融合模块 - 使用与保存模型相同的名称和配置
            self.sim_fc = torch.nn.Sequential(
                torch.nn.Linear(sim_feat_dim, 64),  # 与保存的模型匹配
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(64, 2048),  # 与保存的模型匹配
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3)
            )
            
            # 注意力门控机制
            self.attention_gate = torch.nn.Sequential(
                torch.nn.Conv2d(2048 + 2048, 512, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 2048, kernel_size=1),
                torch.nn.Sigmoid()
            )
            
            # 初始化权重
            self._init_weights()
        
        def _init_weights(self):
            for m in self.sim_fc.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            
            for m in self.attention_gate.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        
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
            sim_proj = self.sim_fc(sim_feat)
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
    
    # 创建模型
    model = EnhancedDeepLab().to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否是完整的checkpoint还是纯模型权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整的checkpoint格式
        print(f"检测到完整checkpoint，加载模型权重...")
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"模型训练到第 {checkpoint['epoch']} 轮")
    else:
        # 纯模型权重格式
        print(f"检测到纯模型权重，直接加载...")
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ 模型加载成功!")
    return model

def infer_and_time(model, img, sim_feat=None, repetitions=20):
    # 预热
    with torch.no_grad():
        if sim_feat is not None:
            _ = model(img, sim_feat)
        else:
            # 如果没有仿真特征，使用零张量
            dummy_sim_feat = torch.zeros(img.size(0), 11).to(img.device)
            _ = model(img, dummy_sim_feat)
    
    times = []
    with torch.no_grad():
        for _ in range(repetitions):
            start = time.perf_counter()
            if sim_feat is not None:
                pred = model(img, sim_feat)
            else:
                dummy_sim_feat = torch.zeros(img.size(0), 11).to(img.device)
                pred = model(img, dummy_sim_feat)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 处理输出
    pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
    
    return avg_time, min_time, max_time, pred_mask

def main():
    parser = argparse.ArgumentParser(description='DeepLab模型推理')
    parser.add_argument('--model_path', type=str, default='models/seg_model.pth',
                        help='模型权重路径')
    parser.add_argument('--img_path', type=str, default='data/combined_dataset/images/tier3/',
                        help='输入图片路径或图片文件夹')
    parser.add_argument('--csv_path', type=str, default='inference/perf_report.csv',
                        help='推理性能表格输出路径')
    parser.add_argument('--no_vis', action='store_true', help='不显示可视化窗口')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_deeplab_model(args.model_path, device)
    
    # 加载仿真特征
    sim_features = load_sim_features()
    
    # 获取图片文件列表
    if os.path.isfile(args.img_path):
        img_files = [args.img_path]
    else:
        img_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            img_files.extend(glob.glob(os.path.join(args.img_path, ext)))
        img_files = sorted(img_files)
    
    if not img_files:
        print(f"❌ 在 {args.img_path} 中未找到图片文件")
        return
    
    print(f"找到 {len(img_files)} 个图片文件")
    
    # 推理
    all_rows = []
    all_avg, all_min, all_max = [], [], []
    
    for img_path in img_files:
        img = load_image(img_path).to(device)
        
        # 获取对应的仿真特征
        filename = os.path.basename(img_path)
        sim_feat = None
        if filename in sim_features:
            sim_feat = torch.tensor(sim_features[filename], dtype=torch.float32).unsqueeze(0).to(device)
            print(f"使用仿真特征: {filename}")
        else:
            print(f"未找到仿真特征: {filename}")
        
        avg_time, min_time, max_time, pred_mask = infer_and_time(model, img, sim_feat)
        
        row = [filename, "deeplab", f"{avg_time:.2f}", f"{min_time:.2f}", f"{max_time:.2f}"]
        all_rows.append(row)
        all_avg.append(avg_time)
        all_min.append(min_time)
        all_max.append(max_time)
        
        print(f"图片: {filename} | 平均: {avg_time:.2f} ms | 最小: {min_time:.2f} ms | 最大: {max_time:.2f} ms")
        
        if not args.no_vis:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(Image.open(img_path))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask > 0.5, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    # 保存性能报告
    if all_rows:
        import pandas as pd
        
        # 创建DataFrame
        df = pd.DataFrame(all_rows, columns=['Image', 'Model_Type', 'Avg_Time(ms)', 'Min_Time(ms)', 'Max_Time(ms)'])
        
        # 保存CSV
        os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
        df.to_csv(args.csv_path, index=False)
        print(f"性能报告已保存到: {args.csv_path}")
        
        # 保存Excel
        excel_path = args.csv_path.replace('.csv', '.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"性能报告已保存到: {excel_path}")
        
        # 打印统计信息
        print(f"\n统计信息:")
        print(f"总图片数: {len(all_rows)}")
        print(f"平均推理时间: {np.mean(all_avg):.2f} ms")
        print(f"最小推理时间: {np.min(all_min):.2f} ms")
        print(f"最大推理时间: {np.max(all_max):.2f} ms")

if __name__ == "__main__":
    import glob
    main() 