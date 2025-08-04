import torch
import segmentation_models_pytorch as smp
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.abspath('.'))

def test_deeplab_loading():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试模型文件
    model_path = "models/seg_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    print(f"测试模型: {model_path}")
    
    try:
        # 定义与保存的模型完全匹配的EnhancedDeepLab模型类
        class EnhancedDeepLab(torch.nn.Module):
            def __init__(self, in_channels=3, num_classes=1, sim_feat_dim=11):
                super().__init__()
                # 使用EfficientNet编码器，与保存的模型匹配
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
        model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查是否是完整的checkpoint还是纯模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整的checkpoint格式
            print(f"检测到完整checkpoint，加载模型权重...")
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"模型训练轮次: {checkpoint['epoch']}")
            if 'best_val_iou' in checkpoint:
                print(f"最佳验证IoU: {checkpoint['best_val_iou']:.4f}")
        else:
            # 纯模型权重格式
            print(f"检测到纯模型权重，直接加载...")
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(device)
        print("✅ 模型加载成功!")
        
        # 测试前向传播
        batch_size = 1
        img = torch.randn(batch_size, 3, 256, 256).to(device)
        sim_feat = torch.randn(batch_size, 11).to(device)
        
        with torch.no_grad():
            output = model(img, sim_feat)
            print(f"✅ 前向传播成功! 输出形状: {output.shape}")
            print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deeplab_loading() 