#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# 测试DualModelEnsemble类
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
        # DeepLab前向传播
        deeplab_output = self.deeplab_model(img, sim_feat)
        
        # LandslideDetector前向传播（不需要sim_feat）
        landslide_output = self.landslide_model(img)
        
        # 确保输出尺寸一致
        if deeplab_output.shape != landslide_output.shape:
            landslide_output = F.interpolate(landslide_output, size=deeplab_output.shape[2:], mode='bilinear', align_corners=False)
        
        # 加权融合
        weight = torch.sigmoid(self.learnable_weight)
        ensemble_output = weight * deeplab_output + (1 - weight) * landslide_output
        
        return ensemble_output

# 测试LandslideDetector类
class LandslideDetector(nn.Module):
    """简化的LandslideDetector模型用于推理"""
    def __init__(self, num_classes=1):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    
    def forward(self, img):
        return self.unet(img)

# 测试EnhancedDeepLab类
class EnhancedDeepLab(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, sim_feat_dim=11):
        super().__init__()
        self.deeplab = smp.DeepLabV3Plus(
            encoder_name="resnext101_32x8d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        self.sim_fusion = nn.Sequential(
            nn.Linear(sim_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.attention_gate = nn.Sequential(
            nn.Conv2d(2048 + 2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        
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
        features = self.deeplab.encoder(img)
        x = features[-1]
        B, C, H, W = x.shape
        
        sim_proj = self.sim_fusion(sim_feat)
        sim_proj = sim_proj.view(B, -1, 1, 1)
        sim_proj = sim_proj.expand(-1, -1, H, W)
        
        combined = torch.cat([x, sim_proj], dim=1)
        attention = self.attention_gate(combined)
        sim_proj = F.interpolate(sim_proj, size=x.shape[2:], mode='nearest')
        fused = x * attention + sim_proj * (1 - attention)
        
        features = list(features)
        features[-1] = fused
        out = self.deeplab.decoder(features)
        out = self.deeplab.segmentation_head(out)
        
        return out

def test_models():
    print("Testing model classes...")
    
    # 创建模型
    deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
    landslide_model = LandslideDetector(num_classes=1)
    ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5)
    
    print("✅ 模型创建成功")
    
    # 测试推理
    dummy_img = torch.randn(1, 3, 256, 256)
    dummy_sim_feat = torch.randn(1, 11)
    
    with torch.no_grad():
        output = ensemble_model(dummy_img, dummy_sim_feat)
        print(f"✅ 推理测试成功，输出形状: {output.shape}")
    
    print("✅ 所有测试通过!")

if __name__ == "__main__":
    test_models() 