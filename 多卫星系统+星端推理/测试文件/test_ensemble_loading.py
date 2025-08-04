#!/usr/bin/env python3
import torch
import sys
import os
sys.path.append(os.path.abspath('.'))

def test_ensemble_loading():
    print("Testing DualModelEnsemble loading...")
    
    # 导入必要的类
    from run_inference import DualModelEnsemble, LandslideDetector
    
    # 定义EnhancedDeepLab类
    import segmentation_models_pytorch as smp
    import torch.nn as nn
    
    class EnhancedDeepLab(torch.nn.Module):
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
            import torch.nn.functional as F
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
    
    # 加载checkpoint
    checkpoint = torch.load('models/best_multimodal_patch_model.pth', map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Model state dict has {len(state_dict)} keys")
        print("First 10 keys:", list(state_dict.keys())[:10])
        
        # 检查是否是DualModelEnsemble
        if any(key.startswith('deeplab_model.') for key in state_dict.keys()):
            print("检测到DualModelEnsemble结构")
            
            # 创建DualModelEnsemble
            deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
            landslide_model = LandslideDetector(num_classes=1)
            ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5)
            
            # 尝试加载
            try:
                ensemble_model.load_state_dict(state_dict, strict=False)
                print("✅ DualModelEnsemble加载成功!")
                
                # 测试推理
                dummy_img = torch.randn(1, 3, 256, 256)
                dummy_sim_feat = torch.randn(1, 11)
                
                with torch.no_grad():
                    output = ensemble_model(dummy_img, dummy_sim_feat)
                    print(f"推理测试成功，输出形状: {output.shape}")
                
                return True
            except Exception as e:
                print(f"❌ DualModelEnsemble加载失败: {e}")
                return False
        else:
            print("不是DualModelEnsemble结构")
            return False
    else:
        print("No model_state_dict found")
        return False

if __name__ == "__main__":
    success = test_ensemble_loading()
    if success:
        print("✅ 测试通过!")
    else:
        print("❌ 测试失败!") 