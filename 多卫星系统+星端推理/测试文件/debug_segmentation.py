import torch
import torch.nn as nn
from models.starlite_cnn import create_segmentation_landslide_model

def debug_segmentation_model():
    """调试分割模型的特征图尺寸"""
    print("=== 调试分割版LandslideDetector ===")
    
    model = create_segmentation_landslide_model(
        num_classes=1,
        use_attention=True,
        use_fpn=True,
        use_dynamic_attention=True,
        use_multi_scale=True
    )
    
    x = torch.randn(1, 3, 224, 224)
    print(f"输入形状: {x.shape}")
    
    # 手动跟踪特征图尺寸
    with torch.no_grad():
        x = model.quant(x)
        
        # 提取特征
        features = []
        for i, layer in enumerate(model.backbone.features):
            x = layer(x)
            if i in model.feature_layers:
                features.append(x)
                print(f"特征层 {i}: {x.shape}")
        
        # 获取最终特征图
        final_feature = x
        print(f"最终特征图: {final_feature.shape}")
        
        # 检查FPN特征
        if model.use_fpn and len(features) >= 3:
            key_features = features[-3:]
            print(f"FPN特征层: {[f.shape for f in key_features]}")
            
            # 自顶向下的特征融合
            P5 = model.fpn['fpn_2'](key_features[2])
            P4 = model.fpn['fpn_1'](key_features[1]) + torch.nn.functional.interpolate(P5, size=key_features[1].shape[2:], mode='bilinear', align_corners=False)
            P3 = model.fpn['fpn_0'](key_features[0]) + torch.nn.functional.interpolate(P4, size=key_features[0].shape[2:], mode='bilinear', align_corners=False)
            
            print(f"FPN处理后: P3={P3.shape}, P4={P4.shape}, P5={P5.shape}")
            
            # 将FPN特征上采样到与最终特征相同的尺寸
            fpn_feature = torch.nn.functional.interpolate(P3, size=final_feature.shape[2:], mode='bilinear', align_corners=False)
            print(f"FPN特征上采样后: {fpn_feature.shape}")
            
            # 拼接主特征和FPN特征
            final_feature = torch.cat([final_feature, fpn_feature], dim=1)
            print(f"拼接后特征: {final_feature.shape}")
        
        # 通过分割解码器
        print(f"解码器输入: {final_feature.shape}")
        output = model.segmentation_decoder(final_feature)
        print(f"解码器输出: {output.shape}")
        
        # 确保输出尺寸与输入相同
        if output.shape[2:] != x.shape[2:]:
            output = torch.nn.functional.interpolate(output, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            print(f"上采样后输出: {output.shape}")
        
        output = model.dequant(output)
        print(f"最终输出: {output.shape}")

if __name__ == "__main__":
    debug_segmentation_model() 