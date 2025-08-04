#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模型维度问题
"""

import torch
import torch.nn as nn
from models.starlite_cnn import create_starlite_model

def debug_model_dimensions():
    """调试模型维度"""
    print("调试模型维度问题...")
    
    # 创建模型
    model = create_starlite_model(enhanced=True, use_attention=True, use_fpn=True)
    
    # 创建测试输入
    input_tensor = torch.randn(2, 3, 224, 224)
    print(f"输入形状: {input_tensor.shape}")
    
    # 获取backbone的in_features
    backbone = model.model.backbone
    in_features = 576  # 默认值
    if hasattr(backbone.classifier, '__getitem__'):
        for m in backbone.classifier:
            if isinstance(m, nn.Linear):
                in_features = m.in_features
                break
    
    print(f"Backbone in_features: {in_features}")
    
    # 检查FPN配置
    if hasattr(model.model, 'fpn'):
        print(f"FPN层数: {len(model.model.fpn)}")
        for name, layer in model.model.fpn.items():
            print(f"  {name}: {layer}")
    
    # 检查分类头
    print(f"分类头: {model.model.classifier}")
    
    # 尝试前向传播
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            print(f"输出形状: {output.shape}")
    except Exception as e:
        print(f"前向传播失败: {e}")
        
        # 调试FPN部分
        print("\n调试FPN部分...")
        features = []
        x = input_tensor
        
        # 模拟backbone前向传播
        for i, layer in enumerate(backbone.features):
            x = layer(x)
            if i in model.model.feature_layers:
                features.append(x)
                print(f"  特征层 {i}: {x.shape}")
        
        # 检查FPN特征
        if len(features) >= 3:
            print(f"\nFPN特征处理:")
            for i, feat in enumerate(features[-3:]):
                print(f"  特征 {i}: {feat.shape}")
                feat_pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat_flattened = torch.flatten(feat_pooled, 1)
                print(f"  池化后: {feat_flattened.shape}")

if __name__ == "__main__":
    debug_model_dimensions() 