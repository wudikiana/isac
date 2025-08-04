#!/usr/bin/env python3
"""
简单验证脚本
"""
import torch
from train_model import (
    DeepLabWithSimFeature, 
    EnhancedDeepLab, 
    process_xview2_mask, 
    postprocess
)

print("="*50)
print("简单验证")
print("="*50)

# 测试模型
model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
print("✅ 模型创建成功")

# 测试掩码处理
test_mask = torch.randn(1, 1, 64, 64)
processed = process_xview2_mask(test_mask, 'all')
print("✅ 掩码处理成功")

# 测试后处理
test_output = torch.randn(1, 1, 64, 64)
postprocessed = postprocess(test_output)
print("✅ 后处理成功")

print("🎉 所有基本功能正常！") 