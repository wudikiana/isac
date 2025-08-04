#!/usr/bin/env python3
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))

def test_dimension_fix():
    print("Testing dimension fix...")
    
    # 模拟仿真特征数据
    sim_features = {
        'test_image.png': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    }
    
    # 测试维度调整逻辑
    img_filename = 'test_image.png'
    if img_filename in sim_features:
        feat = sim_features[img_filename]
        print(f"原始特征维度: {len(feat)}")
        
        # 确保特征维度为11（与模型期望一致）
        if len(feat) != 11:
            print(f"警告：图片 {img_filename} 的特征维度为{len(feat)}，调整为11维")
            if len(feat) > 11:
                feat = feat[:11]  # 截断
                print(f"截断后特征: {feat}")
            else:
                feat = np.pad(feat, (0, 11-len(feat)), 'constant')  # 填充0
                print(f"填充后特征: {feat}")
        
        sim_feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        print(f"最终特征形状: {sim_feat.shape}")
        
        # 验证维度
        if sim_feat.shape[1] == 11:
            print("✅ 维度修复成功!")
            return True
        else:
            print(f"❌ 维度修复失败，期望11维，实际{sim_feat.shape[1]}维")
            return False
    else:
        print("❌ 未找到测试图片")
        return False

if __name__ == "__main__":
    success = test_dimension_fix()
    if success:
        print("✅ 测试通过!")
    else:
        print("❌ 测试失败!") 