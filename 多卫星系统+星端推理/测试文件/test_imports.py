#!/usr/bin/env python3
"""
测试所有导入是否正常
"""
import sys
import os

def test_train_model_imports():
    """测试train_model的导入"""
    print("="*60)
    print("测试train_model的导入")
    print("="*60)
    
    try:
        from train_model import (
            DeepLabWithSimFeature, 
            EnhancedDeepLab, 
            get_multimodal_patch_dataloaders, 
            process_xview2_mask, 
            postprocess,
            load_sim_features,
            custom_collate_fn,
            AdvancedAugmentation,
            DamageAwareDataset,
            HybridPrecisionTrainer,
            BoundaryAwareLoss,
            AdaptiveMiner
        )
        print("✅ train_model 所有导入成功")
        
        # 测试类实例化
        model1 = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
        model2 = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        print("✅ 模型类实例化成功")
        
        # 测试函数调用
        test_mask = torch.randn(1, 1, 64, 64)
        processed = process_xview2_mask(test_mask, 'all')
        print("✅ process_xview2_mask 函数调用成功")
        
        test_output = torch.randn(1, 1, 64, 64)
        postprocessed = postprocess(test_output)
        print("✅ postprocess 函数调用成功")
        
    except Exception as e:
        print(f"❌ train_model 导入失败: {e}")
        import traceback
        traceback.print_exc()

def test_data_utils_imports():
    """测试data_utils的导入"""
    print("\n" + "="*60)
    print("测试data_utils的导入")
    print("="*60)
    
    try:
        from data_utils.data_loader import get_multimodal_patch_dataloaders, optimized_collate
        print("✅ data_utils 导入成功")
        
        # 测试函数调用
        train_loader, val_loader = get_multimodal_patch_dataloaders(
            data_root="data/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=2,
            num_workers=0,
            damage_boost=1
        )
        print("✅ get_multimodal_patch_dataloaders 函数调用成功")
        
    except Exception as e:
        print(f"❌ data_utils 导入失败: {e}")
        import traceback
        traceback.print_exc()

def test_other_files_imports():
    """测试其他文件的导入"""
    print("\n" + "="*60)
    print("测试其他文件的导入")
    print("="*60)
    
    # 测试一些关键文件的导入
    test_files = [
        "test_sim_features.py",
        "test_inference_fix.py", 
        "test_augmentation.py",
        "satellite_inference_server.py",
        "multi_satellite_inference.py"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                # 尝试导入文件中的函数
                if file == "test_sim_features.py":
                    from test_sim_features import test_sim_features_loading
                    print(f"✅ {file} 导入成功")
                elif file == "test_inference_fix.py":
                    from test_inference_fix import test_model_output
                    print(f"✅ {file} 导入成功")
                elif file == "satellite_inference_server.py":
                    from satellite_inference_server import EnhancedDeepLab
                    print(f"✅ {file} 导入成功")
                else:
                    print(f"✅ {file} 存在")
            except Exception as e:
                print(f"❌ {file} 导入失败: {e}")

if __name__ == "__main__":
    import torch
    
    print("开始测试所有导入...")
    
    test_train_model_imports()
    test_data_utils_imports()
    test_other_files_imports()
    
    print("\n" + "="*60)
    print("🎉 导入测试完成！")
    print("="*60) 