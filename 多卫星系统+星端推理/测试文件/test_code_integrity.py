#!/usr/bin/env python3
"""
测试代码完整性脚本
"""

import sys
import os

def test_imports():
    """测试所有必要的导入"""
    print("🔍 测试导入...")
    
    try:
        import torch
        print("✅ PyTorch导入成功")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy导入成功")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib导入成功")
    except ImportError as e:
        print(f"❌ Matplotlib导入失败: {e}")
        return False
    
    try:
        import segmentation_models_pytorch as smp
        print("✅ SMP导入成功")
    except ImportError as e:
        print(f"❌ SMP导入失败: {e}")
        return False
    
    try:
        import albumentations as A
        print("✅ Albumentations导入成功")
    except ImportError as e:
        print(f"❌ Albumentations导入失败: {e}")
        return False
    
    try:
        from skimage import measure
        from skimage.filters import threshold_otsu
        from skimage.morphology import binary_opening, binary_closing, disk
        print("✅ Scikit-image导入成功")
    except ImportError as e:
        print(f"❌ Scikit-image导入失败: {e}")
        return False
    
    try:
        from scipy.ndimage import distance_transform_edt
        print("✅ SciPy导入成功")
    except ImportError as e:
        print(f"❌ SciPy导入失败: {e}")
        return False
    
    return True

def test_syntax():
    """测试主文件语法"""
    print("\n🔍 测试语法...")
    
    try:
        # 尝试编译主文件
        with open('train_model.py', 'r', encoding='utf-8') as f:
            source = f.read()
        
        compile(source, 'train_model.py', 'exec')
        print("✅ 语法检查通过")
        return True
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_classes():
    """测试类定义"""
    print("\n🔍 测试类定义...")
    
    try:
        # 导入主模块
        import train_model
        
        # 测试主要类
        classes_to_test = [
            'CPUAssistedTraining',
            'HybridPrecisionTrainer', 
            'EnhancedDeepLab',
            'MultiScaleDeepLab',
            'HybridLoss',
            'BoundaryAwareLoss',
            'AdaptiveMiner',
            'EnhancedLoss',
            'EnsembleTrainer',
            'PerformanceMonitor'
        ]
        
        for class_name in classes_to_test:
            if hasattr(train_model, class_name):
                print(f"✅ {class_name} 类存在")
            else:
                print(f"❌ {class_name} 类缺失")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 类测试失败: {e}")
        return False

def test_functions():
    """测试函数定义"""
    print("\n🔍 测试函数定义...")
    
    try:
        import train_model
        
        # 测试主要函数
        functions_to_test = [
            'iou_score',
            'dice_score',
            'analyze_performance',
            'enhanced_validation',
            'ensemble_training',
            'validate_ensemble'
        ]
        
        for func_name in functions_to_test:
            if hasattr(train_model, func_name):
                print(f"✅ {func_name} 函数存在")
            else:
                print(f"❌ {func_name} 函数缺失")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 函数测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        import torch
        import train_model
        
        device = torch.device('cpu')
        
        # 测试EnhancedDeepLab
        model1 = train_model.EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        print("✅ EnhancedDeepLab创建成功")
        
        # 测试MultiScaleDeepLab
        model2 = train_model.MultiScaleDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        print("✅ MultiScaleDeepLab创建成功")
        
        # 测试损失函数
        criterion = train_model.HybridLoss()
        print("✅ HybridLoss创建成功")
        
        boundary_criterion = train_model.BoundaryAwareLoss(criterion)
        print("✅ BoundaryAwareLoss创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始代码完整性测试...")
    print("="*50)
    
    tests = [
        test_imports,
        test_syntax,
        test_classes,
        test_functions,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} 失败")
        except Exception as e:
            print(f"❌ {test.__name__} 异常: {e}")
    
    print("\n" + "="*50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！代码完整性良好")
        return True
    else:
        print("⚠️  部分测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 