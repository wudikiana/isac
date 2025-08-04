import torch
import torch.nn as nn
from models.starlite_cnn import create_segmentation_landslide_model
from train_model import EnhancedDeepLab, DualModelEnsemble

def test_segmentation_landslide_model():
    """测试分割版LandslideDetector"""
    print("=== 测试分割版LandslideDetector ===")
    
    model = create_segmentation_landslide_model(
        num_classes=1,
        use_attention=True,
        use_fpn=True,
        use_dynamic_attention=True,
        use_multi_scale=True
    )
    
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 验证输出
    assert output.shape[0] == 2, "批次大小应该保持不变"
    assert output.shape[1] == 1, "输出通道数应该是1"
    assert output.shape[2:] == x.shape[2:], "空间尺寸应该与输入相同"
    print("✅ 分割版LandslideDetector测试通过")

def test_dual_model_ensemble():
    """测试双模型集成"""
    print("\n=== 测试双模型集成 ===")
    
    # 创建两个模型
    deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
    landslide_model = create_segmentation_landslide_model(
        num_classes=1,
        use_attention=True,
        use_fpn=True,
        use_dynamic_attention=True,
        use_multi_scale=True
    )
    
    # 创建集成模型
    ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5)
    
    x = torch.randn(2, 3, 224, 224)
    sim_feat = torch.randn(2, 11)
    
    with torch.no_grad():
        output = ensemble_model(x, sim_feat)
    
    print(f"输入形状: {x.shape}")
    print(f"仿真特征形状: {sim_feat.shape}")
    print(f"输出形状: {output.shape}")
    print(f"融合权重: {ensemble_model.get_fusion_weight():.3f}")
    
    # 验证输出
    assert output.shape[0] == 2, "批次大小应该保持不变"
    assert output.shape[1] == 1, "输出通道数应该是1"
    assert output.shape[2:] == x.shape[2:], "空间尺寸应该与输入相同"
    print("✅ 双模型集成测试通过")

def test_model_compatibility():
    """测试模型兼容性"""
    print("\n=== 测试模型兼容性 ===")
    
    # 测试不同输入尺寸 - 避免batch_size=1的问题
    test_sizes = [(2, 3, 224, 224), (4, 3, 256, 256), (2, 3, 512, 512)]
    
    for batch_size, channels, height, width in test_sizes:
        deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        landslide_model = create_segmentation_landslide_model(
            num_classes=1,
            use_attention=True,
            use_fpn=True,
            use_dynamic_attention=True,
            use_multi_scale=True
        )
        ensemble_model = DualModelEnsemble(deeplab_model, landslide_model)
        
        x = torch.randn(batch_size, channels, height, width)
        sim_feat = torch.randn(batch_size, 11)
        
        with torch.no_grad():
            output = ensemble_model(x, sim_feat)
        
        print(f"输入尺寸 {x.shape} -> 输出尺寸 {output.shape}")
        
        # 验证输出尺寸
        assert output.shape[0] == batch_size, f"批次大小不匹配: {output.shape[0]} != {batch_size}"
        assert output.shape[1] == 1, f"输出通道数不匹配: {output.shape[1]} != 1"
        assert output.shape[2:] == x.shape[2:], f"空间尺寸不匹配: {output.shape[2:]} != {x.shape[2:]}"
    
    print("✅ 模型兼容性测试通过")

def test_fusion_weight():
    """测试融合权重"""
    print("\n=== 测试融合权重 ===")
    
    deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
    landslide_model = create_segmentation_landslide_model(
        num_classes=1,
        use_attention=True,
        use_fpn=True,
        use_dynamic_attention=True,
        use_multi_scale=True
    )
    
    # 测试不同融合权重
    test_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for weight in test_weights:
        ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=weight)
        actual_weight = ensemble_model.get_fusion_weight()
        print(f"设置权重: {weight:.2f} -> 实际权重: {actual_weight:.3f}")
        
        # 验证权重在合理范围内
        assert 0.0 <= actual_weight <= 1.0, f"权重超出范围: {actual_weight}"
    
    print("✅ 融合权重测试通过")

def main():
    """主测试函数"""
    print("开始测试双模型集成...")
    
    try:
        test_segmentation_landslide_model()
        test_dual_model_ensemble()
        test_model_compatibility()
        test_fusion_weight()
        
        print("\n🎉 所有测试通过！双模型集成功能正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 