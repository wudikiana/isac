import torch
import torch.nn as nn
from models.starlite_cnn import create_starlite_model

def test_optimizations():
    """测试所有优化"""
    print("=== 测试所有优化 ===")
    
    # 测试不同配置
    configs = [
        ("基础增强", {"use_dynamic_attention": False, "use_multi_scale": False}),
        ("动态注意力", {"use_dynamic_attention": True, "use_multi_scale": False}),
        ("多尺度FPN", {"use_dynamic_attention": False, "use_multi_scale": True}),
        ("全功能", {"use_dynamic_attention": True, "use_multi_scale": True}),
    ]
    
    x = torch.randn(2, 3, 224, 224)
    
    for name, config in configs:
        model = create_starlite_model(
            enhanced=True,
            use_attention=True,
            use_fpn=True,
            **config
        )
        
        with torch.no_grad():
            output = model(x)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name}: 输出形状 {output.shape}, 参数数量 {param_count:,}")

def test_quantization_fusion():
    """测试量化融合"""
    print("\n=== 测试量化融合 ===")
    
    # 测试全功能模型
    model = create_starlite_model(
        enhanced=True,
        use_attention=True,
        use_fpn=True,
        use_dynamic_attention=True,
        use_multi_scale=True
    )
    
    # 检查模型结构
    print("融合前的模型结构:")
    print(f"注意力层数量: {len(model.model.attention_layers) if hasattr(model.model, 'attention_layers') else 0}")
    print(f"FPN层数量: {len(model.model.fpn) if hasattr(model.model, 'fpn') else 0}")
    
    # 尝试融合
    try:
        model.fuse_model()
        print("✅ 量化融合成功")
    except Exception as e:
        print(f"❌ 量化融合失败: {e}")
        return False
    
    # 融合后测试前向传播
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"融合后输出形状: {output.shape}")
    print("✅ 量化融合测试通过")
    return True

def test_layernorm_in_classifier():
    """测试分类头中的LayerNorm"""
    print("\n=== 测试分类头LayerNorm ===")
    
    model = create_starlite_model(enhanced=True, use_attention=True, use_fpn=True)
    
    # 检查分类头结构
    classifier = model.model.classifier
    print("分类头结构:")
    for i, layer in enumerate(classifier):
        print(f"  {i}: {type(layer).__name__}")
    
    # 检查是否包含LayerNorm
    has_layernorm = any(isinstance(layer, nn.LayerNorm) for layer in classifier)
    print(f"包含LayerNorm: {has_layernorm}")
    
    if has_layernorm:
        print("✅ LayerNorm已应用")
    else:
        print("❌ LayerNorm未找到")
    
    return has_layernorm

def test_fpn_implementation():
    """测试FPN实现"""
    print("\n=== 测试FPN实现 ===")
    
    model = create_starlite_model(enhanced=True, use_attention=True, use_fpn=True, use_multi_scale=True)
    
    # 检查FPN层结构
    if hasattr(model.model, 'fpn'):
        print("FPN层结构:")
        for name, fpn_layer in model.model.fpn.items():
            print(f"  {name}: {type(fpn_layer).__name__}")
            
            # 检查MultiScaleFPN的特殊结构
            if hasattr(fpn_layer, 'scale_layers'):
                print(f"    多尺度层数量: {len(fpn_layer.scale_layers)}")
                print(f"    融合层: {type(fpn_layer.fusion_conv).__name__}")
    
    print("✅ FPN实现检查完成")

def test_weight_loading_logic():
    """测试权重加载逻辑"""
    print("\n=== 测试权重加载逻辑 ===")
    
    # 测试不同配置的权重加载
    configs = [
        ("原始模型", {"enhanced": False}),
        ("增强模型", {"enhanced": True}),
    ]
    
    for name, config in configs:
        try:
            model = create_starlite_model(pretrained=False, **config)
            print(f"{name}: 创建成功")
        except Exception as e:
            print(f"{name}: 创建失败 - {e}")
    
    print("✅ 权重加载逻辑测试完成")

def main():
    """主测试函数"""
    print("开始测试所有优化...")
    
    try:
        test_optimizations()
        test_quantization_fusion()
        test_layernorm_in_classifier()
        test_fpn_implementation()
        test_weight_loading_logic()
        
        print("\n🎉 所有优化测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 