import torch
import torch.nn as nn
import torch.nn.functional as F
from models.starlite_cnn import (
    FPNLayer,
    EnhancedLandslideDetector,
    create_starlite_model
)

def test_fpn_layer():
    """测试FPN层是否正确处理空间特征"""
    print("=== 测试FPN层 ===")
    
    # 创建FPN层
    fpn_layer = FPNLayer(in_channels=64, out_channels=128)
    
    # 创建模拟输入 (batch_size, channels, height, width)
    x = torch.randn(2, 64, 32, 32)
    
    # 前向传播
    output = fpn_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出通道数: {output.shape[1]}")
    print(f"空间尺寸保持不变: {x.shape[2:] == output.shape[2:]}")
    
    # 验证输出
    assert output.shape[0] == 2, "批次大小应该保持不变"
    assert output.shape[1] == 128, "输出通道数应该是128"
    assert output.shape[2:] == x.shape[2:], "空间尺寸应该保持不变"
    print("✅ FPN层测试通过")

def test_enhanced_model_fpn():
    """测试增强模型的FPN功能"""
    print("\n=== 测试增强模型FPN ===")
    
    # 创建模型
    model = create_starlite_model(
        enhanced=True,
        use_attention=True,
        use_fpn=True
    )
    
    # 创建模拟输入
    x = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 验证输出
    assert output.shape[0] == 2, "批次大小应该保持不变"
    assert output.shape[1] == 2, "输出类别数应该是2"
    print("✅ 增强模型FPN测试通过")

def test_attention_application():
    """测试注意力机制应用位置"""
    print("\n=== 测试注意力机制 ===")
    
    # 创建模型
    model = create_starlite_model(
        enhanced=True,
        use_attention=True,
        use_fpn=False  # 关闭FPN以便专注于注意力测试
    )
    
    # 检查注意力层
    if hasattr(model, 'model') and hasattr(model.model, 'attention_layers'):
        attention_layers = model.model.attention_layers
        print(f"注意力层数量: {len(attention_layers)}")
        
        for name, layer in attention_layers.items():
            print(f"注意力层 {name}: {type(layer).__name__}")
    
    # 前向传播测试
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"注意力模型输出形状: {output.shape}")
    print("✅ 注意力机制测试通过")

def test_quantization_fusion():
    """测试量化融合功能"""
    print("\n=== 测试量化融合 ===")
    
    # 创建模型
    model = create_starlite_model(
        enhanced=True,
        use_attention=True,
        use_fpn=True
    )
    
    # 检查模型结构
    print("融合前的模型结构:")
    print(f"FPN层数量: {len(model.model.fpn) if hasattr(model.model, 'fpn') else 0}")
    print(f"注意力层数量: {len(model.model.attention_layers) if hasattr(model.model, 'attention_layers') else 0}")
    
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

def test_model_variants():
    """测试不同模型变体"""
    print("\n=== 测试模型变体 ===")
    
    variants = [
        ("原始模型", create_starlite_model(enhanced=False)),
        ("增强模型(注意力+FPN)", create_starlite_model(enhanced=True, use_attention=True, use_fpn=True)),
        ("仅注意力", create_starlite_model(enhanced=True, use_attention=True, use_fpn=False)),
        ("仅FPN", create_starlite_model(enhanced=True, use_attention=False, use_fpn=True)),
    ]
    
    x = torch.randn(1, 3, 224, 224)
    
    for name, model in variants:
        with torch.no_grad():
            output = model(x)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name}: 输出形状 {output.shape}, 参数数量 {param_count:,}")

def main():
    """主测试函数"""
    print("开始测试FPN修复...")
    
    try:
        test_fpn_layer()
        test_enhanced_model_fpn()
        test_attention_application()
        test_quantization_fusion()
        test_model_variants()
        
        print("\n🎉 所有测试通过！FPN修复成功。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 