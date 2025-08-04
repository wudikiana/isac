import torch
import os

def test_quantized_model():
    model_path = "models/quantized_seg_model.pt"
    print(f"测试量化模型: {model_path}")
    print(f"文件是否存在: {os.path.exists(model_path)}")
    print(f"文件大小: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'}")
    
    try:
        model = torch.jit.load(model_path, map_location='cpu')
        print("✅ 量化模型加载成功")
        
        # 测试推理
        dummy_image = torch.randn(1, 3, 256, 256)
        dummy_sim_feat = torch.randn(1, 11)
        
        with torch.no_grad():
            output = model(dummy_image, dummy_sim_feat)
            print(f"✅ 推理测试成功，输出形状: {output.shape}")
            
    except Exception as e:
        print(f"❌ 量化模型加载失败: {e}")

if __name__ == "__main__":
    test_quantized_model() 