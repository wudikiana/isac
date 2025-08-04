import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_model_loading():
    print("测试模型加载...")
    
    try:
        # 导入run_inference.py中的函数
        from run_inference import load_deeplab_model
        
        device = torch.device("cpu")
        model_path = "models/best_multimodal_patch_model.pth"
        
        print(f"加载模型: {model_path}")
        model = load_deeplab_model(model_path, device)
        
        print("✅ 模型加载成功")
        print(f"模型类型: {type(model)}")
        
        # 检查模型权重
        print("\n检查模型权重...")
        if hasattr(model, 'deeplab_model'):
            print("Deeplab模型权重:")
            for name, param in model.deeplab_model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    print(f"  {name}: {param.shape}, 均值: {param.mean().item():.4f}, 标准差: {param.std().item():.4f}")
        
        if hasattr(model, 'landslide_model'):
            print("Landslide模型权重:")
            for name, param in model.landslide_model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    print(f"  {name}: {param.shape}, 均值: {param.mean().item():.4f}, 标准差: {param.std().item():.4f}")
        
        if hasattr(model, 'learnable_weight'):
            print(f"融合权重: {model.learnable_weight.item():.4f}")
        
        # 测试推理
        dummy_image = torch.randn(1, 3, 256, 256)
        dummy_sim_feat = torch.randn(1, 11)
        
        with torch.no_grad():
            output = model(dummy_image, dummy_sim_feat)
            print(f"\n✅ 推理测试成功，输出形状: {output.shape}")
            print(f"输出统计: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
            
            # 检查输出是否正常
            if output.min().item() < -10 or output.max().item() > 10:
                print("⚠️ 警告：输出值范围异常")
            else:
                print("✅ 输出值范围正常")
                
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading() 