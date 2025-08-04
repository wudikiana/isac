import torch
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.abspath('.'))

def debug_model_loading():
    print("=== 详细调试模型加载 ===")
    
    try:
        from run_inference import load_deeplab_model
        device = torch.device("cpu")
        
        print("1. 开始加载模型...")
        model = load_deeplab_model("models/seg_model.pth", device)
        print("✅ 模型加载成功!")
        
        print("\n2. 测试前向传播...")
        batch_size = 1
        img = torch.randn(batch_size, 3, 256, 256)
        sim_feat = torch.randn(batch_size, 11)
        
        print(f"输入图像形状: {img.shape}")
        print(f"仿真特征形状: {sim_feat.shape}")
        
        with torch.no_grad():
            output = model(img, sim_feat)
            print(f"✅ 前向传播成功! 输出形状: {output.shape}")
        
        print("\n3. 测试推理脚本的其他功能...")
        from run_inference import load_image, infer_and_time
        
        # 测试图像加载
        print("测试图像加载...")
        test_img_path = "data/combined_dataset/images/tier3/"
        import glob
        img_files = glob.glob(os.path.join(test_img_path, "*.png")) + glob.glob(os.path.join(test_img_path, "*.jpg"))
        if img_files:
            test_img = load_image(img_files[0])
            print(f"✅ 图像加载成功! 形状: {test_img.shape}")
            
            # 测试推理和计时
            print("测试推理和计时...")
            avg_time, min_time, max_time, pred_mask = infer_and_time(model, test_img, "original", sim_feat)
            print(f"✅ 推理成功! 平均时间: {avg_time:.2f} ms")
            print(f"预测掩码形状: {pred_mask.shape}")
        else:
            print("❌ 未找到测试图像")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_model_loading()
    if success:
        print("\n🎉 所有测试都通过了!")
    else:
        print("\n💥 还有问题需要解决。") 