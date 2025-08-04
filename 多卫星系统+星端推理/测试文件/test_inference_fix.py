import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from train_model import DeepLabWithSimFeature
from inference.run_inference import load_image, load_deeplab_model

def test_model_output():
    """测试模型输出是否正常"""
    print("🔍 测试模型输出...")
    
    # 创建测试图像
    test_img = torch.randn(1, 3, 256, 256)  # 模拟归一化后的图像
    test_sim_feats = torch.randn(1, 11)     # 模拟仿真特征
    
    # 创建模型
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    model.eval()
    
    with torch.no_grad():
        output = model(test_img, test_sim_feats)
        
    print(f"模型输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"输出均值: {output.mean().item():.3f}")
    print(f"输出标准差: {output.std().item():.3f}")
    
    # 应用sigmoid
    output_sigmoid = torch.sigmoid(output)
    print(f"Sigmoid后范围: [{output_sigmoid.min().item():.3f}, {output_sigmoid.max().item():.3f}]")
    print(f"Sigmoid后均值: {output_sigmoid.mean().item():.3f}")
    
    return output

def test_image_loading():
    """测试图像加载和预处理"""
    print("\n🖼️ 测试图像加载...")
    
    # 查找测试图像
    test_image_path = None
    possible_paths = [
        "data/combined_dataset/images/train2017/guatemala-volcano_00000000_post_disaster.png",
        "data/combined_dataset/images/val2017/guatemala-volcano_00000004_post_disaster.png",
        "data/combined_dataset/images/test2017/guatemala-volcano_00000003_post_disaster.png"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("❌ 未找到测试图像，跳过图像加载测试")
        return None
    
    print(f"使用测试图像: {test_image_path}")
    
    # 加载图像
    img = load_image(test_image_path)
    print(f"加载后图像形状: {img.shape}")
    print(f"图像范围: [{img.min().item():.3f}, {img.max().item():.3f}]")
    print(f"图像均值: {img.mean().item():.3f}")
    print(f"图像标准差: {img.std().item():.3f}")
    
    return img

def test_model_inference():
    """测试完整的模型推理"""
    print("\n🚀 测试完整推理...")
    
    # 加载测试图像
    img = test_image_loading()
    if img is None:
        print("❌ 无法加载测试图像，跳过推理测试")
        return
    
    # 创建模型
    model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
    model.eval()
    
    # 创建仿真特征
    sim_feats = torch.randn(1, 11)
    
    with torch.no_grad():
        output = model(img, sim_feats)
        
    print(f"推理输出形状: {output.shape}")
    print(f"推理输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"推理输出均值: {output.mean().item():.3f}")
    
    # 应用sigmoid
    output_sigmoid = torch.sigmoid(output)
    print(f"Sigmoid后范围: [{output_sigmoid.min().item():.3f}, {output_sigmoid.max().item():.3f}]")
    print(f"Sigmoid后均值: {output_sigmoid.mean().item():.3f}")
    
    # 检查是否有合理的预测
    pred_mask = output_sigmoid.squeeze().numpy()
    high_conf_pixels = (pred_mask > 0.5).sum()
    total_pixels = pred_mask.size
    print(f"高置信度像素数: {high_conf_pixels}/{total_pixels} ({high_conf_pixels/total_pixels:.2%})")
    
    return output_sigmoid

def test_data_consistency():
    """测试数据一致性"""
    print("\n📊 测试数据一致性...")
    
    from data_utils.data_loader import get_multimodal_patch_dataloaders
    
    try:
        train_loader, val_loader = get_multimodal_patch_dataloaders(
            data_root="data/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=2,
            num_workers=0,  # 避免多进程问题
            damage_boost=1
        )
        
        # 获取一个batch
        for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
            if batch_idx >= 1:  # 只测试第一个batch
                break
                
            print(f"训练数据形状:")
            print(f"  图像: {images.shape}")
            print(f"  掩码: {masks.shape}")
            print(f"  仿真特征: {sim_feats.shape}")
            print(f"图像范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
            print(f"图像均值: {images.mean().item():.3f}")
            print(f"图像标准差: {images.std().item():.3f}")
            print(f"掩码范围: [{masks.min().item():.3f}, {masks.max().item():.3f}]")
            print(f"仿真特征范围: [{sim_feats.min().item():.3f}, {sim_feats.max().item():.3f}]")
            
            # 测试模型推理
            model = DeepLabWithSimFeature(in_channels=3, num_classes=1, sim_feat_dim=11)
            model.eval()
            
            with torch.no_grad():
                outputs = model(images, sim_feats)
                
            print(f"模型输出形状: {outputs.shape}")
            print(f"模型输出范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
            print(f"模型输出均值: {outputs.mean().item():.3f}")
            
            # 应用sigmoid
            outputs_sigmoid = torch.sigmoid(outputs)
            print(f"Sigmoid后范围: [{outputs_sigmoid.min().item():.3f}, {outputs_sigmoid.max().item():.3f}]")
            print(f"Sigmoid后均值: {outputs_sigmoid.mean().item():.3f}")
            
            break
            
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")

if __name__ == "__main__":
    print("🧪 开始推理修复测试...")
    print("="*50)
    
    # 测试1: 模型输出
    test_model_output()
    
    # 测试2: 图像加载
    test_image_loading()
    
    # 测试3: 完整推理
    test_model_inference()
    
    # 测试4: 数据一致性
    test_data_consistency()
    
    print("\n" + "="*50)
    print("✅ 推理修复测试完成！") 