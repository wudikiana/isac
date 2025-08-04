import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from inference.run_inference import load_sim_features, get_sim_features_for_image

print("测试文件名匹配功能...")
print("="*50)

# 加载仿真特征
sim_feature_dict = load_sim_features("data/sim_features.csv")

if sim_feature_dict:
    print(f"\n成功加载 {len(sim_feature_dict)} 个特征")
    
    # 显示CSV键的示例
    print(f"\nCSV键示例:")
    for i, key in enumerate(list(sim_feature_dict.keys())[:5]):
        print(f"  {i+1}: {key}")
    
    # 测试几个图像文件
    test_images = [
        "data/combined_dataset/images/tier3/woolsey-fire_00000348_post_disaster.png",
        "data/combined_dataset/images/tier3/woolsey-fire_00000877_post_disaster.png",
        "data/combined_dataset/images/train2017/guatemala-volcano_00000000_post_disaster.png"
    ]
    
    print(f"\n测试图像文件名匹配:")
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n图像: {os.path.basename(test_image)}")
            sim_feats = get_sim_features_for_image(test_image, sim_feature_dict)
            
            if sim_feats.sum() != 0:
                print(f"  ✅ 匹配成功")
                print(f"  特征范围: [{sim_feats.min().item():.3f}, {sim_feats.max().item():.3f}]")
            else:
                print(f"  ❌ 匹配失败，使用零向量")
        else:
            print(f"\n图像不存在: {test_image}")
else:
    print("❌ 仿真特征加载失败") 