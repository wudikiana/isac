import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.abspath('.'))

def test_sim_features():
    print("测试仿真特征加载...")
    
    try:
        from run_inference import load_sim_features
        sim_features = load_sim_features()
        
        print(f"✅ 成功加载 {len(sim_features)} 个仿真特征")
        
        if sim_features:
            # 显示第一个特征
            first_key = list(sim_features.keys())[0]
            first_features = sim_features[first_key]
            print(f"第一个文件名: {first_key}")
            print(f"特征维度: {first_features.shape}")
            print(f"特征值: {first_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ 仿真特征加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sim_features()
    if success:
        print("\n🎉 仿真特征加载修复成功!")
    else:
        print("\n💥 仿真特征加载还有问题。") 