#!/usr/bin/env python3
"""
快速测试多卫星推理系统
"""

import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_satellite_connection():
    """测试卫星连接"""
    print("🔍 测试卫星连接...")
    
    try:
        from multi_satellite_inference import MultiSatelliteInferenceSystem
        
        # 创建系统
        system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
        
        # 发现卫星
        system.discover_satellites()
        
        # 获取系统状态
        status = system.get_system_status()
        
        print(f"✅ 系统状态:")
        print(f"   总卫星数: {status['total_satellites']}")
        print(f"   在线卫星数: {status['online_satellites']}")
        print(f"   队列大小: {status['queue_size']}")
        
        if status['online_satellites'] > 0:
            print("✅ 卫星连接正常")
            return True
        else:
            print("❌ 没有在线卫星")
            return False
            
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

def test_inference():
    """测试推理功能"""
    print("\n🧪 测试推理功能...")
    
    try:
        from multi_satellite_inference import MultiSatelliteInferenceSystem
        
        # 创建系统
        system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
        system.discover_satellites()
        
        # 生成测试数据
        image_data = np.random.rand(3, 256, 256).astype(np.float32)
        sim_features = np.random.rand(11).astype(np.float32)
        
        print(f"   图像数据形状: {image_data.shape}")
        print(f"   仿真特征形状: {sim_features.shape}")
        
        # 提交推理任务
        task_id = system.submit_inference_task(
            image_data=image_data,
            sim_features=sim_features,
            priority=5,
            timeout=30.0
        )
        
        print(f"   提交任务: {task_id}")
        
        # 获取结果
        start_time = time.time()
        result = system.get_inference_result(task_id, timeout=60.0)
        end_time = time.time()
        
        if result:
            print(f"✅ 推理成功:")
            print(f"   状态: {result['status']}")
            print(f"   处理时间: {result['processing_time']:.3f}s")
            print(f"   总耗时: {end_time - start_time:.3f}s")
            print(f"   卫星ID: {result['satellite_id']}")
            print(f"   预测形状: {result['prediction'].shape}")
            return True
        else:
            print("❌ 推理失败")
            return False
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

def test_load_balancing():
    """测试负载均衡"""
    print("\n⚖️ 测试负载均衡...")
    
    try:
        from multi_satellite_inference import MultiSatelliteInferenceSystem
        
        # 创建系统
        system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
        system.discover_satellites()
        
        # 提交多个任务
        task_ids = []
        for i in range(5):
            image_data = np.random.rand(3, 256, 256).astype(np.float32)
            sim_features = np.random.rand(11).astype(np.float32)
            
            task_id = system.submit_inference_task(
                image_data=image_data,
                sim_features=sim_features,
                priority=np.random.randint(1, 10)
            )
            task_ids.append(task_id)
        
        print(f"   提交了 {len(task_ids)} 个任务")
        
        # 收集结果
        results = []
        satellite_counts = {}
        
        for task_id in task_ids:
            result = system.get_inference_result(task_id, timeout=60.0)
            if result:
                results.append(result)
                sat_id = result['satellite_id']
                satellite_counts[sat_id] = satellite_counts.get(sat_id, 0) + 1
        
        print(f"   完成任务: {len(results)}/{len(task_ids)}")
        print(f"   负载分布: {satellite_counts}")
        
        if len(satellite_counts) > 1:
            print("✅ 负载均衡正常工作")
            return True
        else:
            print("⚠️ 负载均衡可能存在问题")
            return False
            
    except Exception as e:
        print(f"❌ 负载均衡测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 多卫星推理系统快速测试")
    print("=" * 50)
    
    # 检查依赖
    print("📋 检查依赖...")
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    try:
        import segmentation_models_pytorch as smp
        print("   segmentation_models_pytorch: ✓")
    except ImportError:
        print("❌ segmentation_models_pytorch未安装")
        print("请运行: pip install segmentation-models-pytorch")
        return
    
    # 检查模型文件
    model_files = [
        "models/best_multimodal_patch_model.pth",
        "models/seg_model.pth",
        "models/seg_model_best.pth"
    ]
    
    model_found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"   模型文件: {model_file} ✓")
            model_found = True
            break
    
    if not model_found:
        print("⚠️ 未找到模型文件")
        print("请确保模型文件存在")
    
    print()
    
    # 运行测试
    tests = [
        ("卫星连接", test_satellite_connection),
        ("推理功能", test_inference),
        ("负载均衡", test_load_balancing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常")
    else:
        print(f"\n⚠️ 有 {total-passed} 个测试失败")
        print("请检查卫星服务器是否正常启动")
    
    print("\n💡 提示:")
    print("1. 确保所有卫星服务器已启动")
    print("2. 检查配置文件中的IP地址和端口")
    print("3. 查看详细日志以获取更多信息")

if __name__ == "__main__":
    main() 