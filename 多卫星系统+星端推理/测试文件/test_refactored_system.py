#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构后的卫星系统
验证所有功能模块是否正常工作
"""

import sys
import os
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试核心模块导入
        from satellite_system import (
            SatelliteStatus, CoverageStatus, EmergencyLevel, WaveformType,
            SatelliteInfo, InferenceTask, TrainingTask, EmergencyBeacon,
            calculate_distance, load_config, get_default_config
        )
        print("   ✅ 核心模块导入成功")
        
        # 测试推理系统导入
        from satellite_system import MultiSatelliteInferenceSystem
        print("   ✅ 推理系统导入成功")
        
        # 测试主系统导入
        from satellite_system import MultiSatelliteSystem
        print("   ✅ 主系统导入成功")
        
        # 测试交互系统导入
        from satellite_system import InteractiveSatelliteSystem
        print("   ✅ 交互系统导入成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_functionality():
    """测试核心功能"""
    print("\n🧪 测试核心功能...")
    
    try:
        from satellite_system import (
            SatelliteStatus, EmergencyLevel, SatelliteInfo, calculate_distance
        )
        
        # 测试枚举
        assert SatelliteStatus.ONLINE.value == "online"
        assert EmergencyLevel.HIGH.value == 3
        print("   ✅ 枚举测试通过")
        
        # 测试数据结构
        sat_info = SatelliteInfo(
            satellite_id="test_sat",
            ip_address="127.0.0.1",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1e12,
            memory_capacity=8192,
            current_load=0.0,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["inference"],
            coverage_area={"lat": [35, 45], "lon": [110, 130]},
            current_position=[40.0, 116.0, 500.0],
            orbit_period=90.0
        )
        print("   ✅ 数据结构测试通过")
        
        # 测试工具函数
        distance = calculate_distance([0, 0, 0], [1, 1, 1])
        assert distance > 0
        print("   ✅ 工具函数测试通过")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 核心功能测试失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")
    
    try:
        from satellite_system import load_config, get_default_config
        
        # 测试默认配置
        default_config = get_default_config()
        assert "satellites" in default_config
        assert "emergency_response" in default_config
        print("   ✅ 默认配置加载成功")
        
        # 测试配置文件加载
        config = load_config("satellite_system/satellite_config.json")
        assert "satellites" in config
        print("   ✅ 配置文件加载成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 配置加载测试失败: {e}")
        return False

def test_inference_system():
    """测试推理系统"""
    print("\n🧪 测试推理系统...")
    
    try:
        from satellite_system import MultiSatelliteInferenceSystem
        
        # 创建推理系统
        system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")
        print("   ✅ 推理系统创建成功")
        
        # 测试卫星发现
        system.discover_satellites()
        print("   ✅ 卫星发现功能正常")
        
        # 测试任务提交
        image_data = np.random.rand(3, 256, 256).astype(np.float32)
        sim_features = np.random.rand(11).astype(np.float32)
        
        task_id = system.submit_inference_task(
            image_data=image_data,
            sim_features=sim_features,
            priority=5,
            location=[39.9, 116.4]
        )
        print(f"   ✅ 推理任务提交成功: {task_id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 推理系统测试失败: {e}")
        return False

def test_main_system():
    """测试主系统"""
    print("\n🧪 测试主系统...")
    
    try:
        from satellite_system import MultiSatelliteSystem, EmergencyLevel
        
        # 创建主系统
        system = MultiSatelliteSystem("satellite_system/satellite_config.json")
        print("   ✅ 主系统创建成功")
        
        # 测试系统状态
        status = system.get_system_status()
        assert "total_satellites" in status
        assert "online_satellites" in status
        print("   ✅ 系统状态获取成功")
        
        # 测试应急功能
        emergency_id = system.trigger_emergency(
            location=[39.9, 116.4, 0],
            emergency_level=EmergencyLevel.HIGH,
            description="测试紧急情况"
        )
        print(f"   ✅ 应急功能测试成功: {emergency_id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 主系统测试失败: {e}")
        return False

def test_system_info():
    """测试系统信息"""
    print("\n🧪 测试系统信息...")
    
    try:
        from satellite_system import get_system_info
        
        info = get_system_info()
        assert "version" in info
        assert "description" in info
        assert "modules" in info
        
        print(f"   ✅ 系统版本: {info['version']}")
        print(f"   ✅ 系统描述: {info['description']}")
        print(f"   ✅ 模块数量: {len(info['modules'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 系统信息测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始测试重构后的卫星系统")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("核心功能", test_core_functionality),
        ("配置加载", test_config_loading),
        ("推理系统", test_inference_system),
        ("主系统", test_main_system),
        ("系统信息", test_system_info)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统重构成功！")
        print("\n💡 下一步操作:")
        print("   python satellite_system/quick_demo.py              # 快速演示")
        print("   python satellite_system/start_system.py --mode demo # 完整演示")
        print("   python satellite_system/start_system.py --mode interactive # 交互模式")
        return True
    else:
        print("⚠️  部分测试失败，请检查系统配置")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试重构后的卫星系统')
    parser.add_argument('--test', type=str, choices=[
        'imports', 'core', 'config', 'inference', 'main', 'info', 'all'
    ], default='all', help='测试类型')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
    else:
        # 运行单个测试
        test_map = {
            'imports': test_imports,
            'core': test_core_functionality,
            'config': test_config_loading,
            'inference': test_inference_system,
            'main': test_main_system,
            'info': test_system_info
        }
        
        test_func = test_map.get(args.test)
        if test_func:
            print(f"🧪 运行 {args.test} 测试...")
            success = test_func()
        else:
            print(f"❌ 未知测试类型: {args.test}")
            success = False
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 