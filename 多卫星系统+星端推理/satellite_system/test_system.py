#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星系统测试脚本
测试重构后的卫星系统功能
"""

import sys
import os
import time
import numpy as np
import threading
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_core_modules():
    """测试核心模块"""
    print("🧪 测试核心模块...")
    
    try:
        from satellite_system import (
            SatelliteStatus, CoverageStatus, EmergencyLevel, WaveformType,
            SatelliteInfo, InferenceTask, TrainingTask, EmergencyBeacon,
            calculate_distance, load_config, create_satellite_info
        )
        
        # 测试枚举
        assert SatelliteStatus.ONLINE.value == "online"
        assert EmergencyLevel.HIGH.value == 3
        
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
        
        # 测试工具函数
        distance = calculate_distance([0, 0, 0], [1, 1, 1])
        assert distance > 0
        
        print("   ✅ 核心模块测试通过")
        return True
        
    except Exception as e:
        print(f"   ❌ 核心模块测试失败: {e}")
        return False

def test_inference_system():
    """测试推理系统"""
    print("🧪 测试推理系统...")
    
    try:
        from satellite_system import MultiSatelliteInferenceSystem
        
        # 创建推理系统
        system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")
        
        # 测试卫星发现
        system.discover_satellites()
        
        # 生成测试数据
        image_data = np.random.rand(3, 256, 256).astype(np.float32)
        sim_features = np.random.rand(11).astype(np.float32)
        
        # 测试任务提交
        task_id = system.submit_inference_task(
            image_data=image_data,
            sim_features=sim_features,
            priority=5,
            location=[39.9, 116.4]
        )
        
        print(f"   ✅ 推理系统测试通过，任务ID: {task_id}")
        return True
        
    except Exception as e:
        print(f"   ❌ 推理系统测试失败: {e}")
        return False

def test_emergency_system():
    """测试应急系统"""
    print("🧪 测试应急系统...")
    
    try:
        from satellite_system import MultiSatelliteSystem, EmergencyLevel
        
        # 创建主系统
        system = MultiSatelliteSystem("satellite_system/satellite_config.json")
        
        # 测试紧急情况触发
        emergency_id = system.trigger_emergency(
            location=[39.9, 116.4, 0],
            emergency_level=EmergencyLevel.HIGH,
            description="测试紧急情况"
        )
        
        # 获取应急历史
        history = system.get_emergency_history(limit=5)
        
        print(f"   ✅ 应急系统测试通过，紧急ID: {emergency_id}")
        print(f"   📈 应急历史数量: {len(history)}")
        return True
        
    except Exception as e:
        print(f"   ❌ 应急系统测试失败: {e}")
        return False

def test_communication_system():
    """测试通信系统"""
    print("🧪 测试通信系统...")
    
    try:
        from satellite_system import CognitiveRadioManager
        
        # 创建认知无线电管理器
        config = {
            'available_bands': [2.4e9, 5.8e9, 12e9],
            'max_bandwidth': 20e6,
            'interference_threshold': 0.1
        }
        
        radio_manager = CognitiveRadioManager(config)
        
        # 测试频谱管理
        terrestrial_users = {
            'user1': {'active': True, 'frequency': 2.4e9}
        }
        
        occupied_bands = radio_manager.detect_primary_users(terrestrial_users)
        
        print(f"   ✅ 通信系统测试通过，占用频段: {len(occupied_bands)}")
        return True
        
    except Exception as e:
        print(f"   ❌ 通信系统测试失败: {e}")
        return False

def test_orbit_system():
    """测试轨道系统"""
    print("🧪 测试轨道系统...")
    
    try:
        from satellite_system import OrbitManager
        
        # 创建轨道管理器
        config = {
            'collision_threshold': 500,
            'formation_threshold': 1000,
            'fuel_efficiency': 0.8
        }
        
        orbit_manager = OrbitManager(config)
        
        # 测试空间碎片管理
        debris_info = {
            'position': [1000, 1000, 1000],
            'velocity': [100, 100, 100],
            'size': 10
        }
        
        orbit_manager.add_space_debris(debris_info)
        debris_count = orbit_manager.get_space_debris_count()
        
        print(f"   ✅ 轨道系统测试通过，碎片数量: {debris_count}")
        return True
        
    except Exception as e:
        print(f"   ❌ 轨道系统测试失败: {e}")
        return False

def test_federated_learning():
    """测试联邦学习系统"""
    print("🧪 测试联邦学习系统...")
    
    try:
        from satellite_system import FederatedLearningManager
        
        # 创建联邦学习管理器
        config = {
            'sync_interval': 600,
            'min_participants': 2,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        fed_manager = FederatedLearningManager(config)
        
        # 测试状态获取
        status = fed_manager.get_federated_status()
        
        print(f"   ✅ 联邦学习测试通过，聚合轮次: {status['aggregation_round']}")
        return True
        
    except Exception as e:
        print(f"   ❌ 联邦学习测试失败: {e}")
        return False

def test_interactive_system():
    """测试交互式系统"""
    print("🧪 测试交互式系统...")
    
    try:
        from satellite_system import InteractiveSatelliteSystem
        
        # 创建交互式系统
        system = InteractiveSatelliteSystem("satellite_system/satellite_config.json")
        
        # 测试系统信息获取
        system_info = system.system.get_system_status() if system.system else {}
        
        print(f"   ✅ 交互式系统测试通过")
        return True
        
    except Exception as e:
        print(f"   ❌ 交互式系统测试失败: {e}")
        return False

def test_system_integration():
    """测试系统集成"""
    print("🧪 测试系统集成...")
    
    try:
        from satellite_system import MultiSatelliteSystem, EmergencyLevel
        
        # 创建完整系统
        system = MultiSatelliteSystem("satellite_system/satellite_config.json")
        
        # 测试系统状态
        status = system.get_system_status()
        
        # 测试推理功能
        image_data = np.random.rand(3, 256, 256).astype(np.float32)
        task_id = system.submit_inference_task(image_data)
        
        # 测试应急功能
        emergency_id = system.trigger_emergency(
            location=[39.9, 116.4, 0],
            emergency_level=EmergencyLevel.MEDIUM,
            description="集成测试"
        )
        
        # 测试轨道功能
        system.add_space_debris({'position': [1000, 1000, 1000], 'velocity': [100, 100, 100]})
        
        print(f"   ✅ 系统集成测试通过")
        print(f"   📊 系统状态: {status['total_satellites']} 卫星")
        print(f"   🤖 推理任务: {task_id}")
        print(f"   🚨 应急ID: {emergency_id}")
        return True
        
    except Exception as e:
        print(f"   ❌ 系统集成测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始卫星系统测试")
    print("=" * 50)
    
    tests = [
        ("核心模块", test_core_modules),
        ("推理系统", test_inference_system),
        ("应急系统", test_emergency_system),
        ("通信系统", test_communication_system),
        ("轨道系统", test_orbit_system),
        ("联邦学习", test_federated_learning),
        ("交互式系统", test_interactive_system),
        ("系统集成", test_system_integration)
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
    print("\n" + "=" * 50)
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
        return True
    else:
        print("⚠️  部分测试失败，请检查系统配置")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='卫星系统测试脚本')
    parser.add_argument('--test', type=str, choices=[
        'core', 'inference', 'emergency', 'communication', 
        'orbit', 'federated', 'interactive', 'integration', 'all'
    ], default='all', help='测试类型')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
    else:
        # 运行单个测试
        test_map = {
            'core': test_core_modules,
            'inference': test_inference_system,
            'emergency': test_emergency_system,
            'communication': test_communication_system,
            'orbit': test_orbit_system,
            'federated': test_federated_learning,
            'interactive': test_interactive_system,
            'integration': test_system_integration
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