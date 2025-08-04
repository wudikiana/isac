#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的系统
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def test_calculate_distance():
    """测试距离计算函数"""
    print("🧪 测试距离计算函数...")
    
    try:
        from satellite_system import calculate_distance
        
        # 测试正常情况
        distance1 = calculate_distance([0, 0, 0], [1, 1, 1])
        print(f"   ✅ 正常情况: {distance1}")
        
        # 测试字符串输入（应该被转换为浮点数）
        distance2 = calculate_distance(['0', '0', '0'], ['1', '1', '1'])
        print(f"   ✅ 字符串输入: {distance2}")
        
        # 测试空列表
        distance3 = calculate_distance([], [])
        print(f"   ✅ 空列表: {distance3}")
        
        # 测试None输入
        distance4 = calculate_distance(None, None)
        print(f"   ✅ None输入: {distance4}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

def test_satellite_creation():
    """测试卫星创建"""
    print("\n🧪 测试卫星创建...")
    
    try:
        from satellite_system import SatelliteInfo, SatelliteStatus, create_satellite_info, load_config
        
        # 测试直接创建
        sat_info = SatelliteInfo(
            satellite_id="test_sat",
            ip_address="127.0.0.1",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1e12,
            memory_capacity=8192,
            current_load=0.0,
            last_heartbeat=0.0,
            model_version="v1.0",
            supported_features=["inference"],
            coverage_area={"lat": [35, 45], "lon": [110, 130]},
            current_position=[40.0, 116.0, 500.0],
            orbit_period=90.0
        )
        print(f"   ✅ 卫星创建成功: {sat_info.satellite_id}")
        
        # 测试从配置创建
        config = load_config("satellite_system/satellite_config.json")
        sat_info2 = create_satellite_info("sat_001", config)
        print(f"   ✅ 从配置创建成功: {sat_info2.satellite_id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emergency_system():
    """测试应急系统"""
    print("\n🧪 测试应急系统...")
    
    try:
        from satellite_system import (
            EmergencyResponseSystem, EmergencyBeacon, EmergencyLevel,
            SatelliteInfo, SatelliteStatus
        )
        
        # 创建配置
        config = {
            'response_timeout': 300,
            'max_concurrent_emergencies': 5,
            'ppo_learning_rate': 0.0003,
            'ppo_clip_ratio': 0.2
        }
        
        # 创建应急系统
        emergency_system = EmergencyResponseSystem(config)
        print("   ✅ 应急系统创建成功")
        
        # 创建卫星
        satellite = SatelliteInfo(
            satellite_id="test_sat",
            ip_address="127.0.0.1",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1e12,
            memory_capacity=8192,
            current_load=0.0,
            last_heartbeat=0.0,
            model_version="v1.0",
            supported_features=["inference"],
            coverage_area={"lat": [35, 45], "lon": [110, 130]},
            current_position=[40.0, 116.0, 500.0],
            orbit_period=90.0
        )
        
        # 注册卫星
        emergency_system.register_satellite(satellite)
        print("   ✅ 卫星注册成功")
        
        # 创建紧急信标
        beacon = EmergencyBeacon(
            beacon_id="test_beacon",
            location=[39.9, 116.4, 0],
            emergency_level=EmergencyLevel.HIGH,
            timestamp=0.0,
            description="测试紧急情况",
            required_resources={}
        )
        
        # 触发紧急情况
        emergency_system.trigger_emergency_beacon(beacon)
        print("   ✅ 紧急情况触发成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
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
        print("   ✅ 卫星发现成功")
        
        # 测试任务提交
        image_data = np.random.rand(3, 256, 256).astype(np.float32)
        sim_features = np.random.rand(11).astype(np.float32)
        
        task_id = system.submit_inference_task(
            image_data=image_data,
            sim_features=sim_features,
            priority=5,
            location=[39.9, 116.4]
        )
        print(f"   ✅ 任务提交成功: {task_id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始测试修复后的系统")
    print("=" * 50)
    
    tests = [
        ("距离计算", test_calculate_distance),
        ("卫星创建", test_satellite_creation),
        ("应急系统", test_emergency_system),
        ("推理系统", test_inference_system)
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
        print("🎉 所有测试通过！修复成功！")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 