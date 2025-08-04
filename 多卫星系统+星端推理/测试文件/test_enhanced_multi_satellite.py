#!/usr/bin/env python3
"""
增强版多卫星推理系统测试脚本
演示分布式训练、覆盖范围管理和负载均衡功能
"""

import numpy as np
import time
import json
import logging
from enhanced_multi_satellite_inference import (
    MultiSatelliteInferenceSystem, 
    SatelliteInfo, 
    SatelliteStatus, 
    CoverageStatus
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_inference():
    """测试基本推理功能"""
    print("\n=== 测试基本推理功能 ===")
    
    # 创建系统实例
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    
    # 模拟卫星发现
    system.discover_satellites()
    
    # 生成测试数据
    image_data = np.random.rand(3, 256, 256).astype(np.float32)
    sim_features = np.random.rand(11).astype(np.float32)
    
    # 提交推理任务
    task_id = system.submit_inference_task(
        image_data=image_data,
        sim_features=sim_features,
        priority=5,
        location=[39.9, 116.4]  # 北京坐标
    )
    
    print(f"提交推理任务: {task_id}")
    
    # 获取结果
    result = system.get_inference_result(task_id, timeout=60.0)
    if result:
        print(f"推理完成: 处理时间 {result['processing_time']:.2f}s")
        print(f"处理卫星: {result['satellite_id']}")
    else:
        print("推理超时或失败")

def test_coverage_management():
    """测试覆盖范围管理"""
    print("\n=== 测试覆盖范围管理 ===")
    
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    system.discover_satellites()
    
    # 模拟更新卫星位置
    beijing_location = [39.9, 116.4, 500]  # 北京上空500km
    shanghai_location = [31.2, 121.5, 500]  # 上海上空500km
    guangzhou_location = [23.1, 113.3, 500]  # 广州上空500km
    
    # 更新卫星位置
    system.update_satellite_position("sat_001", beijing_location)
    system.update_satellite_position("sat_002", shanghai_location)
    system.update_satellite_position("sat_003", guangzhou_location)
    
    # 测试不同位置的卫星选择
    test_locations = [
        ([39.9, 116.4], "北京"),
        ([31.2, 121.5], "上海"),
        ([23.1, 113.3], "广州"),
        ([22.3, 114.2], "深圳")
    ]
    
    for location, city_name in test_locations:
        optimal_satellite = system.get_optimal_satellite_for_location(location)
        print(f"{city_name} ({location[0]:.1f}, {location[1]:.1f}): 最优卫星 {optimal_satellite}")

def test_distributed_training():
    """测试分布式训练功能"""
    print("\n=== 测试分布式训练功能 ===")
    
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    system.discover_satellites()
    
    # 模拟训练数据
    training_data = []
    for i in range(10):
        image = np.random.rand(3, 256, 256).astype(np.float32)
        mask = np.random.rand(256, 256).astype(np.float32)
        training_data.append((image, mask))
    
    sim_features = np.random.rand(11).astype(np.float32)
    
    # 提交训练任务到不同卫星
    for sat_id in ["sat_001", "sat_002", "sat_003"]:
        task_id = system.submit_training_task(
            satellite_id=sat_id,
            training_data=training_data,
            sim_features=sim_features,
            location=[39.9, 116.4]
        )
        print(f"提交训练任务到卫星 {sat_id}: {task_id}")
    
    # 模拟参数同步
    import torch
    mock_parameters = {
        'conv1.weight': torch.randn(64, 3, 7, 7),
        'conv1.bias': torch.randn(64),
        'fc.weight': torch.randn(1000, 512),
        'fc.bias': torch.randn(1000)
    }
    
    system.sync_model_parameters("sat_001", mock_parameters)
    print("已同步卫星 sat_001 的模型参数")

def test_load_balancing():
    """测试负载均衡功能"""
    print("\n=== 测试负载均衡功能 ===")
    
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    system.discover_satellites()
    
    # 模拟不同负载的卫星
    for sat_id, sat_info in system.satellites.items():
        if sat_info.status == SatelliteStatus.ONLINE:
            # 模拟不同负载
            if sat_id == "sat_001":
                sat_info.current_load = 0.3  # 低负载
            elif sat_id == "sat_002":
                sat_info.current_load = 0.7  # 中等负载
            elif sat_id == "sat_003":
                sat_info.current_load = 0.9  # 高负载
    
    # 提交多个任务测试负载均衡
    for i in range(5):
        image_data = np.random.rand(3, 256, 256).astype(np.float32)
        sim_features = np.random.rand(11).astype(np.float32)
        
        task_id = system.submit_inference_task(
            image_data=image_data,
            sim_features=sim_features,
            priority=np.random.randint(1, 10),
            location=[39.9, 116.4]
        )
        print(f"任务 {i+1}: {task_id}")

def test_fault_tolerance():
    """测试故障容错功能"""
    print("\n=== 测试故障容错功能 ===")
    
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    system.discover_satellites()
    
    # 模拟卫星故障
    if "sat_001" in system.satellites:
        system.satellites["sat_001"].status = SatelliteStatus.OFFLINE
        print("模拟卫星 sat_001 故障")
    
    # 提交任务，应该自动选择其他卫星
    image_data = np.random.rand(3, 256, 256).astype(np.float32)
    sim_features = np.random.rand(11).astype(np.float32)
    
    task_id = system.submit_inference_task(
        image_data=image_data,
        sim_features=sim_features,
        priority=8,
        location=[39.9, 116.4]
    )
    
    print(f"故障容错测试任务: {task_id}")
    
    # 获取结果
    result = system.get_inference_result(task_id, timeout=60.0)
    if result:
        print(f"故障容错测试成功: 由 {result['satellite_id']} 处理")

def test_system_status():
    """测试系统状态监控"""
    print("\n=== 测试系统状态监控 ===")
    
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    system.discover_satellites()
    
    # 获取系统状态
    status = system.get_system_status()
    
    print(f"系统状态:")
    print(f"  总卫星数: {status['total_satellites']}")
    print(f"  在线卫星数: {status['online_satellites']}")
    print(f"  队列大小: {status['queue_size']}")
    print(f"  缓存大小: {status['cache_size']}")
    
    print("\n各卫星状态:")
    for sat_id, sat_status in status['satellites'].items():
        print(f"  {sat_id}:")
        print(f"    状态: {sat_status['status']}")
        print(f"    负载: {sat_status['load']:.2f}")
        print(f"    覆盖状态: {sat_status['coverage_status']}")
        print(f"    训练数据数: {sat_status['training_data_count']}")
        print(f"    参数版本: {sat_status['parameter_version']}")

def test_coverage_prediction():
    """测试覆盖预测功能"""
    print("\n=== 测试覆盖预测功能 ===")
    
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    system.discover_satellites()
    
    # 模拟卫星轨迹数据
    current_time = time.time()
    
    # 卫星001的轨迹（从北京到上海）
    for i in range(10):
        lat = 39.9 + (31.2 - 39.9) * i / 9
        lon = 116.4 + (121.5 - 116.4) * i / 9
        position = [lat, lon, 500]
        system.update_satellite_position("sat_001", position)
        time.sleep(0.1)
    
    # 测试不同时间点的覆盖预测
    test_locations = [
        ([39.9, 116.4], "北京"),
        ([31.2, 121.5], "上海"),
        ([23.1, 113.3], "广州")
    ]
    
    for location, city_name in test_locations:
        for sat_id in ["sat_001", "sat_002", "sat_003"]:
            if sat_id in system.coverage_manager.coverage_history:
                coverage_status = system.coverage_manager.predict_coverage(
                    sat_id, location, time_horizon=3600
                )
                print(f"{city_name} - {sat_id}: {coverage_status.value}")

def main():
    """主测试函数"""
    print("增强版多卫星推理系统测试")
    print("=" * 50)
    
    try:
        # 运行各项测试
        test_basic_inference()
        test_coverage_management()
        test_distributed_training()
        test_load_balancing()
        test_fault_tolerance()
        test_system_status()
        test_coverage_prediction()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 