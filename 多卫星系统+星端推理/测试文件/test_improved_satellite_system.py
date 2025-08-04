#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的多卫星推理系统
验证联邦学习集成、模型加载、故障容错等功能
"""

import torch
import numpy as np
import time
import logging
import json
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """测试模型加载功能"""
    logger.info("=== 测试模型加载功能 ===")
    
    from satellite_system.satellite_inference import MultiSatelliteInferenceSystem
    
    # 创建系统实例
    system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")
    
    # 检查本地模型是否加载成功
    if system.local_model is not None:
        logger.info("✅ 本地模型加载成功")
        
        # 测试模型推理
        test_image = np.random.rand(3, 256, 256).astype(np.float32)
        test_sim_features = np.random.rand(11).astype(np.float32)
        
        try:
            # 转换为tensor
            image_tensor = torch.from_numpy(test_image).unsqueeze(0)
            sim_tensor = torch.from_numpy(test_sim_features).unsqueeze(0)
            
            # 移动到GPU
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                sim_tensor = sim_tensor.cuda()
                system.local_model = system.local_model.cuda()
            
            # 推理测试
            with torch.no_grad():
                output = system.local_model(image_tensor, sim_tensor)
                logger.info(f"✅ 模型推理成功，输出形状: {output.shape}")
                
        except Exception as e:
            logger.error(f"❌ 模型推理失败: {e}")
    else:
        logger.warning("⚠️ 本地模型加载失败")

def test_federated_learning_integration():
    """测试联邦学习集成"""
    logger.info("=== 测试联邦学习集成 ===")
    
    from satellite_system.satellite_inference import MultiSatelliteInferenceSystem
    
    system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")
    
    # 检查联邦学习管理器
    if system.federated_manager is not None:
        logger.info("✅ 联邦学习管理器初始化成功")
        
        # 获取联邦学习状态
        federated_status = system.federated_manager.get_federated_status()
        logger.info(f"联邦学习状态: {federated_status}")
        
        # 检查训练协调器
        if system.training_coordinator is not None:
            logger.info("✅ 训练协调器初始化成功")
            
            # 获取训练状态
            training_status = system.training_coordinator.get_training_status()
            logger.info(f"训练状态: {training_status}")
        else:
            logger.warning("⚠️ 训练协调器初始化失败")
    else:
        logger.warning("⚠️ 联邦学习管理器初始化失败")

def test_load_balancer():
    """测试负载均衡器"""
    logger.info("=== 测试负载均衡器 ===")
    
    from satellite_system.satellite_inference import LoadBalancer
    from satellite_system.satellite_core import SatelliteInfo, SatelliteStatus, InferenceTask, CoverageStatus
    
    # 创建负载均衡器
    load_balancer = LoadBalancer()
    
    # 创建模拟卫星
    satellites = {
        "satellite_1": SatelliteInfo(
            satellite_id="satellite_1",
            ip_address="192.168.1.101",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1000.0,
            memory_capacity=8192,
            current_load=0.3,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["inference", "training"],
            coverage_area={},
            current_position=[116.4074, 39.9042, 500],
            orbit_period=90.0,
            federated_score=0.8
        ),
        "satellite_2": SatelliteInfo(
            satellite_id="satellite_2", 
            ip_address="192.168.1.102",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1000.0,
            memory_capacity=8192,
            current_load=0.7,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["inference", "training"],
            coverage_area={},
            current_position=[121.4737, 31.2304, 600],
            orbit_period=90.0,
            federated_score=0.6
        ),
        "satellite_3": SatelliteInfo(
            satellite_id="satellite_3",
            ip_address="192.168.1.103", 
            port=8080,
            status=SatelliteStatus.OFFLINE,
            compute_capacity=1000.0,
            memory_capacity=8192,
            current_load=0.9,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["inference", "training"],
            coverage_area={},
            current_position=[104.0668, 30.5728, 400],
            orbit_period=90.0,
            federated_score=0.4
        )
    }
    
    # 创建测试任务
    task = InferenceTask(
        task_id="test_task",
        image_data=np.random.rand(3, 256, 256),
        sim_features=np.random.rand(11),
        priority=5,
        timestamp=time.time(),
        timeout=30.0,
        location=[116.4074, 39.9042]  # 北京坐标
    )
    
    # 测试不同的负载均衡策略
    strategies = ["coverage_aware", "least_load", "round_robin", "federated_aware"]
    
    for strategy in strategies:
        load_balancer.strategy = strategy
        selected_satellite = load_balancer.select_satellite(satellites, task)
        
        if selected_satellite:
            logger.info(f"✅ {strategy} 策略选择卫星: {selected_satellite.satellite_id}")
        else:
            logger.warning(f"⚠️ {strategy} 策略未选择到卫星")

def test_fault_tolerance():
    """测试故障容错机制"""
    logger.info("=== 测试故障容错机制 ===")
    
    from satellite_system.satellite_inference import FaultToleranceManager, MultiSatelliteInferenceSystem
    from satellite_system.satellite_core import SatelliteInfo, SatelliteStatus, InferenceTask
    
    # 创建故障容错管理器
    fault_tolerance = FaultToleranceManager()
    
    # 创建模拟系统
    system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")
    
    # 创建模拟卫星和任务
    failed_satellite = SatelliteInfo(
        satellite_id="failed_satellite",
        ip_address="192.168.1.104",
        port=8080,
        status=SatelliteStatus.ONLINE,
        compute_capacity=1000.0,
        memory_capacity=8192,
        current_load=0.5,
        last_heartbeat=time.time(),
        model_version="v1.0",
        supported_features=["inference", "training"],
        coverage_area={},
        current_position=[116.4074, 39.9042, 500],
        orbit_period=90.0
    )
    
    task = InferenceTask(
        task_id="fault_test_task",
        image_data=np.random.rand(3, 256, 256),
        sim_features=np.random.rand(11),
        priority=5,
        timestamp=time.time(),
        timeout=30.0
    )
    
    # 测试故障处理
    logger.info("测试故障处理...")
    fault_tolerance.handle_failure(task, failed_satellite, system)
    
    # 检查故障计数
    failure_count = fault_tolerance.failure_count.get("failed_satellite", 0)
    logger.info(f"故障计数: {failure_count}")

def test_coverage_manager():
    """测试覆盖管理器"""
    logger.info("=== 测试覆盖管理器 ===")
    
    from satellite_system.satellite_inference import CoverageManager
    from satellite_system.satellite_core import SatelliteInfo, SatelliteStatus, CoverageStatus
    
    # 创建覆盖管理器
    coverage_manager = CoverageManager()
    
    # 更新卫星位置
    coverage_manager.update_satellite_position("satellite_1", [116.4074, 39.9042, 500], time.time())
    coverage_manager.update_satellite_position("satellite_2", [121.4737, 31.2304, 600], time.time())
    
    # 测试覆盖预测
    target_location = [116.4074, 39.9042]  # 北京
    
    for sat_id in ["satellite_1", "satellite_2"]:
        coverage_status = coverage_manager.predict_coverage(sat_id, target_location)
        logger.info(f"卫星 {sat_id} 覆盖状态: {coverage_status}")
    
    # 创建模拟卫星
    satellites = {
        "satellite_1": SatelliteInfo(
            satellite_id="satellite_1",
            ip_address="192.168.1.101",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1000.0,
            memory_capacity=8192,
            current_load=0.3,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["inference", "training"],
            coverage_area={},
            current_position=[116.4074, 39.9042, 500],
            orbit_period=90.0,
            federated_score=0.8
        ),
        "satellite_2": SatelliteInfo(
            satellite_id="satellite_2",
            ip_address="192.168.1.102", 
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1000.0,
            memory_capacity=8192,
            current_load=0.7,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["inference", "training"],
            coverage_area={},
            current_position=[121.4737, 31.2304, 600],
            orbit_period=90.0,
            federated_score=0.6
        )
    }
    
    # 测试最优卫星选择
    optimal_satellite = coverage_manager.get_optimal_satellite(target_location, satellites)
    if optimal_satellite:
        logger.info(f"✅ 最优卫星: {optimal_satellite}")
    else:
        logger.warning("⚠️ 未找到最优卫星")

def test_inference_task():
    """测试推理任务处理"""
    logger.info("=== 测试推理任务处理 ===")
    
    from satellite_system.satellite_inference import MultiSatelliteInferenceSystem
    
    # 创建系统
    system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")
    
    # 创建测试图像和仿真特征
    test_image = np.random.rand(3, 256, 256).astype(np.float32)
    test_sim_features = np.random.rand(11).astype(np.float32)
    
    # 提交推理任务
    task_id = system.submit_inference_task(
        image_data=test_image,
        sim_features=test_sim_features,
        priority=5,
        timeout=30.0,
        location=[116.4074, 39.9042]
    )
    
    logger.info(f"✅ 提交推理任务: {task_id}")
    
    # 等待一段时间让任务处理
    time.sleep(2)
    
    # 获取系统状态
    status = system.get_system_status()
    logger.info(f"系统状态: {status}")
    
    # 尝试获取结果
    result = system.get_inference_result(task_id, timeout=5.0)
    if result:
        logger.info(f"✅ 获取推理结果成功: {result.get('processed_by', 'unknown')}")
    else:
        logger.warning("⚠️ 未获取到推理结果")

def test_system_integration():
    """测试系统集成"""
    logger.info("=== 测试系统集成 ===")
    
    from satellite_system.satellite_inference import MultiSatelliteInferenceSystem
    
    # 创建系统
    system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")
    
    # 测试系统状态
    status = system.get_system_status()
    logger.info(f"系统状态: {status}")
    
    # 测试卫星位置更新
    system.update_satellite_position("satellite_1", [116.4074, 39.9042, 500])
    logger.info("✅ 卫星位置更新成功")
    
    # 测试最优卫星选择
    optimal_satellite = system.get_optimal_satellite_for_location([116.4074, 39.9042])
    if optimal_satellite:
        logger.info(f"✅ 位置最优卫星: {optimal_satellite}")
    else:
        logger.warning("⚠️ 未找到位置最优卫星")

def main():
    """主测试函数"""
    logger.info("开始测试改进后的多卫星推理系统...")
    
    try:
        # 运行各项测试
        test_model_loading()
        test_federated_learning_integration()
        test_load_balancer()
        test_fault_tolerance()
        test_coverage_manager()
        test_inference_task()
        test_system_integration()
        
        logger.info("所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 