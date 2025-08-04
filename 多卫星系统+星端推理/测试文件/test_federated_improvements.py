#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的联邦学习系统
验证容错机制、数据验证、数值稳定性等功能
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟卫星信息
class MockSatelliteInfo:
    def __init__(self, satellite_id: str, status: str = "ONLINE"):
        self.satellite_id = satellite_id
        self.status = status
        self.parameter_version = 0

# 简单的测试模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_parameter_validation():
    """测试参数验证功能"""
    logger.info("=== 测试参数验证功能 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager
    
    # 创建联邦学习管理器
    config = {
        'min_participants': 2,
        'parameter_validation_enabled': True,
        'min_data_size': 10,
        'max_data_size': 1000,
        'min_loss_threshold': 1e-6,
        'max_loss_threshold': 100.0
    }
    
    federated_manager = FederatedLearningManager(config)
    
    # 初始化全局模型
    model = SimpleModel()
    federated_manager.initialize_global_model(model)
    
    # 测试有效参数
    valid_params = model.state_dict()
    result = federated_manager.submit_local_update("satellite_1", valid_params, 100, 0.5)
    logger.info(f"有效参数提交结果: {result}")
    
    # 测试无效参数格式
    invalid_params = {"invalid_key": torch.randn(5, 5)}
    result = federated_manager.submit_local_update("satellite_2", invalid_params, 100, 0.5)
    logger.info(f"无效参数格式提交结果: {result}")
    
    # 测试无效数据量
    result = federated_manager.submit_local_update("satellite_3", valid_params, 0, 0.5)
    logger.info(f"无效数据量提交结果: {result}")
    
    # 测试无效损失值
    result = federated_manager.submit_local_update("satellite_4", valid_params, 100, -1.0)
    logger.info(f"无效损失值提交结果: {result}")

def test_fault_tolerance():
    """测试容错机制"""
    logger.info("=== 测试容错机制 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager
    
    config = {
        'min_participants': 2,
        'parameter_backup_count': 3
    }
    
    federated_manager = FederatedLearningManager(config)
    model = SimpleModel()
    federated_manager.initialize_global_model(model)
    
    # 提交一些本地更新
    for i in range(3):
        params = model.state_dict()
        # 添加一些噪声模拟训练
        for key in params:
            params[key] += torch.randn_like(params[key]) * 0.1
        
        federated_manager.submit_local_update(f"satellite_{i}", params, 100, 0.5)
    
    # 检查备份
    backups = federated_manager.get_parameter_backups()
    logger.info(f"参数备份数量: {len(backups)}")
    logger.info(f"备份信息: {backups}")
    
    # 模拟聚合失败
    logger.info("模拟聚合失败...")
    federated_manager._rollback_parameters()

def test_numerical_stability():
    """测试数值稳定性"""
    logger.info("=== 测试数值稳定性 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager
    
    config = {
        'min_participants': 2,
        'aggregation_method': 'weighted'
    }
    
    federated_manager = FederatedLearningManager(config)
    model = SimpleModel()
    federated_manager.initialize_global_model(model)
    
    # 测试极端损失值
    extreme_losses = [1e-10, 1e-8, 1.0, 10.0, 100.0]
    
    for i, loss in enumerate(extreme_losses):
        params = model.state_dict()
        # 添加一些噪声
        for key in params:
            params[key] += torch.randn_like(params[key]) * 0.1
        
        result = federated_manager.submit_local_update(f"satellite_{i}", params, 100, loss)
        logger.info(f"极端损失值 {loss} 提交结果: {result}")
    
    # 测试参数裁剪
    logger.info("测试参数裁剪...")
    large_params = model.state_dict()
    for key in large_params:
        large_params[key] = torch.randn_like(large_params[key]) * 10.0  # 大参数
    
    clipped_params = federated_manager._clip_parameters(large_params)
    
    # 检查裁剪效果
    for key in large_params:
        original_norm = torch.norm(large_params[key])
        clipped_norm = torch.norm(clipped_params[key])
        logger.info(f"参数 {key}: 原始范数 {original_norm:.6f}, 裁剪后范数 {clipped_norm:.6f}")

def test_network_sync():
    """测试网络同步功能"""
    logger.info("=== 测试网络同步功能 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager
    
    config = {
        'min_participants': 2,
        'sync_interval': 1  # 1秒同步间隔
    }
    
    federated_manager = FederatedLearningManager(config)
    model = SimpleModel()
    federated_manager.initialize_global_model(model)
    
    # 创建模拟卫星
    satellites = {
        "satellite_1": MockSatelliteInfo("satellite_1", "ONLINE"),
        "satellite_2": MockSatelliteInfo("satellite_2", "ONLINE"),
        "satellite_3": MockSatelliteInfo("satellite_3", "OFFLINE")
    }
    
    # 提交一些更新
    for i in range(2):
        params = model.state_dict()
        for key in params:
            params[key] += torch.randn_like(params[key]) * 0.1
        
        federated_manager.submit_local_update(f"satellite_{i+1}", params, 100, 0.5)
    
    # 测试网络同步
    logger.info("测试网络同步...")
    federated_manager.sync_parameters_across_satellites(satellites)
    
    # 检查网络状态
    status = federated_manager.get_federated_status()
    logger.info(f"网络状态: {status['network_status']}")

def test_data_integrity():
    """测试数据完整性"""
    logger.info("=== 测试数据完整性 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager
    
    config = {
        'min_participants': 2
    }
    
    federated_manager = FederatedLearningManager(config)
    model = SimpleModel()
    federated_manager.initialize_global_model(model)
    
    # 测试参数校验和
    params = model.state_dict()
    checksum1 = federated_manager._calculate_parameter_checksum(params)
    checksum2 = federated_manager._calculate_parameter_checksum(params)
    
    logger.info(f"相同参数校验和: {checksum1 == checksum2}")
    
    # 测试不同参数的校验和
    modified_params = params.copy()
    for key in modified_params:
        modified_params[key] += torch.randn_like(modified_params[key]) * 0.1
    
    checksum3 = federated_manager._calculate_parameter_checksum(modified_params)
    logger.info(f"不同参数校验和: {checksum1 != checksum3}")

def test_comprehensive_scenario():
    """综合测试场景"""
    logger.info("=== 综合测试场景 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager, TrainingCoordinator
    
    # 联邦学习配置
    federated_config = {
        'min_participants': 2,
        'parameter_validation_enabled': True,
        'min_data_size': 10,
        'max_data_size': 1000,
        'min_loss_threshold': 1e-6,
        'max_loss_threshold': 100.0,
        'gradient_clip_norm': 1.0,
        'parameter_backup_count': 3
    }
    
    # 训练配置
    training_config = {
        'sync_interval': 1,
        'max_backup_age': 7200
    }
    
    # 创建训练协调器
    config = {
        'federated_learning': federated_config,
        'training': training_config
    }
    
    coordinator = TrainingCoordinator(config)
    
    # 初始化模型
    model = SimpleModel()
    coordinator.federated_manager.initialize_global_model(model)
    
    # 创建模拟卫星
    satellites = {
        "satellite_1": MockSatelliteInfo("satellite_1", "ONLINE"),
        "satellite_2": MockSatelliteInfo("satellite_2", "ONLINE"),
        "satellite_3": MockSatelliteInfo("satellite_3", "OFFLINE")
    }
    
    # 模拟训练过程
    logger.info("开始模拟训练过程...")
    
    for round_num in range(3):
        logger.info(f"=== 训练轮次 {round_num + 1} ===")
        
        # 每个卫星提交更新
        for i in range(2):
            params = model.state_dict()
            # 模拟训练更新
            for key in params:
                params[key] += torch.randn_like(params[key]) * 0.1
            
            success = coordinator.federated_manager.submit_local_update(
                f"satellite_{i+1}", params, 100 + i*50, 0.5 + i*0.1
            )
            logger.info(f"卫星 {i+1} 提交更新: {'成功' if success else '失败'}")
        
        # 协调训练
        coordinator.coordinate_training(satellites)
        
        # 获取状态
        status = coordinator.get_training_status()
        logger.info(f"联邦学习状态: {status['federated_learning']}")
        logger.info(f"分布式训练状态: {status['distributed_training']}")
        
        time.sleep(0.1)  # 短暂延迟

def main():
    """主测试函数"""
    logger.info("开始测试改进后的联邦学习系统...")
    
    try:
        # 运行各项测试
        test_parameter_validation()
        test_fault_tolerance()
        test_numerical_stability()
        test_network_sync()
        test_data_integrity()
        test_comprehensive_scenario()
        
        logger.info("所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 