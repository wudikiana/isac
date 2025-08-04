#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进后的联邦学习系统演示
展示容错机制、数据验证、数值稳定性等改进功能
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 简单的演示模型
class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def demo_parameter_validation():
    """演示参数验证功能"""
    logger.info("=== 演示参数验证功能 ===")
    
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
    model = DemoModel()
    federated_manager.initialize_global_model(model)
    
    # 演示有效参数提交
    logger.info("1. 提交有效参数...")
    valid_params = model.state_dict()
    success = federated_manager.submit_local_update("satellite_1", valid_params, 100, 0.5)
    logger.info(f"   结果: {'成功' if success else '失败'}")
    
    # 演示无效参数检测
    logger.info("2. 检测无效参数格式...")
    invalid_params = {"invalid_key": torch.randn(5, 5)}
    success = federated_manager.submit_local_update("satellite_2", invalid_params, 100, 0.5)
    logger.info(f"   结果: {'成功' if success else '失败'}")
    
    # 演示数据量验证
    logger.info("3. 验证数据量范围...")
    success = federated_manager.submit_local_update("satellite_3", valid_params, 0, 0.5)
    logger.info(f"   结果: {'成功' if success else '失败'}")
    
    # 演示损失值验证
    logger.info("4. 验证损失值范围...")
    success = federated_manager.submit_local_update("satellite_4", valid_params, 100, -1.0)
    logger.info(f"   结果: {'成功' if success else '失败'}")

def demo_fault_tolerance():
    """演示容错机制"""
    logger.info("=== 演示容错机制 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager
    
    config = {
        'min_participants': 2,
        'parameter_backup_count': 3
    }
    
    federated_manager = FederatedLearningManager(config)
    model = DemoModel()
    federated_manager.initialize_global_model(model)
    
    # 提交多个更新创建备份
    logger.info("1. 创建参数备份...")
    for i in range(3):
        params = model.state_dict()
        # 模拟训练更新
        for key in params:
            params[key] += torch.randn_like(params[key]) * 0.1
        
        federated_manager.submit_local_update(f"satellite_{i}", params, 100, 0.5)
    
    # 显示备份信息
    backups = federated_manager.get_parameter_backups()
    logger.info(f"   备份数量: {len(backups)}")
    for name, info in backups.items():
        logger.info(f"   备份 {name}: 轮次 {info['round']}, 时间 {info['timestamp']:.2f}")
    
    # 演示参数回滚
    logger.info("2. 演示参数回滚...")
    federated_manager._rollback_parameters()
    logger.info("   参数回滚完成")

def demo_numerical_stability():
    """演示数值稳定性"""
    logger.info("=== 演示数值稳定性 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager
    
    config = {
        'min_participants': 2,
        'aggregation_method': 'weighted',
        'gradient_clip_norm': 1.0
    }
    
    federated_manager = FederatedLearningManager(config)
    model = DemoModel()
    federated_manager.initialize_global_model(model)
    
    # 演示参数裁剪
    logger.info("1. 演示参数裁剪...")
    large_params = model.state_dict()
    for key in large_params:
        large_params[key] = torch.randn_like(large_params[key]) * 10.0  # 大参数
    
    clipped_params = federated_manager._clip_parameters(large_params)
    
    for key in large_params:
        original_norm = torch.norm(large_params[key])
        clipped_norm = torch.norm(clipped_params[key])
        logger.info(f"   参数 {key}: {original_norm:.2f} -> {clipped_norm:.2f}")
    
    # 演示损失归一化
    logger.info("2. 演示损失归一化...")
    losses = [0.1, 1.0, 10.0, 100.0]
    min_loss = min(losses)
    max_loss = max(losses)
    
    normalized_losses = []
    for loss in losses:
        if max_loss == min_loss:
            normalized_loss = 1.0
        else:
            normalized_loss = (loss - min_loss) / (max_loss - min_loss) + 1e-6
        normalized_losses.append(normalized_loss)
        logger.info(f"   损失 {loss}: -> {normalized_loss:.6f}")

def demo_comprehensive_scenario():
    """演示综合场景"""
    logger.info("=== 演示综合场景 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager, TrainingCoordinator
    
    # 配置
    federated_config = {
        'min_participants': 2,
        'parameter_validation_enabled': True,
        'gradient_clip_norm': 1.0,
        'parameter_backup_count': 3
    }
    
    training_config = {
        'sync_interval': 1,
        'max_backup_age': 7200
    }
    
    config = {
        'federated_learning': federated_config,
        'training': training_config
    }
    
    # 创建训练协调器
    coordinator = TrainingCoordinator(config)
    model = DemoModel()
    coordinator.federated_manager.initialize_global_model(model)
    
    # 模拟卫星
    satellites = {
        "satellite_1": type('MockSatellite', (), {'status': 'ONLINE', 'parameter_version': 0})(),
        "satellite_2": type('MockSatellite', (), {'status': 'ONLINE', 'parameter_version': 0})(),
        "satellite_3": type('MockSatellite', (), {'status': 'OFFLINE', 'parameter_version': 0})()
    }
    
    logger.info("1. 模拟训练过程...")
    
    for round_num in range(2):
        logger.info(f"   轮次 {round_num + 1}:")
        
        # 每个卫星提交更新
        for i in range(2):
            params = model.state_dict()
            # 模拟训练更新
            for key in params:
                params[key] += torch.randn_like(params[key]) * 0.1
            
            success = coordinator.federated_manager.submit_local_update(
                f"satellite_{i+1}", params, 100 + i*50, 0.5 + i*0.1
            )
            logger.info(f"     卫星 {i+1}: {'成功' if success else '失败'}")
        
        # 协调训练
        coordinator.coordinate_training(satellites)
        
        # 获取状态
        status = coordinator.get_training_status()
        logger.info(f"     聚合轮次: {status['federated_learning']['aggregation_round']}")
        logger.info(f"     备份数量: {status['federated_learning']['parameter_backups_count']}")

def demo_error_handling():
    """演示错误处理"""
    logger.info("=== 演示错误处理 ===")
    
    from satellite_system.satellite_federated import FederatedLearningManager, AggregationError
    
    config = {
        'min_participants': 2
    }
    
    federated_manager = FederatedLearningManager(config)
    model = DemoModel()
    federated_manager.initialize_global_model(model)
    
    # 演示聚合错误处理
    logger.info("1. 演示聚合错误处理...")
    try:
        # 尝试聚合空数据
        federated_manager._federated_averaging()
    except AggregationError as e:
        logger.info(f"   捕获聚合错误: {e}")
    
    # 演示参数验证错误
    logger.info("2. 演示参数验证错误...")
    try:
        federated_manager.update_global_parameters("satellite_1", {"invalid": torch.randn(5, 5)})
    except Exception as e:
        logger.info(f"   捕获验证错误: {e}")

def main():
    """主演示函数"""
    logger.info("开始演示改进后的联邦学习系统...")
    
    try:
        # 运行各项演示
        demo_parameter_validation()
        demo_fault_tolerance()
        demo_numerical_stability()
        demo_comprehensive_scenario()
        demo_error_handling()
        
        logger.info("演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 