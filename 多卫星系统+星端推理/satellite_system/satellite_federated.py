#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星联邦学习系统模块
专注于分布式训练、参数同步和模型聚合
"""

import torch
import numpy as np
import time
import logging
import copy
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .satellite_core import SatelliteInfo, SatelliteStatus

logger = logging.getLogger(__name__)

class ParameterValidationError(Exception):
    """参数验证错误"""
    pass

class AggregationError(Exception):
    """聚合错误"""
    pass

class ParameterVersion(Enum):
    """参数版本状态"""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"

@dataclass
class ParameterUpdate:
    """参数更新数据结构"""
    parameters: Dict[str, torch.Tensor]
    data_size: int
    loss: float
    timestamp: float
    version: int
    checksum: str
    satellite_id: str
    validation_status: ParameterVersion = ParameterVersion.PENDING

class FederatedLearningManager:
    """星间联邦学习管理器 - 改进版"""
    
    def __init__(self, config: dict):
        self.config = config
        self.global_model = None
        self.quantized_global_model = None  # 量化全局模型
        self.local_models = {}
        self.aggregation_round = 0
        self.min_clients = config.get('min_participants', 2)
        self.aggregation_threshold = 0.8
        
        # 量化配置
        self.enable_quantization = config.get('enable_quantization', True)
        self.quantization_method = config.get('quantization_method', 'dynamic')  # dynamic, static
        self.quantized_model_path = config.get('quantized_model_path', 'models/quantized_federated_model.pt')
        
        # 容错机制配置
        self.max_parameter_age = config.get('max_parameter_age', 3600)  # 1小时
        self.parameter_backup_count = config.get('parameter_backup_count', 3)
        self.parameter_backups = {}  # 参数备份
        
        # 数值稳定性配置
        self.min_loss_threshold = config.get('min_loss_threshold', 1e-6)
        self.max_loss_threshold = config.get('max_loss_threshold', 100.0)
        self.gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        
        # 数据验证配置
        self.max_data_size = config.get('max_data_size', 1000000)
        self.min_data_size = config.get('min_data_size', 10)
        self.parameter_validation_enabled = config.get('parameter_validation_enabled', True)
        
        self.federated_config = {
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 32),
            'epochs_per_round': config.get('epochs_per_round', 5),
            'aggregation_method': config.get('aggregation_method', 'fedavg')
        }
        self.sync_interval = config.get('sync_interval', 600)
        self.last_sync_time = time.time()
        
        # 网络状态跟踪
        self.network_status = {}
        self.last_network_check = time.time()
        
    def initialize_global_model(self, model_architecture: torch.nn.Module):
        """初始化全局模型"""
        self.global_model = copy.deepcopy(model_architecture)
        # 创建参数备份
        self._create_parameter_backup("initial")
        logger.info("联邦学习全局模型初始化完成")
        
        # 如果启用量化，创建量化模型
        if self.enable_quantization:
            self._create_quantized_model()
    
    def _create_parameter_backup(self, backup_name: str):
        """创建参数备份"""
        if self.global_model is not None:
            backup = {
                'parameters': copy.deepcopy(self.global_model.state_dict()),
                'timestamp': time.time(),
                'round': self.aggregation_round
            }
            self.parameter_backups[backup_name] = backup
            
            # 限制备份数量
            if len(self.parameter_backups) > self.parameter_backup_count:
                oldest_backup = min(self.parameter_backups.keys(), 
                                  key=lambda k: self.parameter_backups[k]['timestamp'])
                del self.parameter_backups[oldest_backup]
    
    def _validate_parameter_format(self, parameters: Dict[str, torch.Tensor]) -> bool:
        """验证参数格式"""
        if not isinstance(parameters, dict):
            return False
        
        if self.global_model is None:
            return False
        
        global_state = self.global_model.state_dict()
        
        # 检查参数键是否匹配
        if set(parameters.keys()) != set(global_state.keys()):
            return False
        
        # 检查参数形状是否匹配
        for key in parameters:
            if parameters[key].shape != global_state[key].shape:
                return False
            if parameters[key].dtype != global_state[key].dtype:
                return False
        
        return True
    
    def _validate_data_size(self, data_size: int) -> bool:
        """验证数据量"""
        return self.min_data_size <= data_size <= self.max_data_size
    
    def _validate_loss(self, loss: float) -> bool:
        """验证损失值"""
        return self.min_loss_threshold <= loss <= self.max_loss_threshold
    
    def _calculate_parameter_checksum(self, parameters: Dict[str, torch.Tensor]) -> str:
        """计算参数校验和"""
        param_str = ""
        for key in sorted(parameters.keys()):
            param_str += f"{key}:{parameters[key].sum().item()}"
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _clip_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """参数裁剪"""
        clipped_params = {}
        for key, param in parameters.items():
            # 计算参数范数
            param_norm = torch.norm(param)
            if param_norm > self.gradient_clip_norm:
                # 裁剪参数
                clipped_params[key] = param * (self.gradient_clip_norm / param_norm)
                logger.warning(f"参数 {key} 被裁剪，范数: {param_norm:.6f} -> {self.gradient_clip_norm}")
            else:
                clipped_params[key] = param
        return clipped_params
    
    def submit_local_update(self, satellite_id: str, local_parameters: Dict[str, torch.Tensor],
                           data_size: int, loss: float) -> bool:
        """提交本地模型更新 - 改进版"""
        try:
            # 数据验证
            if self.parameter_validation_enabled:
                if not self._validate_parameter_format(local_parameters):
                    logger.error(f"卫星 {satellite_id} 参数格式验证失败")
                    return False
                
                if not self._validate_data_size(data_size):
                    logger.error(f"卫星 {satellite_id} 数据量验证失败: {data_size}")
                    return False
                
                if not self._validate_loss(loss):
                    logger.error(f"卫星 {satellite_id} 损失值验证失败: {loss}")
                    return False
            
            # 参数裁剪
            clipped_parameters = self._clip_parameters(local_parameters)
            
            # 计算校验和
            checksum = self._calculate_parameter_checksum(clipped_parameters)
            
            # 创建参数更新对象
            update = ParameterUpdate(
                parameters=clipped_parameters,
                data_size=data_size,
                loss=loss,
                timestamp=time.time(),
                version=self.aggregation_round,
                checksum=checksum,
                satellite_id=satellite_id,
                validation_status=ParameterVersion.VALID
            )
            
            self.local_models[satellite_id] = update
            
            logger.info(f"卫星 {satellite_id} 提交本地更新，数据量: {data_size}, 损失: {loss:.6f}")
            
            # 检查是否可以进行聚合
            if len(self.local_models) >= self.min_clients:
                self._check_aggregation()
            
            return True
            
        except Exception as e:
            logger.error(f"提交本地更新失败: {e}")
            return False
    
    def _check_aggregation(self):
        """检查是否满足聚合条件"""
        current_time = time.time()
        recent_updates = sum(1 for update in self.local_models.values() 
                           if current_time - update.timestamp < 300)  # 5分钟内
        
        if recent_updates >= self.min_clients:
            self._aggregate_models()
    
    def _aggregate_models(self):
        """聚合模型 - 支持量化"""
        try:
            if not self.local_models:
                logger.warning("没有本地模型可聚合")
                return False
            
            logger.info(f"开始聚合 {len(self.local_models)} 个本地模型...")
            
            # 执行聚合
            if self.federated_config['aggregation_method'] == 'fedavg':
                self._federated_averaging()
            else:
                self._weighted_averaging()
            
            # 更新聚合轮次
            self.aggregation_round += 1
            
            # 创建参数备份
            self._create_parameter_backup(f"round_{self.aggregation_round}")
            
            # 如果启用量化，创建量化模型
            if self.enable_quantization:
                self._create_quantized_model()
            
            logger.info(f"模型聚合完成，轮次: {self.aggregation_round}")
            return True
            
        except Exception as e:
            logger.error(f"模型聚合失败: {e}")
            # 尝试回滚
            self._rollback_parameters()
            return False
    
    def _rollback_parameters(self):
        """参数回滚"""
        if self.parameter_backups:
            # 找到最新的有效备份
            latest_backup = max(self.parameter_backups.keys(), 
                              key=lambda k: self.parameter_backups[k]['timestamp'])
            backup = self.parameter_backups[latest_backup]
            
            if self.global_model is not None:
                self.global_model.load_state_dict(backup['parameters'])
                self.aggregation_round = backup['round']
                logger.info(f"参数回滚到备份: {latest_backup}")
    
    def _federated_averaging(self):
        """联邦平均聚合 - 改进版"""
        total_data_size = sum(update.data_size for update in self.local_models.values())
        
        if total_data_size == 0:
            raise AggregationError("总数据量为0，无法进行聚合")
        
        # 初始化聚合参数
        aggregated_params = {}
        for param_name in self.global_model.state_dict().keys():
            aggregated_params[param_name] = torch.zeros_like(
                self.global_model.state_dict()[param_name])
        
        # 加权平均
        for satellite_id, update in self.local_models.items():
            weight = update.data_size / total_data_size
            for param_name, param_value in update.parameters.items():
                aggregated_params[param_name] += weight * param_value
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_params)
    
    def _weighted_averaging(self):
        """加权平均聚合（基于损失）- 改进版"""
        # 数值稳定性改进
        losses = [update.loss for update in self.local_models.values()]
        min_loss = min(losses)
        max_loss = max(losses)
        
        # 归一化损失值
        normalized_losses = []
        for update in self.local_models.values():
            if max_loss == min_loss:
                normalized_loss = 1.0
            else:
                normalized_loss = (update.loss - min_loss) / (max_loss - min_loss) + 1e-6
            normalized_losses.append(normalized_loss)
        
        total_weight = sum(1.0 / loss for loss in normalized_losses)
        
        if total_weight == 0:
            raise AggregationError("总权重为0，无法进行聚合")
        
        aggregated_params = {}
        for param_name in self.global_model.state_dict().keys():
            aggregated_params[param_name] = torch.zeros_like(
                self.global_model.state_dict()[param_name])
        
        for i, (satellite_id, update) in enumerate(self.local_models.items()):
            weight = (1.0 / normalized_losses[i]) / total_weight
            for param_name, param_value in update.parameters.items():
                aggregated_params[param_name] += weight * param_value
        
        self.global_model.load_state_dict(aggregated_params)
    
    def get_global_model(self) -> Optional[torch.nn.Module]:
        """获取全局模型 - 优先返回量化模型"""
        # 优先返回量化模型
        if self.enable_quantization:
            quantized_model = self.get_quantized_global_model()
            if quantized_model is not None:
                return quantized_model
        
        # 回退到原始模型
        return self.global_model
    
    def get_federated_status(self) -> Dict[str, Any]:
        """获取联邦学习状态"""
        return {
            'aggregation_round': self.aggregation_round,
            'local_models_count': len(self.local_models),
            'min_clients': self.min_clients,
            'global_model_available': self.global_model is not None,
            'last_sync_time': self.last_sync_time,
            'sync_interval': self.sync_interval,
            'parameter_backups_count': len(self.parameter_backups),
            'network_status': self.network_status
        }
    
    def _check_network_connectivity(self, satellites: Dict[str, SatelliteInfo]) -> Dict[str, bool]:
        """检查网络连接状态"""
        current_time = time.time()
        network_status = {}
        
        for sat_id, sat_info in satellites.items():
            # 检查卫星是否在线
            is_online = sat_info.status == SatelliteStatus.ONLINE
            
            # 检查参数版本一致性
            version_consistent = True
            if sat_id in self.local_models:
                expected_version = self.aggregation_round
                actual_version = self.local_models[sat_id].version
                version_consistent = actual_version == expected_version
            
            network_status[sat_id] = {
                'online': is_online,
                'version_consistent': version_consistent,
                'last_check': current_time
            }
        
        self.network_status = network_status
        self.last_network_check = current_time
        return network_status
    
    def sync_parameters_across_satellites(self, satellites: Dict[str, SatelliteInfo]):
        """在卫星间同步参数 - 改进版"""
        current_time = time.time()
        if current_time - self.last_sync_time < self.sync_interval:
            return
        
        # 检查网络连接状态
        network_status = self._check_network_connectivity(satellites)
        
        # 找到最新的参数版本
        latest_satellite = None
        latest_version = -1
        
        for sat_id, sat_info in satellites.items():
            if sat_info.status == SatelliteStatus.ONLINE:
                version = sat_info.parameter_version
                if version > latest_version:
                    latest_version = version
                    latest_satellite = sat_id
        
        if latest_satellite:
            # 将最新参数同步到其他卫星
            latest_params = self.get_global_parameters(latest_satellite)
            if latest_params:
                sync_success_count = 0
                for sat_id, sat_info in satellites.items():
                    if (sat_id != latest_satellite and 
                        sat_info.status == SatelliteStatus.ONLINE):
                        try:
                            self.update_global_parameters(sat_id, latest_params)
                            sat_info.parameter_version = latest_version
                            sync_success_count += 1
                        except Exception as e:
                            logger.error(f"同步参数到卫星 {sat_id} 失败: {e}")
                
                logger.info(f"参数同步完成，成功同步到 {sync_success_count} 个卫星")
        
        self.last_sync_time = current_time
    
    def update_global_parameters(self, satellite_id: str, parameters: Dict[str, torch.Tensor]):
        """更新全局参数 - 改进版"""
        # 验证参数格式
        if not self._validate_parameter_format(parameters):
            raise ParameterValidationError(f"卫星 {satellite_id} 参数格式无效")
        
        # 参数裁剪
        clipped_parameters = self._clip_parameters(parameters)
        
        # 创建参数更新对象
        update = ParameterUpdate(
            parameters=clipped_parameters,
            data_size=100,  # 默认数据量
            loss=0.1,  # 默认损失
            timestamp=time.time(),
            version=self.aggregation_round,
            checksum=self._calculate_parameter_checksum(clipped_parameters),
            satellite_id=satellite_id,
            validation_status=ParameterVersion.VALID
        )
        
        self.local_models[satellite_id] = update
        logger.info(f"更新卫星 {satellite_id} 的全局参数")
    
    def get_global_parameters(self, satellite_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取全局参数"""
        if satellite_id in self.local_models:
            return self.local_models[satellite_id].parameters
        return None
    
    def get_parameter_backups(self) -> Dict[str, Any]:
        """获取参数备份信息"""
        return {
            name: {
                'timestamp': backup['timestamp'],
                'round': backup['round']
            }
            for name, backup in self.parameter_backups.items()
        }

    def _create_quantized_model(self):
        """创建量化模型"""
        try:
            if self.global_model is None:
                logger.warning("全局模型未初始化，无法创建量化模型")
                return
            
            logger.info("开始创建量化模型...")
            
            # 确保模型在CPU上
            model = self.global_model.cpu()
            model.eval()
            
            # 动态量化
            if self.quantization_method == 'dynamic':
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Conv2d, torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("动态量化完成")
            else:
                # 静态量化（需要校准数据）
                logger.warning("静态量化暂未实现，使用动态量化")
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Conv2d, torch.nn.Linear},
                    dtype=torch.qint8
                )
            
            # 准备示例输入进行跟踪
            dummy_image = torch.randn(1, 3, 256, 256)
            dummy_sim_feat = torch.randn(1, 11)
            
            # 跟踪量化模型
            traced_model = torch.jit.trace(quantized_model, (dummy_image, dummy_sim_feat))
            
            # 保存量化模型
            import os
            os.makedirs(os.path.dirname(self.quantized_model_path), exist_ok=True)
            torch.jit.save(traced_model, self.quantized_model_path)
            
            self.quantized_global_model = traced_model
            logger.info(f"量化模型已保存到: {self.quantized_model_path}")
            
        except Exception as e:
            logger.error(f"创建量化模型失败: {e}")
            self.quantized_global_model = None
    
    def load_quantized_model(self):
        """加载量化模型"""
        try:
            if os.path.exists(self.quantized_model_path):
                self.quantized_global_model = torch.jit.load(self.quantized_model_path, map_location='cpu')
                self.quantized_global_model.eval()
                logger.info("量化模型加载成功")
                return True
            else:
                logger.warning(f"量化模型文件不存在: {self.quantized_model_path}")
                return False
        except Exception as e:
            logger.error(f"加载量化模型失败: {e}")
            return False
    
    def get_quantized_global_model(self):
        """获取量化全局模型"""
        if self.quantized_global_model is None:
            # 尝试加载量化模型
            if not self.load_quantized_model():
                # 如果加载失败，尝试重新创建
                if self.global_model is not None:
                    self._create_quantized_model()
        
        return self.quantized_global_model

class DistributedTrainingManager:
    """分布式训练管理器 - 改进版"""
    
    def __init__(self, config: dict):
        self.config = config
        self.parameter_server = {}  # 全局参数服务器
        self.training_queue = []
        self.parameter_sync_interval = config.get('sync_interval', 300)  # 5分钟同步一次
        self.last_sync_time = time.time()
        
        # 容错机制
        self.parameter_backups = {}
        self.max_backup_age = config.get('max_backup_age', 7200)  # 2小时
        
    def update_global_parameters(self, satellite_id: str, parameters: Dict[str, torch.Tensor]):
        """更新全局参数 - 改进版"""
        # 创建参数备份
        backup_key = f"{satellite_id}_{int(time.time())}"
        self.parameter_backups[backup_key] = {
            'parameters': copy.deepcopy(parameters),
            'timestamp': time.time(),
            'satellite_id': satellite_id
        }
        
        # 清理过期备份
        self._cleanup_old_backups()
        
        self.parameter_server[satellite_id] = {
            'parameters': parameters,
            'timestamp': time.time(),
            'version': self.parameter_server.get(satellite_id, {}).get('version', 0) + 1
        }
        logger.info(f"更新卫星 {satellite_id} 的全局参数，版本: {self.parameter_server[satellite_id]['version']}")
    
    def _cleanup_old_backups(self):
        """清理过期备份"""
        current_time = time.time()
        expired_backups = [
            key for key, backup in self.parameter_backups.items()
            if current_time - backup['timestamp'] > self.max_backup_age
        ]
        for key in expired_backups:
            del self.parameter_backups[key]
    
    def get_global_parameters(self, satellite_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取全局参数"""
        if satellite_id in self.parameter_server:
            return self.parameter_server[satellite_id]['parameters']
        return None
    
    def sync_parameters_across_satellites(self, satellites: Dict[str, SatelliteInfo]):
        """在卫星间同步参数 - 改进版"""
        if time.time() - self.last_sync_time < self.parameter_sync_interval:
            return
        
        # 找到最新的参数版本
        latest_satellite = None
        latest_version = -1
        
        for sat_id, sat_info in satellites.items():
            if sat_info.status == SatelliteStatus.ONLINE:
                version = sat_info.parameter_version
                if version > latest_version:
                    latest_version = version
                    latest_satellite = sat_id
        
        if latest_satellite:
            # 将最新参数同步到其他卫星
            latest_params = self.get_global_parameters(latest_satellite)
            if latest_params:
                sync_success_count = 0
                for sat_id, sat_info in satellites.items():
                    if (sat_id != latest_satellite and 
                        sat_info.status == SatelliteStatus.ONLINE):
                        try:
                            self.update_global_parameters(sat_id, latest_params)
                            sat_info.parameter_version = latest_version
                            sync_success_count += 1
                        except Exception as e:
                            logger.error(f"同步参数到卫星 {sat_id} 失败: {e}")
                
                logger.info(f"分布式训练参数同步完成，成功同步到 {sync_success_count} 个卫星")
        
        self.last_sync_time = time.time()
    
    def submit_training_task(self, training_task: Dict[str, Any]):
        """提交训练任务"""
        self.training_queue.append(training_task)
        logger.info(f"提交训练任务: {training_task.get('task_id', 'unknown')} 到卫星 {training_task.get('satellite_id', 'unknown')}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'queue_size': len(self.training_queue),
            'last_sync_time': self.last_sync_time,
            'parameter_versions': {sat_id: info.get('version', 0) 
                                 for sat_id, info in self.parameter_server.items()},
            'backup_count': len(self.parameter_backups)
        }
    
    def clear_training_queue(self):
        """清空训练队列"""
        self.training_queue.clear()
        logger.info("训练队列已清空")

class TrainingCoordinator:
    """训练协调器 - 改进版"""
    
    def __init__(self, config: dict):
        self.config = config
        self.federated_manager = FederatedLearningManager(config.get('federated_learning', {}))
        self.distributed_manager = DistributedTrainingManager(config.get('training', {}))
        
    def coordinate_training(self, satellites: Dict[str, SatelliteInfo]):
        """协调训练过程 - 改进版"""
        try:
            # 联邦学习同步
            self.federated_manager.sync_parameters_across_satellites(satellites)
            
            # 分布式训练同步
            self.distributed_manager.sync_parameters_across_satellites(satellites)
            
            # 更新PPO策略
            # 这里可以添加PPO策略更新逻辑
            
        except Exception as e:
            logger.error(f"训练协调失败: {e}")
            # 触发容错机制
            self._handle_training_failure(e)
    
    def _handle_training_failure(self, error: Exception):
        """处理训练失败"""
        logger.error(f"训练失败，启动容错机制: {error}")
        
        # 可以在这里添加更多的容错逻辑
        # 比如重新初始化模型、通知管理员等
        
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'federated_learning': self.federated_manager.get_federated_status(),
            'distributed_training': self.distributed_manager.get_training_status()
        } 