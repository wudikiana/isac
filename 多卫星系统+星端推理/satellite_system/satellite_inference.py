#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星推理系统模块 - 改进版
专注于AI推理任务处理、负载均衡、故障容错和联邦学习集成
"""

import torch
import numpy as np
import time
import json
import threading
import queue
import socket
import pickle
import struct
import logging
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import os # Added for os.path.exists

from .satellite_core import (
    SatelliteInfo, SatelliteStatus, CoverageStatus, InferenceTask, 
    TrainingTask, calculate_distance, load_config, create_satellite_info
)
from .satellite_federated import FederatedLearningManager, TrainingCoordinator

logger = logging.getLogger(__name__)

class LoadBalancer:
    """负载均衡器 - 改进版"""
    
    def __init__(self):
        self.strategy = "coverage_aware"  # 默认使用覆盖感知策略
        self.round_robin_index = 0
        self.federated_weight = 0.3  # 联邦学习权重
    
    def select_satellite(self, satellites: Dict[str, SatelliteInfo], 
                        task: InferenceTask) -> Optional[SatelliteInfo]:
        """选择卫星 - 改进版"""
        online_satellites = [
            sat for sat in satellites.values() 
            if sat.status == SatelliteStatus.ONLINE
        ]
        
        if not online_satellites:
            return None
        
        if self.strategy == "coverage_aware" and task.location:
            return self._coverage_aware_selection(online_satellites, task)
        elif self.strategy == "least_load":
            return self._least_load(online_satellites, task)
        elif self.strategy == "round_robin":
            return self._round_robin(online_satellites, task)
        elif self.strategy == "federated_aware":
            return self._federated_aware_selection(online_satellites, task)
        else:
            return self._fastest_response(online_satellites, task)
    
    def _coverage_aware_selection(self, satellites: List[SatelliteInfo], 
                                 task: InferenceTask) -> SatelliteInfo:
        """覆盖感知选择 - 改进版"""
        best_satellite = None
        best_score = -1
        
        for sat in satellites:
            # 计算覆盖评分
            coverage_score = 0
            if sat.coverage_status == CoverageStatus.COVERED:
                coverage_score = 1.0
            elif sat.coverage_status == CoverageStatus.WILL_COVER:
                coverage_score = 0.7
            elif sat.coverage_status == CoverageStatus.WAS_COVERED:
                coverage_score = 0.3
            
            # 计算负载评分
            load_score = 1.0 - sat.current_load
            
            # 计算响应时间评分
            response_score = 1.0 / (time.time() - sat.last_heartbeat + 1)
            
            # 计算联邦学习评分
            federated_score = getattr(sat, 'federated_score', 0.5)
            
            # 综合评分
            total_score = (coverage_score * 0.4 + load_score * 0.25 + 
                         response_score * 0.1 + federated_score * 0.25)
            
            if total_score > best_score:
                best_score = total_score
                best_satellite = sat
        
        return best_satellite or satellites[0]
    
    def _federated_aware_selection(self, satellites: List[SatelliteInfo], 
                                  task: InferenceTask) -> SatelliteInfo:
        """联邦学习感知选择"""
        best_satellite = None
        best_score = -1
        
        for sat in satellites:
            # 基础评分
            base_score = 1.0 - sat.current_load
            
            # 联邦学习评分
            federated_score = getattr(sat, 'federated_score', 0.5)
            
            # 模型版本评分
            model_version_score = getattr(sat, 'model_version_score', 0.5)
            
            # 综合评分
            total_score = (base_score * 0.4 + federated_score * 0.4 + 
                         model_version_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_satellite = sat
        
        return best_satellite or satellites[0]
    
    def _least_load(self, satellites: List[SatelliteInfo], task: InferenceTask) -> SatelliteInfo:
        """最少负载策略"""
        return min(satellites, key=lambda x: x.current_load)
    
    def _round_robin(self, satellites: List[SatelliteInfo], task: InferenceTask) -> SatelliteInfo:
        """轮询策略"""
        satellite = satellites[self.round_robin_index % len(satellites)]
        self.round_robin_index += 1
        return satellite
    
    def _fastest_response(self, satellites: List[SatelliteInfo], task: InferenceTask) -> SatelliteInfo:
        """最快响应策略"""
        return min(satellites, key=lambda x: time.time() - x.last_heartbeat)

class FaultToleranceManager:
    """故障容错管理器 - 改进版"""
    
    def __init__(self):
        self.failure_count = {}
        self.recovery_strategies = {
            'retry': self._retry_strategy,
            'fallback': self._fallback_strategy,
            'federated_fallback': self._federated_fallback_strategy
        }
    
    def handle_failure(self, task: InferenceTask, failed_satellite: SatelliteInfo, 
                      system: 'MultiSatelliteInferenceSystem'):
        """处理故障 - 改进版"""
        sat_id = failed_satellite.satellite_id
        self.failure_count[sat_id] = self.failure_count.get(sat_id, 0) + 1
        
        logger.warning(f"卫星 {sat_id} 故障，失败次数: {self.failure_count[sat_id]}")
        
        # 根据失败次数选择恢复策略
        if self.failure_count[sat_id] <= 2:
            strategy = 'retry'
        elif self.failure_count[sat_id] <= 5:
            strategy = 'fallback'
        else:
            strategy = 'federated_fallback'
        
        self.recovery_strategies[strategy](task, failed_satellite, system)
    
    def _retry_strategy(self, task: InferenceTask, failed_satellite: SatelliteInfo, 
                       system: 'MultiSatelliteInferenceSystem'):
        """重试策略"""
        logger.info(f"对任务 {task.task_id} 执行重试策略")
        # 重新提交任务
        system.task_queue.put((task.priority, task))
    
    def _fallback_strategy(self, task: InferenceTask, failed_satellite: SatelliteInfo, 
                          system: 'MultiSatelliteInferenceSystem'):
        """回退策略"""
        logger.info(f"对任务 {task.task_id} 执行回退策略")
        # 使用本地模型处理
        system._process_locally(task)
    
    def _federated_fallback_strategy(self, task: InferenceTask, failed_satellite: SatelliteInfo, 
                                   system: 'MultiSatelliteInferenceSystem'):
        """联邦学习回退策略"""
        logger.info(f"对任务 {task.task_id} 执行联邦学习回退策略")
        # 使用联邦学习模型处理
        if hasattr(system, 'federated_manager') and system.federated_manager:
            system._process_with_federated_model(task)
        else:
            system._process_locally(task)

class SatelliteCommunication:
    """卫星通信管理器 - 改进版"""
    
    def __init__(self):
        self.connection_timeout = 5.0
        self.retry_attempts = 3
        self.federated_sync_enabled = True
    
    def test_connection(self, satellite: SatelliteInfo) -> bool:
        """测试连接 - 改进版"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            result = sock.connect_ex((satellite.ip_address, satellite.port))
            sock.close()
            
            if result == 0:
                # 更新联邦学习状态
                if hasattr(satellite, 'federated_score'):
                    satellite.federated_score = min(1.0, satellite.federated_score + 0.1)
                else:
                    satellite.federated_score = 0.8
                
                return True
            else:
                # 降低联邦学习评分
                if hasattr(satellite, 'federated_score'):
                    satellite.federated_score = max(0.0, satellite.federated_score - 0.2)
                else:
                    satellite.federated_score = 0.3
                
                return False
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
    
    def send_heartbeat(self, satellite: SatelliteInfo) -> bool:
        """发送心跳 - 改进版"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            sock.connect((satellite.ip_address, satellite.port))
            
            # 发送心跳数据
            heartbeat_data = {
                'type': 'heartbeat',
                'timestamp': time.time(),
                'federated_status': getattr(satellite, 'federated_score', 0.5)
            }
            
            # 使用JSON协议
            data = json.dumps(heartbeat_data).encode('utf-8')
            sock.send(data)
            
            # 接收响应
            response_data = sock.recv(1024).decode('utf-8')
            response = json.loads(response_data)
            
            sock.close()
            
            # 更新卫星状态
            satellite.last_heartbeat = time.time()
            if 'federated_score' in response:
                satellite.federated_score = response['federated_score']
            
            return True
        except Exception as e:
            logger.error(f"心跳发送失败: {e}")
            return False
    
    def get_load_info(self, satellite: SatelliteInfo) -> Optional[dict]:
        """获取负载信息 - 改进版"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            sock.connect((satellite.ip_address, satellite.port))
            
            # 发送负载查询请求
            request_data = {
                'type': 'get_load',
                'timestamp': time.time()
            }
            
            # 使用JSON协议
            data = json.dumps(request_data).encode('utf-8')
            sock.send(data)
            
            # 接收响应
            response_data = sock.recv(1024).decode('utf-8')
            response = json.loads(response_data)
            
            sock.close()
            
            # 更新联邦学习状态
            if 'federated_score' in response:
                satellite.federated_score = response['federated_score']
            
            return response
        except Exception as e:
            logger.error(f"负载信息获取失败: {e}")
            return None
    
    def send_inference_task(self, satellite: SatelliteInfo, task_data: dict) -> Optional[dict]:
        """发送推理任务 - 改进版"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30.0)  # 推理任务需要更长的超时时间
            sock.connect((satellite.ip_address, satellite.port))
            
            # 处理numpy数组序列化问题
            serializable_data = {}
            for key, value in task_data.items():
                if isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value
            
            # 发送任务数据 - 使用JSON协议
            data = json.dumps(serializable_data).encode('utf-8')
            sock.send(data)
            
            # 接收结果
            response_data = sock.recv(1024).decode('utf-8')
            response = json.loads(response_data)
            
            sock.close()
            
            # 更新联邦学习状态
            if 'federated_score' in response:
                satellite.federated_score = response['federated_score']
            
            return response
        except Exception as e:
            logger.error(f"推理任务发送失败: {e}")
            return None

class CoverageManager:
    """覆盖管理器 - 改进版"""
    
    def __init__(self):
        self.satellite_positions = {}
        self.coverage_history = {}
        self.federated_coverage_weight = 0.2
    
    def update_satellite_position(self, satellite_id: str, position: List[float], 
                                timestamp: float):
        """更新卫星位置 - 改进版"""
        self.satellite_positions[satellite_id] = {
            'position': position,
            'timestamp': timestamp
        }
        
        # 更新联邦学习覆盖评分
        if satellite_id in self.coverage_history:
            coverage_score = self.coverage_history[satellite_id].get('score', 0.5)
            # 根据位置更新评分
            if len(position) >= 3 and position[2] > 0:  # 高度为正
                coverage_score = min(1.0, coverage_score + 0.1)
            else:
                coverage_score = max(0.0, coverage_score - 0.1)
            
            self.coverage_history[satellite_id]['score'] = coverage_score
    
    def predict_coverage(self, satellite_id: str, target_location: List[float], 
                        time_horizon: float = 3600) -> CoverageStatus:
        """预测覆盖状态 - 改进版"""
        if satellite_id not in self.satellite_positions:
            return CoverageStatus.NOT_COVERED
        
        sat_pos = self.satellite_positions[satellite_id]['position']
        
        # 计算距离
        distance = calculate_distance(sat_pos, target_location)
        
        # 考虑联邦学习覆盖评分
        federated_coverage_score = self.coverage_history.get(satellite_id, {}).get('score', 0.5)
        
        # 综合覆盖判断
        if distance < 1000:  # 近距离
            if federated_coverage_score > 0.7:
                return CoverageStatus.COVERED
            else:
                return CoverageStatus.WILL_COVER
        elif distance < 5000:  # 中等距离
            if federated_coverage_score > 0.5:
                return CoverageStatus.WILL_COVER
            else:
                return CoverageStatus.WAS_COVERED
        else:
            return CoverageStatus.NOT_COVERED
    
    def get_optimal_satellite(self, target_location: List[float], 
                            satellites: Dict[str, SatelliteInfo]) -> Optional[str]:
        """获取最优卫星 - 改进版"""
        best_satellite = None
        best_score = -1
        
        for sat_id, sat_info in satellites.items():
            if sat_info.status != SatelliteStatus.ONLINE:
                continue
            
            # 计算覆盖评分
            coverage_status = self.predict_coverage(sat_id, target_location)
            coverage_score = {
                CoverageStatus.COVERED: 1.0,
                CoverageStatus.WILL_COVER: 0.7,
                CoverageStatus.WAS_COVERED: 0.3,
                CoverageStatus.NOT_COVERED: 0.0
            }[coverage_status]
            
            # 计算联邦学习评分
            federated_score = getattr(sat_info, 'federated_score', 0.5)
            
            # 计算负载评分
            load_score = 1.0 - sat_info.current_load
            
            # 综合评分
            total_score = (coverage_score * 0.5 + federated_score * 0.3 + load_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_satellite = sat_id
        
        return best_satellite
    
    def update_coverage_status(self, satellites: Dict[str, SatelliteInfo]):
        """更新覆盖状态 - 改进版"""
        for sat_id, sat_info in satellites.items():
            if sat_id in self.satellite_positions:
                # 更新联邦学习覆盖评分
                federated_score = getattr(sat_info, 'federated_score', 0.5)
                self.coverage_history[sat_id] = {
                    'score': federated_score,
                    'timestamp': time.time()
                }

class MultiSatelliteInferenceSystem:
    """多卫星推理系统 - 改进版"""
    
    def __init__(self, config_file: str = "satellite_config.json"):
        self.config = load_config(config_file)
        self.satellites = {}
        self.task_queue = queue.PriorityQueue()
        self.results_cache = {}
        self.load_balancer = LoadBalancer()
        self.fault_tolerance = FaultToleranceManager()
        self.communication = SatelliteCommunication()
        self.coverage_manager = CoverageManager()
        
        # 联邦学习集成
        self.federated_manager = None
        self.training_coordinator = None
        self._init_federated_learning()
        
        # 本地模型
        self.local_model = None
        self._load_local_model()
        
        # 启动监控
        self.monitoring_thread = threading.Thread(target=self._start_monitoring, daemon=True)
        self.monitoring_thread.start()
        
        # 发现卫星
        self.discover_satellites()
    
    def _init_federated_learning(self):
        """初始化联邦学习"""
        try:
            federated_config = self.config.get('federated_learning', {})
            self.federated_manager = FederatedLearningManager(federated_config)
            self.training_coordinator = TrainingCoordinator({
                'federated_learning': federated_config,
                'training': self.config.get('training', {})
            })
            logger.info("联邦学习系统初始化成功")
        except Exception as e:
            logger.error(f"联邦学习系统初始化失败: {e}")
    
    def _load_local_model(self):
        """加载本地模型作为备份 - 支持量化模型"""
        try:
            from train_model import EnhancedDeepLab
            
            # 首先尝试加载量化模型
            quantized_model_path = self.config["local_backup"].get("quantized_model_path", "models/quantized_seg_model.pt")
            
            if os.path.exists(quantized_model_path):
                logger.info(f"尝试加载量化模型: {quantized_model_path}")
                try:
                    # 加载量化模型
                    self.local_model = torch.jit.load(quantized_model_path, map_location='cpu')
                    self.local_model.eval()
                    logger.info("量化模型加载成功")
                    return
                except Exception as e:
                    logger.warning(f"量化模型加载失败: {e}，回退到原始模型")
            
            # 回退到原始模型
            logger.info("加载原始模型...")
            model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
            
            # 尝试加载检查点
            checkpoint_path = self.config["local_backup"]["model_path"]
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path, map_location='cuda')
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的模型状态字典格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 改进的状态字典处理
            new_state_dict = {}
            for key, value in state_dict.items():
                # 移除各种前缀
                new_key = key
                prefixes_to_remove = [
                    'deeplab_model.',
                    'landslide_model.',
                    'model.',
                    'module.'
                ]
                
                for prefix in prefixes_to_remove:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                
                new_state_dict[new_key] = value
            
            # 尝试加载模型
            try:
                # 首先尝试严格加载
                model.load_state_dict(new_state_dict, strict=True)
                logger.info("模型严格加载成功")
            except Exception as e:
                logger.warning(f"严格加载失败，尝试非严格加载: {e}")
                # 非严格加载，忽略不匹配的键
                model.load_state_dict(new_state_dict, strict=False)
                logger.info("模型非严格加载成功")
            
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            
            self.local_model = model
            logger.info("本地备份模型加载成功")
            
        except Exception as e:
            logger.error(f"本地备份模型加载失败: {e}")
            self.local_model = None
    
    def register_satellite(self, satellite_info: SatelliteInfo):
        """注册卫星 - 改进版"""
        self.satellites[satellite_info.satellite_id] = satellite_info
        
        # 初始化联邦学习评分
        satellite_info.federated_score = 0.5
        satellite_info.model_version_score = 0.5
        
        logger.info(f"注册卫星: {satellite_info.satellite_id} ({satellite_info.ip_address}:{satellite_info.port})")
    
    def discover_satellites(self):
        """自动发现卫星 - 改进版"""
        logger.info("开始卫星发现...")
        
        for sat_id, sat_config in self.config["satellites"].items():
            # 创建卫星信息对象
            satellite_info = create_satellite_info(sat_id, self.config)
            
            # 测试连接
            if self.communication.test_connection(satellite_info):
                satellite_info.status = SatelliteStatus.ONLINE
                satellite_info.last_heartbeat = time.time()
                logger.info(f"发现在线卫星: {sat_id}")
            
            self.register_satellite(satellite_info)
    
    def submit_inference_task(self, image_data: np.ndarray, 
                            sim_features: Optional[np.ndarray] = None,
                            priority: int = 5,
                            timeout: float = 30.0,
                            location: Optional[List[float]] = None) -> str:
        """提交推理任务 - 改进版"""
        task_id = f"task_{int(time.time() * 1000)}"
        
        task = InferenceTask(
            task_id=task_id,
            image_data=image_data,
            sim_features=sim_features,
            priority=priority,
            timestamp=time.time(),
            timeout=timeout,
            location=location
        )
        
        # 添加到任务队列
        self.task_queue.put((priority, task))
        logger.info(f"提交推理任务: {task_id}, 优先级: {priority}")
        
        return task_id
    
    def get_inference_result(self, task_id: str, timeout: float = 60.0) -> Optional[dict]:
        """获取推理结果"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.results_cache:
                result = self.results_cache.pop(task_id)
                logger.info(f"获取推理结果: {task_id}")
                return result
            
            time.sleep(0.1)
        
        logger.warning(f"获取推理结果超时: {task_id}")
        return None
    
    def _start_monitoring(self):
        """启动监控"""
        def monitor_loop():
            while True:
                try:
                    self._update_satellite_status()
                    self._process_task_queue()
                    
                    # 联邦学习协调
                    if self.training_coordinator:
                        self.training_coordinator.coordinate_training(self.satellites)
                    
                    time.sleep(1)  # 每秒检查一次
                except Exception as e:
                    logger.error(f"监控循环错误: {e}")
                    time.sleep(5)
        
        monitor_loop()
    
    def _update_satellite_status(self):
        """更新卫星状态 - 改进版"""
        for sat_id, sat_info in self.satellites.items():
            try:
                # 发送心跳
                if self.communication.send_heartbeat(sat_info):
                    sat_info.status = SatelliteStatus.ONLINE
                    sat_info.last_heartbeat = time.time()
                else:
                    sat_info.status = SatelliteStatus.OFFLINE
                
                # 获取负载信息
                load_info = self.communication.get_load_info(sat_info)
                if load_info:
                    sat_info.current_load = load_info.get('load', 0.0)
                    sat_info.federated_score = load_info.get('federated_score', 0.5)
                    sat_info.model_version_score = load_info.get('model_version_score', 0.5)
                
            except Exception as e:
                logger.error(f"更新卫星 {sat_id} 状态失败: {e}")
                sat_info.status = SatelliteStatus.OFFLINE
    
    def _process_task_queue(self):
        """处理任务队列 - 改进版"""
        while not self.task_queue.empty():
            try:
                priority, task = self.task_queue.get_nowait()
                
                # 选择卫星
                selected_satellite = self.load_balancer.select_satellite(self.satellites, task)
                
                if selected_satellite:
                    self._submit_to_satellite(task, selected_satellite)
                else:
                    # 没有可用卫星，使用本地处理
                    self._process_locally(task)
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"处理任务队列错误: {e}")
    
    def _submit_to_satellite(self, task: InferenceTask, satellite: SatelliteInfo):
        """提交到卫星 - 改进版"""
        try:
            # 准备任务数据
            task_data = {
                'task_id': task.task_id,
                'image_data': task.image_data,
                'sim_features': task.sim_features,
                'priority': task.priority,
                'timestamp': task.timestamp,
                'timeout': task.timeout,
                'location': task.location
            }
            
            # 发送任务
            result = self.communication.send_inference_task(satellite, task_data)
            
            if result and result.get('success'):
                # 任务成功
                self.results_cache[task.task_id] = result
                logger.info(f"任务 {task.task_id} 在卫星 {satellite.satellite_id} 上成功完成")
            else:
                # 任务失败，触发故障处理
                self.fault_tolerance.handle_failure(task, satellite, self)
                
        except Exception as e:
            logger.error(f"提交任务到卫星失败: {e}")
            self.fault_tolerance.handle_failure(task, satellite, self)
    
    def _process_locally(self, task: InferenceTask):
        """本地处理 - 支持量化模型"""
        try:
            if self.local_model is None:
                logger.error("本地模型不可用")
                return
            
            # 检查是否为量化模型
            is_quantized = isinstance(self.local_model, torch.jit.ScriptModule)
            
            # 准备输入数据
            image_tensor = torch.from_numpy(task.image_data).float()
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
            
            # 处理仿真特征
            if task.sim_features is not None:
                sim_tensor = torch.from_numpy(task.sim_features).float()
                if sim_tensor.dim() == 1:
                    sim_tensor = sim_tensor.unsqueeze(0)  # 添加batch维度
            else:
                # 创建默认仿真特征
                sim_tensor = torch.zeros(1, 11)
            
            # 设备处理
            if is_quantized:
                # 量化模型使用CPU
                device = 'cpu'
                image_tensor = image_tensor.to(device)
                sim_tensor = sim_tensor.to(device)
                logger.info("使用量化模型进行推理")
            else:
                # 原始模型可以使用GPU
                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()
                    sim_tensor = sim_tensor.cuda()
                    self.local_model = self.local_model.cuda()
                    device = 'cuda'
                else:
                    device = 'cpu'
                logger.info("使用原始模型进行推理")
            
            # 推理
            with torch.no_grad():
                output = self.local_model(image_tensor, sim_tensor)
                
                # 后处理
                from train_model import postprocess
                processed_output = postprocess(output)
                
                # 保存结果
                result = {
                    'task_id': task.task_id,
                    'success': True,
                    'output': processed_output,
                    'processed_by': 'quantized_model' if is_quantized else 'local_model',
                    'device': device,
                    'timestamp': time.time()
                }
                
                self.results_cache[task.task_id] = result
                logger.info(f"任务 {task.task_id} 本地处理完成 (设备: {device})")
                
        except Exception as e:
            logger.error(f"本地处理失败: {e}")
    
    def _process_with_federated_model(self, task: InferenceTask):
        """使用联邦学习模型处理 - 支持量化"""
        try:
            if self.federated_manager and self.federated_manager.get_global_model():
                global_model = self.federated_manager.get_global_model()
                
                # 检查是否为量化模型
                is_quantized = isinstance(global_model, torch.jit.ScriptModule)
                
                # 准备输入数据
                image_tensor = torch.from_numpy(task.image_data).float()
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                if task.sim_features is not None:
                    sim_tensor = torch.from_numpy(task.sim_features).float()
                    if sim_tensor.dim() == 1:
                        sim_tensor = sim_tensor.unsqueeze(0)
                else:
                    sim_tensor = torch.zeros(1, 11)
                
                # 设备处理
                if is_quantized:
                    # 量化模型使用CPU
                    device = 'cpu'
                    image_tensor = image_tensor.to(device)
                    sim_tensor = sim_tensor.to(device)
                    logger.info("使用量化联邦学习模型进行推理")
                else:
                    # 原始模型可以使用GPU
                    if torch.cuda.is_available():
                        image_tensor = image_tensor.cuda()
                        sim_tensor = sim_tensor.cuda()
                        global_model = global_model.cuda()
                        device = 'cuda'
                    else:
                        device = 'cpu'
                    logger.info("使用原始联邦学习模型进行推理")
                
                # 推理
                with torch.no_grad():
                    output = global_model(image_tensor, sim_tensor)
                    
                    # 后处理
                    from train_model import postprocess
                    processed_output = postprocess(output)
                    
                    # 保存结果
                    result = {
                        'task_id': task.task_id,
                        'success': True,
                        'output': processed_output,
                        'processed_by': 'quantized_federated_model' if is_quantized else 'federated_model',
                        'device': device,
                        'timestamp': time.time()
                    }
                    
                    self.results_cache[task.task_id] = result
                    logger.info(f"任务 {task.task_id} 联邦学习模型处理完成 (设备: {device})")
                    
        except Exception as e:
            logger.error(f"联邦学习模型处理失败: {e}")
            # 回退到本地处理
            self._process_locally(task)
    
    def get_system_status(self) -> dict:
        """获取系统状态 - 改进版"""
        online_satellites = sum(1 for sat in self.satellites.values() 
                              if sat.status == SatelliteStatus.ONLINE)
        
        federated_status = {}
        if self.federated_manager:
            federated_status = self.federated_manager.get_federated_status()
        
        return {
            'total_satellites': len(self.satellites),
            'online_satellites': online_satellites,
            'queue_size': self.task_queue.qsize(),
            'federated_learning': federated_status,
            'local_model_available': self.local_model is not None,
            'coverage_status': {
                sat_id: sat.coverage_status.value 
                for sat_id, sat in self.satellites.items()
            }
        }
    
    def update_satellite_position(self, satellite_id: str, position: List[float]):
        """更新卫星位置"""
        self.coverage_manager.update_satellite_position(satellite_id, position, time.time())
    
    def get_optimal_satellite_for_location(self, location: List[float]) -> Optional[str]:
        """获取位置的最优卫星"""
        return self.coverage_manager.get_optimal_satellite(location, self.satellites) 