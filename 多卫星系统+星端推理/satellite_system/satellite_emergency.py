#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星应急系统模块
专注于灾害应急响应、PPO强化学习资源分配
"""

import torch
import numpy as np
import time
import json
import queue
import logging
from typing import Dict, List, Tuple, Optional, Any
import math

from .satellite_core import (
    SatelliteInfo, SatelliteStatus, EmergencyLevel, EmergencyBeacon,
    PPOResourceAllocation, calculate_distance
)

logger = logging.getLogger(__name__)

class PPOController:
    """PPO强化学习控制器"""
    
    def __init__(self, learning_rate: float = 0.0003, clip_ratio: float = 0.2):
        self.state_dim = 64  # 状态空间维度
        self.action_dim = 32  # 动作空间维度
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.epsilon = 0.2
        self.clip_ratio = clip_ratio
        
        # 初始化策略网络和价值网络
        self.policy_net = self._build_policy_network()
        self.value_net = self._build_value_network()
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        
        self.memory = []
        self.episode_rewards = []
        
    def _build_policy_network(self):
        """构建策略网络"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_dim),
            torch.nn.Softmax(dim=-1)
        )
    
    def _build_value_network(self):
        """构建价值网络"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def get_state_vector(self, emergency_beacon: EmergencyBeacon, 
                        available_satellites: Dict[str, SatelliteInfo]) -> torch.Tensor:
        """构建状态向量"""
        state = []
        
        # 紧急信标信息
        if emergency_beacon.location:
            state.extend([float(x) for x in emergency_beacon.location])
        else:
            state.extend([0.0, 0.0, 0.0])  # 默认位置
        state.append(float(emergency_beacon.emergency_level.value))
        state.append(float(emergency_beacon.priority))
        
        # 卫星状态信息
        for sat_id, sat_info in available_satellites.items():
            # 确保位置是数值类型
            if sat_info.current_position:
                state.extend([float(x) for x in sat_info.current_position])
            else:
                state.extend([0.0, 0.0, 0.0])  # 默认位置
            state.append(float(sat_info.current_load))
            state.append(float(sat_info.compute_capacity / 1e12))  # 归一化
            state.append(float(sat_info.fuel_level / 100.0))
            # 将状态转换为数值
            status_value = 1.0 if sat_info.status.value == "online" else 0.0
            state.append(status_value)
        
        # 填充到固定长度
        while len(state) < self.state_dim:
            state.append(0.0)
        
        return torch.tensor(state[:self.state_dim], dtype=torch.float32)
    
    def allocate_resources(self, emergency_beacon: EmergencyBeacon,
                          available_satellites: Dict[str, SatelliteInfo]) -> PPOResourceAllocation:
        """使用PPO算法分配资源"""
        logger.info(f"PPO控制器开始为紧急信标 {emergency_beacon.beacon_id} 分配资源")
        
        # 构建状态向量
        state = self.get_state_vector(emergency_beacon, available_satellites)
        
        # 获取动作概率分布
        with torch.no_grad():
            action_probs = self.policy_net(state)
            action = torch.multinomial(action_probs, 1).item()
        
        # 根据动作分配资源
        allocation = self._execute_allocation(action, emergency_beacon, available_satellites)
        
        # 记录经验
        self.memory.append({
            'state': state,
            'action': action,
            'reward': self._calculate_reward(allocation, emergency_beacon),
            'next_state': state  # 简化处理
        })
        
        return allocation
    
    def _execute_allocation(self, action: int, emergency_beacon: EmergencyBeacon,
                           satellites: Dict[str, SatelliteInfo]) -> PPOResourceAllocation:
        """执行资源分配"""
        # 根据动作选择卫星分配策略
        satellite_assignments = {}
        priority_order = []
        
        # 贪婪算法：选择距离最近的可用卫星
        sorted_sats = sorted(satellites.items(), 
                           key=lambda x: calculate_distance(
                               x[1].current_position, emergency_beacon.location) if x[1].current_position and emergency_beacon.location else float('inf'))
        
        for i, (sat_id, sat_info) in enumerate(sorted_sats):
            if sat_info.status == SatelliteStatus.ONLINE and sat_info.current_load < 0.8:
                assignment = {
                    'role': 'primary' if i == 0 else 'support',
                    'compute_allocated': min(0.3, 1.0 - sat_info.current_load),
                    'memory_allocated': min(512, sat_info.memory_capacity // 4),
                    'priority': 10 - i
                }
                satellite_assignments[sat_id] = assignment
                priority_order.append(sat_id)
                
                if len(priority_order) >= 3:  # 最多分配3颗卫星
                    break
        
        return PPOResourceAllocation(
            allocation_id=f"alloc_{int(time.time())}",
            satellite_assignments=satellite_assignments,
            priority_order=priority_order,
            estimated_completion_time=time.time() + 300,  # 5分钟
            resource_utilization={'cpu': 0.7, 'memory': 0.6, 'bandwidth': 0.8}
        )
    
    def _calculate_reward(self, allocation: PPOResourceAllocation, 
                         emergency_beacon: EmergencyBeacon) -> float:
        """计算奖励值"""
        reward = 0.0
        
        # 基于分配的卫星数量
        reward += len(allocation.satellite_assignments) * 10
        
        # 基于紧急程度
        reward += emergency_beacon.emergency_level.value * 5
        
        # 基于资源利用率
        avg_utilization = sum(allocation.resource_utilization.values()) / len(allocation.resource_utilization)
        reward += avg_utilization * 20
        
        return reward
    
    def update_policy(self):
        """更新策略网络"""
        if len(self.memory) < 10:
            return
        
        # 计算优势函数
        states = torch.stack([exp['state'] for exp in self.memory])
        actions = torch.tensor([exp['action'] for exp in self.memory])
        rewards = torch.tensor([exp['reward'] for exp in self.memory])
        
        # 计算价值估计
        with torch.no_grad():
            values = self.value_net(states).squeeze()
        
        # 计算优势
        advantages = rewards - values
        
        # 更新策略网络
        action_probs = self.policy_net(states)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # PPO损失
        ratio = selected_probs / (selected_probs.detach() + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        
        # 更新价值网络
        value_loss = torch.nn.functional.mse_loss(values, rewards)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
        
        # 清空记忆
        self.memory = []
        
        logger.info(f"PPO策略更新完成 - 策略损失: {policy_loss.item():.4f}, 价值损失: {value_loss.item():.4f}")

class EmergencyResponseSystem:
    """灾害应急响应系统"""
    
    def __init__(self, config: dict):
        self.config = config
        self.ppo_controller = PPOController(
            learning_rate=config.get('ppo_learning_rate', 0.0003),
            clip_ratio=config.get('ppo_clip_ratio', 0.2)
        )
        self.satellites = {}
        self.emergency_queue = queue.PriorityQueue()
        self.response_history = []
        self.response_timeout = config.get('response_timeout', 300)
        self.max_concurrent_emergencies = config.get('max_concurrent_emergencies', 5)
        
    def trigger_emergency_beacon(self, beacon: EmergencyBeacon):
        """触发紧急信标"""
        logger.info(f"收到紧急信标: {beacon.beacon_id}, 级别: {beacon.emergency_level.value}")
        
        # 将紧急信标加入队列
        self.emergency_queue.put((beacon.priority, beacon))
        
        # 启动应急响应
        self._process_emergency_response(beacon)
    
    def _process_emergency_response(self, beacon: EmergencyBeacon):
        """处理应急响应"""
        logger.info(f"开始处理紧急信标 {beacon.beacon_id} 的应急响应")
        
        # 1. PPO控制器分配资源
        allocation = self.ppo_controller.allocate_resources(beacon, self.satellites)
        
        # 2. 执行协同调度
        self._execute_coordinated_scheduling(allocation, beacon)
        
        # 3. 记录响应历史
        self.response_history.append({
            'beacon_id': beacon.beacon_id,
            'allocation': allocation,
            'timestamp': time.time(),
            'status': 'completed'
        })
        
        logger.info(f"紧急信标 {beacon.beacon_id} 响应完成")
    
    def _execute_coordinated_scheduling(self, allocation: PPOResourceAllocation, 
                                      beacon: EmergencyBeacon):
        """执行协同调度"""
        logger.info("开始执行协同调度指令")
        
        for sat_id, assignment in allocation.satellite_assignments.items():
            if sat_id in self.satellites:
                satellite = self.satellites[sat_id]
                
                if assignment['role'] == 'primary':
                    # 主卫星执行SAR成像
                    self._execute_sar_imaging(satellite, beacon)
                else:
                    # 支持卫星提供额外算力
                    self._provide_computing_support(satellite, assignment)
        
        # 压缩结果回传
        self._compress_and_transmit_results(allocation, beacon)
    
    def _execute_sar_imaging(self, satellite: SatelliteInfo, beacon: EmergencyBeacon):
        """执行SAR成像"""
        logger.info(f"卫星 {satellite.satellite_id} 开始SAR成像")
        
        # 模拟SAR成像过程
        imaging_time = 120  # 2分钟
        satellite.status = SatelliteStatus.SAR_IMAGING
        
        # 更新卫星负载
        satellite.current_load += 0.3
        
        logger.info(f"SAR成像完成，耗时: {imaging_time}秒")
        satellite.status = SatelliteStatus.ONLINE
    
    def _provide_computing_support(self, satellite: SatelliteInfo, assignment: Dict[str, Any]):
        """提供计算支持"""
        logger.info(f"卫星 {satellite.satellite_id} 提供计算支持")
        
        satellite.status = SatelliteStatus.COMPUTING
        satellite.current_load += assignment['compute_allocated']
        
        # 模拟计算过程
        time.sleep(0.1)  # 模拟计算时间
        
        logger.info(f"计算支持完成")
        satellite.status = SatelliteStatus.ONLINE
    
    def _compress_and_transmit_results(self, allocation: PPOResourceAllocation, 
                                     beacon: EmergencyBeacon):
        """压缩并传输结果"""
        logger.info("开始压缩和传输结果")
        
        # 模拟数据压缩
        compression_ratio = 0.3
        original_size = 100  # MB
        compressed_size = original_size * compression_ratio
        
        # 模拟传输
        transmission_time = compressed_size / 10  # 假设10MB/s传输速率
        
        logger.info(f"结果压缩完成，压缩比: {compression_ratio:.2f}")
        logger.info(f"传输完成，耗时: {transmission_time:.2f}秒")
    
    def register_satellite(self, satellite: SatelliteInfo):
        """注册卫星"""
        self.satellites[satellite.satellite_id] = satellite
        logger.info(f"卫星 {satellite.satellite_id} 注册到应急响应系统")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'satellites_count': len(self.satellites),
            'emergency_queue_size': self.emergency_queue.qsize(),
            'response_history_count': len(self.response_history),
            'ppo_controller_status': 'active',
            'response_timeout': self.response_timeout,
            'max_concurrent_emergencies': self.max_concurrent_emergencies
        }
    
    def get_emergency_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取应急响应历史"""
        return self.response_history[-limit:]
    
    def clear_emergency_queue(self):
        """清空应急队列"""
        while not self.emergency_queue.empty():
            try:
                self.emergency_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("应急队列已清空") 