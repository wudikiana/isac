#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星通信系统模块
专注于星间通信、认知无线电和频谱管理
"""

import socket
import pickle
import struct
import time
import logging
from typing import Dict, List, Optional, Any
import numpy as np

from .satellite_core import (
    SatelliteInfo, SatelliteStatus, WaveformType
)

logger = logging.getLogger(__name__)

class CognitiveRadioManager:
    """认知无线电频谱管理器"""
    
    def __init__(self, config: dict):
        self.ALL_BANDS = config.get('available_bands', [2.4e9, 5.8e9, 12e9, 18e9, 24e9, 30e9])
        self.max_bandwidth = config.get('max_bandwidth', 20e6)
        self.interference_threshold = config.get('interference_threshold', 0.1)
        self.terrestrial_users = {}
        self.satellite_spectrum_usage = {}
        self.spectrum_history = []
        
    def detect_primary_users(self, terrestrial_users: Dict[str, Dict[str, Any]]) -> List[float]:
        """检测地面用户占用频段"""
        occupied_bands = []
        
        for user_id, user_info in terrestrial_users.items():
            if user_info.get('active', False):
                occupied_bands.append(user_info.get('frequency', 2.4e9))
        
        return occupied_bands
    
    def dynamic_spectrum_access(self, satellite: SatelliteInfo, 
                               terrestrial_users: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """动态频谱接入"""
        logger.info(f"卫星 {satellite.satellite_id} 请求动态频谱接入")
        
        # 检测地面用户占用频段
        occupied_bands = self.detect_primary_users(terrestrial_users)
        
        # 选择空闲频段
        available_bands = [band for band in self.ALL_BANDS if band not in occupied_bands]
        
        if not available_bands:
            # 如果没有完全空闲的频段，选择干扰最小的
            available_bands = self.ALL_BANDS
        
        # 选择最优频段
        optimal_band = self._select_optimal_band(available_bands, satellite)
        
        # 计算最大可用带宽
        max_available_bandwidth = self._calculate_available_bandwidth(optimal_band, terrestrial_users)
        
        # 动态调整卫星通信频段
        new_config = {
            'frequency': optimal_band,
            'bandwidth': min(self.max_bandwidth, max_available_bandwidth),
            'power': self._calculate_optimal_power(optimal_band, terrestrial_users),
            'modulation': self._select_modulation(optimal_band)
        }
        
        # 更新卫星配置
        satellite.transceiver_config.update(new_config)
        satellite.current_band = optimal_band
        
        # 记录频谱使用
        self.satellite_spectrum_usage[satellite.satellite_id] = {
            'frequency': optimal_band,
            'bandwidth': new_config['bandwidth'],
            'timestamp': time.time()
        }
        
        logger.info(f"卫星 {satellite.satellite_id} 频谱配置更新: {new_config}")
        return new_config
    
    def _select_optimal_band(self, available_bands: List[float], 
                           satellite: SatelliteInfo) -> float:
        """选择最优频段"""
        # 考虑卫星位置、天气条件、干扰等因素
        optimal_band = available_bands[0]
        min_interference = float('inf')
        
        for band in available_bands:
            interference = self._calculate_interference(band, satellite)
            if interference < min_interference:
                min_interference = interference
                optimal_band = band
        
        return optimal_band
    
    def _calculate_interference(self, frequency: float, satellite: SatelliteInfo) -> float:
        """计算干扰水平"""
        # 简化的干扰计算模型
        base_interference = 0.1
        frequency_factor = frequency / 1e9  # 归一化
        distance_factor = 1.0 / (1.0 + satellite.current_position[2] / 1000)  # 高度因子
        
        return base_interference * frequency_factor * distance_factor
    
    def _calculate_available_bandwidth(self, frequency: float, 
                                     terrestrial_users: Dict[str, Dict[str, Any]]) -> float:
        """计算可用带宽"""
        # 简化的带宽计算
        base_bandwidth = self.max_bandwidth
        
        # 根据地面用户密度调整
        nearby_users = sum(1 for user in terrestrial_users.values() 
                          if abs(user.get('frequency', 0) - frequency) < 1e9)
        
        if nearby_users > 5:
            base_bandwidth *= 0.5
        elif nearby_users > 2:
            base_bandwidth *= 0.8
        
        return base_bandwidth
    
    def _calculate_optimal_power(self, frequency: float, 
                               terrestrial_users: Dict[str, Dict[str, Any]]) -> float:
        """计算最优发射功率"""
        base_power = 10.0  # 10W
        
        # 根据频率调整功率
        if frequency > 20e9:
            base_power *= 1.5
        elif frequency < 5e9:
            base_power *= 0.8
        
        return base_power
    
    def _select_modulation(self, frequency: float) -> str:
        """选择调制方式"""
        if frequency > 20e9:
            return WaveformType.QAM64.value
        elif frequency > 10e9:
            return WaveformType.QAM16.value
        else:
            return WaveformType.QPSK.value
    
    def get_spectrum_usage(self) -> Dict[str, Any]:
        """获取频谱使用情况"""
        return {
            'satellite_usage': self.satellite_spectrum_usage,
            'terrestrial_users': self.terrestrial_users,
            'available_bands': self.ALL_BANDS
        }

class SatelliteCommunication:
    """卫星通信管理器"""
    
    def __init__(self, connection_timeout: float = 5.0, heartbeat_interval: float = 30.0):
        self.connection_timeout = connection_timeout
        self.heartbeat_interval = heartbeat_interval
        self.connection_pool = {}
        
    def test_connection(self, satellite: SatelliteInfo) -> bool:
        """测试连接"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            result = sock.connect_ex((satellite.ip_address, satellite.port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.error(f"连接测试失败 {satellite.satellite_id}: {e}")
            return False
    
    def send_heartbeat(self, satellite: SatelliteInfo) -> bool:
        """发送心跳"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            sock.connect((satellite.ip_address, satellite.port))
            
            # 发送心跳消息
            heartbeat_msg = {
                'type': 'heartbeat',
                'timestamp': time.time()
            }
            data = pickle.dumps(heartbeat_msg)
            sock.send(struct.pack('!I', len(data)) + data)
            
            # 接收响应
            response_size = struct.unpack('!I', sock.recv(4))[0]
            response_data = sock.recv(response_size)
            response = pickle.loads(response_data)
            
            sock.close()
            return response.get('status') == 'ok'
        except Exception as e:
            logger.error(f"心跳发送失败 {satellite.satellite_id}: {e}")
            return False
    
    def get_load_info(self, satellite: SatelliteInfo) -> Optional[dict]:
        """获取负载信息"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            sock.connect((satellite.ip_address, satellite.port))
            
            # 发送负载查询消息
            load_msg = {
                'type': 'get_load',
                'timestamp': time.time()
            }
            data = pickle.dumps(load_msg)
            sock.send(struct.pack('!I', len(data)) + data)
            
            # 接收响应
            response_size = struct.unpack('!I', sock.recv(4))[0]
            response_data = sock.recv(response_size)
            response = pickle.loads(response_data)
            
            sock.close()
            return response
        except Exception as e:
            logger.error(f"获取负载信息失败 {satellite.satellite_id}: {e}")
            return None
    
    def send_inference_task(self, satellite: SatelliteInfo, task_data: dict) -> Optional[dict]:
        """发送推理任务"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30.0)  # 推理任务超时时间更长
            sock.connect((satellite.ip_address, satellite.port))
            
            # 发送推理任务
            data = pickle.dumps(task_data)
            sock.send(struct.pack('!I', len(data)) + data)
            
            # 接收响应
            response_size = struct.unpack('!I', sock.recv(4))[0]
            response_data = sock.recv(response_size)
            response = pickle.loads(response_data)
            
            sock.close()
            return response
        except Exception as e:
            logger.error(f"发送推理任务失败 {satellite.satellite_id}: {e}")
            return None
    
    def send_training_task(self, satellite: SatelliteInfo, task_data: dict) -> Optional[dict]:
        """发送训练任务"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(60.0)  # 训练任务超时时间更长
            sock.connect((satellite.ip_address, satellite.port))
            
            # 发送训练任务
            data = pickle.dumps(task_data)
            sock.send(struct.pack('!I', len(data)) + data)
            
            # 接收响应
            response_size = struct.unpack('!I', sock.recv(4))[0]
            response_data = sock.recv(response_size)
            response = pickle.loads(response_data)
            
            sock.close()
            return response
        except Exception as e:
            logger.error(f"发送训练任务失败 {satellite.satellite_id}: {e}")
            return None
    
    def send_control_command(self, satellite: SatelliteInfo, command: str, params: dict = None) -> Optional[dict]:
        """发送控制命令"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((satellite.ip_address, satellite.port))
            
            # 发送控制命令
            control_msg = {
                'type': 'control',
                'command': command,
                'params': params or {},
                'timestamp': time.time()
            }
            data = pickle.dumps(control_msg)
            sock.send(struct.pack('!I', len(data)) + data)
            
            # 接收响应
            response_size = struct.unpack('!I', sock.recv(4))[0]
            response_data = sock.recv(response_size)
            response = pickle.loads(response_data)
            
            sock.close()
            return response
        except Exception as e:
            logger.error(f"发送控制命令失败 {satellite.satellite_id}: {e}")
            return None
    
    def get_communication_status(self) -> Dict[str, Any]:
        """获取通信状态"""
        return {
            'connection_timeout': self.connection_timeout,
            'heartbeat_interval': self.heartbeat_interval,
            'connection_pool_size': len(self.connection_pool)
        } 