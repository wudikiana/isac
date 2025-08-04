#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星系统核心模块
包含所有共享的基础类、枚举和数据结构
"""

import torch
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 基础枚举 ====================

class SatelliteStatus(Enum):
    """卫星状态枚举"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    EMERGENCY = "emergency"  # 紧急状态
    SAR_IMAGING = "sar_imaging"  # SAR成像中
    COMPUTING = "computing"  # 提供算力中

class CoverageStatus(Enum):
    """覆盖状态枚举"""
    COVERED = "covered"      # 当前可覆盖
    NOT_COVERED = "not_covered"  # 当前不可覆盖
    WILL_COVER = "will_cover"    # 即将覆盖
    WAS_COVERED = "was_covered"  # 之前覆盖过

class EmergencyLevel(Enum):
    """紧急程度枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class WaveformType(Enum):
    """波形类型枚举"""
    QPSK = "qpsk"
    QAM16 = "qam16"
    QAM64 = "qam64"
    OFDM = "ofdm"
    ADAPTIVE = "adaptive"

class TaskType(Enum):
    """任务类型枚举"""
    MONITORING = "monitoring"      # 监测任务
    IMAGING = "imaging"           # 成像任务
    ANALYSIS = "analysis"         # 分析任务
    EMERGENCY = "emergency"       # 紧急任务
    ROUTINE = "routine"           # 例行任务

class DisasterType(Enum):
    """灾害类型枚举"""
    LANDSLIDE = "landslide"       # 滑坡
    FLOOD = "flood"              # 洪水
    FIRE = "fire"                # 火灾
    EARTHQUAKE = "earthquake"     # 地震
    TSUNAMI = "tsunami"          # 海啸
    VOLCANO = "volcano"          # 火山
    UNKNOWN = "unknown"          # 未知

# ==================== 基础数据结构 ====================

@dataclass
class SatelliteInfo:
    """卫星信息数据类"""
    satellite_id: str
    ip_address: str
    port: int
    status: SatelliteStatus
    compute_capacity: float  # FLOPS
    memory_capacity: int     # MB
    current_load: float      # 0-1
    last_heartbeat: float
    model_version: str
    supported_features: List[str]
    
    # 覆盖范围相关
    coverage_area: Dict[str, List[float]]  # 覆盖区域坐标
    current_position: List[float]  # 当前位置 [lat, lon, alt]
    orbit_period: float  # 轨道周期(分钟)
    coverage_status: CoverageStatus = CoverageStatus.NOT_COVERED
    last_coverage_update: float = field(default_factory=time.time)
    
    # 通信相关
    transceiver_config: Dict[str, Any] = field(default_factory=dict)
    current_band: float = 0.0
    
    # 轨道控制相关
    fuel_level: float = 100.0  # 燃料百分比
    thruster_status: Dict[str, bool] = field(default_factory=dict)
    velocity: List[float] = field(default_factory=lambda: [0, 0, 0])  # 速度向量
    
    # 训练相关
    training_data_count: int = 0
    last_training_time: float = 0.0
    model_hash: str = ""
    parameter_version: int = 0
    local_model_hash: str = ""
    
    # 联邦学习相关
    federated_score: float = 0.5  # 联邦学习评分
    model_version_score: float = 0.5  # 模型版本评分
    federated_participation_count: int = 0  # 参与联邦学习次数
    last_federated_update: float = 0.0  # 最后联邦学习更新时间

@dataclass
class InferenceTask:
    """推理任务数据类"""
    task_id: str
    image_data: np.ndarray
    sim_features: Optional[np.ndarray]
    priority: int  # 1-10, 10最高
    timestamp: float
    timeout: float
    callback: Optional[callable] = None
    location: Optional[List[float]] = None  # 任务地理位置

@dataclass
class TrainingTask:
    """训练任务数据类"""
    task_id: str
    satellite_id: str
    training_data: List[Tuple[np.ndarray, np.ndarray]]  # (image, mask) pairs
    sim_features: Optional[np.ndarray]
    training_config: Dict[str, Any]
    timestamp: float
    location: Optional[List[float]] = None

@dataclass
class EmergencyBeacon:
    """紧急信标数据"""
    beacon_id: str
    location: List[float]  # [lat, lon, alt]
    emergency_level: EmergencyLevel
    timestamp: float
    description: str
    required_resources: Dict[str, Any]
    priority: int = 10

@dataclass
class PPOResourceAllocation:
    """PPO资源分配结果"""
    allocation_id: str
    satellite_assignments: Dict[str, Dict[str, Any]]
    priority_order: List[str]
    estimated_completion_time: float
    resource_utilization: Dict[str, float]

@dataclass
class IntentTask:
    """意图理解任务数据类"""
    task_id: str
    original_command: str  # 原始自然语言指令
    task_type: TaskType
    disaster_type: DisasterType
    target_location: List[float]  # 目标位置 [lat, lon, alt]
    priority: int  # 1-10, 10最高
    timestamp: float
    estimated_duration: float  # 预计完成时间(分钟)
    required_satellites: List[str]  # 需要的卫星列表
    imaging_parameters: Dict[str, Any]  # 成像参数
    analysis_parameters: Dict[str, Any]  # 分析参数
    callback: Optional[callable] = None

@dataclass
class RemoteSensingData:
    """遥感数据类"""
    data_id: str
    satellite_id: str
    location: List[float]  # 拍摄位置
    timestamp: float
    data_type: str  # "optical", "sar", "multispectral"
    image_data: np.ndarray
    metadata: Dict[str, Any]  # 元数据
    channel_data: Dict[str, np.ndarray]  # 信道数据
    quality_score: float  # 数据质量评分

@dataclass
class InterpretationResult:
    """智能解译结果类"""
    result_id: str
    task_id: str
    satellite_id: str
    timestamp: float
    disaster_probability: float  # 灾害概率
    disaster_type: DisasterType
    confidence_score: float  # 置信度
    affected_area: float  # 受影响面积(km²)
    risk_level: EmergencyLevel
    key_features: Dict[str, Any]  # 关键特征
    result_slice: np.ndarray  # 结果切片
    analysis_report: str  # 分析报告

# ==================== 工具函数 ====================

def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    """计算两点间距离"""
    try:
        # 确保输入是数值类型
        pos1_float = [float(x) for x in pos1]
        pos2_float = [float(x) for x in pos2]
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1_float, pos2_float)))
    except (ValueError, TypeError) as e:
        logger.error(f"计算距离时参数错误: pos1={pos1}, pos2={pos2}, error={e}")
        return float('inf')  # 返回无穷大表示无效距离

def load_config(config_file: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_file} 不存在，使用默认配置")
        return get_default_config()

def get_default_config() -> dict:
    """获取默认配置"""
    return {
        "satellites": {
            "sat_001": {
                "ip": "192.168.1.101",
                "port": 8080,
                "compute_capacity": 1e12,
                "memory_capacity": 8192,
                "coverage_area": {"lat": [35, 45], "lon": [110, 130]},
                "orbit_period": 90.0,
                "supported_features": ["inference", "training", "communication", "sar_imaging"]
            },
            "sat_002": {
                "ip": "192.168.1.102",
                "port": 8080,
                "compute_capacity": 1.2e12,
                "memory_capacity": 10240,
                "coverage_area": {"lat": [30, 40], "lon": [115, 135]},
                "orbit_period": 90.0,
                "supported_features": ["inference", "training", "communication", "sar_imaging"]
            },
            "sat_003": {
                "ip": "192.168.1.103",
                "port": 8080,
                "compute_capacity": 0.8e12,
                "memory_capacity": 6144,
                "coverage_area": {"lat": [25, 35], "lon": [110, 130]},
                "orbit_period": 90.0,
                "supported_features": ["inference", "training", "communication", "sar_imaging"]
            }
        },
        "training": {
            "sync_interval": 300,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs_per_sync": 5
        },
        "coverage": {
            "update_interval": 60,
            "prediction_horizon": 3600
        },
        "local_backup": {
            "model_path": "models/best_multimodal_patch_model.pth"
        },
        "emergency_response": {
            "response_timeout": 300,
            "max_concurrent_emergencies": 5,
            "ppo_learning_rate": 0.0003,
            "ppo_clip_ratio": 0.2
        },
        "federated_learning": {
            "sync_interval": 600,
            "min_participants": 2,
            "aggregation_method": "fedavg",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs_per_round": 5
        },
        "cognitive_radio": {
            "available_bands": [2.4e9, 5.8e9, 12e9, 18e9, 24e9, 30e9],
            "max_bandwidth": 20e6,
            "interference_threshold": 0.1
        },
        "orbit_control": {
            "collision_threshold": 500,
            "formation_threshold": 1000,
            "fuel_efficiency": 0.8,
            "maneuver_planning": "fuel_optimal"
        },
        "inference": {
            "task_timeout": 30.0,
            "max_queue_size": 1000,
            "cache_size": 100
        }
    }

def create_satellite_info(sat_id: str, config: dict) -> SatelliteInfo:
    """根据配置创建卫星信息对象"""
    sat_config = config["satellites"][sat_id]
    
    return SatelliteInfo(
        satellite_id=sat_id,
        ip_address=sat_config["ip"],
        port=sat_config["port"],
        status=SatelliteStatus.OFFLINE,
        compute_capacity=sat_config["compute_capacity"],
        memory_capacity=sat_config["memory_capacity"],
        current_load=0.0,
        last_heartbeat=0.0,
        model_version="v1.0",
        supported_features=sat_config.get("supported_features", ["inference", "training"]),
        coverage_area=sat_config["coverage_area"],
        current_position=[0.0, 0.0, 0.0],
        orbit_period=sat_config["orbit_period"]
    )

def format_timestamp(timestamp: float) -> str:
    """格式化时间戳"""
    return datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

def validate_location(location: List[float]) -> bool:
    """验证地理位置坐标"""
    if len(location) < 2:
        return False
    
    lat, lon = location[0], location[1]
    return -90 <= lat <= 90 and -180 <= lon <= 180 