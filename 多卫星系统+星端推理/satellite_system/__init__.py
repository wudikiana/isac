#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星系统包
整合所有卫星系统功能模块
"""

# 导出核心模块
from .satellite_core import (
    SatelliteStatus, CoverageStatus, EmergencyLevel, WaveformType,
    TaskType, DisasterType,
    SatelliteInfo, InferenceTask, TrainingTask, EmergencyBeacon, PPOResourceAllocation,
    IntentTask, RemoteSensingData, InterpretationResult,
    calculate_distance, load_config, get_default_config, create_satellite_info,
    format_timestamp, validate_location
)

# 导出推理系统
from .satellite_inference import (
    MultiSatelliteInferenceSystem, LoadBalancer, FaultToleranceManager,
    SatelliteCommunication, CoverageManager
)

# 导出应急系统
from .satellite_emergency import (
    EmergencyResponseSystem, PPOController
)

# 导出通信系统
from .satellite_communication import (
    CognitiveRadioManager, SatelliteCommunication as CommSystem
)

# 导出轨道系统
from .satellite_orbit import (
    OrbitManager, AutonomousOrbitControl, IonThrusterSystem
)

# 导出联邦学习系统
from .satellite_federated import (
    FederatedLearningManager, DistributedTrainingManager, TrainingCoordinator
)

# 导出主系统
from .main_system import MultiSatelliteSystem

# 导出交互系统
from .interactive_system import InteractiveSatelliteSystem

# 导出意图理解模块
from .intent_understanding import IntentUnderstandingEngine, PromptTemplates

# 导出在轨智能解译模块
from .orbital_interpretation import OrbitalInterpreter, FeatureExtractor, PriorKnowledgeBase

# 导出协同任务调度模块
from .cooperative_scheduler import CooperativeScheduler

# 版本信息
__version__ = "2.0.0"
__author__ = "Satellite System Team"
__description__ = "Advanced Multi-Satellite System with AI Inference, Emergency Response, and Federated Learning"

# 主要导出
__all__ = [
    # 核心模块
    'SatelliteStatus', 'CoverageStatus', 'EmergencyLevel', 'WaveformType',
    'TaskType', 'DisasterType',
    'SatelliteInfo', 'InferenceTask', 'TrainingTask', 'EmergencyBeacon', 'PPOResourceAllocation',
    'IntentTask', 'RemoteSensingData', 'InterpretationResult',
    'calculate_distance', 'load_config', 'get_default_config', 'create_satellite_info',
    'format_timestamp', 'validate_location',
    
    # 推理系统
    'MultiSatelliteInferenceSystem', 'LoadBalancer', 'FaultToleranceManager',
    'SatelliteCommunication', 'CoverageManager',
    
    # 应急系统
    'EmergencyResponseSystem', 'PPOController',
    
    # 通信系统
    'CognitiveRadioManager', 'CommSystem',
    
    # 轨道系统
    'OrbitManager', 'AutonomousOrbitControl', 'IonThrusterSystem',
    
    # 联邦学习系统
    'FederatedLearningManager', 'DistributedTrainingManager', 'TrainingCoordinator',
    
    # 意图理解模块
    'IntentUnderstandingEngine', 'PromptTemplates',
    
    # 在轨智能解译模块
    'OrbitalInterpreter', 'FeatureExtractor', 'PriorKnowledgeBase',
    
    # 协同任务调度模块
    'CooperativeScheduler',
    
    # 主系统
    'MultiSatelliteSystem',
    
    # 交互系统
    'InteractiveSatelliteSystem',
]

def get_system_info():
    """获取系统信息"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': [
            'satellite_core',
            'satellite_inference', 
            'satellite_emergency',
            'satellite_communication',
            'satellite_orbit',
            'satellite_federated',
            'intent_understanding',
            'orbital_interpretation',
            'cooperative_scheduler',
            'main_system',
            'interactive_system'
        ]
    }

def create_system(config_file: str = "satellite_config.json"):
    """创建卫星系统实例"""
    return MultiSatelliteSystem(config_file)

def create_interactive_system(config_file: str = "satellite_config.json"):
    """创建交互式卫星系统实例"""
    return InteractiveSatelliteSystem(config_file) 