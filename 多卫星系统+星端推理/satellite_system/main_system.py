#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多卫星系统主模块
整合所有功能模块，提供统一的系统接口
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any

from .satellite_core import (
    SatelliteInfo, SatelliteStatus, EmergencyLevel, EmergencyBeacon,
    TaskType, DisasterType, IntentTask, RemoteSensingData, InterpretationResult,
    load_config, create_satellite_info
)
from .satellite_inference import MultiSatelliteInferenceSystem
from .satellite_emergency import EmergencyResponseSystem
from .satellite_communication import SatelliteCommunication, CognitiveRadioManager
from .satellite_orbit import OrbitManager
from .satellite_federated import TrainingCoordinator
from .intent_understanding import IntentUnderstandingEngine
from .orbital_interpretation import OrbitalInterpreter
from .cooperative_scheduler import CooperativeScheduler

logger = logging.getLogger(__name__)

class MultiSatelliteSystem:
    """多卫星系统主类"""
    
    def __init__(self, config_file: str = "satellite_config.json"):
        self.config = load_config(config_file)
        self.satellites: Dict[str, SatelliteInfo] = {}
        
        # 初始化各个子系统
        self.inference_system = MultiSatelliteInferenceSystem(config_file)
        self.emergency_system = EmergencyResponseSystem(self.config.get('emergency_response', {}))
        self.communication = SatelliteCommunication()
        self.cognitive_radio = CognitiveRadioManager(self.config.get('cognitive_radio', {}))
        self.orbit_manager = OrbitManager(self.config.get('orbit_control', {}))
        self.training_coordinator = TrainingCoordinator(self.config)
        
        # 初始化新功能模块
        self.intent_engine = IntentUnderstandingEngine(self.config)
        self.orbital_interpreter = OrbitalInterpreter(
            self.config.get("model_path", "models/best_multimodal_patch_model.pth"),
            self.config
        )
        self.cooperative_scheduler = CooperativeScheduler(self.config)
        
        # 初始化系统
        self._initialize_system()
        
        # 启动监控线程
        self._start_monitoring()
    
    def _initialize_system(self):
        """初始化系统"""
        logger.info("初始化多卫星系统")
        
        # 初始化卫星
        for sat_id, sat_config in self.config['satellites'].items():
            satellite = create_satellite_info(sat_id, self.config)
            
            self.satellites[sat_id] = satellite
            
            # 注册到各个子系统
            self.inference_system.register_satellite(satellite)
            self.emergency_system.register_satellite(satellite)
            self.orbit_manager.register_satellite(satellite)
            self.cooperative_scheduler.add_satellite(satellite)
        
        logger.info(f"系统初始化完成，注册卫星数: {len(self.satellites)}")
    
    def _start_monitoring(self):
        """启动监控线程"""
        def monitor_loop():
            while True:
                try:
                    # 更新卫星状态
                    self._update_satellite_status()
                    
                    # 轨道控制
                    self.orbit_manager.perform_orbit_control()
                    
                    # 训练协调
                    self.training_coordinator.coordinate_training(self.satellites)
                    
                    # 认知无线电频谱管理
                    self._update_cognitive_radio()
                    
                    time.sleep(30)  # 30秒监控间隔
                    
                except Exception as e:
                    logger.error(f"监控线程异常: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("监控线程启动完成")
    
    def _update_satellite_status(self):
        """更新卫星状态"""
        current_time = time.time()
        
        for sat_id, satellite in self.satellites.items():
            # 模拟卫星状态变化
            if current_time - satellite.last_heartbeat > 60:
                satellite.status = SatelliteStatus.OFFLINE
            else:
                satellite.status = SatelliteStatus.ONLINE
            
            # 模拟负载变化
            if satellite.current_load > 0:
                satellite.current_load = max(0, satellite.current_load - 0.01)
    
    def _update_cognitive_radio(self):
        """更新认知无线电"""
        terrestrial_users = {}  # 这里可以从外部获取地面用户信息
        
        for sat_id, satellite in self.satellites.items():
            if satellite.status == SatelliteStatus.ONLINE:
                # 动态频谱接入
                self.cognitive_radio.dynamic_spectrum_access(satellite, terrestrial_users)
    
    def trigger_emergency(self, location: List[float], emergency_level: EmergencyLevel,
                         description: str = "紧急情况") -> str:
        """触发紧急情况"""
        beacon = EmergencyBeacon(
            beacon_id=f"emergency_{int(time.time())}",
            location=location,
            emergency_level=emergency_level,
            timestamp=time.time(),
            description=description,
            required_resources={'compute': 0.5, 'memory': 1024, 'bandwidth': 100}
        )
        
        self.emergency_system.trigger_emergency_beacon(beacon)
        return beacon.beacon_id
    
    def submit_inference_task(self, image_data, sim_features=None, priority=5, 
                            timeout=30.0, location=None) -> str:
        """提交推理任务"""
        return self.inference_system.submit_inference_task(
            image_data, sim_features, priority, timeout, location
        )
    
    def get_inference_result(self, task_id: str, timeout: float = 60.0) -> Optional[dict]:
        """获取推理结果"""
        return self.inference_system.get_inference_result(task_id, timeout)
    
    def update_satellite_position(self, satellite_id: str, position: List[float]):
        """更新卫星位置"""
        if satellite_id in self.satellites:
            self.inference_system.update_satellite_position(satellite_id, position)
            logger.info(f"更新卫星 {satellite_id} 位置: {position}")
    
    def get_optimal_satellite_for_location(self, location: List[float]) -> Optional[str]:
        """获取指定位置的最优卫星"""
        return self.inference_system.get_optimal_satellite_for_location(location)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        online_count = sum(1 for sat in self.satellites.values() 
                         if sat.status == SatelliteStatus.ONLINE)
        
        return {
            'total_satellites': len(self.satellites),
            'online_satellites': online_count,
            'satellites': {
                sat_id: {
                    'status': sat.status.value,
                    'load': sat.current_load,
                    'position': sat.current_position,
                    'fuel': sat.fuel_level
                } for sat_id, sat in self.satellites.items()
            },
            'emergency_system': self.emergency_system.get_system_status(),
            'inference_system': self.inference_system.get_system_status(),
            'orbit_manager': self.orbit_manager.get_orbit_status(),
            'training_coordinator': self.training_coordinator.get_training_status(),
            'cognitive_radio': self.cognitive_radio.get_spectrum_usage()
        }
    
    def discover_satellites(self):
        """发现卫星"""
        self.inference_system.discover_satellites()
    
    def get_emergency_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取应急响应历史"""
        return self.emergency_system.get_emergency_history(limit)
    
    def add_space_debris(self, debris_info: Dict[str, Any]):
        """添加空间碎片"""
        self.orbit_manager.add_space_debris(debris_info)
    
    def get_space_debris_count(self) -> int:
        """获取空间碎片数量"""
        return self.orbit_manager.get_space_debris_count()
    
    def process_natural_language_command(self, command: str) -> str:
        """处理自然语言指令"""
        logger.info(f"处理自然语言指令: {command}")
        return self.cooperative_scheduler.process_command(command)
    
    def parse_intent(self, command: str) -> IntentTask:
        """解析意图，生成任务"""
        return self.intent_engine.parse_command(command)
    
    def interpret_disaster_scene(self, remote_data: RemoteSensingData, 
                               analysis_params: Dict[str, Any]) -> InterpretationResult:
        """解译灾害场景"""
        return self.orbital_interpreter.interpret_disaster(remote_data, analysis_params)
    
    def get_feature_description(self, feature_name: str, feature_value: float) -> str:
        """获取特征的自然语言描述"""
        return self.orbital_interpreter.knowledge_base.describe_feature(feature_name, feature_value)

def demo_system():
    """演示系统功能"""
    print("=== 多卫星系统演示 ===")
    
    # 创建系统
    system = MultiSatelliteSystem()
    
    print("\n1. 系统初始化状态:")
    status = system.get_system_status()
    print(f"- 总卫星数: {status['total_satellites']}")
    print(f"- 在线卫星数: {status['online_satellites']}")
    
    print("\n2. 触发紧急情况...")
    emergency_id = system.trigger_emergency(
        location=[39.9, 116.4, 0],  # 北京
        emergency_level=EmergencyLevel.HIGH,
        description="地震灾害"
    )
    print(f"- 紧急信标ID: {emergency_id}")
    
    print("\n3. 等待应急响应...")
    time.sleep(2)
    
    print("\n4. 系统状态更新:")
    updated_status = system.get_system_status()
    print(f"- 应急队列大小: {updated_status['emergency_system']['emergency_queue_size']}")
    print(f"- 响应历史数量: {updated_status['emergency_system']['response_history_count']}")
    
    print("\n5. 联邦学习状态:")
    fed_status = updated_status['training_coordinator']['federated_learning']
    print(f"- 聚合轮次: {fed_status['aggregation_round']}")
    print(f"- 本地模型数: {fed_status['local_models_count']}")
    
    print("\n6. 轨道控制状态:")
    orbit_status = updated_status['orbit_manager']
    for sat_id, sat_status in orbit_status.items():
        print(f"- {sat_id}: 燃料 {sat_status['fuel_level']:.1f}%")
    
    print("\n7. 认知无线电状态:")
    radio_status = updated_status['cognitive_radio']
    print(f"- 可用频段数: {len(radio_status['available_bands'])}")
    print(f"- 卫星频谱使用: {len(radio_status['satellite_usage'])}")
    
    print("\n8. 测试意图理解功能...")
    command = "监测山洪风险区"
    intent_task = system.parse_intent(command)
    print(f"- 原始指令: {intent_task.original_command}")
    print(f"- 任务类型: {intent_task.task_type.value}")
    print(f"- 灾害类型: {intent_task.disaster_type.value}")
    print(f"- 目标位置: {intent_task.target_location}")
    print(f"- 优先级: {intent_task.priority}")
    
    print("\n9. 测试协同任务调度...")
    result = system.process_natural_language_command("紧急监测滑坡风险区域")
    print(f"- 任务结果: {result[:200]}...")
    
    print("\n10. 测试特征描述...")
    feature_desc = system.get_feature_description("ndvi", 0.25)
    print(f"- NDVI特征描述: {feature_desc}")
    
    print("\n演示完成！")

if __name__ == "__main__":
    demo_system() 