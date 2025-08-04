#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协同任务调度模块
实现卫星群协同完成SAR成像→星端CNN滑坡概率计算→仅回传结果切片
"""

import time
import uuid
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .satellite_core import (
    SatelliteInfo, IntentTask, RemoteSensingData, InterpretationResult,
    TaskType, DisasterType, EmergencyLevel, calculate_distance, logger
)
from .intent_understanding import IntentUnderstandingEngine
from .orbital_interpretation import OrbitalInterpreter

# 配置日志
logger = logging.getLogger(__name__)

# ==================== 协同任务调度器 ====================

class CooperativeScheduler:
    """协同任务调度器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intent_engine = IntentUnderstandingEngine(config)
        self.orbital_interpreter = OrbitalInterpreter(
            config.get("model_path", "models/best_multimodal_patch_model.pth"),
            config
        )
        self.satellites = {}  # 卫星信息字典
        self.active_tasks = {}  # 活跃任务字典
        self.task_results = {}  # 任务结果字典
        
    def add_satellite(self, satellite: SatelliteInfo):
        """添加卫星到调度系统"""
        self.satellites[satellite.satellite_id] = satellite
        logger.info(f"添加卫星: {satellite.satellite_id}")
    
    def remove_satellite(self, satellite_id: str):
        """从调度系统移除卫星"""
        if satellite_id in self.satellites:
            del self.satellites[satellite_id]
            logger.info(f"移除卫星: {satellite_id}")
    
    def process_command(self, command: str) -> str:
        """处理地面指令"""
        logger.info(f"收到地面指令: {command}")
        
        try:
            # 1. 意图理解
            intent_task = self.intent_engine.parse_command(command)
            logger.info(f"生成意图任务: {intent_task.task_id}")
            
            # 2. 协同任务调度
            result = self._schedule_cooperative_task(intent_task)
            
            # 3. 返回结果
            return result
            
        except Exception as e:
            logger.error(f"处理指令失败: {e}")
            return f"处理指令失败: {str(e)}"
    
    def _schedule_cooperative_task(self, intent_task: IntentTask) -> str:
        """调度协同任务"""
        logger.info(f"开始调度协同任务: {intent_task.task_id}")
        
        start_time = time.time()
        
        # 1. 选择最近的遥感卫星进行拍摄
        imaging_satellite = self._select_imaging_satellite(intent_task)
        if not imaging_satellite:
            return "错误: 没有可用的成像卫星"
        
        # 2. 执行SAR成像
        remote_data = self._perform_sar_imaging(imaging_satellite, intent_task)
        if not remote_data:
            return "错误: SAR成像失败"
        
        # 3. 选择计算卫星进行推理
        computing_satellite = self._select_computing_satellite(intent_task)
        if not computing_satellite:
            return "错误: 没有可用的计算卫星"
        
        # 4. 星端CNN滑坡概率计算
        interpretation_result = self._perform_orbital_interpretation(
            computing_satellite, remote_data, intent_task
        )
        if not interpretation_result:
            return "错误: 星端推理失败"
        
        # 5. 生成救援路线
        rescue_route = self._generate_rescue_route(interpretation_result)
        
        # 6. 回传结果到地面中心
        self._transmit_to_ground(interpretation_result, rescue_route)
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"协同任务完成，总耗时: {total_time:.2f}秒")
        
        # 生成结果报告
        result_report = self._generate_result_report(
            intent_task, remote_data, interpretation_result, 
            rescue_route, total_time
        )
        
        return result_report
    
    def _select_imaging_satellite(self, intent_task: IntentTask) -> Optional[SatelliteInfo]:
        """选择最近的遥感卫星进行拍摄"""
        available_satellites = []
        
        for sat_id, satellite in self.satellites.items():
            # 检查卫星状态
            if satellite.status.value != "online":
                continue
            
            # 检查卫星类型
            if not self._is_imaging_satellite(satellite, intent_task):
                continue
            
            # 检查覆盖范围
            if not self._is_in_coverage(satellite, intent_task.target_location):
                continue
            
            # 计算距离
            distance = calculate_distance(satellite.current_position, intent_task.target_location)
            available_satellites.append((satellite, distance))
        
        if not available_satellites:
            return None
        
        # 选择最近的卫星
        available_satellites.sort(key=lambda x: x[1])
        selected_satellite = available_satellites[0][0]
        
        logger.info(f"选择成像卫星: {selected_satellite.satellite_id}")
        return selected_satellite
    
    def _select_computing_satellite(self, intent_task: IntentTask) -> Optional[SatelliteInfo]:
        """选择计算卫星进行推理"""
        available_satellites = []
        
        for sat_id, satellite in self.satellites.items():
            # 检查卫星状态
            if satellite.status.value != "online":
                continue
            
            # 检查计算能力
            if satellite.compute_capacity < 1e12:  # 1 TFLOPS
                continue
            
            # 检查负载
            if satellite.current_load > 0.8:
                continue
            
            # 计算距离（选择负载较轻的卫星）
            load_factor = satellite.current_load
            available_satellites.append((satellite, load_factor))
        
        if not available_satellites:
            return None
        
        # 选择负载最轻的卫星
        available_satellites.sort(key=lambda x: x[1])
        selected_satellite = available_satellites[0][0]
        
        logger.info(f"选择计算卫星: {selected_satellite.satellite_id}")
        return selected_satellite
    
    def _is_imaging_satellite(self, satellite: SatelliteInfo, intent_task: IntentTask) -> bool:
        """检查是否为成像卫星"""
        # 检查卫星支持的功能
        if "sar_imaging" in satellite.supported_features:
            return True
        
        # 检查卫星ID是否包含成像相关关键词
        imaging_keywords = ["sar", "optical", "imaging", "camera"]
        for keyword in imaging_keywords:
            if keyword in satellite.satellite_id.lower():
                return True
        
        return False
    
    def _is_in_coverage(self, satellite: SatelliteInfo, target_location: List[float]) -> bool:
        """检查目标是否在卫星覆盖范围内"""
        if not satellite.coverage_area:
            # 如果没有覆盖区域信息，假设可以覆盖
            return True
        
        # 简化的覆盖检查
        distance = calculate_distance(satellite.current_position, target_location)
        # 假设覆盖半径为1000km
        coverage_radius = 1000.0
        
        return distance <= coverage_radius
    
    def _perform_sar_imaging(self, satellite: SatelliteInfo, 
                           intent_task: IntentTask) -> Optional[RemoteSensingData]:
        """执行SAR成像"""
        logger.info(f"开始SAR成像: {satellite.satellite_id}")
        
        try:
            # 模拟SAR成像过程
            imaging_params = intent_task.imaging_parameters
            
            # 生成模拟图像数据
            if imaging_params.get("imaging_mode") == "interferometric":
                # 干涉SAR模式
                image_data = self._generate_interferometric_data()
            elif imaging_params.get("imaging_mode") == "flood_mapping":
                # 洪水测绘模式
                image_data = self._generate_flood_mapping_data()
            else:
                # 标准SAR模式
                image_data = self._generate_standard_sar_data()
            
            # 生成信道数据
            channel_data = self._generate_channel_data()
            
            # 计算数据质量评分
            quality_score = self._calculate_data_quality(image_data, channel_data)
            
            # 创建遥感数据对象
            remote_data = RemoteSensingData(
                data_id=f"rs_{uuid.uuid4().hex[:8]}",
                satellite_id=satellite.satellite_id,
                location=intent_task.target_location,
                timestamp=time.time(),
                data_type="sar",
                image_data=image_data,
                metadata={
                    "imaging_mode": imaging_params.get("imaging_mode", "standard"),
                    "resolution": imaging_params.get("resolution", "high"),
                    "coverage_area": imaging_params.get("coverage_area", 100.0),
                    "polarization": imaging_params.get("polarization", "single")
                },
                channel_data=channel_data,
                quality_score=quality_score
            )
            
            logger.info(f"SAR成像完成: {remote_data.data_id}")
            return remote_data
            
        except Exception as e:
            logger.error(f"SAR成像失败: {e}")
            return None
    
    def _perform_orbital_interpretation(self, satellite: SatelliteInfo,
                                      remote_data: RemoteSensingData,
                                      intent_task: IntentTask) -> Optional[InterpretationResult]:
        """执行星端推理"""
        logger.info(f"开始星端推理: {satellite.satellite_id}")
        
        try:
            # 使用在轨智能解译器
            interpretation_result = self.orbital_interpreter.interpret_disaster(
                remote_data, intent_task.analysis_parameters
            )
            
            logger.info(f"星端推理完成: {interpretation_result.result_id}")
            return interpretation_result
            
        except Exception as e:
            logger.error(f"星端推理失败: {e}")
            return None
    
    def _generate_rescue_route(self, interpretation_result: InterpretationResult) -> Dict[str, Any]:
        """生成救援路线"""
        logger.info("生成救援路线")
        
        # 基于灾害类型和位置生成救援路线
        if interpretation_result.disaster_type == DisasterType.LANDSLIDE:
            route = self._generate_landslide_rescue_route(interpretation_result)
        elif interpretation_result.disaster_type == DisasterType.FLOOD:
            route = self._generate_flood_rescue_route(interpretation_result)
        else:
            route = self._generate_general_rescue_route(interpretation_result)
        
        return route
    
    def _transmit_to_ground(self, interpretation_result: InterpretationResult, 
                           rescue_route: Dict[str, Any]):
        """回传结果到地面中心"""
        logger.info("回传结果到地面中心")
        
        # 模拟数据传输
        transmission_data = {
            "result_id": interpretation_result.result_id,
            "disaster_type": interpretation_result.disaster_type.value,
            "disaster_probability": interpretation_result.disaster_probability,
            "risk_level": interpretation_result.risk_level.name,
            "affected_area": interpretation_result.affected_area,
            "confidence_score": interpretation_result.confidence_score,
            "result_slice": interpretation_result.result_slice.tolist(),
            "analysis_report": interpretation_result.analysis_report,
            "rescue_route": rescue_route,
            "timestamp": time.time()
        }
        
        # 这里应该实现实际的数据传输逻辑
        logger.info(f"数据传输完成: {len(str(transmission_data))} 字节")
    
    def _generate_result_report(self, intent_task: IntentTask,
                              remote_data: RemoteSensingData,
                              interpretation_result: InterpretationResult,
                              rescue_route: Dict[str, Any],
                              total_time: float) -> str:
        """生成结果报告"""
        report = f"""
协同任务执行报告
================

原始指令: {intent_task.original_command}
任务ID: {intent_task.task_id}
执行时间: {total_time:.2f}秒

成像阶段:
- 成像卫星: {remote_data.satellite_id}
- 数据质量: {remote_data.quality_score:.2%}
- 成像模式: {remote_data.metadata.get('imaging_mode', 'unknown')}

推理阶段:
- 计算卫星: {interpretation_result.satellite_id}
- 灾害类型: {interpretation_result.disaster_type.value}
- 灾害概率: {interpretation_result.disaster_probability:.2%}
- 置信度: {interpretation_result.confidence_score:.2%}
- 风险等级: {interpretation_result.risk_level.name}
- 受影响面积: {interpretation_result.affected_area:.1f} km²

救援路线:
- 路线类型: {rescue_route.get('route_type', 'unknown')}
- 预计时间: {rescue_route.get('estimated_time', 'unknown')}
- 优先级: {rescue_route.get('priority', 'unknown')}

分析结论:
{interpretation_result.analysis_report}

任务状态: 完成 ✅
"""
        
        return report.strip()
    
    # ==================== 数据生成辅助方法 ====================
    
    def _generate_standard_sar_data(self) -> np.ndarray:
        """生成标准SAR数据"""
        # 模拟SAR图像数据 (多波段)
        data = np.random.rand(4, 256, 256).astype(np.float32)
        # 添加一些结构特征
        data[0] += np.sin(np.linspace(0, 4*np.pi, 256))[:, None]
        data[1] += np.cos(np.linspace(0, 4*np.pi, 256))[None, :]
        return data
    
    def _generate_interferometric_data(self) -> np.ndarray:
        """生成干涉SAR数据"""
        # 模拟干涉SAR数据
        data = np.random.rand(6, 256, 256).astype(np.float32)
        # 添加干涉条纹
        x, y = np.meshgrid(np.linspace(0, 2*np.pi, 256), np.linspace(0, 2*np.pi, 256))
        data[0] += np.sin(x + y) * 0.3
        data[1] += np.cos(x - y) * 0.3
        return data
    
    def _generate_flood_mapping_data(self) -> np.ndarray:
        """生成洪水测绘数据"""
        # 模拟洪水测绘数据
        data = np.random.rand(5, 256, 256).astype(np.float32)
        # 添加水体特征
        data[1] += np.random.rand(256, 256) * 0.5  # 水体指数
        return data
    
    def _generate_channel_data(self) -> Dict[str, np.ndarray]:
        """生成信道数据"""
        return {
            "signal_strength": np.random.rand(256, 256).astype(np.float32),
            "noise_level": np.random.rand(256, 256).astype(np.float32) * 0.1,
            "interference": np.random.rand(256, 256).astype(np.float32) * 0.05
        }
    
    def _calculate_data_quality(self, image_data: np.ndarray, 
                               channel_data: Dict[str, np.ndarray]) -> float:
        """计算数据质量评分"""
        # 基于信噪比计算质量
        signal = np.mean(channel_data["signal_strength"])
        noise = np.mean(channel_data["noise_level"])
        
        if noise == 0:
            snr = 100.0
        else:
            snr = signal / noise
        
        # 归一化到0-1
        quality = min(snr / 100.0, 1.0)
        return quality
    
    def _generate_landslide_rescue_route(self, result: InterpretationResult) -> Dict[str, Any]:
        """生成滑坡救援路线"""
        return {
            "route_type": "landslide_evacuation",
            "estimated_time": "2-4小时",
            "priority": "high",
            "safe_zones": [
                {"lat": result.key_features.get("elevation", {}).get("mean", 0) + 0.1, 
                 "lon": 0.0, "alt": 100.0}
            ],
            "evacuation_paths": [
                {"start": result.key_features.get("elevation", {}).get("mean", 0),
                 "end": result.key_features.get("elevation", {}).get("mean", 0) + 0.1,
                 "distance": 5.0}
            ]
        }
    
    def _generate_flood_rescue_route(self, result: InterpretationResult) -> Dict[str, Any]:
        """生成洪水救援路线"""
        return {
            "route_type": "flood_evacuation",
            "estimated_time": "1-2小时",
            "priority": "critical",
            "safe_zones": [
                {"lat": result.key_features.get("elevation", {}).get("mean", 0) + 0.2, 
                 "lon": 0.0, "alt": 50.0}
            ],
            "evacuation_paths": [
                {"start": result.key_features.get("elevation", {}).get("mean", 0),
                 "end": result.key_features.get("elevation", {}).get("mean", 0) + 0.2,
                 "distance": 3.0}
            ]
        }
    
    def _generate_general_rescue_route(self, result: InterpretationResult) -> Dict[str, Any]:
        """生成通用救援路线"""
        return {
            "route_type": "general_evacuation",
            "estimated_time": "3-6小时",
            "priority": "medium",
            "safe_zones": [
                {"lat": 0.0, "lon": 0.0, "alt": 100.0}
            ],
            "evacuation_paths": [
                {"start": 0.0, "end": 0.1, "distance": 10.0}
            ]
        } 