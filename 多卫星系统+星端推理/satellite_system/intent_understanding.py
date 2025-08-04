#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图理解模块
通过自然语言处理解析地面指令，生成可执行任务清单
"""

import re
import time
import uuid
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .satellite_core import (
    TaskType, DisasterType, EmergencyLevel, IntentTask,
    calculate_distance, logger
)

# 配置日志
logger = logging.getLogger(__name__)

# ==================== 提示模板 ====================

class PromptTemplates:
    """专用提示模板类"""
    
    # 遥感特征转自然语言描述模板
    FEATURE_DESCRIPTION_TEMPLATES = {
        "ndvi": {
            "low": "NDVI<0.3且连续面积>10km² → 植被退化区",
            "medium": "NDVI在0.3-0.5之间 → 植被稀疏区", 
            "high": "NDVI>0.5 → 植被茂密区"
        },
        "moisture": {
            "low": "土壤湿度<20% → 干旱区域",
            "medium": "土壤湿度20-40% → 正常区域",
            "high": "土壤湿度>40% → 湿润区域"
        },
        "elevation": {
            "low": "海拔<500m → 平原区域",
            "medium": "海拔500-2000m → 丘陵区域", 
            "high": "海拔>2000m → 山地区域"
        },
        "slope": {
            "low": "坡度<15° → 平缓区域",
            "medium": "坡度15-30° → 中等坡度区域",
            "high": "坡度>30° → 陡峭区域"
        }
    }
    
    # 任务解析模板
    TASK_PARSING_TEMPLATES = {
        "monitoring": "监测{location}的{disaster_type}风险",
        "imaging": "对{location}进行{data_type}成像",
        "analysis": "分析{location}的{disaster_type}情况",
        "emergency": "紧急处理{location}的{disaster_type}事件"
    }
    
    # 灾害类型关键词映射
    DISASTER_KEYWORDS = {
        "滑坡": DisasterType.LANDSLIDE,
        "山体滑坡": DisasterType.LANDSLIDE,
        "泥石流": DisasterType.LANDSLIDE,
        "洪水": DisasterType.FLOOD,
        "山洪": DisasterType.FLOOD,
        "暴雨": DisasterType.FLOOD,
        "火灾": DisasterType.FIRE,
        "森林火灾": DisasterType.FIRE,
        "地震": DisasterType.EARTHQUAKE,
        "海啸": DisasterType.TSUNAMI,
        "火山": DisasterType.VOLCANO,
        "火山喷发": DisasterType.VOLCANO
    }
    
    # 任务类型关键词映射
    TASK_KEYWORDS = {
        "监测": TaskType.MONITORING,
        "监控": TaskType.MONITORING,
        "观察": TaskType.MONITORING,
        "拍摄": TaskType.IMAGING,
        "成像": TaskType.IMAGING,
        "分析": TaskType.ANALYSIS,
        "评估": TaskType.ANALYSIS,
        "紧急": TaskType.EMERGENCY,
        "应急": TaskType.EMERGENCY,
        "例行": TaskType.ROUTINE,
        "常规": TaskType.ROUTINE
    }

# ==================== 意图理解器 ====================

class IntentUnderstandingEngine:
    """意图理解引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt_templates = PromptTemplates()
        self.location_extractor = LocationExtractor()
        self.task_generator = TaskGenerator(config)
        
    def parse_command(self, command: str) -> IntentTask:
        """解析自然语言指令，生成任务"""
        logger.info(f"开始解析指令: {command}")
        
        # 1. 提取任务类型
        task_type = self._extract_task_type(command)
        
        # 2. 提取灾害类型
        disaster_type = self._extract_disaster_type(command)
        
        # 3. 提取目标位置
        target_location = self.location_extractor.extract_location(command)
        
        # 4. 评估优先级
        priority = self._evaluate_priority(command, task_type, disaster_type)
        
        # 5. 生成任务
        task = self.task_generator.generate_task(
            command, task_type, disaster_type, target_location, priority
        )
        
        logger.info(f"任务生成完成: {task.task_id}")
        return task
    
    def _extract_task_type(self, command: str) -> TaskType:
        """提取任务类型"""
        command_lower = command.lower()
        
        for keyword, task_type in self.prompt_templates.TASK_KEYWORDS.items():
            if keyword in command_lower:
                return task_type
        
        # 默认返回监测任务
        return TaskType.MONITORING
    
    def _extract_disaster_type(self, command: str) -> DisasterType:
        """提取灾害类型"""
        command_lower = command.lower()
        
        for keyword, disaster_type in self.prompt_templates.DISASTER_KEYWORDS.items():
            if keyword in command_lower:
                return disaster_type
        
        return DisasterType.UNKNOWN
    
    def _evaluate_priority(self, command: str, task_type: TaskType, 
                          disaster_type: DisasterType) -> int:
        """评估任务优先级"""
        priority = 5  # 默认优先级
        
        # 紧急任务优先级最高
        if task_type == TaskType.EMERGENCY:
            priority = 10
        elif "紧急" in command or "应急" in command:
            priority = 9
        elif "风险" in command:
            priority = 7
        elif "监测" in command:
            priority = 6
        
        # 根据灾害类型调整优先级
        if disaster_type in [DisasterType.EARTHQUAKE, DisasterType.TSUNAMI]:
            priority = min(priority + 2, 10)
        elif disaster_type in [DisasterType.FLOOD, DisasterType.LANDSLIDE]:
            priority = min(priority + 1, 10)
        
        return priority

# ==================== 位置提取器 ====================

class LocationExtractor:
    """位置信息提取器"""
    
    def __init__(self):
        # 常见地名关键词
        self.location_keywords = [
            "区", "市", "省", "县", "镇", "村", "山", "河", "湖", "海",
            "平原", "丘陵", "山地", "盆地", "高原", "峡谷", "海岸"
        ]
        
        # 坐标模式
        self.coordinate_patterns = [
            r'(\d+\.?\d*)[°度]\s*(\d+\.?\d*)[\'分]\s*(\d+\.?\d*)[\"秒]\s*[NSns]\s*(\d+\.?\d*)[°度]\s*(\d+\.?\d*)[\'分]\s*(\d+\.?\d*)[\"秒]\s*[EWew]',
            r'(\d+\.?\d*)[°度]\s*(\d+\.?\d*)[\'分]\s*[NSns]\s*(\d+\.?\d*)[°度]\s*(\d+\.?\d*)[\'分]\s*[EWew]',
            r'(\d+\.?\d*)[°度]\s*[NSns]\s*(\d+\.?\d*)[°度]\s*[EWew]'
        ]
    
    def extract_location(self, command: str) -> List[float]:
        """从指令中提取位置信息"""
        # 1. 尝试提取坐标
        coordinates = self._extract_coordinates(command)
        if coordinates:
            return coordinates
        
        # 2. 尝试从地名提取（这里简化处理，实际应该连接地理数据库）
        location_name = self._extract_location_name(command)
        if location_name:
            # 这里应该查询地理数据库获取坐标
            # 暂时返回默认坐标
            return [39.9042, 116.4074, 0.0]  # 北京坐标作为示例
        
        # 3. 返回默认坐标
        return [0.0, 0.0, 0.0]
    
    def _extract_coordinates(self, command: str) -> Optional[List[float]]:
        """提取坐标信息"""
        for pattern in self.coordinate_patterns:
            match = re.search(pattern, command)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 6:  # 度分秒格式
                        lat = self._dms_to_decimal(float(groups[0]), float(groups[1]), float(groups[2]), groups[3])
                        lon = self._dms_to_decimal(float(groups[4]), float(groups[5]), float(groups[6]), groups[7])
                    elif len(groups) == 4:  # 度分格式
                        lat = self._dm_to_decimal(float(groups[0]), float(groups[1]), groups[2])
                        lon = self._dm_to_decimal(float(groups[3]), float(groups[4]), groups[5])
                    else:  # 度格式
                        lat = float(groups[0]) if groups[1].upper() == 'N' else -float(groups[0])
                        lon = float(groups[2]) if groups[3].upper() == 'E' else -float(groups[2])
                    
                    return [lat, lon, 0.0]
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_location_name(self, command: str) -> Optional[str]:
        """提取地名"""
        for keyword in self.location_keywords:
            # 简单的关键词匹配，实际应该使用更复杂的NLP技术
            if keyword in command:
                # 提取包含关键词的短语
                words = command.split()
                for i, word in enumerate(words):
                    if keyword in word:
                        # 返回包含关键词的短语
                        start = max(0, i-2)
                        end = min(len(words), i+3)
                        return " ".join(words[start:end])
        
        return None
    
    def _dms_to_decimal(self, degrees: float, minutes: float, seconds: float, direction: str) -> float:
        """度分秒转十进制度"""
        decimal = degrees + minutes/60 + seconds/3600
        return decimal if direction.upper() in ['N', 'E'] else -decimal
    
    def _dm_to_decimal(self, degrees: float, minutes: float, direction: str) -> float:
        """度分转十进制度"""
        decimal = degrees + minutes/60
        return decimal if direction.upper() in ['N', 'E'] else -decimal

# ==================== 任务生成器 ====================

class TaskGenerator:
    """任务生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def generate_task(self, command: str, task_type: TaskType, 
                     disaster_type: DisasterType, target_location: List[float],
                     priority: int) -> IntentTask:
        """生成具体任务"""
        
        # 生成任务ID
        task_id = f"intent_{uuid.uuid4().hex[:8]}"
        
        # 估算任务时长
        estimated_duration = self._estimate_duration(task_type, disaster_type)
        
        # 确定需要的卫星
        required_satellites = self._determine_required_satellites(task_type, disaster_type)
        
        # 设置成像参数
        imaging_parameters = self._get_imaging_parameters(task_type, disaster_type)
        
        # 设置分析参数
        analysis_parameters = self._get_analysis_parameters(task_type, disaster_type)
        
        return IntentTask(
            task_id=task_id,
            original_command=command,
            task_type=task_type,
            disaster_type=disaster_type,
            target_location=target_location,
            priority=priority,
            timestamp=time.time(),
            estimated_duration=estimated_duration,
            required_satellites=required_satellites,
            imaging_parameters=imaging_parameters,
            analysis_parameters=analysis_parameters
        )
    
    def _estimate_duration(self, task_type: TaskType, disaster_type: DisasterType) -> float:
        """估算任务完成时间（分钟）"""
        base_duration = {
            TaskType.MONITORING: 30.0,
            TaskType.IMAGING: 15.0,
            TaskType.ANALYSIS: 45.0,
            TaskType.EMERGENCY: 8.0,  # 紧急任务8分钟内完成
            TaskType.ROUTINE: 60.0
        }
        
        duration = base_duration.get(task_type, 30.0)
        
        # 根据灾害类型调整
        if disaster_type in [DisasterType.EARTHQUAKE, DisasterType.TSUNAMI]:
            duration *= 0.5  # 紧急灾害任务加快
        elif disaster_type == DisasterType.UNKNOWN:
            duration *= 1.2  # 未知类型需要更多时间分析
        
        return duration
    
    def _determine_required_satellites(self, task_type: TaskType, 
                                     disaster_type: DisasterType) -> List[str]:
        """确定需要的卫星类型"""
        satellites = []
        
        if task_type == TaskType.EMERGENCY:
            # 紧急任务需要多种卫星协同
            satellites = ["optical_sat", "sar_sat", "multispectral_sat"]
        elif disaster_type == DisasterType.LANDSLIDE:
            # 滑坡监测需要SAR和光学卫星
            satellites = ["sar_sat", "optical_sat"]
        elif disaster_type == DisasterType.FLOOD:
            # 洪水监测需要SAR卫星
            satellites = ["sar_sat"]
        elif disaster_type == DisasterType.FIRE:
            # 火灾监测需要红外和光学卫星
            satellites = ["thermal_sat", "optical_sat"]
        else:
            # 默认使用光学卫星
            satellites = ["optical_sat"]
        
        return satellites
    
    def _get_imaging_parameters(self, task_type: TaskType, 
                               disaster_type: DisasterType) -> Dict[str, Any]:
        """获取成像参数"""
        params = {
            "resolution": "high",  # 高分辨率
            "coverage_area": 100.0,  # 覆盖面积(km²)
            "imaging_mode": "standard"
        }
        
        if task_type == TaskType.EMERGENCY:
            params.update({
                "resolution": "ultra_high",
                "imaging_mode": "emergency",
                "coverage_area": 50.0  # 紧急任务聚焦小区域
            })
        elif disaster_type == DisasterType.LANDSLIDE:
            params.update({
                "imaging_mode": "interferometric",
                "polarization": "dual"
            })
        elif disaster_type == DisasterType.FLOOD:
            params.update({
                "imaging_mode": "flood_mapping",
                "coverage_area": 200.0  # 洪水需要大范围覆盖
            })
        
        return params
    
    def _get_analysis_parameters(self, task_type: TaskType, 
                                disaster_type: DisasterType) -> Dict[str, Any]:
        """获取分析参数"""
        params = {
            "analysis_type": "standard",
            "confidence_threshold": 0.7,
            "output_format": "slice"
        }
        
        if task_type == TaskType.EMERGENCY:
            params.update({
                "analysis_type": "emergency",
                "confidence_threshold": 0.6,  # 紧急任务降低置信度要求
                "output_format": "slice_and_report"
            })
        elif disaster_type == DisasterType.LANDSLIDE:
            params.update({
                "analysis_type": "landslide_detection",
                "feature_extraction": ["ndvi", "slope", "moisture"]
            })
        elif disaster_type == DisasterType.FLOOD:
            params.update({
                "analysis_type": "flood_mapping",
                "feature_extraction": ["water_index", "elevation"]
            })
        
        return params 