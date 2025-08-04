#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在轨智能解译模块
结合先验知识提取图像特征，实时解译灾害场景
"""

import time
import uuid
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .satellite_core import (
    DisasterType, EmergencyLevel, RemoteSensingData, InterpretationResult,
    calculate_distance, logger
)

# 配置日志
logger = logging.getLogger(__name__)

# ==================== 特征提取器 ====================

class FeatureExtractor:
    """遥感特征提取器"""
    
    def __init__(self):
        self.feature_extractors = {
            "ndvi": self._extract_ndvi,
            "slope": self._extract_slope,
            "moisture": self._extract_moisture,
            "water_index": self._extract_water_index,
            "elevation": self._extract_elevation,
            "texture": self._extract_texture
        }
    
    def extract_features(self, image_data: np.ndarray, 
                        feature_types: List[str]) -> Dict[str, np.ndarray]:
        """提取指定类型的特征"""
        features = {}
        
        for feature_type in feature_types:
            if feature_type in self.feature_extractors:
                try:
                    features[feature_type] = self.feature_extractors[feature_type](image_data)
                except Exception as e:
                    logger.error(f"提取特征 {feature_type} 失败: {e}")
                    features[feature_type] = np.zeros_like(image_data[0])
        
        return features
    
    def _extract_ndvi(self, image_data: np.ndarray) -> np.ndarray:
        """提取NDVI植被指数"""
        # 假设图像数据包含近红外和红光波段
        if image_data.shape[0] >= 4:  # 多光谱图像
            nir = image_data[3]  # 近红外波段
            red = image_data[2]  # 红光波段
        else:  # 模拟NDVI计算
            nir = image_data[0] * 0.8 + image_data[1] * 0.2
            red = image_data[0] * 0.6 + image_data[1] * 0.4
        
        # 避免除零
        denominator = nir + red
        denominator[denominator == 0] = 1e-6
        
        ndvi = (nir - red) / denominator
        return np.clip(ndvi, -1, 1)
    
    def _extract_slope(self, image_data: np.ndarray) -> np.ndarray:
        """提取坡度信息（模拟）"""
        # 使用图像梯度模拟坡度
        if len(image_data.shape) == 3:
            gray = np.mean(image_data, axis=0)
        else:
            gray = image_data[0]
        
        # 计算梯度
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        
        # 计算坡度
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        return slope
    
    def _extract_moisture(self, image_data: np.ndarray) -> np.ndarray:
        """提取土壤湿度信息（模拟）"""
        # 使用短波红外波段模拟土壤湿度
        if image_data.shape[0] >= 5:
            swir = image_data[4]
        else:
            swir = image_data[0] * 0.7 + image_data[1] * 0.3
        
        # 归一化到0-1范围
        moisture = (swir - swir.min()) / (swir.max() - swir.min() + 1e-6)
        return moisture
    
    def _extract_water_index(self, image_data: np.ndarray) -> np.ndarray:
        """提取水体指数"""
        # 使用绿光和近红外波段
        if image_data.shape[0] >= 4:
            green = image_data[1]
            nir = image_data[3]
        else:
            green = image_data[0] * 0.5 + image_data[1] * 0.5
            nir = image_data[0] * 0.8 + image_data[1] * 0.2
        
        # 计算NDWI
        denominator = green + nir
        denominator[denominator == 0] = 1e-6
        
        ndwi = (green - nir) / denominator
        return np.clip(ndwi, -1, 1)
    
    def _extract_elevation(self, image_data: np.ndarray) -> np.ndarray:
        """提取高程信息（模拟）"""
        # 使用图像亮度模拟高程
        if len(image_data.shape) == 3:
            elevation = np.mean(image_data, axis=0)
        else:
            elevation = image_data[0]
        
        # 归一化
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-6)
        return elevation
    
    def _extract_texture(self, image_data: np.ndarray) -> np.ndarray:
        """提取纹理特征"""
        # 使用灰度共生矩阵的简化版本
        if len(image_data.shape) == 3:
            gray = np.mean(image_data, axis=0)
        else:
            gray = image_data[0]
        
        # 计算局部方差作为纹理特征
        from scipy.ndimage import uniform_filter, uniform_filter2d
        mean = uniform_filter2d(gray, size=5)
        mean_sq = uniform_filter2d(gray**2, size=5)
        texture = np.sqrt(mean_sq - mean**2)
        
        return texture

# ==================== 先验知识库 ====================

class PriorKnowledgeBase:
    """先验知识库"""
    
    def __init__(self):
        self.disaster_patterns = {
            DisasterType.LANDSLIDE: {
                "ndvi_threshold": 0.3,
                "slope_threshold": 0.5,  # 约30度
                "moisture_threshold": 0.6,
                "area_threshold": 10.0,  # km²
                "risk_factors": ["high_slope", "low_ndvi", "high_moisture"]
            },
            DisasterType.FLOOD: {
                "water_index_threshold": 0.3,
                "elevation_threshold": 0.3,
                "area_threshold": 5.0,  # km²
                "risk_factors": ["high_water_index", "low_elevation"]
            },
            DisasterType.FIRE: {
                "ndvi_threshold": 0.2,
                "thermal_threshold": 0.7,
                "area_threshold": 2.0,  # km²
                "risk_factors": ["low_ndvi", "high_thermal"]
            }
        }
        
        self.feature_descriptions = {
            "ndvi": {
                "low": "NDVI<0.3且连续面积>10km² → 植被退化区",
                "medium": "NDVI在0.3-0.5之间 → 植被稀疏区",
                "high": "NDVI>0.5 → 植被茂密区"
            },
            "slope": {
                "low": "坡度<15° → 平缓区域",
                "medium": "坡度15-30° → 中等坡度区域",
                "high": "坡度>30° → 陡峭区域"
            },
            "moisture": {
                "low": "土壤湿度<20% → 干旱区域",
                "medium": "土壤湿度20-40% → 正常区域",
                "high": "土壤湿度>40% → 湿润区域"
            }
        }
    
    def get_disaster_pattern(self, disaster_type: DisasterType) -> Dict[str, Any]:
        """获取灾害模式"""
        return self.disaster_patterns.get(disaster_type, {})
    
    def describe_feature(self, feature_name: str, feature_value: float) -> str:
        """将特征值转换为自然语言描述"""
        if feature_name in self.feature_descriptions:
            if feature_value < 0.3:
                return self.feature_descriptions[feature_name]["low"]
            elif feature_value < 0.7:
                return self.feature_descriptions[feature_name]["medium"]
            else:
                return self.feature_descriptions[feature_name]["high"]
        
        return f"{feature_name}: {feature_value:.2f}"

# ==================== 智能解译器 ====================

class OrbitalInterpreter:
    """在轨智能解译器"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model_path = model_path
        self.feature_extractor = FeatureExtractor()
        self.knowledge_base = PriorKnowledgeBase()
        self.model = self._load_model()
        
    def _load_model(self):
        """加载AI模型"""
        try:
            from train_model import EnhancedDeepLab
            model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
            
            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 处理模型权重键名
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state_dict = {}
            
            for key, value in state_dict.items():
                # 移除前缀
                if key.startswith('deeplab_model.'):
                    new_key = key.replace('deeplab_model.', '')
                elif key.startswith('landslide_model.'):
                    new_key = key.replace('landslide_model.', '')
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict)
            model.eval()
            logger.info("AI模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"AI模型加载失败: {e}")
            return None
    
    def interpret_disaster(self, remote_data: RemoteSensingData, 
                          analysis_params: Dict[str, Any]) -> InterpretationResult:
        """解译灾害场景"""
        logger.info(f"开始解译灾害场景: {remote_data.data_id}")
        
        start_time = time.time()
        
        # 1. 特征提取
        feature_types = analysis_params.get("feature_extraction", ["ndvi", "slope", "moisture"])
        features = self.feature_extractor.extract_features(
            remote_data.image_data, feature_types
        )
        
        # 2. 灾害概率计算
        disaster_probability = self._calculate_disaster_probability(
            features, analysis_params
        )
        
        # 3. 灾害类型识别
        disaster_type = self._identify_disaster_type(features, analysis_params)
        
        # 4. 置信度评估
        confidence_score = self._evaluate_confidence(features, disaster_probability)
        
        # 5. 受影响面积计算
        affected_area = self._calculate_affected_area(features, disaster_probability)
        
        # 6. 风险等级评估
        risk_level = self._evaluate_risk_level(disaster_probability, affected_area)
        
        # 7. 关键特征提取
        key_features = self._extract_key_features(features, disaster_type)
        
        # 8. 生成结果切片
        result_slice = self._generate_result_slice(
            remote_data.image_data, disaster_probability, analysis_params
        )
        
        # 9. 生成分析报告
        analysis_report = self._generate_analysis_report(
            features, disaster_type, disaster_probability, 
            confidence_score, affected_area, risk_level
        )
        
        # 计算处理时间
        processing_time = time.time() - start_time
        logger.info(f"解译完成，耗时: {processing_time:.2f}秒")
        
        return InterpretationResult(
            result_id=f"interpret_{uuid.uuid4().hex[:8]}",
            task_id=remote_data.data_id,
            satellite_id=remote_data.satellite_id,
            timestamp=time.time(),
            disaster_probability=disaster_probability,
            disaster_type=disaster_type,
            confidence_score=confidence_score,
            affected_area=affected_area,
            risk_level=risk_level,
            key_features=key_features,
            result_slice=result_slice,
            analysis_report=analysis_report
        )
    
    def _calculate_disaster_probability(self, features: Dict[str, np.ndarray], 
                                      analysis_params: Dict[str, Any]) -> float:
        """计算灾害概率"""
        if self.model is None:
            # 如果没有模型，使用基于规则的简单计算
            return self._rule_based_probability(features, analysis_params)
        
        try:
            # 使用AI模型计算概率
            return self._model_based_probability(features, analysis_params)
        except Exception as e:
            logger.error(f"AI模型推理失败，使用规则方法: {e}")
            return self._rule_based_probability(features, analysis_params)
    
    def _rule_based_probability(self, features: Dict[str, np.ndarray], 
                               analysis_params: Dict[str, Any]) -> float:
        """基于规则的灾害概率计算"""
        probability = 0.0
        weights = {
            "ndvi": 0.3,
            "slope": 0.3,
            "moisture": 0.2,
            "water_index": 0.2
        }
        
        # 计算各特征的贡献
        for feature_name, feature_data in features.items():
            if feature_name in weights:
                # 计算特征的平均值
                feature_mean = np.mean(feature_data)
                
                # 根据特征类型调整概率
                if feature_name == "ndvi" and feature_mean < 0.3:
                    probability += weights[feature_name] * (0.3 - feature_mean) / 0.3
                elif feature_name == "slope" and feature_mean > 0.5:
                    probability += weights[feature_name] * (feature_mean - 0.5) / 0.5
                elif feature_name == "moisture" and feature_mean > 0.6:
                    probability += weights[feature_name] * (feature_mean - 0.6) / 0.4
                elif feature_name == "water_index" and feature_mean > 0.3:
                    probability += weights[feature_name] * (feature_mean - 0.3) / 0.7
        
        return min(probability, 1.0)
    
    def _model_based_probability(self, features: Dict[str, np.ndarray], 
                                analysis_params: Dict[str, Any]) -> float:
        """基于AI模型的灾害概率计算"""
        # 准备输入数据
        if len(features) > 0:
            # 使用第一个特征作为图像输入（简化处理）
            image_input = list(features.values())[0]
            if len(image_input.shape) == 2:
                image_input = np.stack([image_input] * 3, axis=0)
        else:
            # 默认输入
            image_input = np.random.rand(3, 256, 256).astype(np.float32)
        
        # 准备仿真特征
        sim_features = np.random.rand(11).astype(np.float32)
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_input).unsqueeze(0)
        sim_tensor = torch.from_numpy(sim_features).unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            output = self.model(image_tensor, sim_tensor)
            probability = torch.sigmoid(output).item()
        
        return probability
    
    def _identify_disaster_type(self, features: Dict[str, np.ndarray], 
                               analysis_params: Dict[str, Any]) -> DisasterType:
        """识别灾害类型"""
        # 基于特征组合识别灾害类型
        if "water_index" in features and np.mean(features["water_index"]) > 0.3:
            return DisasterType.FLOOD
        elif "slope" in features and np.mean(features["slope"]) > 0.5:
            if "ndvi" in features and np.mean(features["ndvi"]) < 0.3:
                return DisasterType.LANDSLIDE
        elif "ndvi" in features and np.mean(features["ndvi"]) < 0.2:
            return DisasterType.FIRE
        
        return DisasterType.UNKNOWN
    
    def _evaluate_confidence(self, features: Dict[str, np.ndarray], 
                           disaster_probability: float) -> float:
        """评估置信度"""
        # 基于特征质量和灾害概率评估置信度
        confidence = disaster_probability * 0.6  # 基础置信度
        
        # 特征质量评估
        feature_quality = 0.0
        for feature_data in features.values():
            # 计算特征的标准差作为质量指标
            feature_std = np.std(feature_data)
            feature_quality += min(feature_std, 1.0)
        
        feature_quality /= len(features) if features else 1.0
        confidence += feature_quality * 0.4
        
        return min(confidence, 1.0)
    
    def _calculate_affected_area(self, features: Dict[str, np.ndarray], 
                                disaster_probability: float) -> float:
        """计算受影响面积"""
        # 基于灾害概率和图像分辨率计算受影响面积
        if len(features) > 0:
            # 使用第一个特征计算
            feature_data = list(features.values())[0]
            # 计算高概率区域的比例
            high_prob_ratio = np.sum(feature_data > 0.7) / feature_data.size
            
            # 假设图像覆盖100km²
            base_area = 100.0
            affected_area = high_prob_ratio * base_area * disaster_probability
        else:
            affected_area = 0.0
        
        return affected_area
    
    def _evaluate_risk_level(self, disaster_probability: float, 
                           affected_area: float) -> EmergencyLevel:
        """评估风险等级"""
        if disaster_probability > 0.8 and affected_area > 50:
            return EmergencyLevel.CRITICAL
        elif disaster_probability > 0.6 and affected_area > 20:
            return EmergencyLevel.HIGH
        elif disaster_probability > 0.4 and affected_area > 10:
            return EmergencyLevel.MEDIUM
        else:
            return EmergencyLevel.LOW
    
    def _extract_key_features(self, features: Dict[str, np.ndarray], 
                             disaster_type: DisasterType) -> Dict[str, Any]:
        """提取关键特征"""
        key_features = {}
        
        for feature_name, feature_data in features.items():
            feature_mean = np.mean(feature_data)
            feature_std = np.std(feature_data)
            
            key_features[feature_name] = {
                "mean": float(feature_mean),
                "std": float(feature_std),
                "description": self.knowledge_base.describe_feature(feature_name, feature_mean)
            }
        
        return key_features
    
    def _generate_result_slice(self, image_data: np.ndarray, 
                              disaster_probability: float,
                              analysis_params: Dict[str, Any]) -> np.ndarray:
        """生成结果切片"""
        output_format = analysis_params.get("output_format", "slice")
        
        if output_format == "slice":
            # 生成概率图切片
            if len(image_data.shape) == 3:
                result_slice = np.mean(image_data, axis=0)
            else:
                result_slice = image_data[0]
            
            # 应用灾害概率
            result_slice = result_slice * disaster_probability
            
        elif output_format == "slice_and_report":
            # 生成带标注的切片
            result_slice = np.zeros((256, 256, 3), dtype=np.float32)
            
            if len(image_data.shape) == 3:
                result_slice[:, :, 0] = image_data[0]  # R
                result_slice[:, :, 1] = image_data[1] if image_data.shape[0] > 1 else image_data[0]  # G
                result_slice[:, :, 2] = image_data[2] if image_data.shape[0] > 2 else image_data[0]  # B
            else:
                result_slice[:, :, 0] = image_data[0]
                result_slice[:, :, 1] = image_data[0]
                result_slice[:, :, 2] = image_data[0]
            
            # 添加灾害概率作为红色通道的增强
            result_slice[:, :, 0] = np.clip(result_slice[:, :, 0] + disaster_probability * 0.5, 0, 1)
        
        else:
            # 默认切片
            result_slice = np.random.rand(256, 256).astype(np.float32)
        
        return result_slice
    
    def _generate_analysis_report(self, features: Dict[str, np.ndarray],
                                 disaster_type: DisasterType,
                                 disaster_probability: float,
                                 confidence_score: float,
                                 affected_area: float,
                                 risk_level: EmergencyLevel) -> str:
        """生成分析报告"""
        report = f"""
灾害解译分析报告
================

灾害类型: {disaster_type.value}
灾害概率: {disaster_probability:.2%}
置信度: {confidence_score:.2%}
受影响面积: {affected_area:.1f} km²
风险等级: {risk_level.name}

关键特征分析:
"""
        
        for feature_name, feature_info in features.items():
            feature_mean = np.mean(feature_info)
            description = self.knowledge_base.describe_feature(feature_name, feature_mean)
            report += f"- {feature_name}: {description}\n"
        
        report += f"""
结论:
基于遥感数据分析，该区域存在{disaster_type.value}风险，
建议立即启动应急响应机制，预计{affected_area:.1f}平方公里区域可能受到影响。
"""
        
        return report.strip() 