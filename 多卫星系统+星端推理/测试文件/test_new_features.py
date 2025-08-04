#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新功能模块
验证意图理解和在轨智能解译功能
"""

import sys
import os
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def test_intent_understanding():
    """测试意图理解功能"""
    print("🧪 测试意图理解功能...")
    
    try:
        from satellite_system import IntentUnderstandingEngine, TaskType, DisasterType
        
        # 创建意图理解引擎
        config = {
            "model_path": "models/best_multimodal_patch_model.pth",
            "intent_understanding": {
                "confidence_threshold": 0.7,
                "max_tasks": 100
            }
        }
        
        engine = IntentUnderstandingEngine(config)
        
        # 测试不同类型的指令
        test_commands = [
            "监测山洪风险区",
            "紧急拍摄滑坡区域",
            "分析火灾受灾情况",
            "对北京地区进行SAR成像",
            "评估地震灾害影响"
        ]
        
        for command in test_commands:
            print(f"\n   📝 测试指令: {command}")
            intent_task = engine.parse_command(command)
            
            print(f"      ✅ 任务ID: {intent_task.task_id}")
            print(f"      ✅ 任务类型: {intent_task.task_type.value}")
            print(f"      ✅ 灾害类型: {intent_task.disaster_type.value}")
            print(f"      ✅ 目标位置: {intent_task.target_location}")
            print(f"      ✅ 优先级: {intent_task.priority}")
            print(f"      ✅ 预计时长: {intent_task.estimated_duration:.1f}分钟")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 意图理解测试失败: {e}")
        return False

def test_orbital_interpretation():
    """测试在轨智能解译功能"""
    print("\n🧪 测试在轨智能解译功能...")
    
    try:
        from satellite_system import (
            OrbitalInterpreter, RemoteSensingData, 
            DisasterType, EmergencyLevel
        )
        
        # 创建在轨智能解译器
        config = {
            "model_path": "models/best_multimodal_patch_model.pth",
            "orbital_interpretation": {
                "confidence_threshold": 0.6,
                "feature_extraction": ["ndvi", "slope", "moisture"]
            }
        }
        
        interpreter = OrbitalInterpreter(
            config["model_path"], config
        )
        
        # 创建模拟遥感数据
        image_data = np.random.rand(4, 256, 256).astype(np.float32)
        channel_data = {
            "signal_strength": np.random.rand(256, 256).astype(np.float32),
            "noise_level": np.random.rand(256, 256).astype(np.float32) * 0.1
        }
        
        remote_data = RemoteSensingData(
            data_id="test_rs_001",
            satellite_id="test_sat_001",
            location=[39.9, 116.4, 0.0],
            timestamp=time.time(),
            data_type="sar",
            image_data=image_data,
            metadata={"imaging_mode": "standard"},
            channel_data=channel_data,
            quality_score=0.85
        )
        
        # 分析参数
        analysis_params = {
            "analysis_type": "landslide_detection",
            "confidence_threshold": 0.6,
            "output_format": "slice_and_report",
            "feature_extraction": ["ndvi", "slope", "moisture"]
        }
        
        # 执行解译
        print("   🔍 开始灾害场景解译...")
        result = interpreter.interpret_disaster(remote_data, analysis_params)
        
        print(f"      ✅ 结果ID: {result.result_id}")
        print(f"      ✅ 灾害类型: {result.disaster_type.value}")
        print(f"      ✅ 灾害概率: {result.disaster_probability:.2%}")
        print(f"      ✅ 置信度: {result.confidence_score:.2%}")
        print(f"      ✅ 风险等级: {result.risk_level.name}")
        print(f"      ✅ 受影响面积: {result.affected_area:.1f} km²")
        
        # 显示关键特征
        print("      📊 关键特征:")
        for feature_name, feature_info in result.key_features.items():
            print(f"         - {feature_name}: {feature_info['description']}")
        
        # 显示分析报告摘要
        report_lines = result.analysis_report.split('\n')
        print(f"      📋 分析报告摘要: {report_lines[0] if report_lines else '无'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 在轨智能解译测试失败: {e}")
        return False

def test_cooperative_scheduler():
    """测试协同任务调度功能"""
    print("\n🧪 测试协同任务调度功能...")
    
    try:
        from satellite_system import (
            CooperativeScheduler, SatelliteInfo, SatelliteStatus
        )
        
        # 创建协同任务调度器
        config = {
            "model_path": "models/best_multimodal_patch_model.pth",
            "cooperative_scheduler": {
                "max_concurrent_tasks": 10,
                "timeout": 300.0
            }
        }
        
        scheduler = CooperativeScheduler(config)
        
        # 添加模拟卫星
        imaging_sat = SatelliteInfo(
            satellite_id="sar_sat_001",
            ip_address="192.168.1.101",
            port=8081,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1e12,
            memory_capacity=8192,
            current_load=0.3,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["sar_imaging"],
            coverage_area={},
            current_position=[39.9, 116.4, 500.0],
            orbit_period=90.0
        )
        
        computing_sat = SatelliteInfo(
            satellite_id="compute_sat_001",
            ip_address="192.168.1.102",
            port=8082,
            status=SatelliteStatus.ONLINE,
            compute_capacity=5e12,  # 5 TFLOPS
            memory_capacity=16384,
            current_load=0.2,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["ai_inference"],
            coverage_area={},
            current_position=[40.0, 116.5, 500.0],
            orbit_period=90.0
        )
        
        scheduler.add_satellite(imaging_sat)
        scheduler.add_satellite(computing_sat)
        
        # 测试协同任务处理
        test_commands = [
            "紧急监测滑坡风险区域",
            "对洪水灾区进行SAR成像分析"
        ]
        
        for command in test_commands:
            print(f"\n   📝 测试协同任务: {command}")
            result = scheduler.process_command(command)
            
            print(f"      ✅ 任务执行完成")
            print(f"      📄 结果摘要: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 协同任务调度测试失败: {e}")
        return False

def test_feature_description():
    """测试特征描述功能"""
    print("\n🧪 测试特征描述功能...")
    
    try:
        from satellite_system import PromptTemplates
        
        templates = PromptTemplates()
        
        # 测试不同特征值的描述
        test_features = [
            ("ndvi", 0.2),
            ("ndvi", 0.4),
            ("ndvi", 0.6),
            ("slope", 0.1),
            ("slope", 0.3),
            ("slope", 0.7),
            ("moisture", 0.15),
            ("moisture", 0.35),
            ("moisture", 0.55)
        ]
        
        for feature_name, feature_value in test_features:
            description = templates.FEATURE_DESCRIPTION_TEMPLATES[feature_name]
            
            if feature_value < 0.3:
                desc = description["low"]
            elif feature_value < 0.7:
                desc = description["medium"]
            else:
                desc = description["high"]
            
            print(f"   📊 {feature_name}={feature_value:.2f}: {desc}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 特征描述测试失败: {e}")
        return False

def test_integration():
    """测试集成功能"""
    print("\n🧪 测试集成功能...")
    
    try:
        from satellite_system import MultiSatelliteSystem
        
        # 创建主系统
        system = MultiSatelliteSystem("satellite_system/satellite_config.json")
        
        # 测试自然语言指令处理
        print("   📝 测试自然语言指令处理...")
        result = system.process_natural_language_command("监测山洪风险区")
        print(f"      ✅ 指令处理完成: {result[:100]}...")
        
        # 测试意图解析
        print("   🔍 测试意图解析...")
        intent_task = system.parse_intent("紧急拍摄滑坡区域")
        print(f"      ✅ 意图解析完成: {intent_task.task_type.value}")
        
        # 测试特征描述
        print("   📊 测试特征描述...")
        desc = system.get_feature_description("ndvi", 0.25)
        print(f"      ✅ 特征描述: {desc}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试新功能模块")
    print("=" * 50)
    
    test_results = []
    
    # 测试各个功能模块
    test_results.append(("意图理解", test_intent_understanding()))
    test_results.append(("在轨智能解译", test_orbital_interpretation()))
    test_results.append(("协同任务调度", test_cooperative_scheduler()))
    test_results.append(("特征描述", test_feature_description()))
    test_results.append(("集成功能", test_integration()))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("📋 测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！新功能模块工作正常。")
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 