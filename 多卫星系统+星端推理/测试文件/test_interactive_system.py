#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式多卫星系统测试脚本
验证系统的各项功能是否正常工作
"""

import sys
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np

# 导入系统模块
from advanced_multi_satellite_system import (
    AdvancedMultiSatelliteSystem, EmergencyLevel, SatelliteStatus
)
from enhanced_multi_satellite_inference import MultiSatelliteInferenceSystem
from interactive_satellite_system import InteractiveSatelliteSystem

class TestInteractiveSatelliteSystem(unittest.TestCase):
    """交互式多卫星系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.system = InteractiveSatelliteSystem()
        
    def test_system_initialization(self):
        """测试系统初始化"""
        print("🧪 测试系统初始化...")
        
        # 模拟初始化
        with patch.object(self.system, 'advanced_system') as mock_advanced:
            with patch.object(self.system, 'inference_system') as mock_inference:
                result = self.system.initialize_systems()
                self.assertTrue(result)
        
        print("✅ 系统初始化测试通过")
    
    def test_emergency_response(self):
        """测试应急响应功能"""
        print("🧪 测试应急响应功能...")
        
        # 创建模拟系统
        self.system.advanced_system = Mock()
        self.system.advanced_system.trigger_emergency.return_value = "emergency_123"
        
        # 测试触发紧急情况
        args = ['3', '39.9', '116.4', '地震灾害']
        self.system._trigger_emergency(args)
        
        # 验证调用
        self.system.advanced_system.trigger_emergency.assert_called_once()
        
        print("✅ 应急响应测试通过")
    
    def test_satellite_management(self):
        """测试卫星管理功能"""
        print("🧪 测试卫星管理功能...")
        
        # 创建模拟卫星
        mock_satellite = Mock()
        mock_satellite.satellite_id = "sat_001"
        mock_satellite.status = SatelliteStatus.ONLINE
        mock_satellite.current_position = [40.0, 116.0, 500.0]
        mock_satellite.current_load = 0.3
        mock_satellite.fuel_level = 85.0
        mock_satellite.compute_capacity = 1e12
        mock_satellite.memory_capacity = 8192
        mock_satellite.orbit_period = 90.0
        mock_satellite.supported_features = ["sar_imaging", "computing"]
        mock_satellite.coverage_area = {"lat": [35, 45], "lon": [110, 130]}
        
        self.system.advanced_system = Mock()
        self.system.advanced_system.satellites = {"sat_001": mock_satellite}
        
        # 测试卫星列表
        self.system._list_satellites([])
        
        # 测试卫星状态
        self.system._show_satellite_status(['sat_001'])
        
        print("✅ 卫星管理测试通过")
    
    def test_inference_system(self):
        """测试推理系统功能"""
        print("🧪 测试推理系统功能...")
        
        # 创建模拟推理系统
        self.system.inference_system = Mock()
        self.system.inference_system.submit_inference_task.return_value = "task_123"
        self.system.inference_system.get_system_status.return_value = {
            'queue_size': 5,
            'cache_size': 10,
            'online_satellites': 3,
            'training_status': {
                'queue_size': 2,
                'last_sync_time': time.time()
            }
        }
        
        # 测试推理队列显示
        self.system._show_inference_queue([])
        
        print("✅ 推理系统测试通过")
    
    def test_command_parsing(self):
        """测试命令解析"""
        print("🧪 测试命令解析...")
        
        # 测试主菜单命令
        self.system.current_mode = "main"
        self.system._process_command("help")
        self.system._process_command("status")
        
        # 测试应急命令
        self.system.current_mode = "emergency"
        self.system._process_command("help")
        
        # 测试卫星命令
        self.system.current_mode = "satellite"
        self.system._process_command("help")
        
        print("✅ 命令解析测试通过")
    
    def test_configuration_management(self):
        """测试配置管理"""
        print("🧪 测试配置管理...")
        
        # 测试配置保存和加载
        self.system.auto_refresh = True
        self.system.refresh_interval = 10
        self.system.command_history = ["help", "status", "emergency"]
        
        # 保存配置
        self.system._save_configuration()
        
        # 加载配置
        self.system._load_configuration()
        
        print("✅ 配置管理测试通过")

class TestAdvancedSystem(unittest.TestCase):
    """高级系统功能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.system = AdvancedMultiSatelliteSystem()
    
    def test_ppo_controller(self):
        """测试PPO控制器"""
        print("🧪 测试PPO控制器...")
        
        # 创建模拟紧急信标
        from advanced_multi_satellite_system import EmergencyBeacon
        beacon = EmergencyBeacon(
            beacon_id="test_beacon",
            location=[39.9, 116.4, 0],
            emergency_level=EmergencyLevel.HIGH,
            timestamp=time.time(),
            description="测试紧急情况",
            required_resources={'compute': 0.5, 'memory': 1024}
        )
        
        # 测试资源分配
        allocation = self.system.emergency_system.ppo_controller.allocate_resources(
            beacon, self.system.satellites
        )
        
        self.assertIsNotNone(allocation)
        self.assertIn('allocation_id', allocation.__dict__)
        
        print("✅ PPO控制器测试通过")
    
    def test_federated_learning(self):
        """测试联邦学习"""
        print("🧪 测试联邦学习...")
        
        # 创建模拟模型参数
        import torch
        local_parameters = {
            'layer1.weight': torch.randn(64, 64),
            'layer1.bias': torch.randn(64)
        }
        
        # 提交本地更新
        self.system.emergency_system.federated_learning.submit_local_update(
            "sat_001", local_parameters, 100, 0.1
        )
        
        # 检查状态
        status = self.system.emergency_system.federated_learning.get_federated_status()
        self.assertEqual(status['local_models_count'], 1)
        
        print("✅ 联邦学习测试通过")
    
    def test_cognitive_radio(self):
        """测试认知无线电"""
        print("🧪 测试认知无线电...")
        
        # 创建模拟卫星
        from advanced_multi_satellite_system import SatelliteInfo
        satellite = SatelliteInfo(
            satellite_id="test_sat",
            ip_address="192.168.1.100",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1e12,
            memory_capacity=8192,
            current_load=0.0,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["communication"],
            coverage_area={"lat": [35, 45], "lon": [110, 130]},
            current_position=[40.0, 116.0, 500.0],
            orbit_period=90.0
        )
        
        # 测试动态频谱接入
        terrestrial_users = {}
        config = self.system.emergency_system.cognitive_radio.dynamic_spectrum_access(
            satellite, terrestrial_users
        )
        
        self.assertIn('frequency', config)
        self.assertIn('bandwidth', config)
        
        print("✅ 认知无线电测试通过")
    
    def test_orbit_control(self):
        """测试轨道控制"""
        print("🧪 测试轨道控制...")
        
        # 创建模拟卫星
        from advanced_multi_satellite_system import SatelliteInfo
        satellite = SatelliteInfo(
            satellite_id="test_sat",
            ip_address="192.168.1.100",
            port=8080,
            status=SatelliteStatus.ONLINE,
            compute_capacity=1e12,
            memory_capacity=8192,
            current_load=0.0,
            last_heartbeat=time.time(),
            model_version="v1.0",
            supported_features=["orbit_control"],
            coverage_area={"lat": [35, 45], "lon": [110, 130]},
            current_position=[40.0, 116.0, 500.0],
            orbit_period=90.0
        )
        
        # 创建轨道控制器
        from advanced_multi_satellite_system import AutonomousOrbitControl
        controller = AutonomousOrbitControl(satellite)
        
        # 测试编队维持
        reference_sats = []
        controller.maintain_formation(reference_sats)
        
        # 测试碰撞避免
        space_debris = []
        controller.collision_avoidance(space_debris)
        
        print("✅ 轨道控制测试通过")

def run_performance_test():
    """运行性能测试"""
    print("\n🚀 开始性能测试...")
    
    # 测试系统启动时间
    start_time = time.time()
    system = InteractiveSatelliteSystem()
    init_time = time.time() - start_time
    print(f"⏱️  系统初始化时间: {init_time:.3f}秒")
    
    # 测试命令响应时间
    start_time = time.time()
    system._show_help([])
    response_time = time.time() - start_time
    print(f"⏱️  命令响应时间: {response_time:.3f}秒")
    
    # 测试内存使用
    import psutil
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"💾 内存使用: {memory_usage:.1f} MB")
    
    print("✅ 性能测试完成")

def run_integration_test():
    """运行集成测试"""
    print("\n🔗 开始集成测试...")
    
    try:
        # 创建系统
        system = InteractiveSatelliteSystem()
        
        # 模拟初始化
        system.advanced_system = Mock()
        system.inference_system = Mock()
        
        # 测试完整流程
        print("1. 测试系统状态显示...")
        system._show_system_status([])
        
        print("2. 测试应急响应...")
        system._trigger_emergency(['3', '39.9', '116.4', '测试灾害'])
        
        print("3. 测试卫星管理...")
        system._list_satellites([])
        
        print("4. 测试推理任务...")
        system._show_inference_queue([])
        
        print("✅ 集成测试通过")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")

def main():
    """主测试函数"""
    print("🧪 多卫星系统测试套件")
    print("=" * 50)
    
    # 运行单元测试
    print("\n📋 运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    run_performance_test()
    
    # 运行集成测试
    run_integration_test()
    
    print("\n🎉 所有测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    main() 