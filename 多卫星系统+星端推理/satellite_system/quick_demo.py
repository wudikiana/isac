#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示脚本
展示重构后卫星系统的主要功能
"""

import sys
import os
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def demo_core_functionality():
    """演示核心功能"""
    print("🎬 卫星系统 v2.0 快速演示")
    print("=" * 50)
    
    try:
        # 导入系统模块
        from satellite_system import (
            MultiSatelliteSystem, EmergencyLevel, SatelliteStatus,
            get_system_info
        )
        
        # 显示系统信息
        print("\n📋 系统信息:")
        info = get_system_info()
        print(f"   版本: {info['version']}")
        print(f"   描述: {info['description']}")
        print(f"   模块数: {len(info['modules'])}")
        
        # 创建系统实例
        print("\n🚀 创建系统实例...")
        system = MultiSatelliteSystem("satellite_system/satellite_config.json")
        
        # 显示初始状态
        print("\n📊 系统初始状态:")
        status = system.get_system_status()
        print(f"   🛰️  总卫星数: {status['total_satellites']}")
        print(f"   ✅ 在线卫星: {status['online_satellites']}")
        
        # 演示推理功能
        print("\n🤖 演示推理功能:")
        image_data = np.random.rand(3, 256, 256).astype(np.float32)
        sim_features = np.random.rand(11).astype(np.float32)
        
        task_id = system.submit_inference_task(
            image_data=image_data,
            sim_features=sim_features,
            priority=5,
            location=[39.9, 116.4]  # 北京坐标
        )
        print(f"   📝 提交推理任务: {task_id}")
        
        # 演示应急功能
        print("\n🚨 演示应急响应:")
        emergency_id = system.trigger_emergency(
            location=[39.9, 116.4, 0],  # 北京
            emergency_level=EmergencyLevel.HIGH,
            description="地震灾害"
        )
        print(f"   🚨 触发紧急情况: {emergency_id}")
        
        # 等待系统处理
        print("\n⏳ 等待系统处理...")
        time.sleep(2)
        
        # 显示更新后的状态
        print("\n📈 系统状态更新:")
        updated_status = system.get_system_status()
        print(f"   📡 应急队列: {updated_status['emergency_system']['emergency_queue_size']}")
        print(f"   📈 响应历史: {updated_status['emergency_system']['response_history_count']}")
        
        # 演示轨道功能
        print("\n🛰️  演示轨道控制:")
        system.add_space_debris({
            'position': [1000, 1000, 1000],
            'velocity': [100, 100, 100],
            'size': 10
        })
        debris_count = system.get_space_debris_count()
        print(f"   🗑️  空间碎片数量: {debris_count}")
        
        # 获取应急历史
        print("\n📜 应急响应历史:")
        history = system.get_emergency_history(limit=3)
        for i, record in enumerate(history, 1):
            print(f"   {i}. {record['beacon_id']} - {record['status']}")
        
        print("\n🎉 演示完成！")
        print("\n💡 系统功能包括:")
        print("   ✅ AI推理任务处理")
        print("   ✅ 灾害应急响应")
        print("   ✅ 星间联邦学习")
        print("   ✅ 认知无线电频谱管理")
        print("   ✅ 自主轨道控制")
        print("   ✅ 多卫星负载均衡")
        print("   ✅ 故障容错机制")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_interactive_commands():
    """演示交互式命令"""
    print("\n🎮 交互式命令演示:")
    print("   在交互式系统中可以使用以下命令:")
    print("   status          - 显示系统状态")
    print("   emergency       - 进入应急模式")
    print("   satellite       - 进入卫星管理")
    print("   monitoring      - 进入监控模式")
    print("   inference       - 进入推理模式")
    print("   help            - 显示帮助信息")
    print("   exit            - 退出系统")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='卫星系统快速演示')
    parser.add_argument('--interactive', action='store_true', help='启动交互式系统')
    parser.add_argument('--test', action='store_true', help='运行系统测试')
    
    args = parser.parse_args()
    
    if args.interactive:
        # 启动交互式系统
        try:
            from satellite_system import InteractiveSatelliteSystem
            print("🎮 启动交互式系统...")
            system = InteractiveSatelliteSystem("satellite_system/satellite_config.json")
            system.run()
        except Exception as e:
            print(f"❌ 交互式系统启动失败: {e}")
    elif args.test:
        # 运行测试
        try:
            from satellite_system.test_system import run_all_tests
            run_all_tests()
        except Exception as e:
            print(f"❌ 测试运行失败: {e}")
    else:
        # 运行演示
        success = demo_core_functionality()
        if success:
            demo_interactive_commands()
            
            print("\n🚀 启动选项:")
            print("   python satellite_system/quick_demo.py --interactive  # 启动交互式系统")
            print("   python satellite_system/quick_demo.py --test         # 运行系统测试")
            print("   python satellite_system/start_system.py --mode demo  # 运行完整演示")

if __name__ == "__main__":
    main() 