#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星系统启动脚本
用于启动重构后的卫星系统
"""

import sys
import os
import argparse
import time
import threading
import subprocess

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def start_satellite_servers(num_satellites=3, model_path="models/best_multimodal_patch_model.pth"):
    """启动卫星服务器"""
    print(f"🚀 启动 {num_satellites} 个卫星服务器...")
    
    processes = []
    
    for i in range(num_satellites):
        sat_id = f"sat_{i+1:03d}"
        port = 8080 + i
        
        # 启动服务器进程
        cmd = [
            sys.executable, 
            "satellite_system/satellite_server.py",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--satellite_id", sat_id,
            "--model", model_path
        ]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append((sat_id, port, process))
            print(f"   ✅ {sat_id} 启动在端口 {port}")
            time.sleep(2)  # 等待服务器启动
        except Exception as e:
            print(f"   ❌ {sat_id} 启动失败: {e}")
    
    return processes

def start_main_system(config_file="satellite_system/satellite_config.json"):
    """启动主系统"""
    print("\n🌍 启动主卫星系统...")
    
    try:
        from satellite_system import MultiSatelliteSystem
        
        system = MultiSatelliteSystem(config_file)
        print("✅ 主系统启动成功")
        
        # 显示系统状态
        status = system.get_system_status()
        print(f"📊 系统状态:")
        print(f"   🛰️  总卫星数: {status['total_satellites']}")
        print(f"   ✅ 在线卫星: {status['online_satellites']}")
        
        return system
        
    except Exception as e:
        print(f"❌ 主系统启动失败: {e}")
        return None

def start_interactive_system(config_file="satellite_system/satellite_config.json"):
    """启动交互式系统"""
    print("\n🎮 启动交互式系统...")
    
    try:
        from satellite_system import InteractiveSatelliteSystem
        
        system = InteractiveSatelliteSystem(config_file)
        print("✅ 交互式系统启动成功")
        
        # 运行交互式系统
        system.run()
        
    except Exception as e:
        print(f"❌ 交互式系统启动失败: {e}")

def demo_system():
    """演示系统功能"""
    print("\n🎬 系统演示模式")
    
    try:
        from satellite_system import MultiSatelliteSystem, EmergencyLevel
        
        # 创建系统
        system = MultiSatelliteSystem("satellite_system/satellite_config.json")
        
        print("\n1. 系统初始化状态:")
        status = system.get_system_status()
        print(f"   🛰️  总卫星数: {status['total_satellites']}")
        print(f"   ✅ 在线卫星: {status['online_satellites']}")
        
        print("\n2. 触发紧急情况...")
        emergency_id = system.trigger_emergency(
            location=[39.9, 116.4, 0],  # 北京
            emergency_level=EmergencyLevel.HIGH,
            description="地震灾害"
        )
        print(f"   🚨 紧急信标ID: {emergency_id}")
        
        print("\n3. 等待应急响应...")
        time.sleep(2)
        
        print("\n4. 系统状态更新:")
        updated_status = system.get_system_status()
        print(f"   📡 应急队列: {updated_status['emergency_system']['emergency_queue_size']}")
        print(f"   📈 响应历史: {updated_status['emergency_system']['response_history_count']}")
        
        print("\n5. 联邦学习状态:")
        fed_status = updated_status['emergency_system']['federated_learning_status']
        print(f"   🧠 聚合轮次: {fed_status['aggregation_round']}")
        print(f"   📊 本地模型数: {fed_status['local_models_count']}")
        
        print("\n🎬 演示完成！")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='卫星系统启动脚本')
    parser.add_argument('--mode', choices=['servers', 'main', 'interactive', 'demo', 'all'], 
                       default='all', help='启动模式')
    parser.add_argument('--satellites', type=int, default=3, help='卫星数量')
    parser.add_argument('--model', type=str, default='models/best_multimodal_patch_model.pth', 
                       help='模型路径')
    parser.add_argument('--config', type=str, default='satellite_system/satellite_config.json', 
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    print("🌍 卫星系统 v2.0 启动脚本")
    print("=" * 50)
    
    processes = []
    main_system = None
    
    try:
        if args.mode in ['servers', 'all']:
            processes = start_satellite_servers(args.satellites, args.model)
        
        if args.mode in ['main', 'all']:
            main_system = start_main_system(args.config)
        
        if args.mode == 'demo':
            demo_system()
            return
        
        if args.mode in ['interactive', 'all']:
            if args.mode == 'all':
                print("\n⏳ 等待系统初始化完成...")
                time.sleep(5)
            start_interactive_system(args.config)
        
        # 保持运行
        if args.mode == 'all':
            print("\n🔄 系统运行中，按 Ctrl+C 停止...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 收到停止信号")
    
    except KeyboardInterrupt:
        print("\n🛑 收到停止信号")
    except Exception as e:
        print(f"❌ 系统运行异常: {e}")
    finally:
        # 清理资源
        print("\n🧹 正在清理资源...")
        
        # 停止主系统
        if main_system:
            del main_system
        
        # 停止卫星服务器进程
        for sat_id, port, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"   ✅ {sat_id} 已停止")
            except:
                process.kill()
                print(f"   ⚠️  {sat_id} 强制停止")
        
        print("✅ 系统已停止")

if __name__ == "__main__":
    main() 