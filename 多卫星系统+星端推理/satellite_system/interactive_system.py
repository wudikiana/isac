#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式多卫星系统
提供用户友好的命令行界面，整合所有功能模块
"""

import os
import sys
import time
import json
import threading
from typing import Dict, List, Optional, Any
import argparse
from datetime import datetime, timedelta

# 导入系统模块
from .main_system import MultiSatelliteSystem
from .satellite_core import EmergencyLevel, SatelliteStatus

class InteractiveSatelliteSystem:
    """交互式多卫星系统"""
    
    def __init__(self, config_file: str = "satellite_config.json"):
        self.config_file = config_file
        self.system = None
        self.running = False
        self.command_history = []
        self.max_history = 100
        
        # 系统状态
        self.current_mode = "main"  # main, emergency, satellite, monitoring, inference
        self.auto_refresh = True
        self.refresh_interval = 5  # 秒
        
    def initialize_system(self):
        """初始化系统"""
        print("🚀 正在初始化多卫星系统...")
        
        try:
            # 初始化主系统
            self.system = MultiSatelliteSystem(self.config_file)
            print("✅ 多卫星系统初始化完成")
            
            # 启动自动刷新
            if self.auto_refresh:
                self._start_auto_refresh()
            
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False
    
    def _start_auto_refresh(self):
        """启动自动刷新线程"""
        def refresh_loop():
            while self.running:
                try:
                    if self.current_mode == "monitoring":
                        self._display_system_status()
                    time.sleep(self.refresh_interval)
                except Exception as e:
                    print(f"自动刷新异常: {e}")
        
        refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        refresh_thread.start()
    
    def run(self):
        """运行交互式系统"""
        if not self.initialize_system():
            return
        
        self.running = True
        print("\n" + "="*60)
        print("🌍 多卫星系统交互界面 v2.0")
        print("="*60)
        
        while self.running:
            try:
                self._display_prompt()
                command = input("> ").strip()
                
                if command:
                    self._process_command(command)
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  检测到中断信号，正在安全退出...")
                self.running = False
            except EOFError:
                print("\n\n👋 再见！")
                self.running = False
            except Exception as e:
                print(f"❌ 命令执行异常: {e}")
        
        self._cleanup()
    
    def _display_prompt(self):
        """显示命令提示"""
        mode_indicators = {
            "main": "🌍",
            "emergency": "🚨",
            "satellite": "🛰️",
            "monitoring": "📊",
            "inference": "🤖"
        }
        
        indicator = mode_indicators.get(self.current_mode, "❓")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n[{timestamp}] {indicator} {self.current_mode.upper()}", end="")
    
    def _process_command(self, command: str):
        """处理用户命令"""
        # 记录命令历史
        self.command_history.append(command)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
        
        # 解析命令
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        # 根据当前模式处理命令
        if self.current_mode == "main":
            self._handle_main_commands(cmd, args)
        elif self.current_mode == "emergency":
            self._handle_emergency_commands(cmd, args)
        elif self.current_mode == "satellite":
            self._handle_satellite_commands(cmd, args)
        elif self.current_mode == "monitoring":
            self._handle_monitoring_commands(cmd, args)
        elif self.current_mode == "inference":
            self._handle_inference_commands(cmd, args)
    
    def _handle_main_commands(self, cmd: str, args: List[str]):
        """处理主菜单命令"""
        commands = {
            "help": self._show_help,
            "status": self._show_system_status,
            "emergency": self._enter_emergency_mode,
            "satellite": self._enter_satellite_mode,
            "monitoring": self._enter_monitoring_mode,
            "inference": self._enter_inference_mode,
            "exit": self._exit_system,
            "quit": self._exit_system,
            "clear": self._clear_screen,
            "history": self._show_command_history,
            "config": self._show_configuration
        }
        
        if cmd in commands:
            commands[cmd](args)
        else:
            print(f"❓ 未知命令: {cmd}")
            print("💡 输入 'help' 查看可用命令")
    
    def _handle_emergency_commands(self, cmd: str, args: List[str]):
        """处理应急模式命令"""
        commands = {
            "help": self._show_emergency_help,
            "trigger": self._trigger_emergency,
            "list": self._list_emergencies,
            "status": self._show_emergency_status,
            "back": self._back_to_main,
            "exit": self._exit_system
        }
        
        if cmd in commands:
            commands[cmd](args)
        else:
            print(f"❓ 未知应急命令: {cmd}")
            print("💡 输入 'help' 查看应急命令")
    
    def _handle_satellite_commands(self, cmd: str, args: List[str]):
        """处理卫星管理命令"""
        commands = {
            "help": self._show_satellite_help,
            "list": self._list_satellites,
            "status": self._show_satellite_status,
            "control": self._control_satellite,
            "orbit": self._show_orbit_info,
            "back": self._back_to_main,
            "exit": self._exit_system
        }
        
        if cmd in commands:
            commands[cmd](args)
        else:
            print(f"❓ 未知卫星命令: {cmd}")
            print("💡 输入 'help' 查看卫星命令")
    
    def _handle_monitoring_commands(self, cmd: str, args: List[str]):
        """处理监控模式命令"""
        commands = {
            "help": self._show_monitoring_help,
            "refresh": self._refresh_status,
            "auto": self._toggle_auto_refresh,
            "interval": self._set_refresh_interval,
            "back": self._back_to_main,
            "exit": self._exit_system
        }
        
        if cmd in commands:
            commands[cmd](args)
        else:
            print(f"❓ 未知监控命令: {cmd}")
            print("💡 输入 'help' 查看监控命令")
    
    def _handle_inference_commands(self, cmd: str, args: List[str]):
        """处理推理模式命令"""
        commands = {
            "help": self._show_inference_help,
            "submit": self._submit_inference_task,
            "result": self._get_inference_result,
            "queue": self._show_inference_queue,
            "back": self._back_to_main,
            "exit": self._exit_system
        }
        
        if cmd in commands:
            commands[cmd](args)
        else:
            print(f"❓ 未知推理命令: {cmd}")
            print("💡 输入 'help' 查看推理命令")
    
    # ==================== 主菜单命令 ====================
    
    def _show_help(self, args: List[str]):
        """显示帮助信息"""
        print("\n📖 主菜单命令:")
        print("  status      - 显示系统状态")
        print("  emergency   - 进入应急响应模式")
        print("  satellite   - 进入卫星管理模式")
        print("  monitoring  - 进入系统监控模式")
        print("  inference   - 进入推理任务模式")
        print("  config      - 显示系统配置")
        print("  history     - 显示命令历史")
        print("  clear       - 清屏")
        print("  help        - 显示此帮助")
        print("  exit/quit   - 退出系统")
    
    def _show_system_status(self, args: List[str]):
        """显示系统状态"""
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        status = self.system.get_system_status()
        
        print("\n📊 系统状态概览:")
        print(f"  🛰️  总卫星数: {status['total_satellites']}")
        print(f"  ✅ 在线卫星: {status['online_satellites']}")
        print(f"  📡 应急队列: {status['emergency_system']['emergency_queue_size']}")
        print(f"  📈 响应历史: {status['emergency_system']['response_history_count']}")
        
        # 显示卫星详细信息
        print("\n🛰️  卫星状态:")
        for sat_id, sat_info in status['satellites'].items():
            status_icon = "✅" if sat_info['status'] == 'online' else "❌"
            print(f"  {status_icon} {sat_id}: {sat_info['status']} (负载: {sat_info['load']:.2f})")
    
    def _enter_emergency_mode(self, args: List[str]):
        """进入应急模式"""
        self.current_mode = "emergency"
        print("\n🚨 进入应急响应模式")
        self._show_emergency_help([])
    
    def _enter_satellite_mode(self, args: List[str]):
        """进入卫星管理模式"""
        self.current_mode = "satellite"
        print("\n🛰️  进入卫星管理模式")
        self._show_satellite_help([])
    
    def _enter_monitoring_mode(self, args: List[str]):
        """进入监控模式"""
        self.current_mode = "monitoring"
        print("\n📊 进入系统监控模式")
        self._show_monitoring_help([])
        self._display_system_status()
    
    def _enter_inference_mode(self, args: List[str]):
        """进入推理模式"""
        self.current_mode = "inference"
        print("\n🤖 进入推理任务模式")
        self._show_inference_help([])
    
    def _exit_system(self, args: List[str]):
        """退出系统"""
        print("\n👋 正在退出系统...")
        self.running = False
    
    def _clear_screen(self, args: List[str]):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _show_command_history(self, args: List[str]):
        """显示命令历史"""
        print("\n📜 命令历史:")
        for i, cmd in enumerate(self.command_history[-10:], 1):
            print(f"  {i:2d}. {cmd}")
    
    def _show_configuration(self, args: List[str]):
        """显示系统配置"""
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        print("\n⚙️  系统配置:")
        print(f"  📡 卫星数量: {len(self.system.config['satellites'])}")
        print(f"  🚨 应急响应超时: {self.system.config['emergency_response']['response_timeout']}秒")
        print(f"  🤖 联邦学习同步间隔: {self.system.config['federated_learning']['sync_interval']}秒")
        print(f"  🛰️  轨道控制碰撞阈值: {self.system.config['orbit_control']['collision_threshold']}米")
    
    def _back_to_main(self, args: List[str]):
        """返回主菜单"""
        self.current_mode = "main"
        print("\n🏠 返回主菜单")
    
    # ==================== 应急模式命令 ====================
    
    def _show_emergency_help(self, args: List[str]):
        """显示应急帮助"""
        print("\n🚨 应急响应命令:")
        print("  trigger <level> <lat> <lon> <desc> - 触发紧急情况")
        print("  list                                - 列出所有紧急情况")
        print("  status                              - 显示应急系统状态")
        print("  back                                - 返回主菜单")
        print("  help                                - 显示此帮助")
        print("\n紧急级别: LOW(1), MEDIUM(2), HIGH(3), CRITICAL(4)")
        print("示例: trigger 3 39.9 116.4 地震灾害")
    
    def _trigger_emergency(self, args: List[str]):
        """触发紧急情况"""
        if len(args) < 4:
            print("❌ 参数不足，格式: trigger <level> <lat> <lon> <desc>")
            return
        
        try:
            level = int(args[0])
            lat = float(args[1])
            lon = float(args[2])
            desc = " ".join(args[3:])
            
            if level not in [1, 2, 3, 4]:
                print("❌ 紧急级别无效，应为 1-4")
                return
            
            emergency_level = EmergencyLevel(level)
            location = [lat, lon, 0]
            
            emergency_id = self.system.trigger_emergency(
                location, emergency_level, desc
            )
            
            print(f"✅ 紧急情况已触发: {emergency_id}")
            print(f"📍 位置: {lat}, {lon}")
            print(f"🚨 级别: {emergency_level.name}")
            print(f"📝 描述: {desc}")
            
        except ValueError as e:
            print(f"❌ 参数格式错误: {e}")
    
    def _list_emergencies(self, args: List[str]):
        """列出紧急情况"""
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        history = self.system.get_emergency_history()
        if not history:
            print("📭 暂无紧急情况记录")
            return
        
        print("\n🚨 紧急情况历史:")
        for i, record in enumerate(history[-5:], 1):
            print(f"  {i}. {record['beacon_id']} - {record['status']}")
    
    def _show_emergency_status(self, args: List[str]):
        """显示应急系统状态"""
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        status = self.system.get_system_status()
        emergency_status = status['emergency_system']
        
        print("\n🚨 应急系统状态:")
        print(f"  📡 注册卫星: {emergency_status['satellites_count']}")
        print(f"  📋 应急队列: {emergency_status['emergency_queue_size']}")
        print(f"  📈 响应历史: {emergency_status['response_history_count']}")
        print(f"  🤖 PPO控制器: {emergency_status['ppo_controller_status']}")
    
    # ==================== 卫星管理命令 ====================
    
    def _show_satellite_help(self, args: List[str]):
        """显示卫星管理帮助"""
        print("\n🛰️  卫星管理命令:")
        print("  list                    - 列出所有卫星")
        print("  status <sat_id>         - 显示卫星状态")
        print("  control <sat_id> <cmd>  - 控制卫星")
        print("  orbit <sat_id>          - 显示轨道信息")
        print("  back                    - 返回主菜单")
        print("  help                    - 显示此帮助")
    
    def _list_satellites(self, args: List[str]):
        """列出所有卫星"""
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        status = self.system.get_system_status()
        satellites = status['satellites']
        
        print("\n🛰️  卫星列表:")
        for sat_id, sat_info in satellites.items():
            status_icon = "✅" if sat_info['status'] == 'online' else "❌"
            fuel_icon = "🟢" if sat_info['fuel'] > 50 else "🟡" if sat_info['fuel'] > 20 else "🔴"
            
            print(f"  {status_icon} {sat_id}")
            print(f"    📍 位置: {sat_info['position']}")
            print(f"    ⚡ 负载: {sat_info['load']:.2f}")
            print(f"    ⛽ 燃料: {fuel_icon} {sat_info['fuel']:.1f}%")
            print()
    
    def _show_satellite_status(self, args: List[str]):
        """显示卫星状态"""
        if not args:
            print("❌ 请指定卫星ID")
            return
        
        sat_id = args[0]
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        status = self.system.get_system_status()
        if sat_id not in status['satellites']:
            print(f"❌ 卫星 {sat_id} 不存在")
            return
        
        sat_info = status['satellites'][sat_id]
        
        print(f"\n🛰️  卫星 {sat_id} 详细信息:")
        print(f"  📍 位置: {sat_info['position']}")
        print(f"  📡 状态: {sat_info['status']}")
        print(f"  ⚡ 负载: {sat_info['load']:.2f}")
        print(f"  ⛽ 燃料: {sat_info['fuel']:.1f}%")
    
    def _control_satellite(self, args: List[str]):
        """控制卫星"""
        if len(args) < 2:
            print("❌ 格式: control <sat_id> <command>")
            return
        
        sat_id = args[0]
        command = args[1]
        
        print(f"🔄 执行控制命令: {command} 在卫星 {sat_id}")
        # 这里可以添加具体的控制逻辑
    
    def _show_orbit_info(self, args: List[str]):
        """显示轨道信息"""
        if not args:
            print("❌ 请指定卫星ID")
            return
        
        sat_id = args[0]
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        status = self.system.get_system_status()
        orbit_status = status['orbit_manager']
        
        if sat_id not in orbit_status:
            print(f"❌ 卫星 {sat_id} 轨道信息不存在")
            return
        
        sat_orbit = orbit_status[sat_id]
        
        print(f"\n🛰️  卫星 {sat_id} 轨道信息:")
        print(f"  📍 当前位置: {sat_orbit['current_position']}")
        print(f"  ⛽ 燃料状态: {sat_orbit['fuel_level']:.1f}%")
        print(f"  🎯 碰撞阈值: {sat_orbit['collision_threshold']} 米")
        print(f"  📐 编队阈值: {sat_orbit['formation_threshold']} 米")
    
    # ==================== 监控模式命令 ====================
    
    def _show_monitoring_help(self, args: List[str]):
        """显示监控帮助"""
        print("\n📊 监控模式命令:")
        print("  refresh                 - 手动刷新状态")
        print("  auto                    - 切换自动刷新")
        print("  interval <seconds>      - 设置刷新间隔")
        print("  back                    - 返回主菜单")
        print("  help                    - 显示此帮助")
    
    def _refresh_status(self, args: List[str]):
        """手动刷新状态"""
        self._display_system_status()
    
    def _toggle_auto_refresh(self, args: List[str]):
        """切换自动刷新"""
        self.auto_refresh = not self.auto_refresh
        status = "开启" if self.auto_refresh else "关闭"
        print(f"🔄 自动刷新已{status}")
    
    def _set_refresh_interval(self, args: List[str]):
        """设置刷新间隔"""
        if not args:
            print(f"⏱️  当前刷新间隔: {self.refresh_interval} 秒")
            return
        
        try:
            interval = int(args[0])
            if interval < 1:
                print("❌ 刷新间隔不能小于1秒")
                return
            
            self.refresh_interval = interval
            print(f"⏱️  刷新间隔已设置为 {interval} 秒")
        except ValueError:
            print("❌ 无效的间隔时间")
    
    def _display_system_status(self):
        """显示系统状态（监控模式）"""
        if not self.system:
            return
        
        status = self.system.get_system_status()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n[{timestamp}] 📊 系统状态:")
        print(f"  🛰️  卫星: {status['online_satellites']}/{status['total_satellites']} 在线")
        print(f"  🚨 应急队列: {status['emergency_system']['emergency_queue_size']}")
        print(f"  📈 响应历史: {status['emergency_system']['response_history_count']}")
        
        # 显示卫星状态
        online_sats = [sat_id for sat_id, sat_info in status['satellites'].items() 
                      if sat_info['status'] == 'online']
        if online_sats:
            print(f"  ✅ 在线卫星: {', '.join(online_sats)}")
    
    # ==================== 推理模式命令 ====================
    
    def _show_inference_help(self, args: List[str]):
        """显示推理帮助"""
        print("\n🤖 推理任务命令:")
        print("  submit <image_path> [sim_features] - 提交推理任务")
        print("  result <task_id>                    - 获取推理结果")
        print("  queue                               - 显示任务队列")
        print("  back                                - 返回主菜单")
        print("  help                                - 显示此帮助")
    
    def _submit_inference_task(self, args: List[str]):
        """提交推理任务"""
        if not args:
            print("❌ 请指定图像路径")
            return
        
        print("🤖 推理任务提交功能（需要实现图像加载）")
        print("💡 这里可以添加图像处理和推理任务提交逻辑")
    
    def _get_inference_result(self, args: List[str]):
        """获取推理结果"""
        if not args:
            print("❌ 请指定任务ID")
            return
        
        task_id = args[0]
        print(f"🤖 获取推理结果: {task_id}")
        print("💡 这里可以添加结果获取逻辑")
    
    def _show_inference_queue(self, args: List[str]):
        """显示推理队列"""
        if not self.system:
            print("❌ 系统未初始化")
            return
        
        status = self.system.get_system_status()
        inference_status = status['inference_system']
        
        print("\n🤖 推理系统状态:")
        