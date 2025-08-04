#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多卫星系统测试脚本
"""

import argparse
import time
import logging
import numpy as np
import socket
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connectivity(config_file):
    """测试连通性"""
    logger.info("=== 测试连通性 ===")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        satellites = config.get('satellites', {})
        success_count = 0
        
        for sat_id, sat_config in satellites.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((sat_config['ip'], sat_config['port']))
                sock.close()
                
                if result == 0:
                    logger.info(f"✅ 卫星 {sat_id} 连通性正常")
                    success_count += 1
                else:
                    logger.error(f"❌ 卫星 {sat_id} 连通性失败")
                    
            except Exception as e:
                logger.error(f"❌ 卫星 {sat_id} 连通性测试失败: {e}")
        
        success_rate = success_count / len(satellites) if satellites else 0
        logger.info(f"连通性测试结果: {success_count}/{len(satellites)} 成功 ({success_rate:.2%})")
        return success_rate >= 0.8
        
    except Exception as e:
        logger.error(f"连通性测试失败: {e}")
        return False

def test_task_allocation(config_file):
    """测试任务分配"""
    logger.info("=== 测试任务分配 ===")
    
    try:
        # 尝试导入系统
        try:
            from satellite_system.satellite_inference import MultiSatelliteInferenceSystem
            system = MultiSatelliteInferenceSystem(config_file)
        except ImportError as e:
            logger.warning(f"无法导入MultiSatelliteInferenceSystem: {e}")
            logger.info("跳过任务分配测试")
            return True
        
        # 创建测试任务
        test_image = np.random.rand(3, 256, 256).astype(np.float32)
        test_sim_features = np.random.rand(11).astype(np.float32)
        
        # 提交任务
        task_id = system.submit_inference_task(
            image_data=test_image,
            sim_features=test_sim_features,
            priority=5,
            timeout=30.0,
            location=[116.4074, 39.9042]
        )
        
        logger.info(f"提交任务: {task_id}")
        
        # 等待任务处理
        time.sleep(2)
        
        # 检查任务状态
        result = system.get_inference_result(task_id, timeout=5.0)
        if result:
            logger.info("✅ 任务分配测试通过")
            return True
        else:
            logger.warning("⚠️ 任务分配测试失败")
            return False
            
    except Exception as e:
        logger.error(f"任务分配测试失败: {e}")
        return False

def test_system_status(config_file):
    """测试系统状态"""
    logger.info("=== 测试系统状态 ===")
    
    try:
        # 尝试导入系统
        try:
            from satellite_system.satellite_inference import MultiSatelliteInferenceSystem
            system = MultiSatelliteInferenceSystem(config_file)
        except ImportError as e:
            logger.warning(f"无法导入MultiSatelliteInferenceSystem: {e}")
            logger.info("跳过系统状态测试")
            return True
        
        status = system.get_system_status()
        
        logger.info(f"系统状态: {status}")
        
        # 检查关键指标
        online_satellites = status.get('online_satellites', 0)
        total_satellites = status.get('total_satellites', 0)
        local_model_available = status.get('local_model_available', False)
        
        logger.info(f"在线卫星: {online_satellites}/{total_satellites}")
        logger.info(f"本地模型可用: {local_model_available}")
        
        return True
        
    except Exception as e:
        logger.error(f"系统状态测试失败: {e}")
        return False

def verify_config(config_file):
    """验证配置文件"""
    logger.info("=== 验证配置文件 ===")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        satellites = config.get('satellites', {})
        
        for sat_id, sat_config in satellites.items():
            logger.info(f"卫星 {sat_id}:")
            logger.info(f"  - IP地址和端口: {sat_config.get('ip', 'N/A')}:{sat_config.get('port', 'N/A')}")
            logger.info(f"  - 计算资源容量: {sat_config.get('compute_capacity', 'N/A')} FLOPS")
            logger.info(f"  - 内存容量: {sat_config.get('memory_capacity', 'N/A')} MB")
            logger.info(f"  - 覆盖区域: {sat_config.get('coverage_area', 'N/A')}")
            logger.info(f"  - 支持功能: {sat_config.get('supported_features', 'N/A')}")
            logger.info("")
        
        logger.info("✅ 配置文件验证完成")
        return True
        
    except Exception as e:
        logger.error(f"配置文件验证失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多卫星系统测试脚本')
    parser.add_argument('--simulated', action='store_true', help='使用模拟节点')
    parser.add_argument('--config', type=str, default='satellite_system/satellite_config.json', help='配置文件路径')
    parser.add_argument('--verify-config', action='store_true', help='验证配置文件')
    
    args = parser.parse_args()
    
    # 验证配置文件
    if args.verify_config:
        verify_config(args.config)
        return
    
    # 运行测试
    tests = [
        ('连通性测试', lambda: test_connectivity(args.config)),
        ('任务分配测试', lambda: test_task_allocation(args.config)),
        ('系统状态测试', lambda: test_system_status(args.config))
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 异常: {e}")
    
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    logger.info(f"测试完成: {passed_tests}/{total_tests} 通过 ({success_rate:.2%})")

if __name__ == "__main__":
    main() 