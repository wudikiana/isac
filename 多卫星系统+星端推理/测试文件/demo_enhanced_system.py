#!/usr/bin/env python3
"""
增强版多卫星推理系统演示脚本
展示分布式训练、覆盖范围管理和负载均衡功能
"""

import numpy as np
import time
import logging
from enhanced_multi_satellite_inference import MultiSatelliteInferenceSystem

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_basic_functionality():
    """演示基本功能"""
    print("\n=== 增强版多卫星推理系统演示 ===")
    
    # 创建系统实例
    system = MultiSatelliteInferenceSystem("multi_satellite_config.json")
    
    # 发现卫星
    system.discover_satellites()
    
    # 模拟卫星位置更新
    print("\n1. 更新卫星位置...")
    system.update_satellite_position("sat_001", [39.9, 116.4, 500])  # 北京
    system.update_satellite_position("sat_002", [31.2, 121.5, 500])  # 上海
    system.update_satellite_position("sat_003", [23.1, 113.3, 500])  # 广州
    
    # 测试覆盖范围选择
    print("\n2. 测试覆盖范围选择...")
    beijing_location = [39.9, 116.4]
    optimal_satellite = system.get_optimal_satellite_for_location(beijing_location)
    print(f"北京地区最优卫星: {optimal_satellite}")
    
    # 提交推理任务
    print("\n3. 提交推理任务...")
    image_data = np.random.rand(3, 256, 256).astype(np.float32)
    sim_features = np.random.rand(11).astype(np.float32)
    
    task_id = system.submit_inference_task(
        image_data=image_data,
        sim_features=sim_features,
        priority=5,
        location=beijing_location
    )
    print(f"任务ID: {task_id}")
    
    # 获取系统状态
    print("\n4. 系统状态:")
    status = system.get_system_status()
    print(f"在线卫星: {status['online_satellites']}/{status['total_satellites']}")
    print(f"队列大小: {status['queue_size']}")
    
    # 模拟分布式训练
    print("\n5. 模拟分布式训练...")
    import torch
    mock_parameters = {
        'conv1.weight': torch.randn(64, 3, 7, 7),
        'conv1.bias': torch.randn(64)
    }
    system.sync_model_parameters("sat_001", mock_parameters)
    print("已同步卫星 sat_001 的模型参数")
    
    print("\n演示完成！")

if __name__ == "__main__":
    demo_basic_functionality() 