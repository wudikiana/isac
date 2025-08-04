# 卫星系统 v2.0

## 概述

这是一个重构后的高级多卫星系统，整合了AI推理、灾害应急响应、星间联邦学习、认知无线电和自主轨道控制等功能。

## 系统架构

```
satellite_system/
├── __init__.py              # 包初始化文件
├── satellite_core.py        # 核心模块（枚举、数据结构、工具函数）
├── satellite_inference.py   # 推理系统（负载均衡、故障容错）
├── satellite_emergency.py   # 应急系统（PPO强化学习、灾害响应）
├── satellite_communication.py # 通信系统（认知无线电、频谱管理）
├── satellite_orbit.py       # 轨道系统（自主控制、碰撞避免）
├── satellite_federated.py   # 联邦学习系统（分布式训练、参数同步）
├── main_system.py          # 主系统（整合所有功能）
├── interactive_system.py   # 交互式系统（用户界面）
├── satellite_server.py     # 卫星推理服务器
├── start_system.py         # 系统启动脚本
├── test_system.py          # 系统测试脚本
├── satellite_config.json   # 系统配置文件
└── README.md              # 本文件
```

## 主要功能

### 1. AI推理系统
- **多卫星负载均衡**：覆盖感知、最少负载、轮询等策略
- **故障容错**：自动故障检测和任务重分配
- **本地备份**：卫星不可用时使用本地模型处理

### 2. 灾害应急响应
- **PPO强化学习**：智能资源分配和调度
- **协同调度**：多卫星协同执行SAR成像和计算任务
- **实时响应**：紧急情况快速响应和处理

### 3. 星间联邦学习
- **分布式训练**：多卫星协同训练AI模型
- **参数同步**：定期同步和聚合模型参数
- **隐私保护**：本地训练，只共享参数

### 4. 认知无线电
- **动态频谱接入**：根据地面用户占用情况动态选择频段
- **干扰避免**：最小化与地面用户的干扰
- **自适应调制**：根据信道条件选择最优调制方式

### 5. 自主轨道控制
- **编队维持**：多卫星编队飞行控制
- **碰撞避免**：空间碎片检测和避障
- **燃料优化**：最优燃料消耗的机动规划

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision
pip install segmentation-models-pytorch
pip install numpy scipy
pip install psutil  # 可选，用于系统监控
```

### 2. 配置文件

确保 `satellite_config.json` 配置正确：

```json
{
  "satellites": {
    "sat_001": {
      "ip": "192.168.1.101",
      "port": 8080,
      "compute_capacity": 1e12,
      "memory_capacity": 8192,
      "coverage_area": {"lat": [35, 45], "lon": [110, 130]},
      "orbit_period": 90.0
    }
  },
  "emergency_response": {
    "response_timeout": 300,
    "max_concurrent_emergencies": 5
  }
}
```

### 3. 启动系统

#### 方式一：使用启动脚本
```bash
# 启动完整系统
python satellite_system/start_system.py --mode all

# 只启动卫星服务器
python satellite_system/start_system.py --mode servers

# 只启动交互式系统
python satellite_system/start_system.py --mode interactive

# 运行演示
python satellite_system/start_system.py --mode demo
```

#### 方式二：直接启动
```bash
# 启动卫星服务器
python satellite_system/satellite_server.py --satellite_id sat_001 --port 8080

# 启动交互式系统
python satellite_system/interactive_system.py
```

### 4. 测试系统

```bash
# 运行所有测试
python satellite_system/test_system.py

# 运行特定测试
python satellite_system/test_system.py --test inference
python satellite_system/test_system.py --test emergency
```

## 使用示例

### 1. 基本推理任务

```python
from satellite_system import MultiSatelliteInferenceSystem
import numpy as np

# 创建推理系统
system = MultiSatelliteInferenceSystem("satellite_system/satellite_config.json")

# 提交推理任务
image_data = np.random.rand(3, 256, 256).astype(np.float32)
sim_features = np.random.rand(11).astype(np.float32)

task_id = system.submit_inference_task(
    image_data=image_data,
    sim_features=sim_features,
    priority=5,
    location=[39.9, 116.4]  # 北京坐标
)

# 获取结果
result = system.get_inference_result(task_id, timeout=60.0)
if result:
    print(f"推理完成: {result['processing_time']:.3f}s")
```

### 2. 应急响应

```python
from satellite_system import MultiSatelliteSystem, EmergencyLevel

# 创建主系统
system = MultiSatelliteSystem("satellite_system/satellite_config.json")

# 触发紧急情况
emergency_id = system.trigger_emergency(
    location=[39.9, 116.4, 0],  # 北京
    emergency_level=EmergencyLevel.HIGH,
    description="地震灾害"
)

print(f"紧急情况已触发: {emergency_id}")
```

### 3. 交互式操作

```python
from satellite_system import InteractiveSatelliteSystem

# 启动交互式系统
system = InteractiveSatelliteSystem("satellite_system/satellite_config.json")
system.run()

# 在交互界面中可以执行以下命令：
# status          - 显示系统状态
# emergency       - 进入应急模式
# satellite       - 进入卫星管理
# monitoring      - 进入监控模式
# inference       - 进入推理模式
```

## 系统监控

### 1. 系统状态查询

```python
# 获取系统状态
status = system.get_system_status()
print(f"在线卫星: {status['online_satellites']}/{status['total_satellites']}")
print(f"应急队列: {status['emergency_system']['emergency_queue_size']}")
```

### 2. 卫星状态监控

```python
# 获取卫星详细信息
satellites = system.satellites
for sat_id, satellite in satellites.items():
    print(f"{sat_id}: {satellite.status.value}, 负载: {satellite.current_load:.2f}")
```

## 配置说明

### 卫星配置
- `ip`: 卫星IP地址
- `port`: 通信端口
- `compute_capacity`: 计算能力（FLOPS）
- `memory_capacity`: 内存容量（MB）
- `coverage_area`: 覆盖区域坐标
- `orbit_period`: 轨道周期（分钟）

### 应急响应配置
- `response_timeout`: 响应超时时间（秒）
- `max_concurrent_emergencies`: 最大并发紧急情况数
- `ppo_learning_rate`: PPO学习率
- `ppo_clip_ratio`: PPO裁剪比例

### 联邦学习配置
- `sync_interval`: 同步间隔（秒）
- `min_participants`: 最小参与卫星数
- `aggregation_method`: 聚合方法（fedavg/weighted）
- `learning_rate`: 学习率
- `batch_size`: 批次大小

## 故障排除

### 1. 模型加载失败
- 检查模型文件路径是否正确
- 确保模型文件格式兼容
- 检查CUDA环境（如果使用GPU）

### 2. 卫星连接失败
- 检查网络连接
- 确认卫星IP和端口配置
- 检查防火墙设置

### 3. 推理任务超时
- 检查卫星负载情况
- 调整任务超时时间
- 检查网络延迟

### 4. 应急响应失败
- 检查卫星可用性
- 确认PPO模型状态
- 检查资源配置

## 开发指南

### 1. 添加新功能

1. 在相应的模块中添加新类或方法
2. 更新 `__init__.py` 导出新功能
3. 添加相应的测试用例
4. 更新文档

### 2. 扩展卫星类型

1. 在 `satellite_core.py` 中添加新的状态枚举
2. 更新 `SatelliteInfo` 数据结构
3. 修改相关模块的处理逻辑

### 3. 自定义负载均衡策略

1. 在 `satellite_inference.py` 的 `LoadBalancer` 类中添加新策略
2. 实现策略选择逻辑
3. 更新配置文件中的策略选项

## 版本历史

### v2.0.0 (当前版本)
- 重构系统架构，模块化设计
- 增强AI推理功能
- 完善应急响应系统
- 优化联邦学习算法
- 改进认知无线电功能
- 增强轨道控制能力

### v1.0.0
- 基础多卫星系统
- 简单推理功能
- 基本应急响应

## 贡献指南

欢迎提交问题报告和功能请求。如需贡献代码：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目主页：[GitHub Repository]
- 邮箱：[your-email@example.com]
- 问题反馈：[Issues Page]

---

**注意**: 这是一个演示系统，实际部署时需要根据具体需求进行调整和优化。 