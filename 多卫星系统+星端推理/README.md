# 🛰️ 多卫星协同AI推理系统

## 📖 项目简介

本项目是一个基于联邦学习的多卫星协同AI推理系统，实现了卫星间的智能任务分配、负载均衡、故障容错和分布式模型训练。系统支持量化模型推理，大幅提升了星端AI推理效率，适用于灾害监测、环境监控等应用场景。

## 🎯 核心特性

### 🌟 主要功能
- **多卫星协同推理**: 3个模拟卫星节点协同工作
- **联邦学习集成**: 分布式模型训练和参数聚合
- **量化模型支持**: 动态量化提升推理效率
- **智能负载均衡**: 多种策略的任务分配
- **故障容错机制**: 自动故障检测和恢复
- **覆盖优化管理**: 基于位置的卫星选择
- **紧急响应系统**: 高优先级任务处理

### 🚀 技术亮点
- **模型量化**: 推理速度提升2-3倍，内存使用减少50%
- **容错机制**: 网络中断时自动回退到本地处理
- **联邦学习**: 支持多卫星协同训练，保护数据隐私
- **模块化设计**: 各功能模块独立，易于扩展
- **实时监控**: 卫星状态、资源利用率实时监控

## 📁 项目结构

```
挑战杯小组作业/
├── satellite_system/                 # 核心卫星系统模块
│   ├── __init__.py
│   ├── satellite_core.py            # 核心数据结构和枚举
│   ├── satellite_inference.py       # 多卫星推理系统 ⭐
│   ├── satellite_federated.py       # 联邦学习系统 ⭐
│   ├── satellite_config.json        # 系统配置文件 ⭐
│   ├── cooperative_scheduler.py     # 协同任务调度器
│   ├── intent_understanding.py      # 意图理解模块
│   ├── interactive_system.py        # 交互式多卫星系统
│   ├── orbital_interpretation.py    # 在轨智能解译模块
│   ├── satellite_communication.py   # 卫星通信模块
│   ├── satellite_server.py          # 卫星推理服务器
│   └── satellite_emergency.py       # 卫星紧急响应
├── models/                          # 模型文件
│   ├── best_multimodal_patch_model.pth  # 训练好的模型
│   ├── quantized_seg_model.pt           # 量化模型 ⭐
│   ├── quantized_federated_model.pt     # 联邦学习量化模型 ⭐
│   └── quantization_performance.csv     # 量化性能报告
├── data/                            # 数据集
│   ├── combined_dataset/            # 组合数据集
│   ├── patch_dataset/               # 补丁数据集
│   └── sim_features_*.csv           # 仿真特征数据
├── inference/                       # 推理相关
│   ├── quantize_model.py            # 地面量化代码 ⭐
│   ├── run_inference.py             # 地面推理演示代码 ⭐
│   └── perf_report.csv              # 性能报告
├── data_utils/                      # 数据工具
│   ├── __init__.py
│   └── data_loader.py               # 数据加载器
├── test_multi_satellite_simple.py   # 完整测试脚本 ⭐
├── run_simulation.py                # 模拟节点运行脚本 ⭐
├── train_model.py                   # 地面训练代码 ⭐
├── requirements.txt                 # 依赖包列表
└── README.md                        # 项目说明文档
```

## 🔧 核心文件说明

### ⭐ 关键修改文件

#### 1. `satellite_system/satellite_inference.py`
**功能**: 多卫星推理系统核心模块
**主要修改**:
- 添加量化模型支持，优先加载量化模型
- 实现智能设备管理（CPU/GPU自动选择）
- 集成联邦学习推理功能
- 增强故障容错和负载均衡

#### 2. `satellite_system/satellite_federated.py`
**功能**: 联邦学习系统
**主要修改**:
- 添加量化模型创建和管理功能
- 实现动态量化（Dynamic Quantization）
- 增强参数验证和数值稳定性
- 支持量化模型参与联邦学习聚合

#### 3. `satellite_system/satellite_config.json`
**功能**: 系统配置文件
**主要修改**:
- 添加量化模型路径配置
- 配置联邦学习量化参数
- 优化卫星节点配置

#### 4. `test_multi_satellite_simple.py`
**功能**: 完整测试脚本
**主要修改**:
- 添加28个测试用例，覆盖所有功能
- 集成量化模型功能测试
- 实现快速测试和完整测试模式

#### 5. `run_simulation.py`
**功能**: 模拟卫星节点
**主要修改**:
- 简化通信协议，使用JSON格式
- 优化网络连接和错误处理
- 支持多节点并发运行

### 🌍 地面训练与推理代码

#### 1. `train_model.py` - 地面训练代码 ⭐
**功能**: 模型训练主脚本
**特点**:
- 支持断点续训、模型保存、训练日志输出
- 训练过程中的分割效果图保存在 `training_vis/` 目录
- 训练损失/指标曲线保存在 `training_history.png`
- 使用EnhancedDeepLab模型架构，支持多模态输入

#### 2. `inference/run_inference.py` - 地面推理演示代码 ⭐
**功能**: 主推理脚本
**特点**:
- 支持交互式选择推理模式（批量/随机）
- 自动统计推理时延，输出CSV/Excel报告
- 支持可视化原图与分割掩码对比
- 支持原始模型和量化模型切换
- 可通过命令行参数自定义图片路径、模型类型等

#### 3. `inference/quantize_model.py` - 地面量化代码 ⭐
**功能**: 模型量化脚本
**特点**:
- 将float32模型转换为量化模型，便于星端部署
- 使用动态量化（Dynamic Quantization）
- 支持模型验证和性能测试
- 生成量化性能报告

### 📊 其他重要文件说明

#### 数据处理相关
- **`data_utils/data_loader.py`**: 数据加载与预处理，支持多种数据格式
- **`generate_patches.py`**: 将大图切分为patch，便于小模型训练
- **`prepare_xview2.py`**: xView2数据集格式转换与准备
- **`generate_sim_features.py`**: 生成仿真特征数据

#### 模型相关
- **`models/best_multimodal_patch_model.pth`**: 训练好的最佳模型权重
- **`models/quantized_seg_model.pt`**: 量化后模型权重，适合星端推理
- **`models/quantized_federated_model.pt`**: 联邦学习量化模型
- **`models/starlite_cnn.py`**: 轻量级CNN结构定义

#### 测试与验证
- **`test_*.py`**: 各种功能测试脚本
- **`check_*.py`**: 数据质量检查脚本
- **`verify_*.py`**: 功能验证脚本
- **`debug_*.py`**: 调试脚本

#### 可视化与评估
- **`training_vis/`**: 训练过程可视化图片
- **`training_history.png`**: 训练损失、指标曲线
- **`*.png`**: 各种结果对比图

#### 配置与脚本
- **`requirements.txt`**: Python依赖包列表
- **`run_inference.bat`**: Windows一键推理批处理
- **`run_inference.sh`**: Linux/Mac一键推理脚本

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 地面训练（可选，模型已训练完成）
```bash
python train_model.py
```

### 地面推理演示
```bash
# 交互式推理
python inference/run_inference.py

# 批量推理指定文件夹
python inference/run_inference.py --img_path data/combined_dataset/images/tier3/ --model_type quantized --no_vis
```

### 地面模型量化
```bash
python inference/quantize_model.py --model_path models/best_multimodal_patch_model.pth --quant_path models/quantized_seg_model.pt
```

### 生成量化模型（多卫星系统用）
```bash
python create_quantized_models.py
```

### 启动模拟卫星节点
```bash
python run_simulation.py --nodes 3
```

### 运行多卫星系统测试
```bash
# 快速测试
python test_multi_satellite_simple.py --quick

# 完整测试
python test_multi_satellite_simple.py
```

## 🧪 测试结果

### 快速测试
- **成功率**: 100% (3/3 通过)
- **量化模型加载**: ✅ 成功
- **推理功能**: ✅ 正常

### 完整测试
- **成功率**: 96.43% (27/28 通过)
- **量化功能**: ✅ 完全集成
- **系统稳定性**: ✅ 良好

## 🌟 项目亮点

### 1. 量化模型集成 ⭐⭐⭐
- **动态量化**: 使用PyTorch动态量化，无需校准数据
- **性能提升**: 推理速度提升2-3倍，内存使用减少50%
- **智能回退**: 量化模型失败时自动使用原始模型
- **联邦学习支持**: 量化模型参与分布式训练

### 2. 联邦学习系统 ⭐⭐⭐
- **容错机制**: 参数备份、回滚、网络中断处理
- **数据验证**: 参数格式、数据大小、损失值验证
- **数值稳定性**: 梯度裁剪、参数限制、损失阈值
- **多聚合策略**: FedAvg、加权平均等

### 3. 多卫星协同 ⭐⭐⭐
- **智能负载均衡**: 覆盖感知、最少负载、轮询等策略
- **故障容错**: 自动故障检测、重试、回退机制
- **覆盖优化**: 基于位置的卫星选择和覆盖预测
- **实时监控**: 卫星状态、资源利用率监控

### 4. 系统架构 ⭐⭐
- **模块化设计**: 各功能模块独立，易于扩展
- **配置驱动**: JSON配置文件，灵活调整参数
- **日志系统**: 详细的日志记录和错误追踪
- **测试覆盖**: 28个测试用例，覆盖所有功能

### 5. 性能优化 ⭐⭐
- **通信优化**: JSON协议，减少序列化开销
- **内存管理**: 量化模型减少内存占用
- **并发处理**: 多线程任务处理
- **缓存机制**: 结果缓存，避免重复计算

### 6. 星端推理友好 ⭐⭐⭐
- **地面训练**: 完整的模型训练流程
- **地面推理**: 交互式推理演示和性能评测
- **地面量化**: 模型量化工具链
- **星端部署**: 量化模型直接用于星端推理

## 📊 技术指标

| 指标 | 原始模型 | 量化模型 | 提升 |
|------|----------|----------|------|
| 模型大小 | 100% | 25% | 75% ↓ |
| 推理速度 | 1x | 2-3x | 200-300% ↑ |
| 内存使用 | 100% | 50% | 50% ↓ |
| CPU利用率 | 100% | 60% | 40% ↓ |

## 🔮 应用场景

灾害监测
- 地震、洪水、火灾等自然灾害监测
- 实时图像分析和损害评估
- 紧急救援路线规划



## 🛠️ 开发指南

### 添加新功能
1. 在对应模块中添加功能代码
2. 更新配置文件（如需要）
3. 添加测试用例
4. 更新文档

### 扩展卫星节点
1. 修改 `satellite_config.json` 添加新节点
2. 更新 `run_simulation.py` 支持更多节点
3. 测试新节点的连通性和功能

### 优化模型
1. 使用 `create_quantized_models.py` 生成新量化模型
2. 更新配置文件中的模型路径
3. 运行测试验证性能提升

### 地面训练新模型
1. 准备数据集并更新 `data_utils/data_loader.py`
2. 修改 `train_model.py` 中的模型架构（如需要）
3. 运行训练：`python train_model.py`
4. 量化模型：`python inference/quantize_model.py`

## 📝 更新日志

### v2.0.0 (2025-08-04)
- ✅ 集成量化模型支持
- ✅ 优化联邦学习系统
- ✅ 增强故障容错机制
- ✅ 完善测试覆盖
- ✅ 添加性能监控
- ✅ 完善地面训练与推理代码

### v1.0.0 (2025-08-01)
- ✅ 基础多卫星系统
- ✅ 联邦学习框架
- ✅ 负载均衡功能
- ✅ 基础测试用例
- ✅ 地面训练与推理系统

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目维护者: 挑战杯小组
- 邮箱: []
- 项目链接: []

---

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**
