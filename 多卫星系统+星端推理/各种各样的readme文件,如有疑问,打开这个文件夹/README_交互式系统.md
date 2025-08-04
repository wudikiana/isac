# 交互式多卫星系统使用说明

## 快速开始

### 启动系统
```bash
# 交互模式
python interactive_satellite_system.py

# 演示模式
python interactive_satellite_system.py --demo
```

### 主要命令

#### 主菜单
- `help` - 显示帮助
- `status` - 系统状态
- `emergency` - 应急响应模式
- `satellite` - 卫星管理模式
- `monitoring` - 监控模式
- `inference` - 推理模式
- `exit` - 退出

#### 应急响应
- `trigger 3 39.9 116.4 地震灾害` - 触发紧急情况
- `list` - 列出紧急情况
- `status` - 应急状态

#### 卫星管理
- `list` - 卫星列表
- `status sat_001` - 卫星状态
- `control sat_001 restart` - 控制卫星
- `orbit sat_001` - 轨道信息

#### 推理任务
- `submit image.jpg` - 提交任务
- `result task_id` - 获取结果
- `queue` - 任务队列

## 系统功能

✅ **灾害应急响应** - PPO强化学习资源分配  
✅ **星间联邦学习** - 分布式模型训练  
✅ **认知无线电** - 动态频谱管理  
✅ **自主轨道控制** - 编队维持和碰撞避免  
✅ **实时监控** - 系统状态监控  
✅ **推理任务** - 分布式AI处理  

## 配置文件

系统使用 `advanced_satellite_config.json` 配置文件，包含：
- 卫星配置
- 应急响应参数
- 联邦学习设置
- 轨道控制参数

## 故障排除

1. 检查配置文件是否存在
2. 确认依赖包已安装
3. 查看日志输出
4. 检查网络连接

---

**版本**: v2.0 | **支持**: 灾害应急、联邦学习、认知无线电、轨道控制 