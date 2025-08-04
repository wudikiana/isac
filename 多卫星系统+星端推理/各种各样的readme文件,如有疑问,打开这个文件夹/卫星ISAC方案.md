# 卫星边缘计算 ISAC 1 个月冲刺方案

## 1. 项目概览
+ > 本节旨在快速描绘项目背景、核心目标与整体技术路线，帮助新成员在开始动手前对"我们要做什么"形成统一认知。
面向低空智联网应用，以低轨卫星作为空基边缘节点，构建通感算智一体化（ISAC）示范系统。通过 **Python + ns-3/PyTorch** 快速仿真通信-感知联合波形，并使用 **深度强化学习 (Proximal Policy Optimization，PPO)** 优化任务卸载、功率与子载波分配。

## 2. 团队角色与分工
+ > 通过明确角色与职责，确保任务分派清晰、沟通链路最短、资源利用最高。
| 角色 |  | 主要职责 |
|------|------|----------|
| 项目经理/报告 | | 统筹进度、质量；撰写技术报告、PPT |
| 信道&雷达建模 | | 构建 `sat_channel.py`、`radar_echo.py`，生成仿真数据 |
| DRL 开发 | | 编写 Gym 环境 `isac_sat_env.py`、训练 `train_ppo.py` |
| 星端推理 | | 轻量 CNN 模型量化、测试推理延时/功耗 |
| DevOps & 可视化 | | 环境脚本、CI、Plotly/MPL 可视化 |

## 3. 技术栈
+ > 汇总项目全周期将用到的语言、框架与工具包，便于同学们一次性装好依赖，避免环境踩坑。
- Python 3.11
- PyTorch 2.x
- ns-3 5G/LTE 模块（或 Sionna，如果熟悉 TF）
- Gymnasium & stable-baselines3
- Optuna + WandB
- Matplotlib / Plotly

## 4. 环境搭建速查表
+ > 提供从克隆仓库到跑通单元测试的"最短路径"，将环境问题耗时降到最低。
1. 克隆仓库并进入目录
   ```bash
   git clone <repo_url>
   cd repo
   ```
2. 创建 Conda 环境
   ```bash
   conda create -n isac python=3.11 -y
   conda activate isac
   ```
3. 安装依赖
   ```bash
   pip install -r requirements.txt
   bash setup_env.sh   # 编译 ns-3 等
   ```
4. 运行单元测试
   ```bash
   pytest tests/ -q
   ```

## 5. 目录结构说明
+ > 说明代码与文档的组织方式，让团队能够快速定位并修改对应模块，提高协作效率。
```
repo/
 ├ env/                  # 仿真与环境
 │   ├ sat_channel.py    # 卫星信道模型
 │   ├ radar_echo.py     # 雷达回波模型
 │   └ isac_sat_env.py   # Gym 环境封装
 ├ models/
 │   ├ ppo_policy.py     # Actor-Critic 网络
 │   ├ mobilenetv3_quant.pt # 量化模型权重
 │   └ baseline_rule.py  # 贪心基线
 ├ train/
 │   ├ train_ppo.py      # 训练入口
 │   └ evaluate.py       # 性能评估脚本
 ├ utils/
 │   ├ frame_cfg.json    # 联合帧结构参数
 │   └ orbit_tle.txt     # 卫星轨道 TLE
 ├ results/              # 自动生成
 ├ report/               # LaTeX 报告
 └ setup_env.sh          # 安装脚本
```

## 6. 时间线与每日任务
+ > 将 1 个月目标拆解为可执行的每日里程碑，便于跟踪进度并及时调整策略。
### Week0（0.5 天）
+ > **快速启动周**：建立协作工具链与需求文档，确保团队对赛题背景、评审重点有统一理解，为后续开发奠定方向和规范。
- 创建 Git、Issue Board，阅读赛题 → 需求规格 v0.1

### Week1（调研 & 环境）
+ > **调研与环境周**：完成关键论文阅读与环境搭建，确保所有成员具备必要的理论基础并拥有可运行的开发环境。
| Day | 任务 | 交付物 |
|-----|------|-------|
| 1-2 | 安装环境、阅读 4 篇核心论文 | `setup_env.sh` 成功运行, Reading Notes |
| 3   | 完成 `sat_channel.py` (单链路 SNR 曲线) | `fig_snr.png` |
| 4   | 完成 `radar_echo.py` (验证 Range FFT) | `fig_range.png` |
| 5   | 设计 `frame_cfg.json`、更新系统设计 PPT | `design_v0.2.pptx` |

### Week2（基线实现）
+ > **基线落地周**：实现端到端可运行的基线系统，产生可量化的初步结果，为后续算法优化提供对照组。
| Day | 任务 | 交付物 |
|-----|------|-------|
| 6-7 | 编写 `isac_sat_env.py` (obs/action/reward) | 环境通过 `env.check()` |
| 8   | 实现 `baseline_rule.py`，跑 100 episodes | `baseline_results.csv` |
| 9-10| 可视化脚本 `plot_metrics.py`，单元测试覆盖率 80% | `coverage.xml`, `fig_baseline.png` |

+ #### 基线算法说明
+ > 为保证后续算法改进的有效性，本项目提供两种**极易实现**的基准策略，帮助快速跑通流程并形成性能对照。
+
+ | 策略名称 | 思想 | 关键实现片段 | 预期效果 |
+ |----------|------|--------------|-----------|
+ | `RandomPolicy` | 对功率、子载波与卸载比例进行均匀随机选取 | `action = env.action_space.sample()` | 提供最低参考线，验证 DRL 收敛趋势 |
+ | `GreedyPolicy` | 功率固定 25 dBW；按照终端 CQI 降序逐个分配子载波；任务全部卸载到地面 | 参考 `baseline_rule.py` | 快速获得可用链路 & 感知结果，易于实现 |
+
+ 运行示例：
+
+ ```bash
+ # 随机策略评估 100 episodes
+ python train/evaluate.py --policy random --episodes 100 \
+        --output results/random_baseline.csv
+
+ # 贪心策略评估 100 episodes
+ python train/evaluate.py --policy greedy --episodes 100 \
+        --output results/greedy_baseline.csv
+ ```
+
+ 评估脚本将自动输出吞吐/感知精度/能耗/时延四指标曲线，用于与 PPO 结果进行对比。

### Week3（算法优化）
+ > **性能提升周**：引入 DRL、星端推理与多卫星协同等高级特性，大幅提升系统性能并突出项目创新点。
| Day | 任务 | 交付物 |
|-----|------|-------|
| 11-12 | 实现 PPO (`ppo_policy.py`+`train_ppo.py`) | `ppo_logs/` WandB charts |
| 13 | 量化 MobileNetv3，记录推理时延/功耗表 | `table_infer.xlsx` |
| 14 | 加多卫星场景 & 公平性指标 | `env_multisat.patch` |
| 15 | Optuna 超参搜索 50 次 | `optuna_best.json` |

### Week4（集成 & 报告）
+ > **冲刺收官周**：整合全部模块，完成大规模实验与报告/演示，确保成果在评审环节一举拿高分。
| Day | 任务 | 交付物 |
|-----|------|-------|
| 16-17 | 批量仿真 1e4 episodes，生成指标表 | `results/table.xlsx` |
| 18 | 撰写报告、录制 2 分钟演示 | `report/main.pdf`, `demo.mp4` |
| 19 | 内部答辩 & 风险兜底脚本 | `fallback.sh` |
| 20 | 最终提交、代码 tag v1.0 | Git Tag |

## 7. 快速开始示例
+ > 给出最小可运行命令，让同学们"跑起来"而非只停留在阅读阶段，增强成就感。
```bash
# 训练 1000 episodes
python train/train_ppo.py --env isac_sat --total_timesteps 1e6 \
       --config utils/frame_cfg.json --log_dir results/ppo_logs

# 评估模型
python train/evaluate.py --model_path results/ppo_logs/best.ckpt \
       --episodes 200
```

## 8. 推荐阅读与资源
+ > 汇聚论文、工具与数据链接，为深入研究与性能提升提供可靠参考。
1. **论文**
一、通感一体化 + LEO 卫星
1) Integrated Sensing and Communications Enabled Low-Earth-Orbit Satellite Systems
PDF: https://arxiv.org/pdf/2304.00941.pdf
亮点：系统梳理了"LEO-ISAC"链路预算、波束设计、同步机制；阅读后即可着手我们方案中的联合波形模块。
备用：Edge-AI/ISAC 章节较新的综述 https://arxiv.org/pdf/2211.07932.pdf
二、LEO 卫星边缘计算（体系/架构综述）
2) A Comprehensive Survey on Orbital Edge Computing: Systems, Applications, and Algorithms
PDF: https://arxiv.org/pdf/2306.00275.pdf
亮点：按通讯-计算-智能三个维度总结 OEC 现状，对应我们方案中的「系统架构」「资源调度」两大板块。
备用：Celestial: Virtual Software System Testbeds for the LEO Edge（仿真平台）https://arxiv.org/pdf/2207.08037.pdf
三、PPO 在无线资源分配中的应用（可直接复现代码）
3) Proximal Policy Optimization for Integrated Sensing and Communication in mmWave Systems
PDF: https://arxiv.org/pdf/2306.15429.pdf
亮点：给出完整的 PPO 环境/网络结构/奖励设计范例，改一下状态与动作即可迁移到卫星卸载场景。
备用：Distributed PPO for Contention-Based Spectrum Access https://arxiv.org/pdf/2111.09420.pdf
四、星载轻量级视觉模型 / MobileNetV3 相关
4) An FPGA-Based Hardware Accelerator for CNN Inference on-Board Satellites: The CloudScout Case Study
PDF: https://arxiv.org/pdf/2104.15118.pdf
亮点：介绍了在 Φ-Sat-1 上部署 MobileNetV3-Tiny 的全过程（量化 + 推理功耗），对应我们方案中的「星端推理」子任务。
备用：Edge Computing in Low-Earth Orbit — What Could Possibly Go Wrong?（§4.2 有星载 SoC / 辐照讨论）https://arxiv.org/pdf/2302.08952.pdf
2. **代码/工具**
   - Sionna (NVIDIA) – 全链路通信模拟
   - satellite-toolbox-python – 轨道计算
   - WandB – 实验跟踪
3. **数据集**
   - ISAC-Sim Open Dataset（波形 + 雷达回波）
   - COCO 2017 (mini) – 星端推理示例

## 9. FAQ
+ > 收录开发中最常遇到的疑难杂症及解决方案，减少重复沟通与搜索时间。
- **Q：DRL 训练不收敛？**  
  A：先训练 rule-based 策略作为 mentor，使用 imitation learning 预热。
- **Q：ns-3 编译慢？**  
  A：开启 `ccache`，并在 `setup_env.sh` 中 use `-j$(nproc)`。

## 10. 项目特色与得分点
+ > 本节总结项目在技术创新、工程可行与竞赛契合度方面的亮点，并对照评审维度阐述如何拿高分。

### 10.1 技术/创新亮点
1. **通感一体化波形**：采用 OFDM-LFM 复用技术，实现通信与雷达感知在同一频谱高效共存，充分展示 ISAC 核心价值。
2. **卫星边缘计算**：在星端执行 MobileNetV3-Small 量化推理，减少原始数据回传，下行带宽节省 ≥ 80%。
3. **智能资源调度 (PPO)**：多卫星-多终端场景下联合优化功率、子载波与卸载比例，提升系统效能 20%+。
4. **模块化仿真框架**：`env/-models/-train/` 清晰分层，评委可一键验证并复现结果。
5. **可重复实验**：配套 Optuna 超参日志、WandB Dashboard 与 Dockerfile，保证评测环境一致性。

### 10.2 可能的得分点（示例拆分）
| 评审维度 | 比赛权重* | 本项目优势 | 高分策略 |
|---------|-----------|-----------|-----------|
| 技术创新 | 30% | ISAC 波形 + DRL 调度 + 星端 AI | 在报告中突出"三位一体"创新链；增加对比实验 | 
| 工程实现 | 25% | 模块化代码 + 自动化脚本 | 演示一键运行、CI 流水线截图 |
| 性能指标 | 25% | 吞吐↑ 感知精度↑ 能耗↓ 时延↓ | 提供与 Baseline 对比的统计显著性分析 |
| 可复现性 | 10% | Docker + WandB + README | 附赠 `reproduce.sh` 与日志链接 |
| 展示与答辩 | 10% | 可视化动画 + 简洁报告 | 2 分钟动画讲故事，凸显应用价值 |
*注：实际权重以官方评分细则为准。

---
*更新日期：2025-07-05* 

+ **分工执行建议**
+ | 周次 | PM/报告 | 信道&雷达 | DRL 开发 | 星端推理 | DevOps/可视化 |
+ |------|---------|-----------|-----------|-----------|----------------|
+ | Week0 | 创建仓库与需求文档 |  |  |  | 初始化 CI/CD |
+ | Week1 | 每日 Stand-up 记录 | 完成 `sat_channel.py` & `radar_echo.py` | 环境安装脚本沟通 | 协助环境测试 | 打包 Conda & ns-3 镜像 |
+ | Week2 | 审核 Baseline 报告 | 支持信道数据生成 | 实现 `isac_sat_env.py`+`baseline_rule.py` | 准备 MobileNet 数据集 | 绘制 Baseline 曲线 |
+ | Week3 | 跟踪实验里程碑 | 优化仿真效率 | 训练 PPO + Optuna | 量化 & 推理测试 | 维护 WandB Dashboard |
+ | Week4 | 汇总报告 & PPT | Review 模型 | 整理实验日志 | 推理性能对比图 | 合并 Release 分支 & 打标签 |
+ 
+ 通过表格可见，每个角色在各周都有明确 Deliverable，保证并行推进且责任清晰。 

## 11. 典型应用场景：暴雨山洪 & 滑坡快速监测
+ > 通过一个具象化的应急场景，展示本方案在真实业务中的闭环价值，为技术评委和产业评委提供直观参考。
+
+ ### 11.1 背景痛点
+ - 南方山区极端降雨频发，山洪暴发与滑坡常导致通信中断、道路阻断，传统直升机初勘成本高、时效差。  
+ - 现有高空无人机或地面微波链路受天气和地形限制，灾情信息回传延迟≥30 min，错过黄金救援窗口。
+
+ ### 11.2 本方案 10 min 星-地闭环流程
+ ```text
+  T-0 min  暴雨触发地面水文站告警
+  T+1 min  LEO 卫星接收 S 波束紧急信标（覆盖 500 km×500 km 区域）
+  T+2 min  ISAC 波形切换至"窄带通信 + SAR 成像"模式，
+           星载 SoC 立即执行卷积推理
+  T+6 min  PPO 调度判断滑坡概率>80%，
+           仅上传 30 MB 结果切片（节省 92% 下行链路）
+  T+8 min  省应急厅通过 5G-A NTN 终端收到灾害矢量图与最佳进场路线
+  T+10 min 救援队携便携基站抵达，现场通信与指挥恢复
+ ```
+
+ ### 11.3 关键指标对比（仅示例）
+ | 指标 | 传统方案 | 本方案 | 提升 |
+ |------|----------|--------|------|
+ | 灾情告警时延 | ≈30 min | ≤10 min | ↓67% |
+ | 上行数据量 | 400 MB | 30 MB | ↓92% |
+ | 星端算力利用率 | 38% | 74% | ↑36 pts |
+ | 现场 5G-A 覆盖延迟 | ≥20 min | 8 min | ↓60% |
+
+ ### 11.4 社会与经济效益（示例）
+ - 缩短黄金救援窗口 ≥20 min，可挽救生命并减少经济损失。  
+ - 单次灾害现场节省直升机与人力成本约 15 万元。  
+ - 场景可扩展至森林火情早期识别、海上风暴潮监测等多元业务。
+
+ ## 12. 主办方价值对标（含灾害场景卖点）
+ > 结合中国信科集团 / 中信科移动"通感算智"战略需求，量化本方案的落地价值。
+
+ | 主办方关注点 | 对应方案亮点 | 预期收益 |
+ |---------------|--------------|-----------|
+ | ISAC 一体化标杆 | 卫星同时完成通信、SAR 感知、星端推理与 DRL 调度 | 打造业内首个 10 min 灾害闭环 Demo，树立集团技术品牌 |
+ | 5G-A/6G NTN 预研 | 全流程基于 3GPP Rel-18 NTN 信令，可对接中信科移动 5G-A 基带 | 提供标准化接口与实测 KPI，支撑后续商用星座 |
+ | 国家低空经济 & 应急通信 | 快速恢复灾区通信 + 智能灾害识别 | 进入应急专网与无人机通信市场，扩大集团行业版图 |
+ | 可验证 / 可产业化 | ns-3+PyTorch 开源脚本 + 星端昇腾 NPU 参考设计 | 评委可一键复现，企业可二次开发，降低技术门槛 |
+ | 自主可控 | 波形、算法、模型全栈国产替代 | 符合央企安全可控与社会责任要求 |
+ | 经济拉动 | "星算力即服务 (SCaaS)" + 灾害监测订阅 | 300 星规模年新增产值≈4 亿元，带动芯片/终端/服务生态 |
+
+ ---
+ *本章节材料可直接用于申报书"应用场景与项目意义"及路演 PPT，可根据最新仿真结果微调数值。* 