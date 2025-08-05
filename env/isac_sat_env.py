# isac_sat_env.py
import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym import spaces
from collections import deque
from typing import Tuple, Dict, Any, Optional, List
from env.sat_channel import SatelliteChannel
from env.radar_echo import RadarEcho
import wandb

class ISAC_SatEnv(gym.Env):
    """集成通信与感知功能的卫星ISAC环境（完整实现）"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super(ISAC_SatEnv, self).__init__()
        
        # 默认配置（已优化）
        self.config = {
            # 公共参数
            "max_steps": 500,
            "total_power": 80.0,  # 总功率(W)
            "total_bandwidth": 300e6,  # 总带宽(Hz)
            
            # 通信参数
            "comm_freq": 12e9,
            "tx_gain": 32.0,  # 发射增益(dB)
            "rx_gain": 28.0,   # 接收增益(dB)
            "comm_noise_temp": 290.0,
            "comm_snr_thresh": 30.0,  # SNR阈值(dB)
            
            # 雷达参数
            "radar_freq": 10e9,
            "radar_gain": 38.0,
            "target_rcs": 1.5,  # 目标截面积(m²)
            "radar_noise_temp": 290.0,
            "radar_snr_thresh": 5.0,
            
            # 目标参数
            "init_distance": 500.0,  # 初始距离(km)
            "target_speed": 7.8,     # 目标速度(km/s)
            "min_distance": 100.0,   # 最小距离(km)
            
            # 多目标参数
            "num_targets": 2,
            
            # 奖励参数
            "comm_reward_weight": 0.5,
            "radar_reward_weight": 0.4,
            "accuracy_weight": 0.1,
            "fairness_weight": 0.1,
            "action_penalty_weight": 0.001,
            
            # 渲染参数
            "render_interval": 50
        }
        
        if config:
            self.config.update(config)

        # 初始化模块
        self.num_targets = self.config["num_targets"]
        self.comm = [self._init_comm_module(i) for i in range(self.num_targets)]
        self.radar = [self._init_radar_module(i) for i in range(self.num_targets)]

        # 动作空间
        self.action_space = spaces.Box(
            low=0.1, high=0.9, shape=(self.num_targets, 2), dtype=np.float32
        )

        # 观测空间
        obs_low = np.array(
            [-30] * self.num_targets +  # 通信SNR
            [-30] * self.num_targets +  # 雷达SNR
            [0] +  # 归一化步数
            [0.1] * self.num_targets * 2  # 上一个动作
        )
        obs_high = np.array(
            [50] * self.num_targets + 
            [50] * self.num_targets + 
            [1] + 
            [0.9] * self.num_targets * 2
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 环境状态
        self.current_step = 0
        self.target_distance = [self.config["init_distance"]] * self.num_targets
        self.last_action = np.array([[0.5, 0.5]] * self.num_targets)
        self.history = deque(maxlen=1000)
        self.wandb_enabled = False

    def _init_comm_module(self, target_idx: int) -> SatelliteChannel:
        """初始化通信模块"""
        return SatelliteChannel({
            "frequency": self.config["comm_freq"],
            "tx_gain": self.config["tx_gain"],
            "rx_gain": self.config["rx_gain"],
            "noise_temp": self.config["comm_noise_temp"],
            "comm_snr_thresh": self.config["comm_snr_thresh"]
        })

    def _init_radar_module(self, target_idx: int) -> RadarEcho:
        """初始化雷达模块"""
        return RadarEcho({
            "radar_freq": self.config["radar_freq"],
            "radar_gain": self.config["radar_gain"],
            "target_rcs": self.config["target_rcs"],
            "noise_temp": self.config["radar_noise_temp"],
            "radar_snr_thresh": self.config["radar_snr_thresh"]
        })

    def enable_wandb_logging(self, config=None):
        """启用WandB日志记录"""
        if not wandb.run:
            wandb.init(project="satellite-isac", config=config or self.config)
        self.wandb_enabled = True

    def reset(self) -> np.ndarray:
        """重置环境状态"""
        self.current_step = 0
        self.target_distance = [self.config["init_distance"]] * self.num_targets
        self.last_action = np.array([[0.5, 0.5]] * self.num_targets)
        self.history.clear()
        
        for module in self.comm + self.radar:
            module.current_snr = 0.0
            
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一个时间步"""
        # 1. 动作处理
        action = np.clip(action, 0.1, 0.9)
        self.last_action = action.copy()
        
        # 2. 资源分配与状态更新
        comm_snrs, radar_snrs, info_targets = self._allocate_resources(action)
        
        # 3. 计算公平性指标
        fairness_comm = max(0, self._jain_index(comm_snrs) - 0.5) * 2
        fairness_radar = max(0, self._jain_index(radar_snrs) - 0.5) * 2
        
        # 4. 计算奖励
        reward = self._calc_reward(comm_snrs, radar_snrs)
        reward += self.config["fairness_weight"] * (fairness_comm + fairness_radar)
        
        # 5. 收集信息（新增吞吐量、感知精度、时延）
        info = {
            "targets": info_targets,
            "fairness_comm": fairness_comm,
            "fairness_radar": fairness_radar,
            "step": self.current_step,
            "avg_throughput": np.mean([t["comm"]["throughput"] for t in info_targets]),
            "avg_detection_prob": np.mean([t["radar"]["detection_prob"] for t in info_targets]),
            "avg_latency": np.mean([t["comm"]["latency"] for t in info_targets])
        }
        self.history.append(info)
        self.current_step += 1
        
        # 6. 终止条件
        done = self.current_step >= self.config["max_steps"]
        
        # 7. 记录日志
        if self.wandb_enabled:
            self._log_to_wandb(reward, info)
        
        return self._get_obs(), reward, done, info

    def _allocate_resources(self, action: np.ndarray) -> Tuple[List[float], List[float], List[Dict]]:
        comm_snrs = []
        radar_snrs = []
        info_targets = []
        
        total_power = self.config["total_power"]
        total_bw = self.config["total_bandwidth"]
        min_distance = self.config["min_distance"]

        for i in range(self.num_targets):
            # 1. 资源分配
            comm_power = total_power * action[i, 0] / self.num_targets
            radar_power = total_power * (1 - action[i, 0]) / self.num_targets
            comm_bw = total_bw * action[i, 1] / self.num_targets
            radar_bw = total_bw * (1 - action[i, 1]) / self.num_targets
            
            # 2. 更新距离
            self.target_distance[i] += self.config["target_speed"] * (1 + 0.05*np.random.randn())
            self.target_distance[i] = max(min_distance, self.target_distance[i])
            
            # 3. 计算SNR（会自动更新detection_prob）
            comm_snr = np.clip(
                self.comm[i].update(self.target_distance[i], comm_power, comm_bw),
                -30, 50
            )
            radar_snr = np.clip(
                self.radar[i].update(self.target_distance[i], radar_power, radar_bw),
                -30, 50
            )
            
            # 4. 记录信息（添加detection_prob和range_resolution）
            info_targets.append({
                "comm": {
                    "snr": comm_snr,
                    "power": comm_power,
                    "bandwidth": comm_bw,
                    "threshold": self.config["comm_snr_thresh"],
                    "throughput": self.comm[i].throughput / 1e6,
                    "latency": 10 * (1 - np.clip(action[i, 0], 0.1, 0.9))
                },
                "radar": {
                    "snr": radar_snr,
                    "power": radar_power,
                    "bandwidth": radar_bw,
                    "threshold": self.config["radar_snr_thresh"],
                    "detection_prob": self.radar[i].detection_prob,  # 新增检测概率
                    "range_resolution": self.radar[i].get_range_resolution(radar_bw)  # 新增距离分辨率
                },
                "distance": self.target_distance[i],
                "target_id": i
            })
            comm_snrs.append(comm_snr)
            radar_snrs.append(radar_snr)
            
        return comm_snrs, radar_snrs, info_targets

    def _calc_reward(self, comm_snrs, radar_snrs):
        """工业级稳健奖励函数（强制数值范围）"""
        # 1. 基础检查（防止数值溢出）
        comm_snrs = np.clip(comm_snrs, -30, 50).astype(np.float64)
        radar_snrs = np.clip(radar_snrs, -30, 50).astype(np.float64)
        
        # 2. 对数尺度转换（抑制数值爆炸）
        def safe_log(x):
            return np.log10(max(x, 1e-10))  # 防止log(0)
        
        comm_perf = sum(safe_log(s / self.config["comm_snr_thresh"]) for s in comm_snrs)
        radar_perf = sum(safe_log(s / self.config["radar_snr_thresh"]) for s in radar_snrs)
        
        # 3. 动态权重调整（距离自适应）
        avg_distance = np.mean(self.target_distance)
        distance_factor = min(avg_distance / self.config["init_distance"], 3.0)  # 上限3倍
        
        # 4. 严格归一化（双保险机制）
        raw_reward = (
            0.5 * comm_perf * (1 - 0.2 * distance_factor) +
            0.5 * radar_perf * (1 + 0.3 * distance_factor)
        )
        
        # 双重限制：先压缩再裁剪
        compressed = 100 * np.tanh(raw_reward / 1000)  # 使用双曲正切压缩
        return np.clip(compressed, 0, 100)  # 最终限制在0-100

    def _get_obs(self) -> np.ndarray:
        """获取当前观测"""
        obs = np.concatenate([
            [c.current_snr for c in self.comm],  # 通信SNR
            [r.current_snr for r in self.radar], # 雷达SNR
            [self.current_step / self.config["max_steps"]],  # 归一化步数
            self.last_action.flatten()           # 上一个动作
        ])
        return obs.astype(np.float32)

    def _jain_index(self, values: List[float]) -> float:
        """计算Jain公平性指数"""
        values = np.array(values)
        if np.sum(values) <= 0:
            return 0.0
        return (np.sum(values) ** 2) / (len(values) * np.sum(values ** 2) + 1e-8)

    def _log_to_wandb(self, reward: float, info: Dict):
        """记录数据到WandB（新增指标记录）"""
        log_data = {
            "reward": reward,
            "step": info["step"],
            "fairness_comm": info["fairness_comm"],
            "fairness_radar": info["fairness_radar"],
            "avg_throughput": info["avg_throughput"],
            "avg_detection_prob": info["avg_detection_prob"],
            "avg_latency": info["avg_latency"]
        }
        
        for tgt in info["targets"]:
            i = tgt["target_id"]
            log_data.update({
                f"comm_snr_{i}": tgt["comm"]["snr"],
                f"comm_throughput_{i}": tgt["comm"]["throughput"],
                f"comm_latency_{i}": tgt["comm"]["latency"],
                f"radar_snr_{i}": tgt["radar"]["snr"],
                f"radar_det_prob_{i}": tgt["radar"]["detection_prob"],
                f"distance_{i}": tgt["distance"]
            })
            
        wandb.log(log_data)

    def render(self, mode='human'):
        """可视化环境状态（完整子图版，新增指标显示）"""
        if not self.history:
            return None

        # 使用Agg后端确保兼容性
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        
        # ========== 子图1：通信SNR ==========
        plt.subplot(2, 2, 1)
        for i in range(self.num_targets):
            snrs = [t["targets"][i]["comm"]["snr"] for t in self.history]
            plt.plot(snrs, label=f'目标 {i}', linewidth=2)
        plt.axhline(
            self.config["comm_snr_thresh"], 
            color='r', 
            linestyle='--',
            label='阈值'
        )
        plt.title('通信子系统SNR性能', fontsize=12, pad=10)
        plt.xlabel('时间步', fontsize=10)
        plt.ylabel('SNR (dB)', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        # ========== 子图2：雷达SNR ==========
        plt.subplot(2, 2, 2)
        for i in range(self.num_targets):
            snrs = [t["targets"][i]["radar"]["snr"] for t in self.history]
            plt.plot(snrs, label=f'目标 {i}', linewidth=2)
        plt.axhline(
            self.config["radar_snr_thresh"], 
            color='r', 
            linestyle='--',
            label='阈值'
        )
        plt.title('雷达子系统SNR性能', fontsize=12, pad=10)
        plt.xlabel('时间步', fontsize=10)
        plt.ylabel('SNR (dB)', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        # ========== 子图3：目标距离 ==========
        plt.subplot(2, 2, 3)
        for i in range(self.num_targets):
            dists = [t["targets"][i]["distance"] for t in self.history]
            plt.plot(dists, label=f'目标 {i}', linewidth=2)
        plt.axhline(
            self.config["min_distance"], 
            color='r', 
            linestyle='--',
            label='最小距离'
        )
        plt.title('目标距离变化', fontsize=12, pad=10)
        plt.xlabel('时间步', fontsize=10)
        plt.ylabel('距离 (km)', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        # ========== 子图4：资源分配（新增指标标注） ==========
        plt.subplot(2, 2, 4)
        if self.history:
            last_step = self.history[-1]
            bar_width = 0.35
            index = np.arange(self.num_targets)
            
            # 准备数据
            comm_power = [t["comm"]["power"] for t in last_step["targets"]]
            radar_power = [t["radar"]["power"] for t in last_step["targets"]]
            comm_bw = [t["comm"]["bandwidth"]/1e6 for t in last_step["targets"]]
            radar_bw = [t["radar"]["bandwidth"]/1e6 for t in last_step["targets"]]
            
            # 功率分配柱状图
            plt.bar(
                index - bar_width/2, 
                comm_power, 
                bar_width,
                label='通信功率',
                color='#1f77b4'
            )
            plt.bar(
                index + bar_width/2, 
                radar_power, 
                bar_width,
                label='雷达功率',
                color='#ff7f0e'
            )
            
            # 带宽分配折线图（次坐标轴）
            ax2 = plt.gca().twinx()
            ax2.plot(
                index, 
                comm_bw, 
                'o--', 
                color='#2ca02c', 
                label='通信带宽',
                markersize=8
            )
            ax2.plot(
                index, 
                radar_bw, 
                's--', 
                color='#d62728', 
                label='雷达带宽',
                markersize=8
            )
            
            # 新增指标标注
            for i, tgt in enumerate(last_step["targets"]):
                plt.text(
                    i - 0.3, max(comm_power)*1.1, 
                    f"TP:{tgt['comm']['throughput']:.1f}Mbps\nL:{tgt['comm']['latency']:.1f}ms",
                    fontsize=7
                )
                plt.text(
                    i + 0.1, max(radar_power)*1.1,
                    f"DP:{tgt['radar']['detection_prob']:.1%}\nRes:{tgt['radar']['range_resolution']:.1f}m",
                    fontsize=7
                )
            
            plt.title('当前资源分配', fontsize=12, pad=10)
            plt.gca().set_xlabel('目标编号', fontsize=10)
            plt.gca().set_ylabel('功率 (W)', fontsize=10)
            ax2.set_ylabel('带宽 (MHz)', fontsize=10)
            plt.gca().set_xticks(index)
            plt.gca().set_xticklabels([f'目标 {i}' for i in range(self.num_targets)])
            
            # 合并图例
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.gca().legend(
                lines1 + lines2, 
                labels1 + labels2,
                loc='upper right',
                fontsize=8
            )
        
        plt.tight_layout(pad=3.0)
        
        # 处理输出模式
        if mode == 'human':
            plt.show()
            return None
        elif mode == 'rgb_array':
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            width, height = fig.canvas.get_width_height()
            img_array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            img_array = img_array[:, :, :3]  # 转换为RGB
            plt.close(fig)
            return img_array

    def check(self) -> Tuple[bool, dict]:
        """系统自检"""
        report = {
            "modules": {
                "sat_channel": all(c.check()["status"] == "OK" for c in self.comm),
                "radar_echo": all(r.check()["status"] == "OK" for r in self.radar)
            },
            "action_space": {
                "shape": self.action_space.shape,
                "range": (self.action_space.low, self.action_space.high)
            },
            "observation_space": {
                "shape": self.observation_space.shape,
                "range": (self.observation_space.low, self.observation_space.high)
            },
            "config": self.config
        }
        is_ok = all(report["modules"].values())
        return is_ok, report

    def diagnose(self):
        """环境诊断工具（新增指标显示）"""
        print("\n=== Environment Diagnosis ===")
        print("Configuration:")
        print(f"- Targets: {self.num_targets}")
        print(f"- Power: {self.config['total_power']}W")
        print(f"- Bandwidth: {self.config['total_bandwidth']/1e6}MHz")
        print(f"- Comm Threshold: {self.config['comm_snr_thresh']}dB")
        print(f"- Radar Threshold: {self.config['radar_snr_thresh']}dB")
        
        # 测试典型动作
        test_actions = [
            ([[0.1, 0.1]]*self.num_targets, "Min Resources"),
            ([[0.5, 0.5]]*self.num_targets, "Balanced"),
            ([[0.9, 0.9]]*self.num_targets, "Max Resources")
        ]
        
        for action, desc in test_actions:
            print(f"\nTesting {desc}:")
            _, reward, _, info = self.step(np.array(action))
            
            for tgt in info["targets"]:
                print(f"Target {tgt['target_id']}:")
                print(f"  Comm: {tgt['comm']['power']:.1f}W + {tgt['comm']['bandwidth']/1e6:.1f}MHz")
                print(f"    → SNR: {tgt['comm']['snr']:.1f}dB | Throughput: {tgt['comm']['throughput']:.2f}Mbps | Latency: {tgt['comm']['latency']:.2f}ms")
                print(f"  Radar: {tgt['radar']['power']:.1f}W + {tgt['radar']['bandwidth']/1e6:.1f}MHz")
                print(f"    → SNR: {tgt['radar']['snr']:.1f}dB | Detection Prob: {tgt['radar']['detection_prob']:.2%} | Res: {tgt['radar']['range_resolution']:.2f}m")
            
            print(f"Reward: {reward:.2f} | Fairness: C={info['fairness_comm']:.2f}, R={info['fairness_radar']:.2f}")
            print(f"Avg Throughput: {info['avg_throughput']:.2f}Mbps | Avg Detection Prob: {info['avg_detection_prob']:.2%}")

def make_env(config=None):
    """环境创建函数"""
    def _init():
        env = ISAC_SatEnv(config)
        return env
    return _init
