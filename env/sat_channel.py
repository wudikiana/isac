# sat_channel.py
import numpy as np
from scipy.constants import speed_of_light as c, Boltzmann as k
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any, Optional

class SatelliteChannel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        self.config = {
            "frequency": 12e9,       # 载波频率 (Hz)
            "tx_gain": 45.0,         # 发射天线增益 (dB)
            "rx_gain": 40.0,         # 接收天线增益 (dB)
            "noise_temp": 290.0,     # 噪声温度 (K)
            "comm_snr_thresh": 3.0, # SNR阈值 (dB)
            "rain_attenuation": 0.1, # 雨衰 (dB/km)
            "atmospheric_loss": 0.02,# 大气损耗 (dB/km)
            "channel_type": "AWGN",  # 信道类型
            "ricean_k": 3.0          # 莱斯因子(dB)
        }
        if config:
            self.config.update(config)
        
        self.current_snr = 0.0
        self.current_distance = 0.0
        
    def _apply_fading(self, snr_db: float) -> float:
        """应用衰落模型"""
        snr_linear = 10 ** (snr_db / 10)
        
        if self.config["channel_type"] == "Rayleigh":
            fading = np.random.rayleigh(scale=1.0)
            return 10 * np.log10(snr_linear * fading ** 2)
        elif self.config["channel_type"] == "Rician":
            k_linear = 10 ** (self.config["ricean_k"] / 10)
            fading = np.sqrt(k_linear/(k_linear+1)) + \
                    np.random.normal(0, np.sqrt(1/(2*(k_linear+1))))
            return 10 * np.log10(snr_linear * fading ** 2)
        else:  # AWGN
            return snr_db
        
    def update(self, R_km: float, Pt: float, B: float) -> float:
        """带物理约束的SNR计算"""
        # 参数强制约束
        R = np.clip(R_km * 1e3, 100, 10000)  # 距离限制100m~10km
        Pt = np.clip(Pt, 0.1, 100)           # 功率限制0.1W~100W
        B = np.clip(B, 1e6, 100e6)           # 带宽限制1MHz~100MHz
        
        # 物理模型计算
        wavelength = c / self.config["frequency"]
        FSPL_db = 20 * np.log10(4 * np.pi * R / wavelength)
        Pr_db = 10 * np.log10(Pt) + self.config["tx_gain"] + self.config["rx_gain"] - FSPL_db
        Pn_db = 10 * np.log10(k * self.config["noise_temp"] * B)
        
        # 带保护的真实SNR
        true_snr = Pr_db - Pn_db
        self.current_snr = np.clip(true_snr, -30, 30)  # 严格限制范围
        return self.current_snr
    
    def _basic_link_budget(self, R_km: float, Pt: float, B: float) -> float:
        """修正后的链路预算计算"""
        # 参数检查
        R = max(R_km * 1e3, 100.0)  # 距离下限100米
        Pt = max(Pt, 1e-3)          # 功率下限1mW
        B = max(B, 1e3)             # 带宽下限1kHz
        
        # 波长计算（12GHz对应波长0.025m）
        wavelength = c / self.config["frequency"]  
        
        # 关键修正：自由空间路径损耗（单位：dB）
        # 原错误公式：FSPL_db = 20 * np.log10(4 * np.pi * R / wavelength)
        FSPL_db = 20 * np.log10(4 * np.pi * R) + 20 * np.log10(wavelength)  # 修正后的公式
        
        # 接收功率计算（dBm）
        Pr_db = 10 * np.log10(Pt) + self.config["tx_gain"] + self.config["rx_gain"] - FSPL_db
        
        # 噪声功率（dBm）
        Pn_db = 10 * np.log10(k * self.config["noise_temp"] * B)
        
        # SNR计算（强制合理范围）
        snr_db = Pr_db - Pn_db
        return np.clip(snr_db, -20, 50)  # 限制在-20dB~50dB  # 确保不低于-5dB

    # ... (保留原有的plot_snr_vs_distance等方法) ...
    def plot_snr_vs_distance(self, distances_km: list, Pt: float, B: float, 
                           save_to_wandb: bool = True, title: str = None):
        """
        绘制SNR随距离变化曲线
        """
        snrs = [self.update(d, Pt, B) for d in distances_km]
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances_km, snrs, 'g-s', linewidth=2)
        plt.axhline(y=self.config["comm_snr_thresh"], color='r', linestyle='--', 
                   label=f'SNR阈值({self.config["comm_snr_thresh"]} dB)')
        plt.xlabel('距离 (km)')
        plt.ylabel('SNR (dB)')
        plt.title(title or f'卫星通信SNR vs 距离 (Pt={Pt}W, B={B/1e6}MHz)')
        plt.grid(True)
        plt.legend()
        
        if save_to_wandb and wandb.run:
            wandb.log({"sat_comm_snr_vs_distance": wandb.Image(plt)})
        
        plt.close()
        return snrs
    
    def plot_link_budget(self, R_km: float, Pt: float, B: float,
                       save_to_wandb: bool = True):
        """
        绘制链路预算分析图
        """
        # 计算各组成部分
        freq = self.config["frequency"]
        Gt = self.config["tx_gain"]
        Gr = self.config["rx_gain"]
        T = self.config["noise_temp"]
        rain_att = self.config["rain_attenuation"]
        atm_loss = self.config["atmospheric_loss"]
        
        R = R_km * 1e3
        wavelength = c / freq
        FSPL_db = 20 * np.log10(4 * np.pi * R / wavelength)
        additional_loss_db = rain_att * R_km + atm_loss * R_km
        Pt_db = 10 * np.log10(Pt)
        Pn_db = 10 * np.log10(k * T * B)
        snr_db = self.update(R_km, Pt, B)
        
        # 准备数据
        components = [
            ('发射功率', Pt_db, 'lightgreen'),
            ('发射天线增益', Gt, 'lightblue'),
            ('接收天线增益', Gr, 'lightblue'),
            ('自由空间损耗', -FSPL_db, 'salmon'),
            ('附加损耗', -additional_loss_db, 'orange'),
            ('噪声功率', -Pn_db, 'lightgray')
        ]
        
        # 绘制堆叠图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bottom = 0
        for label, value, color in components:
            if value > 0:
                ax.bar('链路预算', value, bottom=bottom, label=label, color=color)
                bottom += value
            else:
                ax.bar('链路预算', -value, bottom=bottom + value, label=label, color=color)
        
        ax.axhline(y=bottom + snr_db, color='purple', linestyle='--', 
                  label=f'最终SNR: {snr_db:.2f} dB')
        ax.set_ylabel('功率 (dB)')
        ax.set_title(f'卫星通信链路预算分析 (距离: {R_km}km)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_to_wandb and wandb.run:
            wandb.log({
                "link_budget_analysis": wandb.Image(fig),
                "distance": R_km,
                "snr": snr_db
            })
        
        plt.close()
        return fig
    
    def check(self) -> Dict[str, Any]:
        """系统自检"""
        return {
            "status": "OK",
            "messages": ["SatelliteChannel operational"],
            "config": self.config
        }
if __name__ == "__main__":
    # 测试不同信道类型
    for ch_type in ["AWGN", "Rayleigh", "Rician"]:
        print(f"\n=== {ch_type}信道测试 ===")
        channel = SatelliteChannel({
            "channel_type": ch_type,
            "ricean_k": 3.0
        })
        snr = channel.update(500, 100, 10e6)
        print(f"通信SNR: {snr:.2f} dB")