import numpy as np
import json
from scipy import special
import matplotlib.pyplot as plt

class EnhancedSatelliteChannel:
    def __init__(self, config_file='frame_cfg.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # 系统参数
        self.fc = self.config['system_parameters']['frequency_GHz'] * 1e9
        self.B = self.config['system_parameters']['bandwidth_MHz'] * 1e6
        self.Ts = self.config['system_parameters']['symbol_duration_us'] * 1e-6
        
        # 通信参数
        self.tx_power = self.config['communication']['tx_power_W']
        self.antenna_gain = self.config['communication']['antenna_gain_dBi']
        self.noise_temp = self.config['communication']['snr_model']['noise_temp_K']
        
        # 新增：雨衰参数
        self.rain_attenuation_db = self.config['communication']['snr_model'].get('rain_attenuation_db', 2.0)
        
        # 状态变量
        self.current_snr = 0.0
        self.channel_coeff = 1.0 + 0j
    
    def free_space_path_loss(self, distance):
        """自由空间路径损耗"""
        wavelength = 3e8 / self.fc
        return 20 * np.log10(4 * np.pi * distance * 1000 / wavelength)
    
    def atmospheric_effects(self, elevation_angle):
        """大气效应(含雨衰)"""
        # 简单仰角模型
        if elevation_angle >= 80:
            return 0.2 + self.rain_attenuation_db
        elif elevation_angle >= 30:
            return 0.6 + self.rain_attenuation_db
        else:
            return 2.0 + self.rain_attenuation_db
    
    def shadow_fading(self):
        """阴影衰落(对数正态分布)"""
        return np.random.normal(0, 2.0)  # 2dB标准差
    
    def multipath_fading(self, num_samples, elevation_angle):
        """多径衰落模型"""
        t = np.arange(num_samples) * self.Ts
        
        # 仰角越高，K因子越大(直射路径越强)
        if elevation_angle >= 80:
            K = 10  # 莱斯K因子(dB)
        elif elevation_angle >= 30:
            K = 5
        else:
            K = 0  # 瑞利衰落
        
        K_linear = 10**(K/10)
        mean = np.sqrt(K_linear / (K_linear + 1))
        std = np.sqrt(1 / (2 * (K_linear + 1)))
        
        inphase = np.random.normal(mean, std, num_samples)
        quadrature = np.random.normal(0, std, num_samples)
        return inphase + 1j * quadrature
    
    def update(self, distance, power=None, bandwidth=None, elevation_angle=60, rain_attenuation=None):
        """更新信道状态"""
        if power is None:
            power = self.tx_power
        if bandwidth is None:
            bandwidth = self.B
        
        # rain_attenuation参数支持
        if rain_attenuation is not None:
            self.rain_attenuation_db = rain_attenuation
        
        # 路径损耗计算
        fspl = self.free_space_path_loss(distance)
        atm_loss = self.atmospheric_effects(elevation_angle)
        shadow = self.shadow_fading()
        
        # 接收功率(dB)
        rx_power_db = (10 * np.log10(power) + 
                      self.antenna_gain * 2 - 
                      fspl - atm_loss + shadow)
        
        # 噪声功率(dB)
        noise_power_db = 10 * np.log10(1.38e-23 * self.noise_temp * bandwidth)
        
        # SNR(dB)
        self.current_snr = rx_power_db - noise_power_db
        
        # 生成信道系数(用于后续信号仿真)
        num_samples = int(self.config['system_parameters']['frame_length_ms'] * 1000 / 
                        self.config['system_parameters']['symbol_duration_us'])
        self.channel_coeff = self.multipath_fading(num_samples, elevation_angle)
        
        return self.current_snr
    
    def apply_channel(self, signal):
        """应用信道效应到信号"""
        if len(signal) != len(self.channel_coeff):
            raise ValueError("Signal length doesn't match channel coefficients")
        return signal * self.channel_coeff
    
    def check(self):
        """系统检查"""
        test_snr = self.update(1000)  # 测试1000km距离
        return {
            "status": "OK" if test_snr > 0 else "WARNING",
            "message": f"Test SNR at 1000km: {test_snr:.2f} dB",
            "current_snr": self.current_snr
        }