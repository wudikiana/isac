import numpy as np
from scipy.fft import fft, fftshift
import json

class EnhancedRadarEcho:
    def __init__(self, config_file='frame_cfg.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # 雷达参数
        self.pri = self.config['radar']['pri_us'] * 1e-6
        self.pulse_width = self.config['radar']['pulse_width_us'] * 1e-6
        self.rcs = self.config['radar']['rcs_m2']
        self.antenna_gain = self.config['radar']['antenna_gain_dBi']
        self.min_snr = self.config['radar']['min_detectable_snr_dB']
        
        # 波形参数
        self.chirp_bw = self.config['radar']['waveform']['chirp_bandwidth_MHz'] * 1e6
        self.subcarrier_ratio = self.config['radar']['waveform']['subcarriers_radar_ratio']
        
        # 状态变量
        self.current_snr = 0.0
        self.last_echo = None
    
    def radar_equation(self, distance, power, bandwidth):
        """雷达方程计算SNR"""
        wavelength = 3e8 / (self.config['system_parameters']['frequency_GHz'] * 1e9)
        G_linear = 10**(self.antenna_gain / 10)
        
        numerator = power * (G_linear**2) * (wavelength**2) * self.rcs
        denominator = (4*np.pi)**3 * (distance*1000)**4 * 1.38e-23 * 290 * bandwidth
        snr_linear = numerator / denominator
        return 10 * np.log10(snr_linear)
    
    def generate_waveform(self, num_samples):
        """生成OFDM-LFM混合波形"""
        # 生成OFDM信号(通信部分)
        num_comm_subcarriers = int((1 - self.subcarrier_ratio) * num_samples)
        comm_signal = np.random.randn(num_comm_subcarriers) + 1j * np.random.randn(num_comm_subcarriers)
        
        # 生成LFM信号(雷达部分)
        t = np.linspace(0, self.pulse_width, num_samples - num_comm_subcarriers)
        f0 = -self.chirp_bw / 2
        f1 = self.chirp_bw / 2
        lfm_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * self.pulse_width)))
        
        # 组合信号
        combined_signal = np.zeros(num_samples, dtype=complex)
        combined_signal[:num_comm_subcarriers] = comm_signal
        combined_signal[num_comm_subcarriers:] = lfm_signal
        
        return combined_signal
    
    def generate_echo(self, distance, velocity, power=None, bandwidth=None):
        """生成目标回波信号"""
        if power is None:
            power = self.config['communication']['tx_power_W']
        if bandwidth is None:
            bandwidth = self.config['system_parameters']['bandwidth_MHz'] * 1e6
        
        # 计算SNR
        self.current_snr = self.radar_equation(distance, power, bandwidth)
        
        # 生成波形
        num_samples = int(self.config['system_parameters']['frame_length_ms'] * 1000 / 
                         self.config['system_parameters']['symbol_duration_us'])
        waveform = self.generate_waveform(num_samples)
        
        # 计算延迟和多普勒
        delay = 2 * distance * 1000 / 3e8  # 往返延迟(s)
        doppler = 2 * velocity * (self.config['system_parameters']['frequency_GHz'] * 1e9) / 3e8
        
        # 生成回波
        delay_samples = int(delay / self.pri * num_samples)
        t = np.arange(num_samples) * self.pri
        echo = np.roll(waveform, delay_samples) * np.exp(1j * 2 * np.pi * doppler * t)
        
        # 功率调整
        echo = echo * np.sqrt(power / self.config['communication']['tx_power_W'])
        self.last_echo = echo
        
        return echo
    
    def range_processing(self, echo_signal=None):
        """距离处理(FFT)"""
        if echo_signal is None:
            if self.last_echo is None:
                raise ValueError("No echo signal available")
            echo_signal = self.last_echo
        
        # 加窗处理
        window = np.hamming(len(echo_signal))
        windowed_signal = echo_signal * window
        
        # FFT处理
        fft_result = fftshift(fft(windowed_signal))
        return fft_result
    
    def check(self):
        """系统检查"""
        test_snr = self.radar_equation(50, self.config['communication']['tx_power_W'], 
                                      self.config['system_parameters']['bandwidth_MHz'] * 1e6)
        detectable = test_snr >= self.min_snr
        return {
            "status": "OK" if detectable else "WARNING",
            "message": f"Test SNR at 50km: {test_snr:.2f} dB (Detectable: {detectable})",
            "current_snr": self.current_snr
        }