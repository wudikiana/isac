import numpy as np
import json
from sat_channel import EnhancedSatelliteChannel
from radar_echo import EnhancedRadarEcho
import matplotlib.pyplot as plt

class ISACSimulator:
    def __init__(self, config_file='frame_cfg.json'):
        self.config = json.load(open(config_file))
        
        # 初始化子系统
        self.comm_channel = EnhancedSatelliteChannel(config_file)
        self.radar_system = EnhancedRadarEcho(config_file)
        
        # 资源配置
        self.time_slots = self.config['joint_resource_allocation']['time_sharing']
        self.freq_allocation = self.config['joint_resource_allocation']['frequency_sharing']
        
        # 仿真状态
        self.current_step = 0
        self.target_distance = 500  # km
        self.target_velocity = 7.8  # km/s
    
    def generate_frame(self):
        """生成一个完整的ISAC帧"""
        num_symbols = int(self.config['system_parameters']['frame_length_ms'] * 1000 / 
                         self.config['system_parameters']['symbol_duration_us'])
        frame = np.zeros(num_symbols, dtype=complex)
        
        # 生成通信数据
        comm_data = np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
        
        # 应用时间资源分配
        for slot in self.time_slots:
            if slot['function'] == 'comm':
                frame[slot['symbols']] = comm_data[slot['symbols']]
            elif slot['function'] == 'radar':
                # 生成雷达波形
                radar_waveform = self.radar_system.generate_waveform(len(slot['symbols']))
                frame[slot['symbols']] = radar_waveform
        
        return frame
    
    def simulate_transmission(self, distance=None, velocity=None, rain_attenuation=None):
        """模拟一次完整的ISAC传输"""
        if distance is None:
            distance = self.target_distance
        if velocity is None:
            velocity = self.target_velocity
        
        # rain_attenuation参数支持
        comm_snr = self.comm_channel.update(distance, rain_attenuation=rain_attenuation)
        
        # 1. 生成发射帧
        tx_frame = self.generate_frame()
        
        # 2. 更新信道状态
        radar_echo = self.radar_system.generate_echo(distance, velocity)
        
        # 3. 应用信道效应
        rx_frame = self.comm_channel.apply_channel(tx_frame) + radar_echo
        
        # 4. 添加噪声
        noise_power = 1.38e-23 * self.comm_channel.noise_temp * self.comm_channel.B
        noise = np.random.normal(0, np.sqrt(noise_power/2), len(rx_frame)) + \
                1j * np.random.normal(0, np.sqrt(noise_power/2), len(rx_frame))
        rx_frame += noise
        
        # 5. 处理雷达回波
        radar_profile = self.radar_system.range_processing(radar_echo)
        
        return {
            'tx_frame': tx_frame,
            'rx_frame': rx_frame,
            'radar_profile': radar_profile,
            'comm_snr': comm_snr,
            'radar_snr': self.radar_system.current_snr,
            'distance': distance,
            'velocity': velocity
        }
    
    def run_simulation(self, num_frames=10):
        """运行多帧仿真"""
        results = []
        
        for _ in range(num_frames):
            # 更新目标距离
            self.target_distance += self.target_velocity * \
                                  (self.config['system_parameters']['frame_length_ms'] / 1000)
            
            # 仿真传输
            result = self.simulate_transmission()
            results.append(result)
            
            # 打印进度
            print(f"Frame {self.current_step}: Distance={result['distance']:.2f}km, "
                  f"Comm SNR={result['comm_snr']:.2f}dB, "
                  f"Radar SNR={result['radar_snr']:.2f}dB")
            
            self.current_step += 1
        
        return results
    
    def visualize_results(self, results):
        """可视化仿真结果"""
        plt.figure(figsize=(15, 10))
        
        # SNR变化曲线
        plt.subplot(2, 2, 1)
        plt.plot([r['comm_snr'] for r in results], 'b-', label='Comm SNR')
        plt.plot([r['radar_snr'] for r in results], 'r-', label='Radar SNR')
        plt.axhline(self.config['communication']['modulation']['mcs_table'][0]['required_snr_dB'], 
                   color='b', linestyle='--', label='Comm Threshold')
        plt.axhline(self.config['radar']['min_detectable_snr_dB'], 
                   color='r', linestyle='--', label='Radar Threshold')
        plt.xlabel('Frame Index')
        plt.ylabel('SNR (dB)')
        plt.title('Communication and Radar SNR')
        plt.legend()
        plt.grid()
        
        # 距离变化
        plt.subplot(2, 2, 2)
        plt.plot([r['distance'] for r in results], 'g-')
        plt.xlabel('Frame Index')
        plt.ylabel('Distance (km)')
        plt.title('Target Distance')
        plt.grid()
        
        # 雷达距离剖面(最后一帧)
        plt.subplot(2, 2, 3)
        plt.plot(np.abs(results[-1]['radar_profile']))
        plt.xlabel('Range Bin')
        plt.ylabel('Amplitude')
        plt.title('Radar Range Profile (Last Frame)')
        plt.grid()
        
        # 信号星座图(最后一帧)
        plt.subplot(2, 2, 4)
        plt.scatter(np.real(results[-1]['rx_frame']), 
                   np.imag(results[-1]['rx_frame']), s=5)
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.title('Received Signal Constellation')
        plt.grid()
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    simulator = ISACSimulator()
    results = simulator.run_simulation(num_frames=20)
    simulator.visualize_results(results)