import numpy as np
import json
from sat_channel import SatelliteChannel
from radar_echo import EnhancedRadarEcho
import matplotlib.pyplot as plt
from tqdm import tqdm

class ISACSimulator:
    def __init__(self, config_file='frame_cfg.json'):
        with open(config_file) as f:
            self.config = json.load(f)
        
        # 初始化模块（显式传递所有参数）
        self.radar = EnhancedRadarEcho(config_file)
        self.comm = SatelliteChannel({
            "frequency": self.config['system_parameters']['frequency_GHz'] * 1e9,
            "tx_gain": self.config['communication']['antenna_gain_dBi'],
            "rx_gain": self.config['communication']['antenna_gain_dBi'],
            "noise_temp": self.config['communication']['snr_model']['noise_temp_K'],
            "rain_attenuation": self.config['communication']['snr_model']['rain_attenuation_db'] / 50,
            "channel_type": "Rician",
            "ricean_k": 5.0,
            "shadowing_std": 2.0  # 新增阴影衰落参数
        })
        
        # 仿真参数
        self.distances = np.linspace(50, 1000, 50)
        self.velocities = np.array([7.8, 3.1])
        self.num_targets = 2

    def run_simulation(self, num_frames=100):
        """主仿真函数（包含全部所需参数）"""
        results = {
            # 基础参数
            'channel_type': [],
            'path_loss': np.zeros((len(self.distances), num_frames)),
            'shadow_fading': np.zeros((len(self.distances), num_frames)),
            'rain_attenuation': np.zeros((len(self.distances), num_frames)),
            
            # 目标参数
            'target_rcs': np.full((len(self.distances), num_frames, self.num_targets), 
                                self.config['radar']['rcs_m2']),
            
            # 通信参数
            'bandwidth': np.zeros((len(self.distances), num_frames, 2)),  # [comm, radar]
            'modulation': [],
            'ber': np.zeros((len(self.distances), num_frames)),
            
            # 性能指标
            'comm_snr': np.zeros((len(self.distances), num_frames)),
            'radar_snr': np.zeros((len(self.distances), num_frames, self.num_targets)),
        }

        for frame_idx in tqdm(range(num_frames)):
            for dist_idx, distance in enumerate(self.distances):
                # 记录环境参数
                results['channel_type'].append(self.comm.config["channel_type"])
                results['path_loss'][dist_idx, frame_idx] = self._calculate_path_loss(distance)
                results['shadow_fading'][dist_idx, frame_idx] = np.random.normal(0, self.comm.config["shadowing_std"])
                results['rain_attenuation'][dist_idx, frame_idx] = self.comm.config["rain_attenuation"] * distance
                
                # 资源分配
                action = np.random.uniform(0.3, 0.7, size=(self.num_targets, 2))
                total_bw = self.config['system_parameters']['bandwidth_MHz'] * 1e6
                results['bandwidth'][dist_idx, frame_idx, 0] = total_bw * np.mean(action[:, 1])  # 通信带宽
                results['bandwidth'][dist_idx, frame_idx, 1] = total_bw * np.mean(1 - action[:, 1])  # 雷达带宽
                
                # 通信性能
                snr = self.comm.update(distance, 
                                      self.config['communication']['tx_power_W'], 
                                      results['bandwidth'][dist_idx, frame_idx, 0])
                results['comm_snr'][dist_idx, frame_idx] = snr
                
                # 调制和BER
                mcs = self._select_mcs(snr)
                results['modulation'].append(mcs['mod'])
                results['ber'][dist_idx, frame_idx] = self._calculate_ber(snr, mcs)
                
                # 雷达性能
                for tgt in range(self.num_targets):
                    radar_snr = self.radar.radar_equation(
                        distance,
                        self.config['communication']['tx_power_W'] * (1 - action[tgt, 0]),
                        results['bandwidth'][dist_idx, frame_idx, 1]
                    )
                    results['radar_snr'][dist_idx, frame_idx, tgt] = radar_snr

        # 转换列表为数组
        results['channel_type'] = np.array(results['channel_type'])
        results['modulation'] = np.array(results['modulation'])
        
        return results

    def _calculate_path_loss(self, distance):
        """计算自由空间路径损耗"""
        wavelength = 3e8 / (self.config['system_parameters']['frequency_GHz'] * 1e9)
        return 20 * np.log10(4 * np.pi * distance * 1000 / wavelength)

    def _select_mcs(self, snr):
        """选择调制方式（返回包含mod和code_rate的字典）"""
        for mcs in reversed(self.config['communication']['modulation']['mcs_table']):
            if snr >= mcs['required_snr_dB']:
                return mcs
        return self.config['communication']['modulation']['mcs_table'][0]

    def _calculate_ber(self, snr, mcs):
        """计算理论BER"""
        snr_linear = 10**(snr/10)
        if mcs['mod'] == "QPSK":
            return 0.5 * (1 - np.sqrt(snr_linear/(1 + snr_linear)))
        elif mcs['mod'] == "16QAM":
            return 0.5 * (1 - np.sqrt(snr_linear/(5 + snr_linear)))
        else:
            return 0.5 * np.exp(-snr_linear/2)  # BPSK近似

    def save_results(self, results, filename='isac_simulation_data.npz'):
        """保存数据（包含所有指定参数）"""
        np.savez_compressed(
            filename,
            # 环境参数
            distances=self.distances,
            channel_type=results['channel_type'],
            path_loss=results['path_loss'],
            shadow_fading=results['shadow_fading'],
            rain_attenuation=results['rain_attenuation'],
            
            # 目标参数
            target_rcs=results['target_rcs'],
            
            # 资源参数
            bandwidth=results['bandwidth'],
            modulation=results['modulation'],
            
            # 性能指标
            ber=results['ber'],
            comm_snr=results['comm_snr'],
            radar_snr=results['radar_snr'],
            
            # 元数据
            config=json.dumps(self.config)
        )
        print(f"包含全部参数的数据已保存到 {filename}")

if __name__ == "__main__":
    simulator = ISACSimulator()
    results = simulator.run_simulation(num_frames=200)
    simulator.save_results(results)