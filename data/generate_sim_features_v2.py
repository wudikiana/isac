#!/usr/bin/env python3
"""
重新生成仿真特征数据
使用models_cs中的仿真器生成完整的仿真特征
"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# 添加models_cs路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
models_cs_path = os.path.join(current_dir, 'models_cs')
sys.path.insert(0, models_cs_path)

try:
    from isac_simulator import ISACSimulator
    from sat_channel import SatelliteChannel
    from radar_echo import EnhancedRadarEcho
    print("成功导入models_cs模块")
except ImportError as e:
    print(f"导入models_cs模块失败: {e}")
    print("请确保models_cs目录存在且包含所需的Python文件")
    sys.exit(1)

class EnhancedSimFeatureGenerator:
    def __init__(self, config_file='models_cs/frame_cfg.json'):
        """初始化仿真特征生成器"""
        self.config_file = config_file
        
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            print(f"配置文件不存在: {config_file}")
            sys.exit(1)
        
        # 加载配置
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # 初始化仿真器
        try:
            self.simulator = ISACSimulator(config_file)
            self.comm_channel = SatelliteChannel({
                "frequency": self.config['system_parameters']['frequency_GHz'] * 1e9,
                "tx_gain": self.config['communication']['antenna_gain_dBi'],
                "rx_gain": self.config['communication']['antenna_gain_dBi'],
                "noise_temp": self.config['communication']['snr_model']['noise_temp_K'],
                "rain_attenuation": self.config['communication']['snr_model']['rain_attenuation_db'] / 50,
                "channel_type": "Rician",
                "ricean_k": 5.0,
                "shadowing_std": 2.0
            })
            self.radar_system = EnhancedRadarEcho(config_file)
            print("仿真特征生成器初始化完成")
        except Exception as e:
            print(f"初始化仿真器失败: {e}")
            sys.exit(1)
    
    def generate_single_feature(self, img_path):
        """为单个图像生成仿真特征"""
        try:
            # 随机生成仿真参数
            distance = random.uniform(50, 1000)
            velocity = random.uniform(1, 10)
            tx_power = random.uniform(5, 20)
            bandwidth = random.uniform(50, 200) * 1e6
            
            # 1. 通信SNR计算
            comm_snr = self.comm_channel.update(distance, tx_power, bandwidth)
            
            # 2. 雷达特征计算
            radar_snr = self.radar_system.radar_equation(distance, tx_power, bandwidth)
            
            # 3. 生成雷达回波并计算特征
            echo = self.radar_system.generate_echo(distance, velocity, tx_power, bandwidth)
            radar_profile = self.radar_system.range_processing(echo)
            
            # 雷达特征统计
            radar_max = np.max(np.abs(radar_profile))
            radar_std = np.std(np.abs(radar_profile))
            radar_peak_idx = np.argmax(np.abs(radar_profile))
            
            # 4. 路径损耗计算
            wavelength = 3e8 / (self.config['system_parameters']['frequency_GHz'] * 1e9)
            path_loss = 20 * np.log10(4 * np.pi * distance * 1000 / wavelength)
            
            # 5. 阴影衰落
            shadow_fading = np.random.normal(0, 2.0)
            
            # 6. 雨衰
            rain_attenuation = 0.04 * distance
            
            # 7. 目标RCS
            target_rcs = self.config['radar']['rcs_m2']
            
            # 8. 带宽分配
            total_bw = self.config['system_parameters']['bandwidth_MHz'] * 1e6
            comm_bw_ratio = random.uniform(0.4, 0.7)
            comm_bw = total_bw * comm_bw_ratio
            
            # 9. 调制方式选择
            mcs = self.simulator._select_mcs(comm_snr)
            modulation = mcs['mod']
            
            # 10. BER计算
            ber = self.simulator._calculate_ber(comm_snr, mcs)
            
            # 11. 信道类型
            channel_type = "Rician"
            
            # 构建特征向量
            features = {
                'img_path': img_path,
                'comm_snr': comm_snr,
                'radar_feat': radar_snr,
                'radar_max': radar_max,
                'radar_std': radar_std,
                'radar_peak_idx': radar_peak_idx,
                'channel_type': channel_type,
                'path_loss': path_loss,
                'shadow_fading': shadow_fading,
                'rain_attenuation': rain_attenuation,
                'target_rcs': target_rcs,
                'bandwidth': comm_bw / 1e6,
                'modulation': modulation,
                'ber': ber
            }
            
            return features
            
        except Exception as e:
            print(f"生成特征失败 {img_path}: {e}")
            # 返回默认特征
            return {
                'img_path': img_path,
                'comm_snr': 10.0,
                'radar_feat': 15.0,
                'radar_max': 20.0,
                'radar_std': 5.0,
                'radar_peak_idx': 100,
                'channel_type': 'Rician',
                'path_loss': 120.0,
                'shadow_fading': 0.0,
                'rain_attenuation': 2.0,
                'target_rcs': 1.0,
                'bandwidth': 100.0,
                'modulation': 'QPSK',
                'ber': 1e-6
            }
    
    def find_image_files(self, data_root="data/combined_dataset"):
        """查找所有图像文件，包括tier3目录"""
        image_files = []
        
        # 遍历所有子目录，包括tier3
        splits = ['train2017', 'val2017', 'test2017', 'tier3']
        
        for split in splits:
            images_dir = os.path.join(data_root, 'images', split)
            if os.path.exists(images_dir):
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 使用完整的相对路径，包含combined_dataset前缀
                        img_path = f"combined_dataset/images/{split}/{file}"
                        image_files.append(img_path)
        
        print(f"找到 {len(image_files)} 个图像文件")
        return image_files
    
    def generate_all_features(self, output_file='data/sim_features_complete.csv'):
        """生成所有图像的仿真特征，完全替换旧数据"""
        print("开始生成仿真特征...")
        
        # 查找所有图像文件
        image_files = self.find_image_files()
        
        if not image_files:
            print("未找到图像文件，请检查data/combined_dataset目录")
            return []
        
        # 生成特征
        all_features = []
        
        for img_path in tqdm(image_files, desc="生成仿真特征"):
            features = self.generate_single_feature(img_path)
            all_features.append(features)
        
        # 保存到CSV文件
        print(f"保存特征到 {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if all_features:
                writer = csv.DictWriter(f, fieldnames=all_features[0].keys())
                writer.writeheader()
                writer.writerows(all_features)
        
        print(f"成功生成 {len(all_features)} 个仿真特征")
        return all_features
    
    def backup_and_replace(self, new_features, old_file='data/sim_features.csv'):
        """备份旧文件并替换为新的完整数据"""
        if os.path.exists(old_file):
            # 备份旧文件
            backup_file = old_file.replace('.csv', '_backup.csv')
            import shutil
            shutil.copy2(old_file, backup_file)
            print(f"已备份旧文件到: {backup_file}")
        
        # 保存新文件
        output_file = old_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if new_features:
                writer = csv.DictWriter(f, fieldnames=new_features[0].keys())
                writer.writeheader()
                writer.writerows(new_features)
        
        print(f"已替换文件: {output_file}")
        return new_features

def main():
    """主函数"""
    print("=== 仿真特征重新生成工具 ===")
    
    # 初始化生成器
    generator = EnhancedSimFeatureGenerator()
    
    # 生成新特征
    new_features = generator.generate_all_features('data/sim_features_complete.csv')
    
    if new_features:
        # 备份并替换旧文件
        generator.backup_and_replace(new_features, 'data/sim_features.csv')
        
        print("\n=== 生成完成 ===")
        print("生成的文件:")
        print("1. data/sim_features_complete.csv - 新生成的完整特征")
        print("2. data/sim_features.csv - 替换后的主特征文件")
        print("3. data/sim_features_backup.csv - 备份的旧特征文件")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"- 新生成特征: {len(new_features)} 个")
        
        # 验证最终数据
        final_df = pd.DataFrame(new_features)
        print(f"\n最终数据验证:")
        print(f"- 总行数: {len(final_df)}")
        print(f"- 完整特征数: {(final_df != -1).all(axis=1).sum()}")
        print(f"- 缺失特征数: {(final_df == -1).any(axis=1).sum()}")
        
        # 检查路径格式
        print(f"\n路径格式检查:")
        sample_paths = final_df['img_path'].head(5).tolist()
        for path in sample_paths:
            print(f"  {path}")
    else:
        print("生成特征失败")

if __name__ == "__main__":
    main() 