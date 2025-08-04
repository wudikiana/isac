import sys
sys.path.append(r'C:/Users/rhy/Desktop/挑战杯小组作业/shujushengcheng/shujushengcheng/models')
from isac_simulator import ISACSimulator
import os
import glob
import json
import numpy as np
from tqdm import tqdm

# 假设 ISACSimulator 已在 PYTHONPATH 下
# from isac_simulator import ISACSimulator

def extract_sim_params_from_json(json_path):
    with open(json_path, 'r') as f:
        meta = json.load(f)
    metadata = meta.get('metadata', {})
    import math
    off_nadir = metadata.get('off_nadir_angle', 0)
    sat_height = 700  # km
    try:
        distance = sat_height / math.cos(math.radians(float(off_nadir)))
    except Exception:
        distance = 700
    disaster_type = metadata.get('disaster_type', 'unknown')
    if disaster_type == 'wind':
        velocity = 10.0
    elif disaster_type == 'flood':
        velocity = 2.0
    elif disaster_type == 'landslide':
        velocity = 1.0
    else:
        velocity = 5.0
    capture_date = metadata.get('capture_date', '2000-01-01T00:00:00Z')
    try:
        month = int(capture_date[5:7])
    except Exception:
        month = 1
    if disaster_type == 'flood' or month in [6,7,8]:
        rain_attenuation = 10.0
    else:
        rain_attenuation = 2.0
    return {
        'distance': distance,
        'velocity': velocity,
        'rain_attenuation': rain_attenuation
    }

def main():
    simulator = ISACSimulator(config_file='shujushengcheng/shujushengcheng/models/frame_cfg.json')
    out_csv = 'data/sim_features.csv'
    img_dirs = [
        'data/combined_dataset/images/train2017',
        'data/combined_dataset/images/val2017',
        'data/combined_dataset/images/test2017',
        'data/combined_dataset/images/tier3'
    ]
    all_rows = []
    for img_dir in img_dirs:
        img_files = glob.glob(os.path.join(img_dir, '*.*'))
        img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_path in tqdm(img_files, desc=f'Processing {img_dir}'):
            json_path = os.path.splitext(img_path)[0] + '.json'
            if not os.path.exists(json_path):
                print(f'Warning: JSON not found for {img_path}')
                continue
            sim_params = extract_sim_params_from_json(json_path)
            result = simulator.simulate_transmission(
                distance=sim_params['distance'],
                velocity=sim_params['velocity'],
                rain_attenuation=sim_params['rain_attenuation']
            )
            comm_snr = result.get('comm_snr', -1)
            radar_profile = result.get('radar_profile', None)
            if radar_profile is not None:
                radar_profile = np.array(radar_profile)
                radar_feat = float(np.mean(np.abs(radar_profile)))
                radar_max = float(np.max(np.abs(radar_profile)))
                radar_std = float(np.std(radar_profile))
                radar_peak_idx = int(np.argmax(np.abs(radar_profile)))
            else:
                radar_feat = radar_max = radar_std = radar_peak_idx = -1
            channel_type = result.get('channel_type', 'unknown')
            path_loss = result.get('path_loss', -1)
            shadow_fading = result.get('shadow_fading', -1)
            rain_attenuation = result.get('rain_attenuation', -1)
            target_rcs = result.get('target_rcs', -1)
            bandwidth = result.get('bandwidth', -1)
            modulation = result.get('modulation', 'unknown')
            ber = result.get('ber', -1)
            all_rows.append([
                os.path.relpath(img_path, 'data'), comm_snr, radar_feat, radar_max, radar_std, radar_peak_idx,
                channel_type, path_loss, shadow_fading, rain_attenuation, target_rcs, bandwidth, modulation, ber
            ])
    # 保存为CSV
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('img_path,comm_snr,radar_feat,radar_max,radar_std,radar_peak_idx,channel_type,path_loss,shadow_fading,rain_attenuation,target_rcs,bandwidth,modulation,ber\n')
        for row in all_rows:
            f.write(','.join([str(x) for x in row]) + '\n')
    print(f'已保存所有仿真特征到 {out_csv}')

if __name__ == '__main__':
    main() 