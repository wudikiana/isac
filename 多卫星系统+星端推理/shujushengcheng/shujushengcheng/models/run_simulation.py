#!/usr/bin/env python3
"""
卫星ISAC系统主运行脚本
功能：执行端到端仿真，生成通信雷达联合数据，支持多种运行模式
"""

import argparse
import time
import h5py
import numpy as np
from pathlib import Path
from isac_simulator import ISACSimulator
from sat_channel import EnhancedSatelliteChannel
from radar_echo import EnhancedRadarEcho

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='卫星ISAC系统仿真主程序')
    
    # 仿真参数
    parser.add_argument('--frames', type=int, default=100,
                      help='仿真帧数 (默认: 100)')
    parser.add_argument('--config', type=str, default='frame_cfg.json',
                      help='配置文件路径 (默认: frame_cfg.json)')
    parser.add_argument('--mode', choices=['baseline', 'random', 'fixed'], 
                      default='baseline', help='资源分配模式 (默认: baseline)')
    
    # 输出参数
    parser.add_argument('--output', type=str, required=True,
                      help='输出HDF5文件路径')
    parser.add_argument('--plot', action='store_true',
                      help='是否生成实时可视化图表')
    parser.add_argument('--log', action='store_true',
                      help='是否输出详细日志')
    
    # 目标场景参数
    parser.add_argument('--distance', type=float, default=500.0,
                      help='初始目标距离(km) (默认: 500)')
    parser.add_argument('--velocity', type=float, default=7.8,
                      help='目标速度(km/s) (默认: 7.8)')
    
    # 新增：雨衰参数
    parser.add_argument('--rain_attenuation', type=float, default=None,
                      help='雨衰(dB)，如不指定则用配置文件默认值')
    
    return parser.parse_args()

def initialize_systems(config_file):
    """初始化各子系统"""
    print("[初始化] 加载系统配置...")
    simulator = ISACSimulator(config_file)
    comm_channel = EnhancedSatelliteChannel(config_file)
    radar_system = EnhancedRadarEcho(config_file)
    
    # 系统自检
    print("\n[系统自检]")
    print("卫星信道:", comm_channel.check())
    print("雷达系统:", radar_system.check())
    
    return simulator, comm_channel, radar_system

def generate_resource_allocation(mode, num_frames, config):
    """生成资源分配策略"""
    print(f"\n[资源分配] 模式: {mode}")
    
    if mode == 'fixed':
        # 固定分配 (通信:60%, 雷达:40%)
        return np.tile([0.6, 0.6], (num_frames, 1))
    
    elif mode == 'random':
        # 随机分配
        return np.random.uniform(0.3, 0.7, (num_frames, 2))
    
    else:  # baseline
        # 基于SNR的自适应基线
        allocations = []
        for _ in range(num_frames):
            # 简单逻辑：SNR低时分配更多资源
            comm_snr = np.random.uniform(5, 30)
            ratio = 0.3 + 0.5 * (comm_snr / 30)  # SNR越高分配越多资源给通信
            allocations.append([ratio, 1-ratio])
        return np.array(allocations)

def run_simulation(simulator, num_frames, allocation_plan, args):
    """运行主仿真循环"""
    print(f"\n[开始仿真] 总帧数: {num_frames}")
    start_time = time.time()
    results = []
    
    for frame_idx in range(num_frames):
        frame_start = time.time()
        
        # 更新目标距离
        simulator.target_distance += simulator.target_velocity * \
                                   (simulator.config['system_parameters']['frame_length_ms'] / 1000)
        
        # 应用资源分配 (仅示例，实际应集成到仿真器中)
        if args.mode != 'baseline':
            tx_power = simulator.config['communication']['tx_power_W'] * allocation_plan[frame_idx][0]
            bandwidth = simulator.config['system_parameters']['bandwidth_MHz'] * 1e6 * allocation_plan[frame_idx][1]
        else:
            tx_power = None
            bandwidth = None
        
        # 执行仿真
        result = simulator.simulate_transmission(
            distance=simulator.target_distance,
            velocity=simulator.target_velocity,
            rain_attenuation=args.rain_attenuation
        )
        
        # 添加资源分配信息
        result.update({
            'power_allocation': allocation_plan[frame_idx][0],
            'bandwidth_allocation': allocation_plan[frame_idx][1],
            'frame_idx': frame_idx
        })
        results.append(result)
        
        # 日志输出
        if args.log or frame_idx % 10 == 0:
            print(f"Frame {frame_idx:04d}: "
                  f"Dist={result['distance']:.1f}km | "
                  f"Comm SNR={result['comm_snr']:.1f}dB | "
                  f"Radar SNR={result['radar_snr']:.1f}dB | "
                  f"Power={allocation_plan[frame_idx][0]:.2f} | "
                  f"BW={allocation_plan[frame_idx][1]:.2f} | "
                  f"Time={time.time()-frame_start:.3f}s")
    
    total_time = time.time() - start_time
    print(f"\n[仿真完成] 总耗时: {total_time:.2f}s | 平均每帧: {total_time/num_frames:.3f}s")
    return results

def save_results(results, output_path):
    """保存仿真结果到HDF5文件"""
    print(f"\n[保存结果] 路径: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as hf:
        # 保存标量数据
        grp_scalar = hf.create_group('scalars')
        scalar_fields = ['comm_snr', 'radar_snr', 'distance', 'velocity', 
                        'power_allocation', 'bandwidth_allocation', 'frame_idx']
        
        for field in scalar_fields:
            grp_scalar.create_dataset(field, data=[r[field] for r in results])
        
        # 保存信号数据(最后一帧)
        grp_signals = hf.create_group('signals')
        last_frame = results[-1]
        grp_signals.create_dataset('tx_frame', data=last_frame['tx_frame'])
        grp_signals.create_dataset('rx_frame', data=last_frame['rx_frame'])
        grp_signals.create_dataset('radar_profile', data=last_frame['radar_profile'])
        
        # 保存元数据
        hf.attrs['simulation_time'] = time.ctime()
        hf.attrs['total_frames'] = len(results)
        hf.attrs['config_file'] = str(args.config)

def generate_realtime_plots(results):
    """生成实时可视化图表"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 8))
    
    # SNR曲线
    plt.subplot(2, 2, 1)
    plt.plot([r['comm_snr'] for r in results], 'b-', label='Comm SNR')
    plt.plot([r['radar_snr'] for r in results], 'r-', label='Radar SNR')
    plt.xlabel('Frame Index')
    plt.ylabel('SNR (dB)')
    plt.title('Communication & Radar SNR')
    plt.legend()
    plt.grid()
    
    # 资源分配
    plt.subplot(2, 2, 2)
    plt.stackplot(
        range(len(results)),
        [r['power_allocation'] for r in results],
        [r['bandwidth_allocation'] for r in results],
        labels=['Power', 'Bandwidth']
    )
    plt.xlabel('Frame Index')
    plt.ylabel('Resource Ratio')
    plt.title('Resource Allocation')
    plt.legend()
    plt.grid()
    
    # 距离变化
    plt.subplot(2, 2, 3)
    plt.plot([r['distance'] for r in results], 'g-')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance (km)')
    plt.title('Target Distance')
    plt.grid()
    
    # 雷达距离剖面(最后一帧)
    plt.subplot(2, 2, 4)
    plt.plot(np.abs(results[-1]['radar_profile']))
    plt.xlabel('Range Bin')
    plt.ylabel('Amplitude')
    plt.title('Radar Range Profile (Last Frame)')
    plt.grid()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 解析参数
    args = parse_arguments()
    
    try:
        # 初始化系统
        simulator, comm_channel, radar_system = initialize_systems(args.config)
        simulator.target_distance = args.distance
        simulator.target_velocity = args.velocity
        
        # 生成资源分配计划
        allocation_plan = generate_resource_allocation(
            args.mode, args.frames, simulator.config
        )
        
        # 运行仿真
        results = run_simulation(simulator, args.frames, allocation_plan, args)
        
        # 保存结果
        save_results(results, args.output)
        
        # 可视化
        if args.plot:
            generate_realtime_plots(results)
            
        print("\n[仿真成功完成]")
        
    except Exception as e:
        print(f"\n[错误] 仿真失败: {str(e)}")
        raise