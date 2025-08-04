# baseline_rule.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import csv
import os
from env.isac_sat_env import ISAC_SatEnv
from typing import Dict, List, Tuple, Any

def random_policy(env) -> np.ndarray:
    """随机策略"""
    return env.action_space.sample()

def greedy_policy(env) -> np.ndarray:
    """基础贪心策略"""
    return np.array([[0.5, 0.5]] * env.num_targets, dtype=np.float32)

def distance_aware_policy(env) -> np.ndarray:
    """基于距离感知的动态资源分配策略"""
    actions = []
    for i in range(env.num_targets):
        # 计算距离比率 (0-1)
        distance_ratio = (env.target_distance[i] - env.config["min_distance"]) / \
                       (env.config["init_distance"] - env.config["min_distance"])
        
        # 动态功率分配
        comm_power_ratio = 0.4 + 0.4 * distance_ratio
        
        # 动态带宽分配
        radar_bw_ratio = 0.6 - 0.4 * abs(distance_ratio - 0.5)
        
        actions.append([
            np.clip(comm_power_ratio, 0.1, 0.9),
            np.clip(radar_bw_ratio, 0.1, 0.9)
        ])
    return np.array(actions, dtype=np.float32)

def run_episode(env, policy_fn) -> Dict[str, Any]:
    """运行单个episode并收集性能指标"""
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    comm_snr_list = []
    radar_snr_list = []
    comm_success_count = 0
    radar_success_count = 0
    fairness_comm_list = []
    fairness_radar_list = []
    
    while not done:
        action = policy_fn(env)
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        fairness_comm_list.append(info.get('fairness_comm', 0))
        fairness_radar_list.append(info.get('fairness_radar', 0))
        comm_snr_list += [t['comm']['snr'] for t in info['targets']]
        radar_snr_list += [t['radar']['snr'] for t in info['targets']]
        comm_success_count += sum([t['comm']['snr'] >= t['comm']['threshold'] for t in info['targets']])
        radar_success_count += sum([t['radar']['snr'] >= t['radar']['threshold'] for t in info['targets']])
    
    # 计算平均性能指标
    avg_comm_snr = np.mean(comm_snr_list) if comm_snr_list else 0
    avg_radar_snr = np.mean(radar_snr_list) if radar_snr_list else 0
    comm_success_rate = comm_success_count / (steps * env.num_targets) if steps > 0 else 0
    radar_success_rate = radar_success_count / (steps * env.num_targets) if steps > 0 else 0
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'avg_comm_snr': avg_comm_snr,
        'avg_radar_snr': avg_radar_snr,
        'comm_success_rate': comm_success_rate,
        'radar_success_rate': radar_success_rate,
        'fairness_comm': np.mean(fairness_comm_list),
        'fairness_radar': np.mean(fairness_radar_list)
    }

def run_baseline(policy_name: str, num_episodes: int, config: dict = None) -> List[Dict[str, Any]]:
    """运行指定策略的多个episode"""
    env = ISAC_SatEnv(config=config)
    results = []
    
    # 选择策略
    if policy_name == 'random':
        policy_fn = random_policy
    elif policy_name == 'greedy':
        policy_fn = greedy_policy
    elif policy_name == 'distance_aware':
        policy_fn = distance_aware_policy
    else:
        raise ValueError(f"未知策略: {policy_name}")
    
    print(f"开始运行 {policy_name} 策略 ({num_episodes} episodes)...")
    
    for episode in range(num_episodes):
        metrics = run_episode(env, policy_fn)
        metrics['episode'] = episode
        results.append(metrics)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean([r['total_reward'] for r in results[-10:]])
            print(f"  Episode {episode + 1}/{num_episodes} 完成. 平均奖励: {avg_reward:.2f}")
    
    return results

def save_results_to_csv(results: List[Dict[str, Any]], filename: str):
    """将结果保存到CSV文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'total_reward', 'steps', 
                     'avg_comm_snr', 'avg_radar_snr',
                     'comm_success_rate', 'radar_success_rate',
                     'fairness_comm', 'fairness_radar']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"结果已保存至: {filename}")

def main():
    """主函数：运行三种策略并比较结果"""
    optimized_config = {
    # 物理参数
    "frequency": 2.4e9,
    "tx_gain": 30.0,
    "rx_gain": 25.0,
    
    # 雷达增强参数
    "radar_gain": 35.0,
    "target_rcs": 5.0,
    "radar_snr_thresh": -10.0,  # 降低雷达阈值
    
    # 资源分配
    "total_power": 100.0,       # 提高总功率
    "total_bandwidth": 50e6,
    
    # 环境参数
    "init_distance": 50.0,
    "max_steps": 200,
    
    # 奖励权重
    "comm_reward_weight": 0.5,
    "radar_reward_weight": 0.5,
    "fairness_weight": 0.2,
    "action_penalty_weight": 0.01
    }
    
    # 测试三种策略
    policies = ['random', 'greedy', 'distance_aware']
    all_results = {}
    
    for policy in policies:
        results = run_baseline(policy, 50, optimized_config)
        all_results[policy] = results
        save_results_to_csv(results, f'results/{policy}_results.csv')
    
    # 打印比较结果
    print("\n策略性能比较:")
    for policy in policies:
        rewards = [r['total_reward'] for r in all_results[policy]]
        comm_success = [r['comm_success_rate'] for r in all_results[policy]]
        radar_success = [r['radar_success_rate'] for r in all_results[policy]]
        
        print(f"{policy}策略:")
        print(f"  平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  通信成功率: {np.mean(comm_success):.1%}")
        print(f"  雷达成功率: {np.mean(radar_success):.1%}\n")

if __name__ == "__main__":
    main()