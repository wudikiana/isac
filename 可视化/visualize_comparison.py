# visualize_comparison.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_results(policy_names: List[str], results_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """加载所有策略的结果"""
    results = {}
    for policy in policy_names:
        try:
            # 尝试加载CSV结果文件
            csv_path = f"{results_dir}/{policy}_metrics.csv"
            if os.path.exists(csv_path):
                results[policy] = pd.read_csv(csv_path)
                logger.info(f"成功加载 {policy} 策略结果")
            else:
                logger.warning(f"未找到 {policy} 策略的结果文件: {csv_path}")
        except Exception as e:
            logger.error(f"加载 {policy} 策略结果时出错: {str(e)}")
    return results

def plot_metrics_comparison(results_dict: Dict[str, pd.DataFrame], 
                          save_path: str = "results/comparison_plots") -> None:
    """绘制四种核心指标的对比图"""
    os.makedirs(save_path, exist_ok=True)
    
    # 定义核心指标和显示名称
    metrics = {
        'throughput': '通信吞吐量 (Mbps)',
        'sensing_accuracy': '感知精度 (%)',
        'energy_consumption': '能耗 (W)',
        'latency': '时延 (ms)',
        'reward': '平均奖励'
    }
    
    # 准备对比数据
    comparison_data = []
    for policy_name, df in results_dict.items():
        for metric, display_name in metrics.items():
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    comparison_data.append({
                        'Policy': policy_name,
                        'Metric': display_name,
                        'Mean': np.mean(values),
                        'Std': np.std(values),
                        'Median': np.median(values)
                    })
    
    if not comparison_data:
        logger.error("没有有效数据可绘制")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # 1. 绘制柱状图对比
    plt.figure(figsize=(15, 10))
    for i, (metric, display_name) in enumerate(metrics.items()):
        plt.subplot(2, 3, i+1)
        subset = df[df['Metric'] == display_name]
        if not subset.empty:
            sns.barplot(x='Policy', y='Mean', data=subset, palette="viridis", 
                        order=["随机策略", "贪心策略", "距离感知策略", "PPO策略"])
            plt.errorbar(x=range(len(subset)), y=subset['Mean'], yerr=subset['Std'], 
                        fmt='none', c='black', capsize=5)
            plt.title(display_name)
            plt.ylabel('')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/metrics_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("已保存指标对比柱状图")
    
    # 2. 绘制雷达图对比
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # 准备雷达图数据
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    
    for policy_name in results_dict.keys():
        policy_data = df[df['Policy'] == policy_name]
        if not policy_data.empty:
            values = []
            for display_name in metrics.values():
                val = policy_data[policy_data['Metric'] == display_name]['Mean'].values
                if len(val) > 0:
                    # 归一化到0-1范围
                    max_val = df[df['Metric'] == display_name]['Mean'].max()
                    min_val = df[df['Metric'] == display_name]['Mean'].min()
                    norm_val = (val[0] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    values.append(norm_val)
                else:
                    values.append(0)
            
            values = np.concatenate((values, [values[0]]))  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=policy_name)
            ax.fill(angles, values, alpha=0.25)
    
    ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics.values())
    ax.set_title("策略性能对比雷达图", size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(f"{save_path}/radar_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("已保存雷达对比图")
    
    # 3. 绘制训练曲线对比
    plt.figure(figsize=(12, 8))
    for policy_name, df in results_dict.items():
        if 'reward' in df.columns:
            sns.lineplot(data=df, x=df.index, y='reward', 
                        label=policy_name, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("平均奖励")
    plt.title("不同策略的奖励曲线对比")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/reward_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("已保存奖励曲线对比图")

def plot_individual_metrics(results_dict: Dict[str, pd.DataFrame], 
                          save_path: str = "results/comparison_plots") -> None:
    """为每个策略绘制详细的指标分布"""
    os.makedirs(save_path, exist_ok=True)
    
    metrics = ['throughput', 'sensing_accuracy', 'energy_consumption', 'latency']
    titles = ['通信吞吐量 (Mbps)', '感知精度 (%)', '能耗 (W)', '时延 (ms)']
    
    for policy_name, df in results_dict.items():
        plt.figure(figsize=(12, 8))
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in df.columns:
                plt.subplot(2, 2, i+1)
                sns.histplot(df[metric], kde=True, bins=20)
                plt.title(f"{policy_name} - {title}")
                plt.xlabel('')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/{policy_name}_metrics_dist.png", bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"已保存 {policy_name} 策略的指标分布图")

def main():
    """主函数：加载结果并生成可视化"""
    # 定义要比较的策略
    policy_names = ["随机策略", "贪心策略", "距离感知策略", "PPO策略"]
    
    # 加载结果
    results_dict = load_results(policy_names)
    
    if not results_dict:
        logger.error("没有可用的结果数据，请先运行评估脚本")
        return
    
    # 生成可视化
    plot_metrics_comparison(results_dict)
    plot_individual_metrics(results_dict)
    
    logger.info("所有可视化图表已生成并保存至 results/comparison_plots 目录")

if __name__ == "__main__":
    main()