import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import os

# ==================== 文件路径配置 ====================
# 获取脚本所在目录（可视化文件夹）
script_dir = os.path.dirname(os.path.abspath(__file__))

# 方法1：假设models文件夹与可视化文件夹同级
project_root = os.path.dirname(script_dir)  # 获取可视化文件夹的父目录
models_dir = os.path.join(project_root, 'models')
csv_path = os.path.join(models_dir, 'quantization_performance.csv')

# 方法2：或者直接指定绝对路径（取消下面一行的注释）
# csv_path = r'E:\你的路径\models\quantization_performance.csv'

# 检查文件是否存在
if not os.path.exists(csv_path):
    print(f"错误：找不到文件 {csv_path}")
    print("请检查：")
    print(f"1. models文件夹是否在 {project_root} 目录下")
    print("2. 或者使用绝对路径直接指定csv位置")
    exit()

# 读取数据
df = pd.read_csv(csv_path)

# 创建results文件夹
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# ==================== 数据预处理 ====================
# 转换时间格式
df['test_time'] = pd.to_datetime(df['test_time'])

# 提取模型名称
df['model_name'] = df['model_path'].str.extract(r'models/(.*)\.pt')

# 清理模型名称显示
df['model_name'] = df['model_name'].str.replace('quantized_', '').str.replace('_model', '')

# ==================== 可视化函数 ====================
def plot_performance_comparison():
    """ 性能对比柱状图 """
    performance_df = df.groupby('model_name').agg({
        'avg_latency_ms': 'mean',
        'fps': 'mean',
        'memory_usage_mb': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 延迟对比
    sns.barplot(data=performance_df, x='model_name', y='avg_latency_ms', 
                ax=axes[0], palette='viridis')
    axes[0].set_title('平均延迟 (ms)', fontproperties='SimHei')
    axes[0].set_xlabel('模型', fontproperties='SimHei')
    axes[0].set_ylabel('延迟 (ms)', fontproperties='SimHei')
    axes[0].tick_params(axis='x', rotation=45)
    
    # FPS对比
    sns.barplot(data=performance_df, x='model_name', y='fps', 
                ax=axes[1], palette='viridis')
    axes[1].set_title('帧率 (FPS)', fontproperties='SimHei')
    axes[1].set_xlabel('模型', fontproperties='SimHei')
    axes[1].set_ylabel('FPS', fontproperties='SimHei')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 内存对比
    sns.barplot(data=performance_df, x='model_name', y='memory_usage_mb', 
                ax=axes[2], palette='viridis')
    axes[2].set_title('内存占用 (MB)', fontproperties='SimHei')
    axes[2].set_xlabel('模型', fontproperties='SimHei')
    axes[2].set_ylabel('内存 (MB)', fontproperties='SimHei')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '性能对比.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series():
    """ 时间序列趋势图 """
    time_df = df.sort_values('test_time')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 延迟趋势
    sns.lineplot(data=time_df, x='test_time', y='avg_latency_ms', 
                 hue='model_name', ax=axes[0], marker='o', palette='viridis')
    axes[0].set_title('延迟时间变化趋势', fontproperties='SimHei')
    axes[0].set_xlabel('测试时间', fontproperties='SimHei')
    axes[0].set_ylabel('延迟 (ms)', fontproperties='SimHei')
    axes[0].legend(title='模型')
    
    # FPS趋势
    sns.lineplot(data=time_df, x='test_time', y='fps', 
                 hue='model_name', ax=axes[1], marker='o', palette='viridis')
    axes[1].set_title('帧率变化趋势', fontproperties='SimHei')
    axes[1].set_xlabel('测试时间', fontproperties='SimHei')
    axes[1].set_ylabel('FPS', fontproperties='SimHei')
    axes[1].legend(title='模型')
    
    # 内存趋势
    sns.lineplot(data=time_df, x='test_time', y='memory_usage_mb', 
                 hue='model_name', ax=axes[2], marker='o', palette='viridis')
    axes[2].set_title('内存占用变化趋势', fontproperties='SimHei')
    axes[2].set_xlabel('测试时间', fontproperties='SimHei')
    axes[2].set_ylabel('内存 (MB)', fontproperties='SimHei')
    axes[2].legend(title='模型')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '时间序列分析.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart():
    """ 模型性能雷达图 """
    radar_df = df.groupby('model_name').agg({
        'avg_latency_ms': 'mean',
        'fps': 'mean',
        'memory_usage_mb': 'mean',
        'std_latency_ms': 'mean'
    }).reset_index()
    
    # 数据归一化
    radar_df['latency_norm'] = 1 - (radar_df['avg_latency_ms'] / radar_df['avg_latency_ms'].max())
    radar_df['fps_norm'] = radar_df['fps'] / radar_df['fps'].max()
    radar_df['memory_norm'] = 1 - (radar_df['memory_usage_mb'] / radar_df['memory_usage_mb'].max())
    radar_df['stability_norm'] = 1 - (radar_df['std_latency_ms'] / radar_df['std_latency_ms'].max())
    
    categories = ['延迟', '帧率', '内存', '稳定性']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    for idx, row in radar_df.iterrows():
        values = [
            row['latency_norm'],
            row['fps_norm'],
            row['memory_norm'],
            row['stability_norm']
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['model_name'])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontproperties='SimHei')
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('模型性能雷达图 (归一化)', fontproperties='SimHei', y=1.1)
    
    plt.savefig(os.path.join(results_dir, '性能雷达图.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 执行可视化 ====================
if __name__ == "__main__":
    print("正在生成可视化图表...")
    plot_performance_comparison()
    plot_time_series()
    plot_radar_chart()
    
    print(f"\n生成结果已保存到: {results_dir}")
    print("包含以下文件:")
    print("  - 性能对比.png")
    print("  - 时间序列分析.png")
    print("  - 性能雷达图.png")
    
    # 在Windows系统中自动打开结果文件夹
    if os.name == 'nt':
        os.startfile(results_dir)