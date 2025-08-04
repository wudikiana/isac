import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math

# 设置页面配置
st.set_page_config(
    page_title="卫星ISAC灾害监测系统",
    page_icon="🛰️",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .satellite-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def parse_real_tle_data():
    """从orbit_tle.txt解析真实TLE数据"""
    tle_data = []
    try:
        with open('orbit_tle.txt', 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        st.error("无法找到orbit_tle.txt文件，请确保它和脚本在同一目录")
        return []
    
    i = 0
    while i < len(lines):
        if not lines[i].startswith('1 ') and not lines[i].startswith('2 '):
            name = lines[i]
            if i+2 < len(lines) and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
                line1 = lines[i+1]
                line2 = lines[i+2]
                
                try:
                    # 解析轨道参数
                    inclination = float(line2[8:16].strip() or 0)  # 轨道倾角(度)
                    
                    # 离心率处理
                    ecc_str = line2[26:33].strip()
                    eccentricity = float("0." + ecc_str) if ecc_str else 0.0
                    
                    # 改进的平均运动解析
                    mean_motion_str = line2[52:63].replace(' ', '')
                    mean_motion = float(mean_motion_str) if mean_motion_str else 0.0
                    
                    # 计算轨道周期(分钟) - 添加最小值保护
                    period = max(0.1, 1440 / mean_motion) if mean_motion > 0 else 0.1
                    
                    # 计算轨道高度(km) - 添加保护措施
                    try:
                        altitude = max(0, (398600.4418**(1/3) * (period * 60 / (2 * math.pi))**(2/3)) - 6378)
                    except:
                        altitude = 500  # 默认值
                    
                    # 确定卫星类型
                    if altitude < 2000:
                        sat_type = 'LEO'
                        coverage_radius = 1000 + altitude * 0.5
                    elif 2000 <= altitude < 35786:
                        sat_type = 'MEO'
                        coverage_radius = 3000 + altitude * 0.3
                    else:
                        sat_type = 'GEO'
                        coverage_radius = 18000
                    
                    # 确保覆盖半径为正值
                    coverage_radius = max(100, coverage_radius)
                    coverage_area = math.pi * (coverage_radius ** 2)
                    
                    tle_data.append({
                        'name': name,
                        'type': sat_type,
                        'altitude': round(altitude),
                        'inclination': inclination,
                        'period': round(period, 2),
                        'eccentricity': eccentricity,
                        'coverage_radius': round(coverage_radius),
                        'coverage_area': round(coverage_area),
                        'color': 'red' if sat_type == 'LEO' else 'blue' if sat_type == 'MEO' else 'green'
                    })
                    
                except Exception as e:
                    st.warning(f"解析卫星 {name} 数据时出错: {str(e)}")
                    continue
                
                i += 3
            else:
                i += 1
        else:
            i += 1
            
    return tle_data

def calculate_satellite_position(satellite, time_hours):
    """计算卫星位置"""
    earth_radius = 6371  # km
    altitude = satellite['altitude']
    period = satellite['period']  # minutes
    
    # 简化的轨道计算
    orbit_radius = earth_radius + altitude
    angular_velocity = 2 * np.pi / (period * 60)  # rad/s
    
    # 计算轨道上的位置
    angle = angular_velocity * time_hours * 3600  # 当前角度
    
    # 考虑轨道倾角
    inclination_rad = np.radians(satellite['inclination'])
    
    # 计算3D坐标
    x = orbit_radius * np.cos(angle) * np.cos(inclination_rad)
    y = orbit_radius * np.sin(angle)
    z = orbit_radius * np.cos(angle) * np.sin(inclination_rad)
    
    return x, y, z

def create_multi_satellite_3d_scene():
    """创建多卫星3D场景"""
    satellites = parse_real_tle_data()
    
    fig = go.Figure()
    
    # 添加地球
    earth_radius = 6371
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(-np.pi/2, np.pi/2, 25)
    theta, phi = np.meshgrid(theta, phi)
    
    x_earth = earth_radius * np.cos(phi) * np.cos(theta)
    y_earth = earth_radius * np.cos(phi) * np.sin(theta)
    z_earth = earth_radius * np.sin(phi)
    
    fig.add_trace(go.Surface(
        x=x_earth,
        y=y_earth,
        z=z_earth,
        colorscale='Blues',
        opacity=0.8,
        name='地球',
        showscale=False
    ))
    
    # 添加卫星轨道和位置
    time_hours = 2  # 当前时间（小时）
    
    for sat in satellites:
        # 计算轨道
        orbit_points = []
        for t in np.linspace(0, 2*np.pi, 100):
            x, y, z = calculate_satellite_position(sat, t * sat['period'] / (2*np.pi) / 60)
            orbit_points.append([x, y, z])
        
        orbit_points = np.array(orbit_points)
        
        # 添加轨道线
        fig.add_trace(go.Scatter3d(
            x=orbit_points[:, 0],
            y=orbit_points[:, 1],
            z=orbit_points[:, 2],
            mode='lines',
            line=dict(color=sat['color'], width=2),
            name=f"{sat['name']} 轨道",
            showlegend=True
        ))
        
        # 添加卫星当前位置
        x, y, z = calculate_satellite_position(sat, time_hours)
        fig.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='markers',
            marker=dict(size=8, color=sat['color'], symbol='diamond'),
            name=f"{sat['name']} 当前位置",
            showlegend=False
        ))
    
    # 修改灾害区域的标记符号为支持的3D符号
    fig.add_trace(go.Scatter3d(
        x=[0, 0, 0],
        y=[6371, 6371.5, 6371],
        z=[0, 0.5, 0],
        mode='markers',
        marker=dict(size=15, color='orange', symbol='cross'),
        name='灾害区域'
    ))
    
    fig.update_layout(
        title='多卫星ISAC系统3D场景 (基于真实TLE数据)',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        ),
        width=1000,
        height=700
    )
    
    return fig

def create_satellite_coverage_analysis():
    """创建基于真实数据的卫星覆盖分析"""
    satellites = parse_real_tle_data()
    
    df = pd.DataFrame([{
        '卫星': sat['name'],
        '类型': sat['type'],
        '覆盖半径(km)': sat['coverage_radius'],
        '覆盖面积(km²)': sat['coverage_area'],
        '高度(km)': sat['altitude'],
        '倾角(°)': sat['inclination'],
        '周期(min)': sat['period']
    } for sat in satellites])
    
    # 按覆盖面积排序
    df = df.sort_values('覆盖面积(km²)', ascending=False)
    
    # 创建覆盖面积对比图
    fig = px.bar(
        df, 
        x='卫星', 
        y='覆盖面积(km²)',
        color='类型',
        title='卫星覆盖面积对比(基于真实TLE数据)',
        color_discrete_map={'LEO': 'red', 'MEO': 'blue', 'GEO': 'green'},
        hover_data=['高度(km)', '倾角(°)', '周期(min)']
    )
    
    fig.update_layout(height=500)
    return fig, df

def create_satellite_performance_dashboard():
    """创建卫星性能仪表板（修复负数问题版）"""
    satellites = parse_real_tle_data()
    
    performance_data = []
    for sat in satellites:
        # 确保关键参数有效
        altitude = max(0, sat['altitude'])  # 高度不小于0
        
        # 安全计算性能指标
        try:
            if sat['type'] == 'LEO':
                comm_capacity = 100 + math.log1p(altitude) * 20
                sensing_range = 800 + altitude * 0.6
                processing_power = 15 + altitude * 0.03
                energy_efficiency = 0.75 + min(altitude * 0.0008, 0.24)
            elif sat['type'] == 'MEO':
                comm_capacity = 150 + math.log1p(altitude) * 15
                sensing_range = 2500 + altitude * 0.4
                processing_power = 25 + altitude * 0.015
                energy_efficiency = 0.82 + min(altitude * 0.0003, 0.17)
            else:  # GEO
                comm_capacity = 300 + math.log1p(altitude) * 10
                sensing_range = 6000 + altitude * 0.05
                processing_power = 40 + altitude * 0.002
                energy_efficiency = 0.88 + min(altitude * 0.0001, 0.11)
                
            # 确保最小值
            comm_capacity = max(10, comm_capacity)
            sensing_range = max(100, sensing_range)
            processing_power = max(5, processing_power)
            energy_efficiency = max(0.5, min(energy_efficiency, 0.99))
            
        except Exception as e:
            st.warning(f"计算卫星 {sat['name']} 性能时出错: {str(e)}")
            continue
            
        performance_data.append({
            '卫星': sat['name'],
            '通信容量(Mbps)': round(comm_capacity, 2),
            '感知范围(km)': round(sensing_range, 2),
            '处理能力(TFLOPS)': round(processing_power, 2),
            '能效比': round(energy_efficiency, 3)
        })
    
    # ... 其余代码保持不变 ...
    
    df_perf = pd.DataFrame(performance_data)
    
    # 创建性能对比图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('通信容量', '感知范围', '处理能力', '能效比')
    )
    
    # 通信容量
    fig.add_trace(
        go.Bar(x=df_perf['卫星'], y=df_perf['通信容量(Mbps)'], name='通信容量'),
        row=1, col=1
    )
    
    # 感知范围
    fig.add_trace(
        go.Bar(x=df_perf['卫星'], y=df_perf['感知范围(km)'], name='感知范围'),
        row=1, col=2
    )
    
    # 处理能力
    fig.add_trace(
        go.Bar(x=df_perf['卫星'], y=df_perf['处理能力(TFLOPS)'], name='处理能力'),
        row=2, col=1
    )
    
    # 能效比
    fig.add_trace(
        go.Bar(x=df_perf['卫星'], y=df_perf['能效比'], name='能效比'),
        row=2, col=2
    )
    
    fig.update_layout(
        title='多卫星ISAC性能对比 (基于轨道特性)',
        height=600,
        showlegend=False
    )
    
    return fig, df_perf

def create_coordination_scenario():
    """创建多卫星协同场景"""
    satellites = parse_real_tle_data()
    leo_sats = [sat['name'] for sat in satellites if sat['type'] == 'LEO']
    meo_sats = [sat['name'] for sat in satellites if sat['type'] == 'MEO']
    geo_sats = [sat['name'] for sat in satellites if sat['type'] == 'GEO']
    
    # 模拟协同任务分配（基于卫星类型）
    coordination_data = {
        '任务': ['灾害区域扫描', '通信链路建立', '数据处理', '结果回传', '应急响应'],
        '主卫星': [
            np.random.choice(leo_sats),
            np.random.choice(meo_sats if meo_sats else leo_sats),
            np.random.choice(leo_sats),
            np.random.choice(geo_sats if geo_sats else meo_sats if meo_sats else leo_sats),
            np.random.choice(leo_sats)
        ],
        '辅助卫星': [
            np.random.choice(leo_sats),
            np.random.choice(leo_sats),
            np.random.choice(meo_sats if meo_sats else leo_sats),
            np.random.choice(leo_sats),
            np.random.choice(meo_sats if meo_sats else leo_sats)
        ],
        '完成时间(min)': [3, 2, 4, 1, 5],
        '状态': ['进行中', '已完成', '进行中', '等待中', '准备中']
    }
    
    df_coord = pd.DataFrame(coordination_data)
    
    # 创建甘特图样式的任务时间线
    fig = go.Figure()
    
    colors = {'进行中': 'orange', '已完成': 'green', '等待中': 'gray', '准备中': 'blue'}
    
    for i, task in enumerate(df_coord['任务']):
        status = df_coord.iloc[i]['状态']
        duration = df_coord.iloc[i]['完成时间(min)']
        satellite = df_coord.iloc[i]['主卫星']
        
        fig.add_trace(go.Bar(
            x=[duration],
            y=[task],
            orientation='h',
            name=satellite,
            marker_color=colors[status],
            text=f"{satellite}<br>{duration}min",
            textposition='auto'
        ))
    
    fig.update_layout(
        title='多卫星协同任务时间线 (动态分配)',
        xaxis_title='时间 (分钟)',
        yaxis_title='任务',
        height=400,
        showlegend=False
    )
    
    return fig, df_coord

def main():
    """主函数"""
    # 主标题
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ 多卫星ISAC灾害监测系统</h1>
        <p>基于真实TLE数据的通感算智一体化低轨卫星边缘计算平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.header("🎛️ 系统控制面板")
    
    # 场景选择
    scenario = st.sidebar.selectbox(
        "选择监测场景",
        ["暴雨山洪监测", "滑坡灾害预警", "森林火情识别", "海上风暴监测"]
    )
    
    # 卫星选择
    st.sidebar.header("🛰️ 卫星选择")
    satellites = parse_real_tle_data()
    selected_satellites = st.sidebar.multiselect(
        "选择参与卫星",
        [sat['name'] for sat in satellites],
        default=[sat['name'] for sat in satellites if sat['type'] == 'LEO'][:3]
    )
    
    # 时间控制
    st.sidebar.header("⏱️ 时间控制")
    current_time = st.sidebar.slider(
        "当前时间 (小时)",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )
    
    # 计算关键指标
    selected_sats_data = [sat for sat in satellites if sat['name'] in selected_satellites]
    total_coverage = sum(sat['coverage_area'] for sat in selected_sats_data) if selected_sats_data else 0
    avg_response_time = 5 + len(selected_satellites) * 0.5  # 模拟响应时间计算
    coordination_efficiency = 80 + len(selected_satellites) * 2  # 模拟协同效率
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>参与卫星</h4>
            <h3>{len(selected_satellites)}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>总覆盖面积</h4>
            <h3>{total_coverage/1e6:.2f}M km²</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>响应时间</h4>
            <h3>≤{avg_response_time:.1f} min</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>协同效率</h4>
            <h3>{min(coordination_efficiency, 95)}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # 主内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 3D场景", "📡 覆盖分析", "⚡ 性能对比", "🤝 协同任务", "📋 卫星信息"
    ])
    
    with tab1:
        st.header("🌍 多卫星ISAC系统3D场景 (基于真实TLE数据)")
        st.write("展示多颗卫星的轨道分布和当前位置")
        
        fig_3d = create_multi_satellite_3d_scene()
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 场景说明
        st.markdown("""
        ### 场景说明
        - **不同颜色轨道**: 红色=LEO, 蓝色=MEO, 绿色=GEO
        - **菱形标记**: 当前卫星位置
        - **橙色十字标记**: 灾害监测区域
        - **蓝色球体**: 地球表面
        - **所有轨道参数**: 来自真实TLE数据
        """)
    
    with tab2:
        st.header("📡 卫星覆盖分析 (真实覆盖范围)")
        st.write("分析各卫星的真实覆盖能力和范围")
        
        fig_coverage, df_coverage = create_satellite_coverage_analysis()
        st.plotly_chart(fig_coverage, use_container_width=True)
        
        st.markdown("### 覆盖数据详情")
        st.dataframe(df_coverage, use_container_width=True)
    
    with tab3:
        st.header("⚡ 多卫星性能对比")
        st.write("基于轨道特性的卫星ISAC性能指标对比")
        
        fig_perf, df_perf = create_satellite_performance_dashboard()
        st.plotly_chart(fig_perf, use_container_width=True)
        
        st.markdown("### 性能数据详情")
        st.dataframe(df_perf, use_container_width=True)
    
    with tab4:
        st.header("🤝 多卫星协同任务")
        st.write("基于卫星类型的动态任务分配时间线")
        
        fig_coord, df_coord = create_coordination_scenario()
        st.plotly_chart(fig_coord, use_container_width=True)
        
        st.markdown("### 协同任务详情")
        st.dataframe(df_coord, use_container_width=True)
    
    with tab5:
        st.header("📋 卫星详细信息")
        st.write("显示各卫星的真实轨道参数和技术规格")
        
        for sat in satellites:
            if sat['name'] in selected_satellites:
                st.markdown(f"""
                <div class="satellite-info">
                    <h4>🛰️ {sat['name']}</h4>
                    <p><strong>类型:</strong> {sat['type']}</p>
                    <p><strong>轨道高度:</strong> {sat['altitude']} km</p>
                    <p><strong>轨道倾角:</strong> {sat['inclination']}°</p>
                    <p><strong>轨道周期:</strong> {sat['period']} 分钟</p>
                    <p><strong>覆盖半径:</strong> {sat['coverage_radius']} km</p>
                    <p><strong>离心率:</strong> {sat['eccentricity']:.6f}</p>
                    <p><strong>状态:</strong> {'🟢 在线' if sat['type'] == 'LEO' else '🟡 待机' if sat['type'] == 'MEO' else '🔵 同步'}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🛰️ 基于真实TLE数据的卫星ISAC技术方案 | 多卫星协同通感算智一体化平台</p>
        <p>📊 支持LEO/MEO/GEO多轨道卫星协同，实现全域灾害监测与应急响应</p>
        <p>🕒 数据更新时间: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()