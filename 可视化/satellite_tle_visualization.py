# satellite_tle_visualization.py
import numpy as np
import pyvista as pv
from pyvista import examples
import wandb
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, utc
import warnings

class TLEVisualizer:
    def __init__(self, tle_file="orbit_tle.txt"):
        # 初始化参数
        self.earth_radius = 6371  # 地球半径(km)
        self.fov_angle = 60       # 视场角(度)
        
        # 颜色定义
        self.colors = {
            'earth': '#1f77b4',
            'satellite': '#ff7f0e',
            'coverage': '#d62728',
            'orbit': '#7f7f7f',
            'disaster_area': '#ff0000'
        }
        
        # 加载TLE文件
        self.satellites = self._load_tle_file(tle_file)
        
        # 灾害区域坐标 (示例: 四川山区)
        self.disaster_area = {
            'latitude': 30.5,
            'longitude': 103.0,
            'radius': 150  # 灾害影响半径(km)
        }
        
        # 初始化WandB
        wandb.init(project="satellite-disaster-monitoring", 
                 config={
                     "visualization_type": "Disaster Monitoring",
                     "earth_radius": self.earth_radius,
                     "fov_angle": self.fov_angle,
                     "disaster_area": self.disaster_area
                 })

    def _load_tle_file(self, tle_file):
        """加载TLE文件并解析卫星数据"""
        satellites = []
        with open(tle_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                name = lines[i].strip()
                line1 = lines[i+1].strip()
                line2 = lines[i+2].strip()
                satellites.append({
                    'name': name,
                    'satellite': EarthSatellite(line1, line2, name),
                    'color': np.random.rand(3)  # 为每颗卫星分配随机颜色
                })
        return satellites

    def _calculate_orbit(self, satellite, steps=100, days=1):
        """计算卫星轨道位置"""
        ts = load.timescale()
        now = datetime.utcnow().replace(tzinfo=utc)
        times = [ts.utc(now + timedelta(minutes=15*i)) 
                for i in range(steps)]
        
        positions = []
        for time in times:
            geocentric = satellite.at(time)
            subpoint = geocentric.subpoint()
            positions.append([
                subpoint.longitude.degrees,
                subpoint.latitude.degrees,
                subpoint.elevation.km
            ])
        
        return np.array(positions)

    def _calculate_coverage(self, position, radius_km=1000):
        """计算卫星覆盖区域"""
        lon, lat, alt = position
        radius_deg = np.degrees(radius_km / self.earth_radius)
        
        # 创建覆盖区域网格
        theta = np.linspace(0, 2*np.pi, 50)
        phi = np.linspace(0, np.radians(self.fov_angle), 25)
        theta, phi = np.meshgrid(theta, phi)
        
        # 转换为笛卡尔坐标
        x = (self.earth_radius + alt) * np.sin(phi) * np.cos(theta)
        y = (self.earth_radius + alt) * np.sin(phi) * np.sin(theta)
        z = (self.earth_radius + alt) * np.cos(phi)
        
        # 旋转到正确位置
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        
        # 旋转矩阵
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(lat_rad), -np.sin(lat_rad)],
            [0, np.sin(lat_rad), np.cos(lat_rad)]
        ])
        
        rot_z = np.array([
            [np.cos(lon_rad), -np.sin(lon_rad), 0],
            [np.sin(lon_rad), np.cos(lon_rad), 0],
            [0, 0, 1]
        ])
        
        # 应用旋转
        xyz = np.stack([x, y, z], axis=-1)
        xyz = np.dot(xyz, rot_x.T)
        xyz = np.dot(xyz, rot_z.T)
        
        return xyz[..., 0], xyz[..., 1], xyz[..., 2]

    def _create_disaster_area(self):
        """创建灾害区域3D模型"""
        theta = np.linspace(0, 2*np.pi, 50)
        r = np.linspace(0, self.disaster_area['radius'], 10)
        theta, r = np.meshgrid(theta, r)
        
        # 转换为笛卡尔坐标
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(x)
        
        # 转换为地球表面坐标
        lat_rad = np.radians(self.disaster_area['latitude'])
        lon_rad = np.radians(self.disaster_area['longitude'])
        
        x_earth = self.earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
        y_earth = self.earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
        z_earth = self.earth_radius * np.sin(lat_rad)
        
        # 旋转到正确位置
        rot_matrix = np.array([
            [-np.sin(lon_rad), -np.sin(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.cos(lon_rad)],
            [np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)*np.sin(lon_rad)],
            [0, np.cos(lat_rad), np.sin(lat_rad)]
        ])
        
        xyz = np.stack([x, y, z], axis=-1)
        xyz = np.dot(xyz, rot_matrix.T)
        x = xyz[..., 0] + x_earth
        y = xyz[..., 1] + y_earth
        z = xyz[..., 2] + z_earth
        
        return x, y, z

    def create_pyvista_visualization(self):
        """使用PyVista创建3D可视化"""
        try:
            # 创建绘图窗口（优先使用交互模式调试）
            plotter = pv.Plotter(off_screen=False)
            
            # 方法1：使用示例地球模型（兼容旧版）
            try:
                earth = examples.load_globe()
                texture = examples.load_globe_texture()
                plotter.add_mesh(earth, texture=texture)
            except:
                # 方法2：手动创建地球模型
                earth = pv.Sphere(radius=self.earth_radius,
                                 theta_resolution=120,
                                 phi_resolution=120)
                plotter.add_mesh(earth, color=self.colors['earth'])
            
            # 添加灾害区域
            da_x, da_y, da_z = self._create_disaster_area()
            disaster_area = pv.StructuredGrid(da_x, da_y, da_z)
            plotter.add_mesh(disaster_area, color=self.colors['disaster_area'], opacity=0.7)
            
            # 添加卫星轨道和位置（简化版）
            for i, sat in enumerate(self.satellites[:3]):  # 先只显示3颗卫星测试
                positions = self._calculate_orbit(sat['satellite'])
                x = (self.earth_radius + positions[:,2]) * np.cos(np.radians(positions[:,1])) * np.cos(np.radians(positions[:,0]))
                y = (self.earth_radius + positions[:,2]) * np.cos(np.radians(positions[:,1])) * np.sin(np.radians(positions[:,0]))
                z = (self.earth_radius + positions[:,2]) * np.sin(np.radians(positions[:,1]))
                
                orbit = pv.lines_from_points(np.column_stack([x, y, z]))
                plotter.add_mesh(orbit, color=tuple(sat['color']), line_width=2)
                
                sat_pos = pv.Sphere(radius=500, center=(x[0], y[0], z[0]))
                plotter.add_mesh(sat_pos, color=tuple(sat['color']))
            
            # 设置视图
            plotter.camera_position = [(0, -20000, 10000), (0, 0, 0), (0, 0, 1)]
            plotter.add_text("卫星灾害监测系统", position="upper_left", font_size=18)
            
            # 显示窗口
            plotter.show(auto_close=False)
            
            # 保存截图
            screenshot = "satellite_disaster_pyvista.png"
            plotter.screenshot(screenshot)
            print(f"截图已保存到: {screenshot}")
            wandb.log({"visualization": wandb.Image(screenshot)})
            
            # 保持窗口打开
            input("按Enter键关闭可视化窗口...")
            plotter.close()
            
        except Exception as e:
            print(f"可视化生成错误: {str(e)}")
            raise

    def generate_visualizations(self):
        """生成所有可视化"""
        print(f"正在为{len(self.satellites)}颗卫星生成灾害监测可视化...")
        try:
            print("生成PyVista可视化...")
            self.create_pyvista_visualization()
        except Exception as e:
            print(f"可视化生成失败: {str(e)}")
            wandb.log({"error": str(e)})
        finally:
            wandb.finish()

if __name__ == "__main__":
    try:
        # 检查PyVista版本
        print(f"PyVista版本: {pv.__version__}")
        
        # 创建可视化器实例
        visualizer = TLEVisualizer("orbit_tle.txt")
        visualizer.generate_visualizations()
    except Exception as e:
        print(f"程序运行错误: {str(e)}")
        input("按Enter键退出...")