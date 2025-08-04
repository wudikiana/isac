#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星推理服务器
专门为重构后的卫星系统设计的推理服务器
"""

import torch
import numpy as np
import time
import json
import socket
import threading
import logging
import pickle
import struct
from typing import Dict, Any, Optional
from dataclasses import dataclass
import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 检查必要的依赖
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("❌ 缺少依赖: segmentation_models_pytorch")
    print("请运行: pip install segmentation-models-pytorch")
    sys.exit(1)

try:
    from train_model import EnhancedDeepLab
except ImportError as e:
    print(f"❌ 导入EnhancedDeepLab失败: {e}")
    print("请确保train_model.py文件存在且包含EnhancedDeepLab类")
    sys.exit(1)

@dataclass
class InferenceRequest:
    """推理请求数据类"""
    task_id: str
    image_data: np.ndarray
    sim_features: Optional[np.ndarray]
    priority: int
    timestamp: float

@dataclass
class InferenceResponse:
    """推理响应数据类"""
    task_id: str
    prediction: np.ndarray
    processing_time: float
    status: str
    error_message: Optional[str] = None

class SatelliteInferenceServer:
    """卫星端推理服务器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, 
                 model_path: str = "models/best_multimodal_patch_model.pth",
                 satellite_id: str = "sat_001"):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.satellite_id = satellite_id
        
        # 初始化模型
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 服务器状态
        self.running = False
        self.current_load = 0.0
        self.processed_tasks = 0
        self.failed_tasks = 0
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 启动监控线程
        self._start_monitoring()
    
    def _load_model(self) -> Optional[EnhancedDeepLab]:
        """加载推理模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                self.logger.error(f"模型文件不存在: {self.model_path}")
                self.logger.info("请确保模型文件存在，或使用 --model 参数指定正确的模型路径")
                return None
            
            model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
            
            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 检查checkpoint格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果checkpoint直接是state_dict
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # 移动到设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            self.logger.info(f"模型加载成功: {self.model_path}")
            self.logger.info(f"使用设备: {device}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.logger.info("请检查模型文件格式是否正确")
            return None
    
    def _start_monitoring(self):
        """启动监控线程"""
        def monitor_loop():
            while self.running:
                try:
                    # 更新负载信息
                    self._update_load_info()
                    time.sleep(30)  # 每30秒更新一次
                except Exception as e:
                    self.logger.error(f"监控线程错误: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("监控线程已启动")
    
    def _update_load_info(self):
        """更新负载信息"""
        try:
            import psutil
            self.current_load = psutil.cpu_percent() / 100.0
        except ImportError:
            # 如果没有psutil，使用简单的负载估算
            self.current_load = min(1.0, self.processed_tasks / 1000.0)
    
    def start_server(self):
        """启动服务器"""
        if self.model is None:
            self.logger.error("模型未加载，无法启动服务器")
            self.logger.info("请检查模型文件路径是否正确")
            return
        
        self.running = True
        self.logger.info(f"启动推理服务器: {self.host}:{self.port}")
        self.logger.info(f"卫星ID: {self.satellite_id}")
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.host, self.port))
                server_socket.listen(5)
                
                self.logger.info("服务器启动成功，等待连接...")
                
                while self.running:
                    try:
                        client_socket, address = server_socket.accept()
                        self.logger.info(f"接受连接: {address}")
                        
                        # 为每个客户端创建处理线程
                        client_thread = threading.Thread(
                            target=self._handle_client,
                            args=(client_socket, address)
                        )
                        client_thread.daemon = True
                        client_thread.start()
                        
                    except Exception as e:
                        self.logger.error(f"接受连接错误: {e}")
                        
        except Exception as e:
            self.logger.error(f"服务器启动失败: {e}")
        finally:
            self.running = False
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """处理客户端连接"""
        try:
            while self.running:
                # 接收数据长度
                length_data = client_socket.recv(4)
                if not length_data:
                    break
                
                data_length = struct.unpack('!I', length_data)[0]
                
                # 接收数据
                data = b''
                while len(data) < data_length:
                    chunk = client_socket.recv(data_length - len(data))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) != data_length:
                    self.logger.warning(f"数据接收不完整: {len(data)}/{data_length}")
                    continue
                
                # 解析请求
                try:
                    request_data = pickle.loads(data)
                    
                    # 处理不同类型的请求
                    if request_data.get('type') == 'heartbeat':
                        response = self._handle_heartbeat(request_data)
                    elif request_data.get('type') == 'get_load':
                        response = self._handle_get_load(request_data)
                    elif request_data.get('type') == 'inference':
                        request = InferenceRequest(**request_data)
                        response = self._process_inference(request)
                    else:
                        response = InferenceResponse(
                            task_id="unknown",
                            prediction=np.array([]),
                            processing_time=0.0,
                            status="error",
                            error_message="Unknown request type"
                        )
                    
                    # 发送响应
                    response_data = pickle.dumps(response.__dict__)
                    response_length = struct.pack('!I', len(response_data))
                    client_socket.send(response_length + response_data)
                    
                except Exception as e:
                    self.logger.error(f"处理请求错误: {e}")
                    # 发送错误响应
                    error_response = InferenceResponse(
                        task_id="unknown",
                        prediction=np.array([]),
                        processing_time=0.0,
                        status="error",
                        error_message=str(e)
                    )
                    response_data = pickle.dumps(error_response.__dict__)
                    response_length = struct.pack('!I', len(response_data))
                    client_socket.send(response_length + response_data)
                    
        except Exception as e:
            self.logger.error(f"客户端处理错误 {address}: {e}")
        finally:
            client_socket.close()
            self.logger.info(f"客户端连接关闭: {address}")
    
    def _handle_heartbeat(self, request_data: dict) -> dict:
        """处理心跳请求"""
        return {
            'type': 'heartbeat_response',
            'status': 'ok',
            'satellite_id': self.satellite_id,
            'timestamp': time.time(),
            'load': self.current_load
        }
    
    def _handle_get_load(self, request_data: dict) -> dict:
        """处理获取负载信息请求"""
        return {
            'type': 'load_response',
            'status': 'ok',
            'satellite_id': self.satellite_id,
            'load': self.current_load,
            'processed_tasks': self.processed_tasks,
            'failed_tasks': self.failed_tasks,
            'training_data_count': 0,  # 可以扩展
            'last_training_time': 0.0,
            'model_hash': "model_hash_placeholder",
            'parameter_version': 1
        }
    
    def _process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """处理推理请求"""
        start_time = time.time()
        
        try:
            # 预处理图像数据
            img_tensor = torch.tensor(request.image_data, dtype=torch.float32)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # 移动到设备
            img_tensor = img_tensor.to(self.device)
            
            # 准备仿真特征
            if request.sim_features is not None:
                sim_feats = torch.tensor(request.sim_features, dtype=torch.float32)
                if sim_feats.dim() == 1:
                    sim_feats = sim_feats.unsqueeze(0)
                sim_feats = sim_feats.to(self.device)
            else:
                sim_feats = torch.zeros(1, 11, dtype=torch.float32, device=self.device)
            
            # 执行推理
            with torch.no_grad():
                pred = self.model(img_tensor, sim_feats)
                pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self.processed_tasks += 1
            
            self.logger.info(f"推理完成: {request.task_id}, 耗时: {processing_time:.3f}s")
            
            return InferenceResponse(
                task_id=request.task_id,
                prediction=pred_mask,
                processing_time=processing_time,
                status="success"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_tasks += 1
            
            self.logger.error(f"推理失败: {request.task_id}, 错误: {e}")
            
            return InferenceResponse(
                task_id=request.task_id,
                prediction=np.array([]),
                processing_time=processing_time,
                status="error",
                error_message=str(e)
            )
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务器状态"""
        return {
            "satellite_id": self.satellite_id,
            "running": self.running,
            "current_load": self.current_load,
            "processed_tasks": self.processed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (self.processed_tasks - self.failed_tasks) / max(self.processed_tasks, 1),
            "device": str(self.device),
            "model_loaded": self.model is not None
        }
    
    def stop_server(self):
        """停止服务器"""
        self.running = False
        self.logger.info("服务器停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='卫星推理服务器')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口')
    parser.add_argument('--model', type=str, default='models/best_multimodal_patch_model.pth', 
                       help='模型路径')
    parser.add_argument('--satellite_id', type=str, default='sat_001', help='卫星ID')
    
    args = parser.parse_args()
    
    # 创建服务器
    server = SatelliteInferenceServer(
        host=args.host,
        port=args.port,
        model_path=args.model,
        satellite_id=args.satellite_id
    )
    
    print(f"🚀 启动卫星推理服务器")
    print(f"   卫星ID: {args.satellite_id}")
    print(f"   地址: {args.host}:{args.port}")
    print(f"   模型: {args.model}")
    print(f"   设备: {server.device}")
    print(f"   模型状态: {'已加载' if server.model else '未加载'}")
    
    try:
        # 启动服务器
        server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 收到停止信号")
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
    finally:
        server.stop_server()
        print("✅ 服务器已停止")


if __name__ == "__main__":
    main() 