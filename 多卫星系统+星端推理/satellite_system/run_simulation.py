#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多卫星节点模拟启动脚本
"""

import argparse
import socket
import threading
import time
import json
import logging
import numpy as np
import torch
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from satellite_system.satellite_core import SatelliteStatus, CoverageStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimulatedSatelliteNode:
    """模拟卫星节点"""
    
    def __init__(self, satellite_id: str, config: dict):
        self.satellite_id = satellite_id
        self.config = config
        self.status = SatelliteStatus.ONLINE
        self.current_load = 0.0
        self.federated_score = 0.8
        self.model_version_score = 0.9
        
        # 网络配置
        self.ip_address = config['ip']
        self.port = config['port']
        self.server_socket = None
        
        # 任务队列
        self.task_queue = []
        self.results_cache = {}
        
        # 模拟模型
        self.local_model = None
        self._load_simulated_model()
        
        logger.info(f"模拟卫星节点 {satellite_id} 初始化完成")
    
    def _load_simulated_model(self):
        """加载模拟模型"""
        try:
            from train_model import EnhancedDeepLab
            self.local_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
            self.local_model.eval()
            logger.info(f"卫星 {self.satellite_id} 模型加载成功")
        except Exception as e:
            logger.warning(f"卫星 {self.satellite_id} 模型加载失败: {e}")
            self.local_model = None
    
    def start_server(self):
        """启动服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.ip_address, self.port))
            self.server_socket.listen(5)
            
            logger.info(f"卫星 {self.satellite_id} 服务器启动在 {self.ip_address}:{self.port}")
            
            # 启动监听线程
            listen_thread = threading.Thread(target=self._listen_for_connections, daemon=True)
            listen_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"卫星 {self.satellite_id} 服务器启动失败: {e}")
            return False
    
    def _listen_for_connections(self):
        """监听连接"""
        while True:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"卫星 {self.satellite_id} 接受连接: {address}")
                
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                logger.error(f"卫星 {self.satellite_id} 连接处理错误: {e}")
                break
    
    def _handle_client(self, client_socket, address):
        """处理客户端连接"""
        try:
            # 接收请求
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                try:
                    request = json.loads(data)
                    response = self._process_request(request)
                except json.JSONDecodeError:
                    # 如果不是JSON，发送默认心跳响应
                    response = self._get_heartbeat_response()
            else:
                response = self._get_heartbeat_response()
            
            # 发送JSON响应
            response_str = json.dumps(response)
            client_socket.send(response_str.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"卫星 {self.satellite_id} 客户端处理错误: {e}")
        finally:
            client_socket.close()
    
    def _process_request(self, request: dict) -> dict:
        """处理请求"""
        request_type = request.get('type', 'heartbeat')
        
        if request_type == 'heartbeat':
            return self._get_heartbeat_response()
        elif request_type == 'get_load':
            return self._get_load_response()
        elif request_type == 'inference':
            return self._get_inference_response(request)
        else:
            return self._get_heartbeat_response()
    
    def _get_heartbeat_response(self) -> dict:
        """获取心跳响应"""
        return {
            'status': 'ok',
            'satellite_id': self.satellite_id,
            'timestamp': time.time(),
            'federated_score': self.federated_score,
            'current_load': self.current_load,
            'status': self.status.value,
            'type': 'heartbeat'
        }
    
    def _get_load_response(self) -> dict:
        """获取负载响应"""
        return {
            'status': 'ok',
            'load': self.current_load,
            'federated_score': self.federated_score,
            'model_version_score': self.model_version_score,
            'type': 'load_info'
        }
    
    def _get_inference_response(self, request: dict) -> dict:
        """获取推理响应"""
        try:
            task_id = request.get('task_id', 'unknown')
            
            # 模拟推理过程
            time.sleep(0.1)  # 模拟处理时间
            
            return {
                'status': 'success',
                'task_id': task_id,
                'output': [0.5],  # 简化的输出
                'processing_time': 0.1,
                'satellite_id': self.satellite_id,
                'type': 'inference'
            }
            
        except Exception as e:
            logger.error(f"推理执行错误: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'type': 'inference'
            }
    
    def stop(self):
        """停止卫星节点"""
        if self.server_socket:
            self.server_socket.close()
        logger.info(f"卫星 {self.satellite_id} 已停止")

def load_config(config_file: str = "satellite_system/satellite_config.json") -> dict:
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return {}

def create_simulated_nodes(config: dict, num_nodes: int = 3):
    """创建模拟卫星节点"""
    nodes = []
    satellite_configs = config.get('satellites', {})
    
    for i, (sat_id, sat_config) in enumerate(list(satellite_configs.items())[:num_nodes]):
        node = SimulatedSatelliteNode(sat_id, sat_config)
        nodes.append(node)
    
    return nodes

def test_connectivity(nodes):
    """测试节点连通性"""
    logger.info("测试节点连通性...")
    
    for node in nodes:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((node.ip_address, node.port))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ 节点 {node.satellite_id} 连通性正常")
            else:
                logger.error(f"❌ 节点 {node.satellite_id} 连通性失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 节点 {node.satellite_id} 连通性测试失败: {e}")
            return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多卫星节点模拟启动脚本')
    parser.add_argument('--nodes', type=int, default=3, help='模拟节点数量')
    parser.add_argument('--config', type=str, default='satellite_system/satellite_config.json', help='配置文件路径')
    parser.add_argument('--duration', type=int, default=600, help='运行持续时间(秒)')  # 增加到10分钟
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        logger.error("配置加载失败，退出")
        return
    
    # 创建模拟节点
    nodes = create_simulated_nodes(config, args.nodes)
    logger.info(f"创建了 {len(nodes)} 个模拟卫星节点")
    
    # 启动所有节点
    for node in nodes:
        if not node.start_server():
            logger.error(f"节点 {node.satellite_id} 启动失败")
            return
    
    # 等待服务器启动
    time.sleep(3)
    
    # 测试连通性
    if not test_connectivity(nodes):
        logger.error("连通性测试失败，退出")
        return
    logger.info("✅ 所有节点连通性测试通过")
    
    logger.info(f"模拟卫星节点启动完成，运行 {args.duration} 秒...")
    logger.info("按 Ctrl+C 可提前停止")
    
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    finally:
        for node in nodes:
            node.stop()
        logger.info("所有模拟节点已停止")

if __name__ == "__main__":
    main() 