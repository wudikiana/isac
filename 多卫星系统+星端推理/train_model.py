import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

# 添加当前目录到Python路径，确保可以导入data_utils模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset  # 修复基类缺失问题
from PIL import Image  # 后续代码需要 PIL 库
import csv 
import glob  # 用于文件路径匹配
import albumentations as A  # 用于图像增强
from albumentations.pytorch import ToTensorV2  # 用于转换到张量
import torchvision.transforms as T  # 用于图像变换
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入后处理相关库
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk, ball
from scipy.ndimage import distance_transform_edt

# 导入我们的LandslideDetector模型
from models.starlite_cnn import create_starlite_model, create_enhanced_model, create_segmentation_landslide_model
# 启用TF32计算
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ====== 后处理函数 ======
def postprocess(output, min_area=100, merge_distance=10, debug_mode=False):
    """
    增强的后处理流程：阈值化 -> 连通域分析 -> 小区域过滤 -> 边界优化
    增加了动态范围检查和异常处理
    
    Args:
        output: 模型输出
        min_area: 最小区域面积
        merge_distance: 合并距离
        debug_mode: 是否启用调试模式（打印警告信息）
    """
    # 设置调试模式
    postprocess._debug_mode = debug_mode
    # 确保输入是tensor并转换为numpy
    if torch.is_tensor(output):
        prob = torch.sigmoid(output).cpu().numpy()
    else:
        prob = output
    
    # 确保prob是2D数组
    if prob.ndim > 2:
        # 如果是4D [B, C, H, W]，取第一个样本的第一个通道
        if prob.ndim == 4:
            prob = prob[0, 0] if prob.shape[1] == 1 else prob[0, :, :, 0]
        # 如果是3D [C, H, W]，取第一个通道
        elif prob.ndim == 3:
            prob = prob[0] if prob.shape[0] == 1 else prob[0, :, :]
    
    # 添加动态范围检查 - 使用自适应阈值
    prob_range = prob.max() - prob.min()
    if prob_range < 0.01:  # 值域过小
        # 只在调试模式下打印警告，避免训练时过多输出
        if hasattr(postprocess, '_debug_mode') and postprocess._debug_mode:
            print(f"警告：输出概率范围过小({prob_range:.4f})，使用自适应阈值")
        
        # 改进的自适应阈值逻辑
        prob_mean = prob.mean()
        if prob_mean > 0.7:  # 高概率区域
            binary = np.ones_like(prob, dtype=np.uint8)
        elif prob_mean < 0.3:  # 低概率区域
            binary = np.zeros_like(prob, dtype=np.uint8)
        else:  # 中等概率区域，使用0.5作为阈值
            binary = (prob > 0.5).astype(np.uint8)
    else:
        try:
            thresh = threshold_otsu(prob)
            binary = (prob > thresh).astype(np.uint8)
        except:
            if hasattr(postprocess, '_debug_mode') and postprocess._debug_mode:
                print("Otsu阈值失败，使用中值阈值")
            binary = (prob > np.median(prob)).astype(np.uint8)
    
    # 添加面积过滤 - 改进的保守过滤
    if binary.mean() > 0.95:  # 超过95%区域被预测为损坏
        if hasattr(postprocess, '_debug_mode') and postprocess._debug_mode:
            print("警告：预测区域占比过高，进行保守过滤")
        # 使用更高阈值但保留高置信度区域
        high_conf_thresh = max(0.8, np.percentile(prob, 90))
        binary = (prob > high_conf_thresh).astype(np.uint8)
    
    # 连通域分析
    try:
        labels = measure.label(binary)
        
        # 移除小区域
        properties = measure.regionprops(labels)
        for prop in properties:
            if prop.area < min_area:
                labels[labels == prop.label] = 0
        
        # 合并邻近区域 - 修复逻辑
        if labels.max() > 0:  # 只有当存在有效区域时才进行合并
            distance = distance_transform_edt(labels == 0)
            close_mask = distance < merge_distance
            labels[close_mask] = 1
        
        # 边界优化
        refined = np.zeros_like(binary)
        for i in range(1, labels.max() + 1):
            region = (labels == i)
            # 形态学优化 - 确保结构元素维度匹配
            try:
                # 确保region是2D数组
                if region.ndim > 2:
                    region = region.squeeze()
                
                # 创建与输入维度匹配的结构元素
                if region.ndim == 2:
                    # 2D图像，使用2D结构元素
                    selem_open = disk(1)
                    selem_close = disk(2)
                    
                    # 检查结构元素大小是否合适
                    if selem_open.shape[0] > region.shape[0] or selem_open.shape[1] > region.shape[1]:
                        print("结构元素过大，跳过形态学优化")
                        refined[region] = 1
                        continue
                    
                    region = binary_opening(region, footprint=selem_open)
                    region = binary_closing(region, footprint=selem_close)
                else:
                    print(f"不支持的维度: {region.ndim}，跳过形态学优化")
                    refined[region] = 1
                    continue
                    
            except Exception as morph_error:
                print(f"形态学操作失败: {morph_error}，跳过形态学优化")
                # 如果形态学操作失败，直接使用原始区域
                pass
            refined[region] = 1
        
        # 添加更精细的警告级别
        if hasattr(postprocess, '_debug_mode') and postprocess._debug_mode:
            refined_sum = refined.sum()
            refined_size = refined.size
            
            if refined_sum == 0:
                # 检查原始概率是否全低
                if prob.max() < 0.3:  # 低概率区域
                    print("信息：后处理输出全零（低概率区域）")
                elif prob.mean() > 0.7:  # 高概率区域被保守过滤
                    print("信息：后处理输出全零（高概率区域被保守过滤）")
                else:  # 中等概率区域
                    print("信息：后处理输出全零（中等概率区域）")
            elif refined_sum == refined_size:
                # 检查原始概率是否全高
                if prob.min() > 0.7:  # 高概率区域
                    print("信息：后处理输出全一（高概率区域）")
                else:  # 低概率但被保留
                    print("警告：后处理输出全一（低概率区域被保留）")
            elif refined_sum > 0 and refined_sum < refined_size:
                # 正常输出情况 - 检查概率分布
                prob_mean = prob.mean()
                prob_std = prob.std()
                if prob_std > 0.1:  # 高方差
                    print(f"信息：后处理输出正常 (均值={prob_mean:.2f}, 标准差={prob_std:.2f})")
                else:  # 低方差
                    print(f"信息：后处理输出正常 (低方差, 均值={prob_mean:.2f})")
            else:
                # 异常情况
                print(f"警告：后处理输出异常 (sum={refined_sum}, size={refined_size})")
        
        return torch.from_numpy(refined).float()
        
    except Exception as e:
        print(f"后处理异常，返回原始二值化结果: {e}")
        return torch.from_numpy(binary).float()

def simple_postprocess(output, threshold=0.5, adaptive=True):
    """
    简化的后处理函数，用于训练时的快速计算
    避免复杂的形态学操作，提高训练速度
    
    Args:
        output: 模型输出
        threshold: 固定阈值（当adaptive=False时使用）
        adaptive: 是否使用自适应阈值
    """
    if torch.is_tensor(output):
        prob = torch.sigmoid(output).cpu().numpy()
    else:
        prob = output
    
    # 确保prob是2D数组
    if prob.ndim > 2:
        if prob.ndim == 4:
            prob = prob[0, 0] if prob.shape[1] == 1 else prob[0, :, :, 0]
        elif prob.ndim == 3:
            prob = prob[0] if prob.shape[0] == 1 else prob[0, :, :]
    
    # 智能阈值化
    if adaptive:
        # 检查概率分布
        prob_mean = prob.mean()
        prob_std = prob.std()
        
        # 低方差情况 - 根据均值和标准差动态调整阈值
        if prob_std < 0.05:  # 更严格的标准差判断
            if prob_mean > 0.7:  # 高均值区域
                binary = np.ones_like(prob, dtype=np.uint8)
            elif prob_mean < 0.3:  # 低均值区域
                binary = np.zeros_like(prob, dtype=np.uint8)
            elif prob_mean > 0.55:  # 中等高值区域（如0.59-0.65）
                # 保留高概率区域，使用0.5作为阈值
                binary = (prob > 0.5).astype(np.uint8)
            elif prob_mean < 0.45:  # 中等低值区域
                # 过滤低概率区域，使用0.5作为阈值
                binary = (prob > 0.5).astype(np.uint8)
            else:  # 接近0.5的中间值使用均值作为阈值
                binary = (prob > prob_mean).astype(np.uint8)
        else:
            # 正常情况使用固定阈值
            binary = (prob > threshold).astype(np.uint8)
    else:
        # 使用固定阈值
        binary = (prob > threshold).astype(np.uint8)
    
    # 检查是否需要保守过滤 - 改进逻辑
    if binary.mean() > 0.95:  # 超过95%区域被预测为损坏
        # 对于低方差数据，使用更温和的过滤策略
        if prob.std() < 0.05:
            # 低方差数据：使用75%分位数作为阈值
            high_conf_thresh = np.percentile(prob, 75)
        else:
            # 高方差数据：使用90%分位数作为阈值
            high_conf_thresh = max(0.8, np.percentile(prob, 90))
        binary = (prob > high_conf_thresh).astype(np.uint8)
    
    # 添加警告系统
    if hasattr(simple_postprocess, '_debug_mode') and simple_postprocess._debug_mode:
        if binary.sum() == 0:
            # 检查原始概率是否全低
            if prob.max() < 0.3:  # 低概率区域
                print("信息：后处理输出全零（低概率区域）")
            else:  # 高概率但被过滤
                print("警告：后处理输出全零（高概率区域被过滤）")
        elif binary.sum() == binary.size:
            # 检查原始概率是否全高
            if prob.min() > 0.7:  # 高概率区域
                print("信息：后处理输出全一（高概率区域）")
            else:  # 低概率但被保留
                print("警告：后处理输出全一（低概率区域被保留）")
        elif binary.sum() > 0 and binary.sum() < binary.size:
            # 正常输出情况 - 检查概率分布
            prob_mean = prob.mean()
            prob_std = prob.std()
            if prob_std > 0.1:  # 高方差
                print(f"信息：后处理输出正常 (均值={prob_mean:.2f}, 标准差={prob_std:.2f})")
            else:  # 低方差
                print(f"信息：后处理输出正常 (低方差, 均值={prob_mean:.2f})")
    
    return torch.from_numpy(binary).float()

try:
    from data_utils.data_loader import optimized_collate
except ImportError:
    # 如果导入失败，定义一个简单的collate函数作为替代
    def optimized_collate(batch):
        """简单的collate函数，确保tensor维度一致"""
        images, masks, feats = zip(*batch)
        
        # 确保所有图像tensor维度一致 [B, C, H, W]
        images = list(images)
        for i, img in enumerate(images):
            if img.dim() == 2:
                images[i] = img.unsqueeze(0)  # [H, W] -> [1, H, W]
            elif img.dim() == 3 and img.shape[0] not in [1, 3]:
                images[i] = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        # 确保所有mask tensor维度一致 [B, 1, H, W]
        masks = list(masks)
        for i, mask in enumerate(masks):
            if mask.dim() == 2:
                masks[i] = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
            elif mask.dim() == 1:
                masks[i] = mask.unsqueeze(0).unsqueeze(0)  # [H] -> [1, 1, H]
        
        # 确保所有sim_feats tensor维度一致 [B, 11]
        feats = list(feats)
        for i, sim_feat in enumerate(feats):
            if sim_feat.dim() == 0:
                feats[i] = sim_feat.unsqueeze(0)  # [] -> [1]
        
        # 使用torch.stack进行批处理
        try:
            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            feats = torch.stack(feats, dim=0)
        except Exception as e:
            print(f"Collate错误: {e}")
            print(f"图像形状: {[img.shape for img in images]}")
            print(f"掩码形状: {[mask.shape for mask in masks]}")
            print(f"特征形状: {[sim_feat.shape for sim_feat in feats]}")
            raise e
        
        return images, masks, feats

# ====== CPU-GPU协同优化类 ======
class CPUAssistedTraining:
    """CPU辅助GPU训练优化器"""
    def __init__(self, model, device='cuda', num_cpu_workers=8):
        self.model = model
        self.device = device
        self.num_cpu_workers = num_cpu_workers
        self.cpu_executor = ThreadPoolExecutor(max_workers=num_cpu_workers)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 启动CPU工作线程
        self.cpu_workers = []
        for i in range(num_cpu_workers):
            worker = threading.Thread(target=self._cpu_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.cpu_workers.append(worker)
    
    def _cpu_worker(self, worker_id):
        """CPU工作线程"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # 停止信号
                    break
                
                task_type, data = task
                if task_type == 'compute_metrics':
                    result = self._compute_metrics_on_cpu(data)
                elif task_type == 'data_preprocessing':
                    result = self._preprocess_data_on_cpu(data)
                elif task_type == 'feature_extraction':
                    result = self._extract_features_on_cpu(data)
                else:
                    result = None
                
                self.result_queue.put((task_type, result))
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"CPU工作线程 {worker_id} 错误: {e}")
    
    def _compute_metrics_on_cpu(self, data):
        """在CPU上计算评估指标"""
        outputs, masks = data
        # 将数据移到CPU进行计算
        outputs_cpu = outputs.cpu().detach()
        masks_cpu = masks.cpu().detach()
        
        # 计算IoU和Dice
        preds = (torch.sigmoid(outputs_cpu) > 0.5).float()
        masks_cpu = masks_cpu.float()
        
        intersection = (preds * masks_cpu).sum()
        union = (preds + masks_cpu).sum() - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        
        dice = (2. * intersection + 1e-5) / (preds.sum() + masks_cpu.sum() + 1e-5)
        
        # 计算更多统计指标
        accuracy = (preds == masks_cpu).float().mean()
        precision = (preds * masks_cpu).sum() / (preds.sum() + 1e-5)
        recall = (preds * masks_cpu).sum() / (masks_cpu.sum() + 1e-5)
        
        # 计算一些特征统计
        output_stats = {
            'mean': outputs_cpu.mean().item(),
            'std': outputs_cpu.std().item(),
            'min': outputs_cpu.min().item(),
            'max': outputs_cpu.max().item()
        }
        
        # 模拟一些额外的CPU计算
        #time.sleep(0.001)  # 模拟1ms的CPU计算时间
        
        return {
            'iou': iou.item(), 
            'dice': dice.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'output_stats': output_stats
        }
    
    def _preprocess_data_on_cpu(self, data):
        """在CPU上进行数据预处理"""
        images, masks = data
        # 在CPU上进行额外的数据预处理
        processed_images = []
        processed_masks = []
        
        for img, mask in zip(images, masks):
            # 归一化处理
            img_norm = (img - img.mean()) / (img.std() + 1e-8)
            
            # 计算一些图像统计信息
            img_stats = {
                'mean': img.mean().item(),
                'std': img.std().item(),
                'min': img.min().item(),
                'max': img.max().item()
            }
            
            # 简单的数据增强
            if random.random() > 0.5:
                img_norm = torch.flip(img_norm, dims=[-1])  # 水平翻转
                mask = torch.flip(mask, dims=[-1])
            
            # 计算边缘特征
            if img.dim() == 3:
                # 简单的边缘检测
                grad_x = torch.diff(img_norm, dim=-1, prepend=img_norm[:, :, :1])
                grad_y = torch.diff(img_norm, dim=-2, prepend=img_norm[:, :1, :])
                edge_feature = torch.sqrt(grad_x**2 + grad_y**2)
                edge_strength = edge_feature.mean().item()
            else:
                edge_strength = 0.0
            
            processed_images.append(img_norm)
            processed_masks.append(mask)
        
        # 模拟一些额外的CPU计算时间
        #time.sleep(0.002)  # 模拟2ms的CPU计算时间
        
        return torch.stack(processed_images), torch.stack(processed_masks)
    
    def _extract_features_on_cpu(self, data):
        """在CPU上提取特征"""
        images = data
        features = []
        
        for img in images:
            # 简单的特征提取（例如边缘检测）
            img_cpu = img.cpu().detach()
            # 计算梯度特征
            grad_x = torch.diff(img_cpu, dim=-1, prepend=img_cpu[:, :, :1])
            grad_y = torch.diff(img_cpu, dim=-2, prepend=img_cpu[:, :1, :])
            edge_feature = torch.sqrt(grad_x**2 + grad_y**2)
            features.append(edge_feature.mean())
        
        return torch.stack(features)
    
    def submit_cpu_task(self, task_type, data):
        """提交CPU任务"""
        self.task_queue.put((task_type, data))
    
    def get_cpu_result(self, timeout=0.1):
        """获取CPU计算结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def shutdown(self):
        """关闭CPU工作线程"""
        for _ in range(self.num_cpu_workers):
            self.task_queue.put(None)
        self.cpu_executor.shutdown(wait=True)

class AsyncDataProcessor:
    """异步数据处理器"""
    def __init__(self, num_workers=12):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = []
    
    def submit_preprocessing(self, data_batch):
        """提交数据预处理任务"""
        future = self.executor.submit(self._preprocess_batch, data_batch)
        self.futures.append(future)
        return future
    
    def _preprocess_batch(self, data_batch):
        """预处理数据批次"""
        images, masks, sim_feats = data_batch
        
        # 在CPU上进行预处理
        processed_images = []
        for img in images:
            # 数据增强和预处理
            img_cpu = img.cpu()
            
            # 计算图像统计信息
            img_mean = img_cpu.mean()
            img_std = img_cpu.std()
            
            # 归一化处理
            img_norm = (img_cpu - img_mean) / (img_std + 1e-8)
            
            # 计算一些额外的特征
            if img_cpu.dim() == 3:
                # 计算梯度特征
                grad_x = torch.diff(img_norm, dim=-1, prepend=img_norm[:, :, :1])
                grad_y = torch.diff(img_norm, dim=-2, prepend=img_norm[:, :1, :])
                edge_feature = torch.sqrt(grad_x**2 + grad_y**2)
                
                # 计算纹理特征
                texture_feature = torch.std(img_norm, dim=(1, 2))
            else:
                edge_feature = torch.zeros(1)
                texture_feature = torch.zeros(1)
            
            processed_images.append(img_norm)
        
        # 模拟一些CPU计算时间
        #time.sleep(0.003)  # 模拟3ms的CPU计算时间
        
        return torch.stack(processed_images), masks, sim_feats
    
    def get_completed_results(self):
        """获取已完成的结果"""
        completed = []
        remaining = []
        
        for future in self.futures:
            if future.done():
                try:
                    result = future.result()
                    completed.append(result)
                except Exception as e:
                    print(f"预处理任务失败: {e}")
            else:
                remaining.append(future)
        
        self.futures = remaining
        return completed
    
    def shutdown(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)

class HybridPrecisionTrainer:
    """混合精度训练器"""
    def __init__(self, model, optimizer, criterion, device='cuda', num_cpu_workers=8, num_async_workers=8):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()
        
        # 创建CPU辅助训练器
        self.cpu_assistant = CPUAssistedTraining(model, device, num_cpu_workers=num_cpu_workers)
        
        # 创建异步数据处理器
        self.async_processor = AsyncDataProcessor(num_workers=num_async_workers)
        
        # 性能统计
        self.gpu_time = 0
        self.cpu_time = 0
        self.total_batches = 0
    
    def train_step(self, images, masks, sim_feats):
        """混合精度训练步骤"""
        start_time = time.time()
        
        # 异步提交CPU预处理任务
        cpu_future = self.async_processor.submit_preprocessing((images, masks, sim_feats))
        
        # GPU前向传播 - 使用自动混合精度
        gpu_start = time.time()
        self.optimizer.zero_grad()
        
        with autocast('cuda'):  # 👈 自动混合精度
            outputs = self.model(images, sim_feats)
            loss = self.criterion(outputs, masks)  # 使用HybridLoss
        
        # 缩放梯度避免下溢
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        gpu_time = time.time() - gpu_start
        self.gpu_time += gpu_time
        
        # 获取CPU预处理结果并计算CPU时间
        cpu_start = time.time()
        cpu_results = self.async_processor.get_completed_results()
        
        # 提交CPU指标计算任务
        self.cpu_assistant.submit_cpu_task('compute_metrics', (outputs, masks))
        
        # 执行一些CPU计算任务
        cpu_compute_start = time.time()
        # 在CPU上进行一些额外的计算
        with torch.no_grad():
            # 计算一些统计信息
            outputs_cpu = outputs.cpu().detach()
            masks_cpu = masks.cpu().detach()
            
            # 计算一些统计指标
            preds = (torch.sigmoid(outputs_cpu) > 0.5).float()
            accuracy = (preds == masks_cpu).float().mean()
            
            # 计算一些特征统计
            feature_stats = {
                'output_mean': outputs_cpu.mean().item(),
                'output_std': outputs_cpu.std().item(),
                'mask_mean': masks_cpu.mean().item(),
                'accuracy': accuracy.item()
            }
            
            # 执行一些额外的CPU计算来增加CPU时间
            # 计算边缘特征
            if outputs_cpu.dim() == 4:
                edge_features = []
                for i in range(outputs_cpu.shape[0]):
                    img = outputs_cpu[i, 0]  # 取第一个通道
                    grad_x = torch.diff(img, dim=-1, prepend=img[:, :1])
                    grad_y = torch.diff(img, dim=-2, prepend=img[:1, :])
                    edge = torch.sqrt(grad_x**2 + grad_y**2)
                    edge_features.append(edge.mean().item())
            
            # 计算纹理特征
            texture_features = []
            for i in range(outputs_cpu.shape[0]):
                img = outputs_cpu[i, 0]
                texture = torch.std(img)
                texture_features.append(texture.item())
        
        cpu_time = time.time() - cpu_start
        self.cpu_time += cpu_time
        
        self.total_batches += 1
        
        return loss.item(), outputs
    
    def get_performance_stats(self):
        """获取性能统计"""
        if self.total_batches == 0:
            return {}
        
        avg_gpu_time = self.gpu_time / self.total_batches
        avg_cpu_time = self.cpu_time / self.total_batches
        gpu_utilization = avg_gpu_time / (avg_gpu_time + avg_cpu_time) * 100
        
        return {
            'avg_gpu_time': avg_gpu_time,
            'avg_cpu_time': avg_cpu_time,
            'gpu_utilization': gpu_utilization,
            'total_batches': self.total_batches
        }
    
    def shutdown(self):
        """关闭训练器"""
        self.cpu_assistant.shutdown()
        self.async_processor.shutdown()

# ====== 自定义collate函数 ======
def custom_collate_fn(batch):
    """
    自定义collate函数，确保所有tensor维度一致
    """
    images, masks, sim_feats = zip(*batch)
    
    # 确保所有图像tensor维度一致 [B, C, H, W]
    images = list(images)
    for i, img in enumerate(images):
        if img.dim() == 2:
            images[i] = img.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif img.dim() == 3 and img.shape[0] not in [1, 3]:
            images[i] = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    # 确保所有mask tensor维度一致 [B, 1, H, W]
    masks = list(masks)
    for i, mask in enumerate(masks):
        if mask.dim() == 2:
            masks[i] = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif mask.dim() == 1:
            masks[i] = mask.unsqueeze(0).unsqueeze(0)  # [H] -> [1, 1, H]
    
    # 确保所有sim_feats tensor维度一致 [B, 11]
    sim_feats = list(sim_feats)
    for i, sim_feat in enumerate(sim_feats):
        if sim_feat.dim() == 0:
            sim_feats[i] = sim_feat.unsqueeze(0)  # [] -> [1]
    
    # 使用torch.stack进行批处理
    try:
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        sim_feats = torch.stack(sim_feats, dim=0)
    except Exception as e:
        print(f"Collate错误: {e}")
        print(f"图像形状: {[img.shape for img in images]}")
        print(f"掩码形状: {[mask.shape for mask in masks]}")
        print(f"特征形状: {[sim_feat.shape for sim_feat in sim_feats]}")
        raise e
    
    return images, masks, sim_feats

# ====== 内存预加载数据集类 ======
class MemoryCachedDataset(Dataset):
    """将数据预加载到内存中的数据集包装器"""
    def __init__(self, base_dataset, device='cpu', cache_sim_features=True):
        self.base_dataset = base_dataset
        self.device = device
        self.cache_sim_features = cache_sim_features
        
        print(f"正在将数据集预加载到内存中...")
        self.cached_data = []
        
        for i in tqdm(range(len(base_dataset)), desc="预加载数据"):
            try:
                img, mask, sim_feat = base_dataset[i]
                
                # 将数据移动到指定设备
                if isinstance(img, torch.Tensor):
                    img = img.to(device, non_blocking=True)
                if isinstance(mask, torch.Tensor):
                    mask = mask.to(device, non_blocking=True)
                if isinstance(sim_feat, torch.Tensor) and self.cache_sim_features:
                    sim_feat = sim_feat.to(device, non_blocking=True)
                
                self.cached_data.append((img, mask, sim_feat))
                
            except Exception as e:
                print(f"预加载样本 {i} 时出错: {e}")
                # 创建一个空的替代样本
                if len(self.cached_data) > 0:
                    # 使用第一个成功加载的样本作为模板
                    template_img, template_mask, template_sim = self.cached_data[0]
                    self.cached_data.append((template_img.clone(), template_mask.clone(), template_sim.clone()))
                else:
                    # 如果还没有成功加载的样本，创建一个默认样本
                    default_img = torch.zeros(3, 64, 64, device=device)
                    default_mask = torch.zeros(1, 64, 64, device=device)
                    default_sim = torch.zeros(11, device=device)
                    self.cached_data.append((default_img, default_mask, default_sim))
        
        print(f"✅ 数据预加载完成！共加载 {len(self.cached_data)} 个样本到 {device}")
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

class GPUPreloadedDataset(Dataset):
    """将数据预加载到GPU显存中的数据集包装器"""
    def __init__(self, base_dataset, device='cuda', batch_size=32):
        self.base_dataset = base_dataset
        self.device = device
        self.batch_size = batch_size
        
        print(f"正在将数据集预加载到GPU显存中...")
        self.cached_batches = []
        
        # 按批次预加载
        for i in tqdm(range(0, len(base_dataset), batch_size), desc="预加载GPU批次"):
            batch_end = min(i + batch_size, len(base_dataset))
            batch_data = []
            
            for j in range(i, batch_end):
                try:
                    img, mask, sim_feat = base_dataset[j]
                    
                    # 确保数据是tensor格式
                    if not isinstance(img, torch.Tensor):
                        img = torch.tensor(img, dtype=torch.float32)
                    if not isinstance(mask, torch.Tensor):
                        mask = torch.tensor(mask, dtype=torch.float32)
                    if not isinstance(sim_feat, torch.Tensor):
                        sim_feat = torch.tensor(sim_feat, dtype=torch.float32)
                    
                    batch_data.append((img, mask, sim_feat))
                    
                except Exception as e:
                    print(f"预加载样本 {j} 时出错: {e}")
                    # 创建默认样本
                    default_img = torch.zeros(3, 64, 64, dtype=torch.float32)
                    default_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
                    default_sim = torch.zeros(11, dtype=torch.float32)
                    batch_data.append((default_img, default_mask, default_sim))
            
            # 将批次数据移动到GPU
            if batch_data:
                batch_imgs = torch.stack([item[0] for item in batch_data]).to(device, non_blocking=True)
                batch_masks = torch.stack([item[1] for item in batch_data]).to(device, non_blocking=True)
                batch_sims = torch.stack([item[2] for item in batch_data]).to(device, non_blocking=True)
                
                self.cached_batches.append((batch_imgs, batch_masks, batch_sims))
        
        print(f"✅ GPU数据预加载完成！共加载 {len(self.cached_batches)} 个批次到 {device}")
    
    def __len__(self):
        return len(self.cached_batches) * self.batch_size
    
    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        item_idx = idx % self.batch_size
        
        if batch_idx < len(self.cached_batches):
            batch_imgs, batch_masks, batch_sims = self.cached_batches[batch_idx]
            if item_idx < batch_imgs.shape[0]:
                return batch_imgs[item_idx], batch_masks[item_idx], batch_sims[item_idx]
        
        # 返回默认值
        return torch.zeros(3, 64, 64, device=self.device), torch.zeros(1, 64, 64, device=self.device), torch.zeros(11, device=self.device)

# ====== 优化后的Patch级别增强类 ======
class AdvancedAugmentation:
    def __init__(self, is_training=True):
        self.is_training = is_training
        
        # 训练集增强 - 简化为更稳定的变换
        self.train_transform = A.Compose([
            # 基础几何变换
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            
            # 轻微缩放
            A.Affine(scale=(0.9, 1.1), keep_ratio=True, p=0.3),
            
            # 基础颜色变换
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
                
            # 轻微噪声
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            
            # 格式转换
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
        
        # 验证集转换 - 仅基础预处理
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __call__(self, image, mask):
        if self.is_training:
            augmented = self.train_transform(image=image, mask=mask)
        else:
            augmented = self.val_transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

class DamageAwareDataset(Dataset):
    def __init__(self, base_dataset, damage_boost=5, normal_ratio=0.05, allow_synthetic=True):
        self.base_dataset = base_dataset
        self.damage_indices = []
        self.normal_indices = []
        self.normal_ratio = normal_ratio  # 正常样本比例，默认5%
        self.allow_synthetic = allow_synthetic  # 是否允许合成正常样本
        
        # 预扫描数据集统计损坏样本
        for i in tqdm(range(len(base_dataset)), desc="Scanning damage samples"):
            _, mask, *_ = base_dataset[i]
            if mask.sum() > 0:
                self.damage_indices.append(i)
            else:
                self.normal_indices.append(i)
        
        # 如果没有找到正常样本，创建一些正常样本
        if len(self.normal_indices) == 0 and self.allow_synthetic:
            print("警告: 没有找到正常样本，将创建合成正常样本")
            # 从损坏样本中随机选择一些，将其掩码清零作为正常样本
            num_normal_needed = max(1, int(len(self.damage_indices) * self.normal_ratio))
            selected_indices = random.sample(self.damage_indices, min(num_normal_needed, len(self.damage_indices)))
            self.normal_indices = selected_indices
            print(f"创建了 {len(self.normal_indices)} 个合成正常样本")
        elif len(self.normal_indices) == 0 and not self.allow_synthetic:
            print("警告: 没有找到正常样本，且不允许合成样本")
        
        # 确保正常样本数量符合比例要求
        target_normal_count = int(len(self.damage_indices) * self.normal_ratio)
        if len(self.normal_indices) > target_normal_count:
            # 如果正常样本太多，随机选择
            self.normal_indices = random.sample(self.normal_indices, target_normal_count)
        elif len(self.normal_indices) < target_normal_count and self.allow_synthetic:
            # 如果正常样本太少，从损坏样本中复制一些并清零掩码
            additional_needed = target_normal_count - len(self.normal_indices)
            additional_indices = random.sample(self.damage_indices, min(additional_needed, len(self.damage_indices)))
            self.normal_indices.extend(additional_indices)
        
        self.damage_boost = damage_boost
        print(f"Found {len(self.damage_indices)} damage samples and {len(self.normal_indices)} normal samples")
        print(f"Normal sample ratio: {len(self.normal_indices) / (len(self.damage_indices) + len(self.normal_indices)) * 100:.1f}%")

    def __len__(self):
        return len(self.normal_indices) + len(self.damage_indices) * self.damage_boost

    def __getitem__(self, idx):
        if idx < len(self.damage_indices) * self.damage_boost:
            # 损坏样本
            damage_idx = idx % len(self.damage_indices)
            return self.base_dataset[self.damage_indices[damage_idx]]
        else:
            # 正常样本
            normal_idx = (idx - len(self.damage_indices) * self.damage_boost) % len(self.normal_indices)
            img, mask, sim_feat = self.base_dataset[self.normal_indices[normal_idx]]
            
            # 对于正常样本，将掩码清零（确保没有损坏区域）
            if mask.sum() > 0:
                mask = torch.zeros_like(mask)
            
            return img, mask, sim_feat

class YOLOLandslideDataset(Dataset):
    """YOLO格式的山体滑坡数据集"""
    def __init__(self, images_dir, labels_dir, transform=None, disaster_class_ids=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # 灾害相关类别ID (COCO 80类中的相关ID)
        self.disaster_class_ids = disaster_class_ids or [
            0, 1, 2, 3, 5, 6, 7, 8, 10, 24, 25, 27, 28, 29, 33, 
            44, 56, 57, 58, 59, 60, 62, 63, 67, 73
        ]
        
        # 获取所有图像文件
        self.image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        if not self.image_files:
            # 尝试其他图像格式
            self.image_files = glob.glob(os.path.join(images_dir, "*.jpeg")) + \
                             glob.glob(os.path.join(images_dir, "*.png"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # 如果图像损坏，创建替代图像
            img = Image.new('RGB', (224, 224), 
                           color=(random.randint(0, 255), 
                                  random.randint(0, 255), 
                                  random.randint(0, 255)))
        
        # 获取对应的标注文件路径
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
        
        # 二分类标签: 1=包含灾害相关物体，0=不包含
        label = 0
        
        # 如果标注文件存在，检查是否包含灾害类别
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(parts[0])
                            if class_id in self.disaster_class_ids:
                                label = 1
                                break  # 只要有一个灾害物体就标记为1
                        except ValueError:
                            continue
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class CombinedLandslideDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(images_dir, '*.*'))
        self.image_files = [f for f in self.image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)
        label = 0
        if "xview2" in base_name.lower() and "post" in base_name.lower():
            label = 1
        elif "xview2" in base_name.lower() and "pre" in base_name.lower():
            label = 0
        else:
            label_path = os.path.join(self.labels_dir, os.path.splitext(base_name)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            label = 1
                            break
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        if self.transform:
            img = self.transform(img)
        return img, label

def get_segmentation_dataloaders(data_root="data/combined_dataset", batch_size=32, num_workers=8, show_warnings=False, skip_problematic_samples=False):
    """
    获取分割任务的数据加载器
    返回: (train_loader, val_loader, test_loader)
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        show_warnings: 是否显示数据质量警告
        skip_problematic_samples: 是否跳过有问题的样本
    """
    # 数据增强和转换
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集 - 应用选项1：隐藏警告但保留所有数据
    train_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "train2017"),
        os.path.join(data_root, "masks", "train2017"),
        transform=train_transform,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )
    
    val_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "val2017"),
        os.path.join(data_root, "masks", "val2017"), 
        transform=val_transform,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )
    
    test_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "test2017"),
        os.path.join(data_root, "masks", "test2017"),
        transform=val_transform,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def get_multi_class_dataloaders(data_root="data/combined_dataset", batch_size=32, num_workers=8, 
                               damage_level='categorical', show_warnings=False, skip_problematic_samples=False):
    """
    获取多类别分类任务的数据加载器
    支持5个类别：背景(0)、未损坏(1)、轻微损坏(2)、中等损坏(3)、严重损坏(4)
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        damage_level: 掩码处理方式，推荐使用'categorical'进行多类别分类
        show_warnings: 是否显示数据质量警告
        skip_problematic_samples: 是否跳过有问题的样本
    """
    # 数据增强和转换
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集 - 使用多类别掩码处理
    train_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "train2017"),
        os.path.join(data_root, "masks", "train2017"),
        transform=train_transform,
        damage_level=damage_level,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )
    
    val_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "val2017"),
        os.path.join(data_root, "masks", "val2017"), 
        transform=val_transform,
        damage_level=damage_level,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )
    
    test_dataset = XView2SegmentationDataset(
        os.path.join(data_root, "images", "test2017"),
        os.path.join(data_root, "masks", "test2017"),
        transform=val_transform,
        damage_level=damage_level,
        show_warnings=show_warnings,
        skip_problematic_samples=skip_problematic_samples
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"[多类别数据加载] 使用damage_level='{damage_level}'")
    print(f"[多类别数据加载] 训练集大小: {len(train_dataset)}")
    print(f"[多类别数据加载] 验证集大小: {len(val_dataset)}")
    print(f"[多类别数据加载] 测试集大小: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def get_landslide_dataloaders(data_root="data/combined_dataset", batch_size=4):
    """
    获取山体滑坡分类数据加载器
    适用于二分类任务（滑坡/非滑坡）
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def make_loader(split):
        images_dir = os.path.join(data_root, "images", split)
        labels_dir = os.path.join(data_root, "labels", split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"警告: {split} 目录不存在: {images_dir} 或 {labels_dir}")
            return None
        
        dataset = CombinedLandslideDataset(images_dir, labels_dir, transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train2017"), num_workers=4, pin_memory=True)
    
    return make_loader("train2017"), make_loader("val2017"), make_loader("test2017")

def get_calibration_loader(data_root="data/combined_dataset", batch_size=32, num_samples=100):
    train_images_dir = os.path.join(data_root, "images", "train2017")
    train_labels_dir = os.path.join(data_root, "labels", "train2017")
    dataset = CombinedLandslideDataset(train_images_dir, train_labels_dir, transform=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    indices = random.sample(range(len(dataset)), num_samples)
    calib_subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def load_sim_features(sim_feature_csv='data/sim_features.csv', normalize=True):
    """
    加载并归一化仿真特征
    Args:
        sim_feature_csv: 仿真特征CSV文件路径
        normalize: 是否进行归一化处理
    """
    sim_dict = {}
    all_features = []  # 用于计算归一化参数
    
    try:
        with open(sim_feature_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 数值型特征
                float_feats = []
                for col in ['comm_snr', 'radar_feat', 'radar_max', 'radar_std', 
                           'radar_peak_idx', 'path_loss', 'shadow_fading', 
                           'rain_attenuation', 'target_rcs', 'bandwidth', 'ber']:
                    try:
                        value = float(row[col])
                        # 处理异常值
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0
                        float_feats.append(value)
                    except (ValueError, KeyError):
                        float_feats.append(0.0)
                
                # 字符串型特征
                str_feats = [row.get('channel_type', ''), row.get('modulation', '')]
                sim_dict[row['img_path']] = (float_feats, str_feats)
                all_features.append(float_feats)
        
        # 归一化处理
        if normalize and all_features:
            all_features = np.array(all_features)
            # 计算每个特征的统计信息
            feature_means = np.mean(all_features, axis=0)
            feature_stds = np.std(all_features, axis=0)
            
            # 避免除零
            feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
            
            # 归一化所有特征
            for img_path, (float_feats, str_feats) in sim_dict.items():
                float_feats = np.array(float_feats)
                normalized_feats = (float_feats - feature_means) / feature_stds
                sim_dict[img_path] = (normalized_feats.tolist(), str_feats)
            
            print(f"仿真特征归一化完成，均值: {feature_means}, 标准差: {feature_stds}")
        
        print(f"成功加载 {len(sim_dict)} 个仿真特征")
        
    except FileNotFoundError:
        print(f"警告: 仿真特征文件 {sim_feature_csv} 不存在")
    except Exception as e:
        print(f"加载仿真特征时出错: {e}")
    
    return sim_dict

def process_xview2_mask(mask_tensor, damage_level='all', sample_ratio=None):
    """
    增强的xView2掩码处理函数，支持根据样本比例动态调整
    
    Args:
        mask_tensor: 输入掩码张量
        damage_level: 损坏级别处理方式
            'all': 所有损坏级别(2,3,4)都标记为1
            'light': 轻微损坏(2)标记为0.3，中等(3)标记为0.6，严重(4)标记为1.0
            'binary': 轻微(2)标记为0，中等和严重(3,4)标记为1
            'multi': 轻微(2)标记为1，中等(3)标记为2，严重(4)标记为3
            'progressive': 渐进式权重：轻微(2)=0.25，中等(3)=0.5，严重(4)=1.0
            'categorical': 多类别分类：背景(0)=0，未损坏(1)=1，轻微(2)=2，中等(3)=3，严重(4)=4
            'damage_only': 只关注损坏区域：轻微(2)=1，中等(3)=2，严重(4)=3，其他=0
            'severity_weighted': 严重程度加权：轻微(2)=0.2，中等(3)=0.5，严重(4)=1.0
            'adaptive': 根据样本比例动态调整权重
        sample_ratio: 样本比例信息，用于自适应调整
            - 如果为None，使用默认权重
            - 如果提供，根据比例调整权重
    """
    if damage_level == 'all':
        # 原始行为：所有损坏级别都标记为1
        return (mask_tensor >= 2).float()
    elif damage_level == 'light':
        # 根据损坏程度分配不同权重
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 2] = 0.3  # 轻微损坏
        result[mask_tensor == 3] = 0.6  # 中等损坏
        result[mask_tensor == 4] = 1.0  # 严重损坏
        return result
    elif damage_level == 'binary':
        # 轻微损坏不算损坏，只有中等和严重损坏才算
        return (mask_tensor >= 3).float()
    elif damage_level == 'multi':
        # 多级分类：轻微=1，中等=2，严重=3
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 2] = 1.0  # 轻微损坏
        result[mask_tensor == 3] = 2.0  # 中等损坏
        result[mask_tensor == 4] = 3.0  # 严重损坏
        return result
    elif damage_level == 'progressive':
        # 渐进式权重：更细致的损坏程度区分
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 2] = 0.25  # 轻微损坏
        result[mask_tensor == 3] = 0.5   # 中等损坏
        result[mask_tensor == 4] = 1.0   # 严重损坏
        return result
    elif damage_level == 'categorical':
        # 多类别分类：保持原始类别标签
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 0] = 0.0  # 背景
        result[mask_tensor == 1] = 1.0  # 未损坏
        result[mask_tensor == 2] = 2.0  # 轻微损坏
        result[mask_tensor == 3] = 3.0  # 中等损坏
        result[mask_tensor == 4] = 4.0  # 严重损坏
        return result
    elif damage_level == 'damage_only':
        # 只关注损坏区域，忽略背景和未损坏区域
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 2] = 1.0  # 轻微损坏
        result[mask_tensor == 3] = 2.0  # 中等损坏
        result[mask_tensor == 4] = 3.0  # 严重损坏
        return result
    elif damage_level == 'severity_weighted':
        # 严重程度加权：更强调严重损坏
        result = torch.zeros_like(mask_tensor, dtype=torch.float32)
        result[mask_tensor == 2] = 0.2  # 轻微损坏
        result[mask_tensor == 3] = 0.5  # 中等损坏
        result[mask_tensor == 4] = 1.0  # 严重损坏
        return result
    elif damage_level == 'adaptive':
        # 根据样本比例动态调整权重
        if sample_ratio is None:
            # 默认权重
            result = torch.zeros_like(mask_tensor, dtype=torch.float32)
            result[mask_tensor == 2] = 0.3  # 轻微损坏
            result[mask_tensor == 3] = 0.6  # 中等损坏
            result[mask_tensor == 4] = 1.0  # 严重损坏
        else:
            # 根据样本比例调整权重
            # 如果损坏样本比例低，增加权重以平衡
            # 如果损坏样本比例高，降低权重以避免过拟合
            if sample_ratio < 0.1:  # 损坏样本比例很低
                result = torch.zeros_like(mask_tensor, dtype=torch.float32)
                result[mask_tensor == 2] = 0.5  # 增加轻微损坏权重
                result[mask_tensor == 3] = 0.8  # 增加中等损坏权重
                result[mask_tensor == 4] = 1.0  # 保持严重损坏权重
            elif sample_ratio < 0.3:  # 损坏样本比例较低
                result = torch.zeros_like(mask_tensor, dtype=torch.float32)
                result[mask_tensor == 2] = 0.4  # 适度增加轻微损坏权重
                result[mask_tensor == 3] = 0.7  # 适度增加中等损坏权重
                result[mask_tensor == 4] = 1.0  # 保持严重损坏权重
            elif sample_ratio > 0.7:  # 损坏样本比例很高
                result = torch.zeros_like(mask_tensor, dtype=torch.float32)
                result[mask_tensor == 2] = 0.2  # 降低轻微损坏权重
                result[mask_tensor == 3] = 0.5  # 降低中等损坏权重
                result[mask_tensor == 4] = 0.8  # 降低严重损坏权重
            else:  # 损坏样本比例适中
                result = torch.zeros_like(mask_tensor, dtype=torch.float32)
                result[mask_tensor == 2] = 0.3  # 标准轻微损坏权重
                result[mask_tensor == 3] = 0.6  # 标准中等损坏权重
                result[mask_tensor == 4] = 1.0  # 标准严重损坏权重
        return result
    else:
        # 默认行为
        return (mask_tensor >= 2).float()

class XView2SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, sim_feature_dict=None, transform=None, mask_transform=None, 
                 damage_sample_txts=None, damage_prob=0.7, is_training=True, damage_level='all',
                 show_warnings=True, skip_problematic_samples=False):
        """
        xView2数据集加载器
        数据集特性：
        - 包含灾前(pre-disaster)和灾后(post-disaster)图像
        - 掩码用于定位和损伤评估任务
        - 掩码是单通道PNG图像，值含义：
            0: 背景
            1: 未损坏
            2: 损坏
        
        Args:
            show_warnings: 是否显示数据质量警告
            skip_problematic_samples: 是否跳过有问题的样本（灾后图像无损坏区域）
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.sim_feature_dict = sim_feature_dict
        self.transform = transform
        self.mask_transform = mask_transform
        self.is_training = is_training
        self.damage_level = damage_level
        self.show_warnings = show_warnings
        self.skip_problematic_samples = skip_problematic_samples

        # 智能文件筛选：根据文件名判断是否为灾后图像
        self.image_files = []
        problematic_files = []
        
        for f in os.listdir(images_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_name = os.path.splitext(f)[0] + "_target.png"
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(mask_path):
                    # 检查掩码内容，确保灾后图像确实包含损坏区域
                    try:
                        mask = Image.open(mask_path)
                        if mask.mode != 'L':
                            mask = mask.convert('L')
                        mask_np = np.array(mask)
                        has_damage = (mask_np >= 2).sum() > 0
                        
                        # 如果是灾后图像但没有损坏区域，记录警告
                        if 'post_disaster' in f and not has_damage:
                            if self.show_warnings:
                                print(f"警告：灾后图像 {f} 未检测到损坏区域")
                            problematic_files.append(f)
                            # 如果选择跳过有问题的样本，则不添加到文件列表
                            if not self.skip_problematic_samples:
                                self.image_files.append(f)
                        else:
                            self.image_files.append(f)
                    except Exception as e:
                        if self.show_warnings:
                            print(f"警告：无法读取掩码 {mask_path}: {e}")
                        continue
        
        if self.skip_problematic_samples and problematic_files:
            print(f"已跳过 {len(problematic_files)} 个有问题的样本")

        self.use_weighted_sampling = False
        if damage_sample_txts is not None:
            # damage_sample_txts: (has_damage_txt, no_damage_txt)
            has_damage_txt, no_damage_txt = damage_sample_txts
            with open(has_damage_txt) as f:
                self.has_damage = [line.strip() for line in f if line.strip()]
            with open(no_damage_txt) as f:
                self.no_damage = [line.strip() for line in f if line.strip()]
            self.use_weighted_sampling = True
            self.damage_prob = damage_prob
            self.length = len(self.has_damage) + len(self.no_damage)
        else:
            self.length = len(self.image_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            if self.use_weighted_sampling:
                # 增强采样：保证85%概率采样有损坏区域
                found_damage = False
                for _ in range(10):
                    if random.random() < 0.85 and len(self.has_damage) > 0:
                        mask_name = random.choice(self.has_damage)
                    else:
                        mask_name = random.choice(self.no_damage)
                    img_name = mask_name.replace('_target', '')
                    img_path = os.path.join(self.images_dir, img_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        mask = Image.open(mask_path)
                        if mask.mode != 'L':
                            mask = mask.convert('L')
                        mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                        mask_np = np.array(mask)
                        if (mask_np >= 2).sum() > 0:
                            found_damage = True
                            break
                if not found_damage:
                    # fallback到无损坏样本
                    mask_name = random.choice(self.no_damage)
                    img_name = mask_name.replace('_target', '')
                    img_path = os.path.join(self.images_dir, img_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    mask = Image.open(mask_path)
                    if mask.mode != 'L':
                        mask = mask.convert('L')
                    mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                    mask_np = np.array(mask)
            else:
                img_name = self.image_files[idx]
                mask_name = os.path.splitext(img_name)[0] + "_target.png"
                img_path = os.path.join(self.images_dir, img_name)
                mask_path = os.path.join(self.masks_dir, mask_name)
                mask = Image.open(mask_path)
                if mask.mode != 'L':
                    mask = mask.convert('L')
                mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
                mask_np = np.array(mask)  # uint8, 0/1/2/3/4
                mask_bin = (mask_np >= 2).astype(np.float32)  # 2/3/4为损坏
                mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # [1, H, W]
            image = Image.open(img_path).convert('RGB')
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_tensor = T.ToTensor()(mask)
            # 使用增强的掩码处理函数，支持多种损坏级别处理方式
            damage_level = getattr(self, 'damage_level', 'all')  # 默认使用'all'模式
            mask_tensor = process_xview2_mask(mask_tensor, damage_level)
            
            # 只在第一个样本时显示处理信息，避免刷屏
            if idx == 0:
                unique_values = torch.unique(mask_tensor)
                print(f"[掩码处理] 使用damage_level='{damage_level}', 处理后唯一值: {unique_values.tolist()}")
            if self.transform:
                if 'albumentations' in str(type(self.transform)):
                    augmented = self.transform(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']
                else:
                    image = self.transform(image)
            if self.use_weighted_sampling and idx == 0:
                print(f"采样掩码: {mask_name}, 损坏像素数: {(mask_np == 2).sum()}")
            if idx == 0 and not self.use_weighted_sampling:
                print(f"\n第一个样本调试信息:")
                print(f"图像路径: {img_path}")
                print(f"掩码路径: {mask_path}")
                print(f"原始掩码值范围: min={mask_np.min()}, max={mask_np.max()}")
                print(f"原始掩码唯一值: {np.unique(mask_np)}")
                print(f"处理后掩码形状: {mask_tensor.shape}")
                print(f"处理后掩码唯一值: {torch.unique(mask_tensor)}")
            # 加载sim特征
            sim_feat_tensor = torch.zeros(11)
            str_feats = ["", ""]
            if hasattr(self, 'sim_feature_dict') and self.sim_feature_dict is not None:
                key = os.path.basename(img_path)
                if key in self.sim_feature_dict:
                    sim_feats = self.sim_feature_dict[key]
                    sim_feat_tensor = torch.tensor(sim_feats[:11], dtype=torch.float32)
                    str_feats = sim_feats[11:]
            # 自动跳过全为0的掩码（无损坏像素）
            return image, mask_tensor, sim_feat_tensor, str_feats
        except Exception as e:
            print(f"[警告] 加载样本 {idx} 时出错: {e}, 自动跳过，尝试下一个样本。")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

class PatchSegmentationDataset(Dataset):
    """
    用于加载npy格式的patch图像和掩码。
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不一致"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        image = np.load(img_path)  # shape: (C, H, W) or (H, W, C)
        mask = np.load(mask_path)  # shape: (H, W)
        # 保证image为float32, mask为float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        # 如果image是(H, W, C)，转为(C, H, W)
        if image.ndim == 3 and image.shape[0] != 1 and image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        # 确保mask维度一致
        mask_tensor = torch.from_numpy(mask)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [H] -> [1, 1, H]
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, mask_tensor


def get_patch_dataloaders(data_root="data/patch_dataset", batch_size=4, num_workers=8):
    """
    获取patch分割任务的数据加载器
    返回: (train_loader, val_loader)
    """
    train_images_dir = os.path.join(data_root, "train/images")
    train_masks_dir = os.path.join(data_root, "train/masks")
    val_images_dir = os.path.join(data_root, "val/images")
    val_masks_dir = os.path.join(data_root, "val/masks")

    train_dataset = PatchSegmentationDataset(train_images_dir, train_masks_dir)
    val_dataset = PatchSegmentationDataset(val_images_dir, val_masks_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

class MultiModalPatchSegmentationDataset(Dataset):
    """
    支持多模态patch分割：每个patch加载patch、掩码、原图仿真特征。
    """
    def __init__(self, images_dir, masks_dir, patch_index_csv, sim_feature_dict, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.sim_feature_dict = sim_feature_dict
        # 加载patch->原图索引
        self.patch2img = {}
        with open(patch_index_csv, 'r') as f:
            next(f)
            for line in f:
                patch, img = line.strip().split(',')
                self.patch2img[patch] = img
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
        assert len(self.image_files) == len(self.mask_files), "图像和掩码数量不一致"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_patch_name = self.image_files[idx]
        mask_patch_name = self.mask_files[idx]
        img_patch_path = os.path.join(self.images_dir, img_patch_name)
        mask_patch_path = os.path.join(self.masks_dir, mask_patch_name)
        image = np.load(img_patch_path)
        mask = np.load(mask_patch_path)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if image.ndim == 3 and image.shape[0] != 1 and image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        # 确保mask维度一致
        mask_tensor = torch.from_numpy(mask)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [H] -> [1, 1, H]
        # 查找原图名并加载仿真特征
        origin_img = self.patch2img.get(img_patch_name, None)
        if origin_img is not None and self.sim_feature_dict is not None:
            # 尝试多种路径格式来匹配sim_feature_dict中的键
            possible_keys = [
                origin_img,  # 原始格式
                f"combined_dataset/images/train2017/{origin_img}",  # 训练集路径
                f"combined_dataset/images/val2017/{origin_img}",    # 验证集路径
                f"combined_dataset/images/test2017/{origin_img}",   # 测试集路径
                origin_img.replace('/', '\\'),  # Windows路径格式
                f"combined_dataset\\images\\train2017\\{origin_img}",
                f"combined_dataset\\images\\val2017\\{origin_img}",
                f"combined_dataset\\images\\test2017\\{origin_img}"
            ]
            
            sim_feats = None
            for key in possible_keys:
                if key in self.sim_feature_dict:
                    sim_feats_tuple = self.sim_feature_dict[key]
                    # 只取数值型特征，忽略字符串特征
                    sim_feats = sim_feats_tuple[0] if isinstance(sim_feats_tuple, tuple) else sim_feats_tuple
                    break
            
            if sim_feats is None:
                # 如果找不到匹配的键，使用零向量
                sim_feats = np.zeros(11, dtype=np.float32)
        
            # 对sim_feats进行归一化处理，防止数值过大
            sim_feats = np.array(sim_feats, dtype=np.float32)
            if np.std(sim_feats) > 0:
                sim_feats = (sim_feats - np.mean(sim_feats)) / np.std(sim_feats)
        
            sim_feats_tensor = torch.tensor(sim_feats, dtype=torch.float32)
        else:
            sim_feats = np.zeros(11, dtype=np.float32)
            # 对sim_feats进行归一化处理，防止数值过大
            sim_feats = np.array(sim_feats, dtype=np.float32)
            if np.std(sim_feats) > 0:
                sim_feats = (sim_feats - np.mean(sim_feats)) / np.std(sim_feats)
        
            sim_feats_tensor = torch.tensor(sim_feats, dtype=torch.float32)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, mask_tensor, sim_feats_tensor


# 数据增强包装类
class AugmentedDataset(Dataset):
    def __init__(self, dataset, augment_fn):
        self.dataset = dataset
        self.augment_fn = augment_fn
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, mask, sim_feat = self.dataset[idx]
        
        # 检查输入数据
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"[警告] AugmentedDataset: 输入图像包含NaN/Inf!")
            return self.__getitem__((idx + 1) % len(self))  # 尝试下一个样本
        
        # 检查图像数据类型和范围
        if img.dtype == torch.float32:
            # 如果已经是float32，需要反归一化到0-255范围
            if img.min() >= -3 and img.max() <= 3:
                # 反归一化：从ImageNet标准化范围转换回0-255
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = img * 255.0
                img = torch.clamp(img, 0, 255)
        
        # 转换为NumPy数组 (H, W, C) - 保持float类型用于归一化
        img = img.permute(1, 2, 0).numpy()
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # 确保mask维度一致 - 修复维度问题
        if mask.dim() == 3:
            mask = mask.squeeze(0)  # 从 [1, H, W] 转为 [H, W]
        elif mask.dim() == 1:
            mask = mask.unsqueeze(0)  # 从 [H] 转为 [1, H]
        mask = mask.numpy()
        
        # 检查转换后的数据
        if np.isnan(img).any() or np.isinf(img).any():
            print(f"[警告] AugmentedDataset: 转换后图像包含NaN/Inf!")
            return self.__getitem__((idx + 1) % len(self))  # 尝试下一个样本
        
        # 应用增强
        try:
            img, mask = self.augment_fn(img, mask)
        except Exception as e:
            print(f"[警告] AugmentedDataset: 数据增强失败: {e}")
            return self.__getitem__((idx + 1) % len(self))  # 尝试下一个样本
        
        # 确保返回的mask是3D tensor [1, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        # 调试：检查增强后的数据
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"[警告] AugmentedDataset: 增强后图像包含NaN/Inf!")
            print(f"img shape: {img.shape}, dtype: {img.dtype}")
            print(f"img range: [{img.min().item():.4f}, {img.max().item():.4f}]")
            # 返回一个安全的替代图像
            img = torch.zeros_like(img)
        
        return img, mask, sim_feat

def get_multimodal_patch_dataloaders(data_root="data/patch_dataset", 
                                    sim_feature_csv="data/sim_features.csv", 
                                    batch_size=4, 
                                    num_workers=4,
                                    damage_boost=5,
                                    normal_ratio=0.05,
                                    preload_to_memory=False,
                                    preload_to_gpu=False,
                                    device='cuda',
                                    pin_memory=True,
                                    persistent_workers=True,
                                    prefetch_factor=8,
                                    drop_last=True):
    """
    获取多模态patch数据加载器，支持内存和GPU预加载
    
    Args:
        data_root: 数据根目录
        sim_feature_csv: 仿真特征CSV文件路径
        batch_size: 批次大小
        num_workers: 数据加载进程数
        damage_boost: 损坏样本增强倍数
        normal_ratio: 正常样本比例
        preload_to_memory: 是否预加载到内存
        preload_to_gpu: 是否预加载到GPU显存
        device: 设备类型
    """
    
    sim_feature_dict = load_sim_features(sim_feature_csv)
    
    # 创建基础数据集
    train_dataset = MultiModalPatchSegmentationDataset(
        os.path.join(data_root, "train/images"),
        os.path.join(data_root, "train/masks"),
        os.path.join(data_root, "patch_index_train.csv"),
        sim_feature_dict,
        transform=None  # 稍后应用增强
    )
    
    val_dataset = MultiModalPatchSegmentationDataset(
        os.path.join(data_root, "val/images"),
        os.path.join(data_root, "val/masks"),
        os.path.join(data_root, "patch_index_val.csv"),
        sim_feature_dict,
        transform=None
    )
    
    # 应用过采样 - 启用数据增强
    train_aug = AdvancedAugmentation(is_training=True)
    val_aug = AdvancedAugmentation(is_training=False)  # 验证集也需要归一化
    
    train_dataset = DamageAwareDataset(train_dataset, damage_boost=damage_boost, normal_ratio=normal_ratio)
    
    # 启用增强
    train_dataset = AugmentedDataset(train_dataset, train_aug)
    val_dataset = AugmentedDataset(val_dataset, val_aug)
    
    # 根据预加载选项处理数据集
    # if preload_to_gpu:
    #     print("🚀 启用GPU显存预加载...")
    #     train_dataset = GPUPreloadedDataset(train_dataset, device=device, batch_size=batch_size)
    #     val_dataset = GPUPreloadedDataset(val_dataset, device=device, batch_size=batch_size)
    #     # GPU预加载后，不需要DataLoader的pin_memory和num_workers
    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=4,  # GPU预加载不需要多进程
    #         pin_memory=False,  # 数据已在GPU
    #         drop_last=True,
    #         collate_fn=custom_collate_fn
    #     )
    #     val_loader = DataLoader(
    #         val_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=4,
    #         pin_memory=False,
    #         collate_fn=custom_collate_fn
    #     )
    # elif preload_to_memory:
    #     print("💾 启用内存预加载...")
    #     train_dataset = MemoryCachedDataset(train_dataset, device='cpu')
    #     val_dataset = MemoryCachedDataset(val_dataset, device='cpu')
    #     # 内存预加载后，仍然需要pin_memory来快速传输到GPU
    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #         persistent_workers=persistent_workers,
    #         prefetch_factor=prefetch_factor,
    #         drop_last=drop_last,
    #         collate_fn=custom_collate_fn
    #     )
    #     val_loader = DataLoader(
    #         val_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #         persistent_workers=persistent_workers,
    #         prefetch_factor=prefetch_factor,
    #         drop_last=drop_last,
    #         collate_fn=custom_collate_fn
    #     )
    # else:
    print("📁 使用标准数据加载...")
    # 标准数据加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=optimized_collate  # 使用从data_loader导入的优化collate函数
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=optimized_collate  # 使用从data_loader导入的优化collate函数
    )
    return train_loader, val_loader

# 多模态分割模型 - 增强版本
class EnhancedDeepLab(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, sim_feat_dim=11):
        super().__init__()
        # 使用更强大的编码器
        self.deeplab = smp.DeepLabV3Plus(
            encoder_name="resnext101_32x8d",  # 更强的预训练编码器
            encoder_weights="imagenet",       # 使用预训练权重
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        # 改进的特征融合模块
        self.sim_fusion = nn.Sequential(
            nn.Linear(sim_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2048),  # 修改为2048以匹配encoder输出维度
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 注意力门控机制
        self.attention_gate = nn.Sequential(
            nn.Conv2d(2048 + 2048, 512, kernel_size=1),  # 修改输入通道为2048+2048=4096
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),  # 修改输出通道为2048以匹配x的维度
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.sim_fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.attention_gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, img, sim_feat):
        # 输入检查
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"[警告] 模型输入图像包含NaN/Inf!")
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(sim_feat).any() or torch.isinf(sim_feat).any():
            print(f"[警告] 模型输入sim特征包含NaN/Inf!")
            sim_feat = torch.nan_to_num(sim_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 提取图像特征
        features = self.deeplab.encoder(img)
        x = features[-1]
        B, C, H, W = x.shape
        
        # 处理仿真特征
        sim_proj = self.sim_fusion(sim_feat)
        sim_proj = sim_proj.view(B, -1, 1, 1)
        sim_proj = sim_proj.expand(-1, -1, H, W)
        
        # 注意力融合
        combined = torch.cat([x, sim_proj], dim=1)
        attention = self.attention_gate(combined)
        # 确保sim_proj维度与x匹配
        sim_proj = F.interpolate(sim_proj, size=x.shape[2:], mode='nearest')
        fused = x * attention + sim_proj * (1 - attention)
        
        # 检查融合后的特征
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print(f"[警告] 融合特征包含NaN/Inf，使用原始特征!")
            fused = x
        
        # 解码器
        features = list(features)
        features[-1] = fused
        out = self.deeplab.decoder(features)
        out = self.deeplab.segmentation_head(out)
        
        # 检查最终输出
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[警告] 模型输出包含NaN/Inf，返回零张量!")
            out = torch.zeros_like(out)
        
        return out

class MultiClassDeepLab(nn.Module):
    """
    支持多类别分类的DeepLab模型
    输出5个类别：背景(0)、未损坏(1)、轻微损坏(2)、中等损坏(3)、严重损坏(4)
    """
    def __init__(self, in_channels=3, num_classes=5, sim_feat_dim=11):
        super().__init__()
        # 使用更强大的编码器
        self.deeplab = smp.DeepLabV3Plus(
            encoder_name="resnext101_32x8d",  # 更强的预训练编码器
            encoder_weights="imagenet",       # 使用预训练权重
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        # 改进的特征融合模块
        self.sim_fusion = nn.Sequential(
            nn.Linear(sim_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2048),  # 修改为2048以匹配encoder输出维度
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 注意力门控机制
        self.attention_gate = nn.Sequential(
            nn.Conv2d(2048 + 2048, 512, kernel_size=1),  # 修改输入通道为2048+2048=4096
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),  # 修改输出通道为2048以匹配x的维度
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.sim_fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.attention_gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, img, sim_feat):
        # 输入检查
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"[警告] 模型输入图像包含NaN/Inf!")
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(sim_feat).any() or torch.isinf(sim_feat).any():
            print(f"[警告] 模型输入sim特征包含NaN/Inf!")
            sim_feat = torch.nan_to_num(sim_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 提取图像特征
        features = self.deeplab.encoder(img)
        x = features[-1]
        B, C, H, W = x.shape
        
        # 处理仿真特征
        sim_proj = self.sim_fusion(sim_feat)
        sim_proj = sim_proj.view(B, -1, 1, 1)
        sim_proj = sim_proj.expand(-1, -1, H, W)
        
        # 注意力融合
        combined = torch.cat([x, sim_proj], dim=1)
        attention = self.attention_gate(combined)
        # 确保sim_proj维度与x匹配
        sim_proj = F.interpolate(sim_proj, size=x.shape[2:], mode='nearest')
        fused = x * attention + sim_proj * (1 - attention)
        
        # 检查融合后的特征
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print(f"[警告] 融合特征包含NaN/Inf，使用原始特征!")
            fused = x
        
        # 解码器
        features = list(features)
        features[-1] = fused
        out = self.deeplab.decoder(features)
        out = self.deeplab.segmentation_head(out)
        
        # 检查最终输出
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[警告] 模型输出包含NaN/Inf，返回零张量!")
            out = torch.zeros_like(out)
        
        return out

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # 数据合法性检查
        if torch.isnan(inputs).any():
            print("[警告] inputs中有nan值")
            inputs = torch.nan_to_num(inputs, nan=0.0)
        if torch.isnan(targets).any():
            print("[警告] targets中有nan值")
            targets = torch.nan_to_num(targets, nan=0.0)
        
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 确保输入在合理范围内
        inputs = torch.clamp(inputs, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 检查计算结果
        if torch.isnan(dice) or torch.isinf(dice):
            print(f"[警告] Dice计算异常: dice={dice}, intersection={intersection}, inputs_sum={inputs.sum()}, targets_sum={targets.sum()}")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        return 1 - dice

# 评估指标
def iou_score(outputs, masks, smooth=1e-5):
    # 数据合法性检查
    if torch.isnan(outputs).any():
        outputs = torch.nan_to_num(outputs, nan=0.0)
    if torch.isnan(masks).any():
        masks = torch.nan_to_num(masks, nan=0.0)
    
    preds = (torch.sigmoid(outputs) > 0.5).float()
    masks = masks.float()
    intersection = (preds * masks).sum()
    union = (preds + masks).sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    # 检查计算结果
    if torch.isnan(iou) or torch.isinf(iou):
        return torch.tensor(0.0, device=outputs.device)
    
    return iou

def dice_score(outputs, masks, smooth=1e-5):
    # 数据合法性检查
    if torch.isnan(outputs).any():
        outputs = torch.nan_to_num(outputs, nan=0.0)
    if torch.isnan(masks).any():
        masks = torch.nan_to_num(masks, nan=0.0)
    
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    masks = masks.float()
    intersection = (outputs * masks).sum()
    dice = (2. * intersection + smooth) / (outputs.sum() + masks.sum() + smooth)
    
    # 检查计算结果
    if torch.isnan(dice) or torch.isinf(dice):
        return torch.tensor(0.0, device=outputs.device)
    
    return dice

# 组合损失函数：Dice + Focal + Boundary
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.dice = DiceLoss()
        self.alpha = alpha
        self.gamma = gamma
        
    def focal_loss(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
    def boundary_loss(self, pred, target):
        # 保证输入为4维 [B, 1, H, W]
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        # 计算边界梯度差异
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return (loss_x + loss_y) * 0.5
    
    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        # 边界损失需要sigmoid输出
        with torch.no_grad():
            pred_sigmoid = torch.sigmoid(inputs)
        
        boundary = self.boundary_loss(pred_sigmoid, targets)
        
        # 加权组合
        return self.alpha * dice + (1 - self.alpha) * focal + 0.3 * boundary

# 边界感知训练 (Boundary-Aware Training)
class BoundaryAwareLoss(nn.Module):
    def __init__(self, main_loss_fn, alpha=0.3, beta=0.2, kernel_size=3, use_sobel=True):
        super().__init__()
        self.main_loss = main_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.use_sobel = use_sobel
        self.kernel = torch.ones(1, 1, kernel_size, kernel_size).float() / (kernel_size**2)
        
        # Sobel算子 - 更精确的边界检测
        if use_sobel:
            self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                      dtype=torch.float32).view(1, 1, 3, 3)
            self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                      dtype=torch.float32).view(1, 1, 3, 3)
    
    def to(self, device):
        self.kernel = self.kernel.to(device)
        if self.use_sobel:
            self.sobel_x = self.sobel_x.to(device)
            self.sobel_y = self.sobel_y.to(device)
        return super().to(device)
    
    def compute_boundary_sobel(self, mask):
        # 保证sobel算子和mask在同一设备
        sobel_x = self.sobel_x.to(mask.device)
        sobel_y = self.sobel_y.to(mask.device)
        
        # 调试信息
        #print(f"[调试] 输入mask形状: {mask.shape}")
        
        # 处理多通道输入：如果mask有多个通道，取平均值
        if mask.size(1) > 1:
            mask = mask.mean(dim=1, keepdim=True)
            #print(f"[调试] 多通道处理后mask形状: {mask.shape}")
        
        # 确保mask是4维张量 [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
            #print(f"[调试] 3维转4维后mask形状: {mask.shape}")
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            #print(f"[调试] 2维转4维后mask形状: {mask.shape}")
        
        # 确保通道数为1
        if mask.size(1) != 1:
            mask = mask.mean(dim=1, keepdim=True)
            #print(f"[调试] 通道数调整后mask形状: {mask.shape}")
        
        #print(f"[调试] 最终mask形状: {mask.shape}")
        #print(f"[调试] sobel_x形状: {sobel_x.shape}")
        
        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        #print(f"[调试] 输出grad形状: {grad.shape}")
        return grad
    
    def compute_boundary_conv(self, mask):
        """使用卷积核计算边界 - 原始方法"""
        # 处理多通道输入：如果mask有多个通道，取平均值
        if mask.size(1) > 1:
            mask = mask.mean(dim=1, keepdim=True)
        
        # 确保mask是4维张量 [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        # 确保通道数为1
        if mask.size(1) != 1:
            mask = mask.mean(dim=1, keepdim=True)
        
        # 计算边界图
        avg_pooled = F.conv2d(mask, self.kernel, padding=self.kernel.size(-1)//2)
        boundary = torch.abs(mask - avg_pooled)
        return boundary
    
    def compute_boundary_multi_scale(self, mask, scales=[1, 2, 4]):
        """多尺度边界检测"""
        boundaries = []
        original_size = mask.shape[-2:]  # 保存原始尺寸
        
        for scale in scales:
            if scale == 1:
                scaled_mask = mask
            else:
                # 下采样 - 使用更精确的尺寸计算
                h, w = mask.shape[-2:]
                new_h, new_w = h // scale, w // scale
                scaled_mask = F.adaptive_avg_pool2d(mask, (new_h, new_w))
            
            # 计算边界
            if self.use_sobel:
                boundary = self.compute_boundary_sobel(scaled_mask)
            else:
                boundary = self.compute_boundary_conv(scaled_mask)
            
            # 确保所有边界都上采样到原始尺寸
            if boundary.shape[-2:] != original_size:
                boundary = F.interpolate(boundary, size=original_size, 
                                       mode='bilinear', align_corners=True)
            
            boundaries.append(boundary)
        
        # 确保所有边界张量尺寸一致后再stack
        if len(boundaries) > 0:
            # 强制所有边界张量都调整到原始尺寸
            target_size = original_size
            for i, boundary in enumerate(boundaries):
                if boundary.shape[-2:] != target_size:
                    print(f"调整边界张量{i}尺寸从{boundary.shape}到目标尺寸{target_size}")
                    boundary = F.interpolate(boundary, size=target_size, 
                                           mode='bilinear', align_corners=True)
                    boundaries[i] = boundary
            
            # 验证所有边界张量尺寸一致
            first_shape = boundaries[0].shape
            for i, boundary in enumerate(boundaries):
                if boundary.shape != first_shape:
                    print(f"错误：边界张量{i}尺寸仍不匹配，期望{first_shape}，实际{boundary.shape}")
                    # 最后一次强制调整
                    boundary = F.interpolate(boundary, size=first_shape[-2:], 
                                           mode='bilinear', align_corners=True)
                    boundaries[i] = boundary
            
            # 多尺度边界融合
            multi_scale_boundary = torch.mean(torch.stack(boundaries), dim=0)
        else:
            # 如果没有边界，返回零张量
            multi_scale_boundary = torch.zeros_like(mask)
        
        return multi_scale_boundary
    
    def compute_boundary(self, mask):
        """主边界计算方法"""
        if self.use_sobel:
            # 使用多尺度边界检测
            return self.compute_boundary_multi_scale(mask)
        else:
            return self.compute_boundary_conv(mask)
    
    def forward(self, pred, target):
        # 主损失
        main_loss = self.main_loss(pred, target)
        
        # 预测边界
        pred_prob = torch.sigmoid(pred)
        pred_boundary = self.compute_boundary(pred_prob)
        
        # 真实边界
        target_boundary = self.compute_boundary(target)
        
        # 边界损失
        boundary_loss = F.l1_loss(pred_boundary, target_boundary)
        
        # 边界IoU损失
        pred_boundary_bin = (pred_boundary > 0.1).float()
        target_boundary_bin = (target_boundary > 0.1).float()
        boundary_iou = 1 - iou_score(pred_boundary_bin, target_boundary_bin)
        
        # 组合损失
        total_loss = main_loss + self.alpha * boundary_loss + self.beta * boundary_iou
        
        return total_loss

# 自适应难样本挖掘 (Adaptive Hard Example Mining)
class AdaptiveMiner:
    def __init__(self, base_loss_fn, start_epoch=5, max_ratio=0.3, warmup_epochs=10):
        self.base_loss = base_loss_fn
        self.start_epoch = start_epoch
        self.max_ratio = max_ratio
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
    
    def __call__(self, pred, target):
        # 计算每个像素的损失
        pixel_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 前几个epoch使用标准损失
        if self.current_epoch < self.start_epoch:
            return pixel_loss.mean()
        
        # 计算难样本比例（随epoch线性增加）
        ratio = min(self.max_ratio, 
                  self.max_ratio * (self.current_epoch - self.start_epoch) / self.warmup_epochs)
        
        # 识别难样本
        with torch.no_grad():
            # 预测概率
            pred_prob = torch.sigmoid(pred)
            # 预测错误区域
            incorrect_mask = (pred_prob > 0.5).float() != target
            # 边界区域
            boundary = self.compute_boundary(target)
            boundary_mask = boundary > 0.1
            
            # 难样本掩码
            hard_mask = incorrect_mask | boundary_mask
            
        # 计算难样本损失
        hard_loss = (pixel_loss * hard_mask).sum() / (hard_mask.sum() + 1e-5)
        
        # 计算易样本损失
        easy_loss = (pixel_loss * ~hard_mask).sum() / ((~hard_mask).sum() + 1e-5)
        
        # 加权组合
        return ratio * hard_loss + (1 - ratio) * easy_loss
    
    def compute_boundary(self, mask, kernel_size=3):
        # 处理多通道输入：如果mask有多个通道，取平均值
        if mask.size(1) > 1:
            mask = mask.mean(dim=1, keepdim=True)
        
        # 确保mask是4维张量 [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        # 确保通道数为1
        if mask.size(1) != 1:
            mask = mask.mean(dim=1, keepdim=True)
        
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device) / (kernel_size**2)
        avg_pooled = F.conv2d(mask, kernel, padding=kernel_size//2)
        boundary = torch.abs(mask - avg_pooled)
        return boundary

# 增强的损失函数组合
class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        super().__init__()
        self.alpha = alpha  # Dice权重
        self.beta = beta    # Focal权重
        self.gamma = gamma  # Boundary权重
        self.delta = delta  # IoU权重
        
        self.dice_loss = DiceLoss()
        
    def focal_loss(self, inputs, targets, gamma=2.0):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**gamma * BCE_loss
        return F_loss.mean()
    
    def boundary_loss(self, pred, target):
        # 计算边界损失
        pred_prob = torch.sigmoid(pred)
        
        # 处理多通道输入：如果pred_prob有多个通道，取平均值
        if pred_prob.size(1) > 1:
            pred_prob = pred_prob.mean(dim=1, keepdim=True)
        
        # 处理多通道输入：如果target有多个通道，取平均值
        if target.size(1) > 1:
            target = target.mean(dim=1, keepdim=True)
        
        # 确保是4维张量 [B, 1, H, W]
        if pred_prob.dim() == 3:
            pred_prob = pred_prob.unsqueeze(1)
        elif pred_prob.dim() == 2:
            pred_prob = pred_prob.unsqueeze(0).unsqueeze(0)
            
        if target.dim() == 3:
            target = target.unsqueeze(1)
        elif target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        
        # 使用Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        pred_grad_x = F.conv2d(pred_prob, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_prob, sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        return F.l1_loss(pred_grad, target_grad)
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        
        # IoU损失
        iou = 1 - iou_score(inputs, targets)
        
        # 组合损失
        total_loss = (self.alpha * dice + 
                     self.beta * focal + 
                     self.gamma * boundary + 
                     self.delta * iou)
        
        return total_loss

class MultiClassLoss(nn.Module):
    """
    多类别分类损失函数
    支持5个类别：背景(0)、未损坏(1)、轻微损坏(2)、中等损坏(3)、严重损坏(4)
    """
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1, class_weights=None):
        super().__init__()
        self.alpha = alpha  # 交叉熵损失权重
        self.beta = beta    # Dice损失权重
        self.gamma = gamma  # Focal损失权重
        self.delta = delta  # 边界损失权重
        
        # 类别权重：更关注损坏类别
        if class_weights is None:
            # 背景权重较低，损坏类别权重较高
            self.class_weights = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        else:
            self.class_weights = class_weights
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.focal_loss = self._focal_loss
        self.dice_loss = self._dice_loss
        self.boundary_loss = self._boundary_loss
    
    def _focal_loss(self, inputs, targets, gamma=2.0, alpha=0.25):
        """Focal Loss for multi-class classification"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
    def _dice_loss(self, inputs, targets, smooth=1e-5):
        """Dice Loss for multi-class classification"""
        # 将输入转换为概率分布
        probs = F.softmax(inputs, dim=1)
        
        # 将目标转换为one-hot编码
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 计算每个类别的Dice损失
        dice_loss = 0
        for i in range(num_classes):
            intersection = (probs[:, i] * targets_one_hot[:, i]).sum()
            union = probs[:, i].sum() + targets_one_hot[:, i].sum()
            dice_loss += 1 - (2 * intersection + smooth) / (union + smooth)
        
        return dice_loss / num_classes
    
    def _boundary_loss(self, pred, target):
        """边界损失：关注损坏区域的边界"""
        # 将输入转换为概率分布
        probs = F.softmax(pred, dim=1)
        
        # 创建损坏区域的mask（类别2,3,4）
        damage_mask = (target >= 2).float()
        
        # 计算边界
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        boundary = F.conv2d(damage_mask.unsqueeze(1), kernel, padding=1)
        boundary = ((boundary > 0) & (boundary < 9)).float()
        
        # 在边界区域计算损失
        boundary_loss = 0
        for i in range(2, 5):  # 只考虑损坏类别
            class_mask = (target == i).float()
            boundary_class = boundary * class_mask
            if boundary_class.sum() > 0:
                prob_class = probs[:, i]
                boundary_loss += F.binary_cross_entropy(prob_class, boundary_class)
        
        return boundary_loss
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] 模型输出
            targets: [B, H, W] 目标标签 (0-4)
        """
        # 数据合法性检查
        if torch.isnan(inputs).any():
            print(f"[警告] 损失函数输入包含NaN!")
            inputs = torch.nan_to_num(inputs, nan=0.0)
        
        if torch.isnan(targets).any():
            print(f"[警告] 损失函数目标包含NaN!")
            targets = torch.nan_to_num(targets, nan=0.0)
        
        # 确保目标在有效范围内
        targets = torch.clamp(targets, 0, 4).long()
        
        # 计算各种损失
        ce_loss = self.ce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        boundary_loss = self.boundary_loss(inputs, targets)
        
        # 组合损失
        total_loss = (self.alpha * ce_loss + 
                     self.beta * dice_loss + 
                     self.gamma * focal_loss + 
                     self.delta * boundary_loss)
        
        return total_loss

def multi_class_iou_score(outputs, masks, num_classes=5, smooth=1e-5):
    """
    多类别分类的IoU评分
    Args:
        outputs: [B, C, H, W] 模型输出
        masks: [B, H, W] 目标标签
        num_classes: 类别数量
        smooth: 平滑因子
    """
    # 数据合法性检查
    if torch.isnan(outputs).any():
        print(f"[警告] IoU计算输入包含NaN!")
        outputs = torch.nan_to_num(outputs, nan=0.0)
    
    if torch.isnan(masks).any():
        print(f"[警告] IoU计算目标包含NaN!")
        masks = torch.nan_to_num(masks, nan=0.0)
    
    # 获取预测类别
    pred = torch.argmax(outputs, dim=1)
    
    # 计算每个类别的IoU
    iou_scores = []
    for i in range(num_classes):
        pred_mask = (pred == i).float()
        target_mask = (masks == i).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    
    # 返回平均IoU和每个类别的IoU
    mean_iou = sum(iou_scores) / len(iou_scores)
    return mean_iou, iou_scores

def multi_class_dice_score(outputs, masks, num_classes=5, smooth=1e-5):
    """
    多类别分类的Dice评分
    Args:
        outputs: [B, C, H, W] 模型输出
        masks: [B, H, W] 目标标签
        num_classes: 类别数量
        smooth: 平滑因子
    """
    # 数据合法性检查
    if torch.isnan(outputs).any():
        print(f"[警告] Dice计算输入包含NaN!")
        outputs = torch.nan_to_num(outputs, nan=0.0)
    
    if torch.isnan(masks).any():
        print(f"[警告] Dice计算目标包含NaN!")
        masks = torch.nan_to_num(masks, nan=0.0)
    
    # 获取预测类别
    pred = torch.argmax(outputs, dim=1)
    
    # 计算每个类别的Dice分数
    dice_scores = []
    for i in range(num_classes):
        pred_mask = (pred == i).float()
        target_mask = (masks == i).float()
        
        intersection = (pred_mask * target_mask).sum()
        dice = (2 * intersection + smooth) / (pred_mask.sum() + target_mask.sum() + smooth)
        dice_scores.append(dice.item())
    
    # 返回平均Dice和每个类别的Dice
    mean_dice = sum(dice_scores) / len(dice_scores)
    return mean_dice, dice_scores

def evaluate_multi_class_performance(outputs, masks, num_classes=5):
    """
    多类别分类性能评估
    Args:
        outputs: [B, C, H, W] 模型输出
        masks: [B, H, W] 目标标签
        num_classes: 类别数量
    Returns:
        dict: 包含各种评估指标
    """
    # 获取预测类别
    pred = torch.argmax(outputs, dim=1)
    
    # 计算IoU和Dice
    mean_iou, class_ious = multi_class_iou_score(outputs, masks, num_classes)
    mean_dice, class_dices = multi_class_dice_score(outputs, masks, num_classes)
    
    # 计算准确率
    accuracy = (pred == masks).float().mean().item()
    
    # 计算损坏类别的性能（类别2,3,4）
    damage_pred = (pred >= 2).float()
    damage_target = (masks >= 2).float()
    damage_iou = ((damage_pred * damage_target).sum() + 1e-5) / (damage_pred.sum() + damage_target.sum() - (damage_pred * damage_target).sum() + 1e-5)
    
    # 计算严重损坏的性能（类别4）
    severe_pred = (pred == 4).float()
    severe_target = (masks == 4).float()
    severe_iou = ((severe_pred * severe_target).sum() + 1e-5) / (severe_pred.sum() + severe_target.sum() - (severe_pred * severe_target).sum() + 1e-5)
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'accuracy': accuracy,
        'damage_iou': damage_iou.item(),
        'severe_damage_iou': severe_iou.item(),
        'class_ious': class_ious,
        'class_dices': class_dices,
        'class_names': ['背景', '未损坏', '轻微损坏', '中等损坏', '严重损坏']
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型选择配置
    use_multiscale = False  # 设置为True使用多尺度模型，False使用原始模型
    use_landslide_detector = True  # 设置为True使用LandslideDetector模型
    use_ensemble = True  # 设置为True使用模型集成训练
    
    print("="*60)
    print("🚀 模型配置")
    print("="*60)
    if use_ensemble:
        print("🎯 使用模型集成训练 (DeepLab + LandslideDetector)")
    elif use_landslide_detector:
        print("📊 使用优化的LandslideDetector模型")
    elif use_multiscale:
        print("📊 使用多尺度特征融合模型 (MultiScaleDeepLab)")
        print("⚠️  注意：多尺度模型与现有检查点不兼容，将从头开始训练")
    else:
        print("📊 使用原始增强模型 (EnhancedDeepLab)")
    print("="*60)
    
    # 数据加载配置
    print("🚀 数据加载配置")
    print("="*60)
    
    # 根据可用内存选择预加载策略
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    print(f"系统内存: {total_memory_gb:.1f} GB")
    print(f"GPU显存: {gpu_memory_gb:.1f} GB")
    
    # === 强制动态加载+pin memory ===
    preload_to_memory = False
    preload_to_gpu = False
    print("📁 [强制] 使用标准数据加载 (动态加载+pin memory)")
    
    # 自动选择预加载策略
    if total_memory_gb >= 32:  # 32GB以上内存
        print("💾 检测到充足内存，启用内存预加载")
        preload_to_memory = True
    elif gpu_memory_gb >= 16:  # 16GB以上显存
        print("🚀 检测到充足显存，启用GPU预加载")
        preload_to_gpu = True
    else:
        print("📁 使用标准数据加载")
    
    # 动态加载配置
    batch_size = 32  # 默认batch size，可根据GPU显存调整
    num_workers = 12  # 使用12个工作进程
    pin_memory = True  # 启用pin memory
    
    # 根据GPU显存自动调整batch size
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = torch.cuda.memory_reserved(0)
        if total_memory >= 16 * 1024**3:  # 16GB以上显存
            batch_size =64
        elif total_memory >= 8 * 1024**3:  # 8GB显存
            batch_size = 32
        else:  # 小于8GB显存
            batch_size = 16
        print(f"自动设置batch size为: {batch_size} (根据GPU显存 {total_memory/1024**3:.1f}GB)")
    persistent_workers = True  # 保持工作进程存活
    prefetch_factor = 2  # 预取2个batch
    drop_last = True  # 丢弃最后一个不完整的batch
    
    print(f"批次大小: {batch_size}")
    print(f"工作进程数: {num_workers}")
    print("="*60)
    
    train_loader, val_loader = get_multimodal_patch_dataloaders(
        data_root="data/patch_dataset",
        sim_feature_csv="data/sim_features.csv",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        damage_boost=5,
        normal_ratio=0.05,
        preload_to_memory=preload_to_memory,
        preload_to_gpu=preload_to_gpu,
        device=device
    )
    
    # 根据配置选择模型
    if use_ensemble:
        # 创建模型集成
        deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11).to(device)
        landslide_model = create_segmentation_landslide_model(
            num_classes=1,
            use_attention=True,
            use_fpn=True,
            use_dynamic_attention=True,
            use_multi_scale=True
        ).to(device)
        
        # 创建双模型集成
        ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5).to(device)
        # 为了兼容性，将ensemble_model赋值给model
        model = ensemble_model
        print("✅ 双模型集成创建成功 (DeepLab + 分割版LandslideDetector)")
        
    elif use_landslide_detector:
        model = create_segmentation_landslide_model(
            num_classes=1,
            use_attention=True,
            use_fpn=True,
            use_dynamic_attention=True,
            use_multi_scale=True
        ).to(device)
        print("✅ 分割版LandslideDetector模型创建成功")
        
    elif use_multiscale:
        model = MultiScaleDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11).to(device)
        print("✅ 多尺度特征融合模型创建成功")
    else:
        model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11).to(device)
        print("✅ 原始增强模型创建成功")
    
    # 使用增强的损失函数 - 修复：使用更稳定的损失函数
    if use_ensemble:
        # 对于集成模型，使用更简单的损失函数以避免数值不稳定
        criterion = DiceLoss()
        print("✅ 使用DiceLoss以确保数值稳定性")
    else:
        base_criterion = HybridLoss()
        criterion = BoundaryAwareLoss(base_criterion, alpha=0.3, beta=0.2)
        
        # 创建自适应难样本挖掘器
        adaptive_miner = AdaptiveMiner(base_criterion, start_epoch=5, max_ratio=0.3, warmup_epochs=10)
        print("✅ 边界感知损失和自适应难样本挖掘已启用")
    
    # === 掩码处理和后处理配置 ===
    print("🔧 掩码处理和后处理配置")
    print("="*60)
    
    # 掩码处理配置
    damage_level = 'all'  # 可选: 'all', 'light', 'binary', 'multi'
    print(f"📊 掩码处理方式: {damage_level}")
    print("  - all: 所有损坏级别(2,3,4)都标记为1")
    print("  - light: 轻微损坏(2)标记为0.3，中等(3)标记为0.6，严重(4)标记为1.0")
    print("  - binary: 轻微(2)标记为0，中等和严重(3,4)标记为1")
    print("  - multi: 轻微(2)标记为1，中等(3)标记为2，严重(4)标记为3")
    
    # 后处理配置
    enable_postprocess = True  # 是否启用后处理
    postprocess_min_area = 100  # 最小连通域面积
    postprocess_merge_distance = 10  # 合并距离
    print(f"🔧 后处理配置:")
    print(f"  - 启用后处理: {enable_postprocess}")
    print(f"  - 最小连通域面积: {postprocess_min_area}")
    print(f"  - 合并距离: {postprocess_merge_distance}")
    print("="*60)
    
    # 分层学习率配置
    encoder_params = []
    decoder_params = []
    fusion_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'decoder' in name or 'segmentation_head' in name:
            decoder_params.append(param)
        else:
            fusion_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},    # 预训练编码器用较小LR
        {'params': decoder_params, 'lr': 1e-4},
        {'params': fusion_params, 'lr': 5e-4}
    ], weight_decay=1e-4)
    
    # 临时调度器，稍后更新
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-4, 5e-3, 2.5e-3],  # 对应encoder/decoder/fusion的max_lr
        total_steps=20 * len(train_loader),  # 临时值
        pct_start=0.3,
        anneal_strategy='linear',
        final_div_factor=10000,
        three_phase=False
    )
    
    # 创建混合精度训练器
    print("🚀 初始化CPU-GPU协同训练器...")
    hybrid_trainer = HybridPrecisionTrainer(
        model, 
        optimizer,
        criterion,
        device,
        num_cpu_workers=8,
        num_async_workers=8
    )
    print("✅ CPU-GPU协同训练器初始化完成")
    print(f"配置参数: 12个CPU工作线程, 8个异步数据预处理线程 ")

    # === 智能模型恢复相关 ===
    checkpoint_path = "models/checkpoint.pth"
    best_model_path = "models/best_multimodal_patch_model.pth"
    
    # 智能恢复训练状态 - 优先从最佳模型文件恢复
    start_epoch = 1
    best_val_iou = 0.0
    iou_log = []
    
    # 首先尝试从最佳模型文件恢复（如果它是检查点格式）
    if os.path.exists(best_model_path):
        try:
            print(f"🔍 检查最佳模型文件: {best_model_path}")
            best_model_data = torch.load(best_model_path, map_location=device)
            
            # 检查是否是最佳模型文件包含检查点信息
            if isinstance(best_model_data, dict) and 'model_state_dict' in best_model_data:
                print("✅ 发现最佳模型文件包含完整训练状态")
                print(f"   - Epoch: {best_model_data.get('epoch', 'N/A')}")
                print(f"   - 最佳IoU: {best_model_data.get('best_val_iou', 'N/A'):.4f}")
                print(f"   - 包含优化器状态: {'optimizer_state_dict' in best_model_data}")
                print(f"   - 包含调度器状态: {'scheduler_state_dict' in best_model_data}")
                
                # 从最佳模型文件恢复
                if use_ensemble:
                    ensemble_model.load_state_dict(best_model_data['model_state_dict'])
                else:
                    model.load_state_dict(best_model_data['model_state_dict'])
                
                if 'optimizer_state_dict' in best_model_data:
                    optimizer.load_state_dict(best_model_data['optimizer_state_dict'])
                if 'scheduler_state_dict' in best_model_data:
                    scheduler.load_state_dict(best_model_data['scheduler_state_dict'])
                if 'scaler_state_dict' in best_model_data:
                    hybrid_trainer.scaler.load_state_dict(best_model_data['scaler_state_dict'])
                
                start_epoch = best_model_data.get('epoch', 0) + 1
                best_val_iou = best_model_data.get('best_val_iou', 0.0)
                iou_log = best_model_data.get('iou_log', [])
                
                print(f"✅ 成功从最佳模型文件恢复训练状态!")
                print(f"  从epoch {start_epoch} 继续训练")
                print(f"  历史最佳IoU: {best_val_iou:.4f}")
                print(f"  已训练epoch数: {len(iou_log)}")
                
                # 同时更新检查点文件
                torch.save(best_model_data, checkpoint_path)
                print(f"✅ 已同步更新检查点文件: {checkpoint_path}")
                
            else:
                print("⚠️ 最佳模型文件不是检查点格式，尝试从检查点文件恢复...")
                raise Exception("最佳模型文件不是检查点格式")
                
        except Exception as e:
            print(f"❌ 从最佳模型文件恢复失败: {e}")
            print("尝试从检查点文件恢复...")
            
            # 如果从最佳模型文件恢复失败，尝试从检查点文件恢复
            if os.path.exists(checkpoint_path):
                try:
                    print(f"发现检查点文件: {checkpoint_path}")
                    print("正在尝试恢复训练状态...")
                    
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # 检查模型结构兼容性
                    if use_ensemble:
                        ensemble_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if 'scaler_state_dict' in checkpoint:
                        hybrid_trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    best_val_iou = checkpoint.get('best_val_iou', 0.0)
                    iou_log = checkpoint.get('iou_log', [])
                    
                    print(f"✅ 成功从检查点文件恢复训练状态!")
                    print(f"  从epoch {start_epoch} 继续训练")
                    print(f"  历史最佳IoU: {best_val_iou:.4f}")
                    print(f"  已训练epoch数: {len(iou_log)}")
                    
                except Exception as e2:
                    print(f"❌ 恢复检查点失败: {e2}")
                    print("将从头开始训练...")
                    start_epoch = 1
                    best_val_iou = 0.0
                    iou_log = []
            else:
                print("未发现检查点文件，将从头开始训练")
    else:
        print("未发现最佳模型文件，尝试从检查点文件恢复...")
        if os.path.exists(checkpoint_path):
            try:
                print(f"发现检查点文件: {checkpoint_path}")
                print("正在尝试恢复训练状态...")
                
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 检查模型结构兼容性
                if use_ensemble:
                    ensemble_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    hybrid_trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_iou = checkpoint.get('best_val_iou', 0.0)
                iou_log = checkpoint.get('iou_log', [])
                
                print(f"✅ 成功从检查点文件恢复训练状态!")
                print(f"  从epoch {start_epoch} 继续训练")
                print(f"  历史最佳IoU: {best_val_iou:.4f}")
                print(f"  已训练epoch数: {len(iou_log)}")
                
            except Exception as e:
                print(f"❌ 恢复检查点失败: {e}")
                print("将从头开始训练...")
                start_epoch = 1
                best_val_iou = 0.0
                iou_log = []
        else:
            print("未发现检查点文件，将从头开始训练")

    # === 分阶段训练逻辑 ===
    # 计算当前阶段 - 修复：根据实际start_epoch计算正确的阶段
    current_stage = (start_epoch - 1) // 20 + 1
    stage_start_epoch = (current_stage - 1) * 20 + 1
    stage_end_epoch = current_stage * 20
    
    print(f"\n当前训练阶段: {current_stage}")
    print(f"阶段范围: epoch {stage_start_epoch} - {stage_end_epoch}")
    print(f"实际开始epoch: {start_epoch}")
    
    # 更新调度器的total_steps
    total_steps = (stage_end_epoch - start_epoch + 1) * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-4, 5e-3, 2.5e-3],  # 对应encoder/decoder/fusion的max_lr
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='linear',
        final_div_factor=10000,
        three_phase=False
    )
    
    # 检查是否需要训练 - 修复：只有当start_epoch超出当前阶段范围时才跳过训练
    if start_epoch > stage_end_epoch:
        print(f"\n当前阶段已完成（从epoch {start_epoch}恢复），直接进行量化...")
    else:
        # 正常训练循环（当前阶段）
        print(f"开始训练阶段 {current_stage}，从epoch {start_epoch} 到 {stage_end_epoch}")
        
        # 根据模型类型选择训练方式
        if use_ensemble:
            print("🎯 使用双模型集成训练")
            # 为集成模型创建优化器和调度器 - 使用更保守的学习率和自适应调整
            ensemble_optimizer = optim.AdamW(ensemble_model.parameters(), lr=1e-6, weight_decay=1e-3)
            ensemble_scheduler = optim.lr_scheduler.OneCycleLR(
                ensemble_optimizer,
                max_lr=5e-5,  # 进一步降低最大学习率
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='linear',
                final_div_factor=10000
            )
            ensemble_scaler = GradScaler()
            
            for epoch in range(start_epoch, stage_end_epoch + 1):
                # 使用双模型集成训练
                train_loss, train_iou = train_dual_model_epoch(
                    ensemble_model, train_loader, ensemble_optimizer, criterion, device, epoch, ensemble_scaler
                )
                
                # 验证
                val_loss, val_iou = val_dual_model_epoch(
                    ensemble_model, val_loader, criterion, device, ensemble_scaler
                )
                
                # 更新学习率
                ensemble_scheduler.step()
                
                print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
                print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
                print(f"Fusion Weight: {ensemble_model.get_fusion_weight():.3f}")
                
                # 保存最佳模型 - 增强版：包含完整训练状态
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ensemble_model.state_dict(),
                        'optimizer_state_dict': ensemble_optimizer.state_dict(),
                        'scheduler_state_dict': ensemble_scheduler.state_dict(),
                        'scaler_state_dict': ensemble_scaler.state_dict(),
                        'best_val_iou': best_val_iou,
                        'iou_log': iou_log,
                        'fusion_weight': ensemble_model.get_fusion_weight()
                    }, best_model_path)
                    print(f"✅ 保存最佳集成模型，IoU: {val_iou:.4f}")
                    print(f"   已保存完整训练状态到: {best_model_path}")
                
                iou_log.append(val_iou)
        else:
            # 原始单模型训练
            for epoch in range(start_epoch, stage_end_epoch + 1):
                model.train()
                total_loss = 0
                total_iou = 0
                total_dice = 0
                
                # 更新自适应难样本挖掘器的epoch
                adaptive_miner.update_epoch(epoch)
                
                # 性能监控
                epoch_start_time = time.time()
                gpu_compute_time = 0
                cpu_compute_time = 0
                
                for batch_idx, (images, masks, sim_feats) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{stage_end_epoch} - Training")):
                    # 数据移动到GPU
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    sim_feats = sim_feats.to(device, non_blocking=True)
                
                # 掩码处理 - 支持动态调整
                if damage_level != 'all':
                    # 计算当前batch的样本比例
                    current_sample_ratio = None
                    if damage_level == 'adaptive':
                        # 计算损坏样本比例
                        damage_pixels = (masks >= 2).sum().float()
                        total_pixels = masks.numel()
                        current_sample_ratio = damage_pixels / total_pixels
                        if batch_idx % 100 == 0:  # 每100个batch打印一次，减少刷屏
                            print(f"[动态调整] Batch {batch_idx} 损坏样本比例: {current_sample_ratio:.4f}")
                    
                    masks = process_xview2_mask(masks, damage_level, current_sample_ratio)
                
                # 使用混合精度训练器
                gpu_start = time.time()
                
                # 前向传播
                with autocast('cuda'):
                    outputs = model(images, sim_feats)
                    # 使用自适应难样本挖掘损失
                    loss = adaptive_miner(outputs, masks)
                
                # 反向传播
                hybrid_trainer.optimizer.zero_grad()
                hybrid_trainer.scaler.scale(loss).backward()
                hybrid_trainer.scaler.step(hybrid_trainer.optimizer)
                hybrid_trainer.scaler.update()
                
                gpu_compute_time += time.time() - gpu_start
                
                # 后处理优化IoU
                with torch.no_grad():
                    if enable_postprocess:
                        try:
                            processed_outputs = torch.stack([
                                postprocess(out, min_area=postprocess_min_area, merge_distance=postprocess_merge_distance)
                                for out in outputs.detach().cpu()
                            ]).to(device)
                            batch_iou = iou_score(processed_outputs, masks).item()
                            batch_dice = dice_score(processed_outputs, masks).item()
                        except Exception as e:
                            print(f'[警告] 训练后处理失败: {e}, 使用原始输出计算IoU')
                            batch_iou = iou_score(outputs, masks).item()
                            batch_dice = dice_score(outputs, masks).item()
                    else:
                        # 获取CPU辅助计算的指标
                        cpu_result = hybrid_trainer.cpu_assistant.get_cpu_result()
                        if cpu_result:
                            task_type, result = cpu_result
                            if task_type == 'compute_metrics':
                                batch_iou = result['iou']
                                batch_dice = result['dice']
                            else:
                                batch_iou = iou_score(outputs, masks).item()
                                batch_dice = dice_score(outputs, masks).item()
                        else:
                            batch_iou = iou_score(outputs, masks).item()
                            batch_dice = dice_score(outputs, masks).item()
                
                # 修复：添加更详细的调试信息
                if batch_idx == 0 and epoch % 10 == 0:  # 每10个epoch打印第一个batch的信息，减少刷屏
                    print(f"\n[调试] Epoch {epoch} 训练第一个batch:")
                    print(f"  outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                    print(f"  loss: {loss:.4f}, IoU: {batch_iou:.4f}, Dice: {batch_dice:.4f}")
                    print(f"  掩码处理方式: {damage_level}, 后处理启用: {enable_postprocess}")
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"[警告] 训练 outputs存在NaN/Inf! batch_idx={batch_idx}")
                    continue  # 跳过这个batch
                
                # 调整输出维度
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                outputs = outputs.squeeze(1) if outputs.shape[1] == 1 else outputs
                masks = masks.squeeze(1) if masks.shape[1] == 1 else masks
                
                total_loss += loss
                total_iou += batch_iou
                total_dice += batch_dice
                
                # 每100个batch打印性能统计 - 已移除以减少刷屏
            
            # OneCycleLR在每个batch后step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / len(train_loader)
            avg_iou = total_iou / len(train_loader)
            avg_dice = total_dice / len(train_loader)
            
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f} IoU: {avg_iou:.4f} Dice: {avg_dice:.4f}")
            print(f"Epoch时间: {epoch_time:.2f}s, 学习率: {current_lr:.2e}")
            
            # 打印详细的性能统计
            stats = hybrid_trainer.get_performance_stats()
            if stats:
                print(f"📊 性能统计:")
                print(f"  GPU利用率: {stats['gpu_utilization']:.1f}%")
                print(f"  平均GPU时间: {stats['avg_gpu_time']:.3f}s")
                print(f"  平均CPU时间: {stats['avg_cpu_time']:.3f}s")
                print(f"  总批次数: {stats['total_batches']}")
                print(f"  TF32加速: 已启用")
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f} IoU: {avg_iou:.4f} Dice: {avg_dice:.4f}")
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_iou = 0
            val_dice = 0
            
            with torch.no_grad():
                for batch_idx, (images, masks, sim_feats) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{stage_end_epoch} - Validation")):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    sim_feats = sim_feats.to(device, non_blocking=True)
                    
                    # 掩码处理
                    if damage_level != 'all':
                        masks = process_xview2_mask(masks, damage_level)
                    
                    with autocast('cuda'):
                        outputs = model(images, sim_feats)
                    
                    # 修复：添加更详细的验证调试信息
                    if batch_idx == 0 and epoch % 10 == 0:  # 每10个epoch打印第一个batch的信息，减少刷屏
                        print(f"\n[调试] Epoch {epoch} 验证第一个batch:")
                        print(f"  掩码唯一值: {torch.unique(masks)}")
                        print(f"  outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                        print(f"  掩码处理方式: {damage_level}, 后处理启用: {enable_postprocess}")
                    
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"[警告] 验证 outputs存在NaN/Inf! batch_idx={batch_idx}")
                        continue  # 跳过这个batch
                    
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    outputs = outputs.squeeze(1) if outputs.shape[1] == 1 else outputs
                    masks = masks.squeeze(1) if masks.shape[1] == 1 else masks
                    
                    # 后处理优化IoU
                    if enable_postprocess:
                        try:
                            processed_outputs = torch.stack([
                                postprocess(out, min_area=postprocess_min_area, merge_distance=postprocess_merge_distance)
                                for out in outputs.detach().cpu()
                            ]).to(device)
                            batch_iou = iou_score(processed_outputs, masks).item()
                            batch_dice = dice_score(processed_outputs, masks).item()
                        except Exception as e:
                            print(f'[警告] 验证后处理失败: {e}, 使用原始输出计算IoU')
                            batch_iou = iou_score(outputs, masks).item()
                            batch_dice = dice_score(outputs, masks).item()
                    else:
                        # 使用CPU辅助计算验证指标
                        hybrid_trainer.cpu_assistant.submit_cpu_task('compute_metrics', (outputs, masks))
                        cpu_result = hybrid_trainer.cpu_assistant.get_cpu_result()
                        
                        if cpu_result:
                            task_type, result = cpu_result
                            if task_type == 'compute_metrics':
                                batch_iou = result['iou']
                                batch_dice = result['dice']
                            else:
                                batch_iou = iou_score(outputs, masks).item()
                                batch_dice = dice_score(outputs, masks).item()
                        else:
                            batch_iou = iou_score(outputs, masks).item()
                            batch_dice = dice_score(outputs, masks).item()
                    
                    loss = hybrid_trainer.criterion(outputs, masks)
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[调试] 验证 loss为NaN/Inf! batch_idx={batch_idx}")
                        continue
                    
                    val_loss += loss.item()
                    val_iou += batch_iou
                    val_dice += batch_dice
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f} IoU: {avg_val_iou:.4f} Dice: {avg_val_dice:.4f}")
            iou_log.append(avg_val_iou)
            
            # 保存最佳模型逻辑 - 增强版：保存完整训练状态
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                
                # 保存完整训练状态到最佳模型文件
                if use_ensemble:
                    best_model_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': ensemble_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': hybrid_trainer.scaler.state_dict(),
                        'best_val_iou': best_val_iou,
                        'iou_log': iou_log
                    }
                else:
                    best_model_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': hybrid_trainer.scaler.state_dict(),
                        'best_val_iou': best_val_iou,
                        'iou_log': iou_log
                    }
                
                torch.save(best_model_checkpoint, best_model_path)
                print(f"[保存] 新最佳模型，Val IoU: {best_val_iou:.4f} (历史最佳)")
                print(f"   已保存完整训练状态到: {best_model_path}")
            else:
                print(f"[未保存] 当前IoU {avg_val_iou:.4f} < 历史最佳 {best_val_iou:.4f}")
            
            # 更新学习率调度器
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
            
            # 保存检查点（包括模型、优化器、scaler等状态）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': hybrid_trainer.scaler.state_dict(),
                'best_val_iou': best_val_iou,
                'iou_log': iou_log
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
        
        # 保存IoU历史
        with open("iou_history.csv", "w") as f:
            f.write("epoch,iou\n")
            for i, iou in enumerate(iou_log):
                f.write(f"{i+1},{iou:.6f}\n")
    
    # === 分阶段量化逻辑 ===
    # 检查当前阶段是否完成
    if start_epoch > stage_end_epoch or epoch == stage_end_epoch:
        # 关闭混合精度训练器
        print("🔄 关闭CPU-GPU协同训练器...")
        hybrid_trainer.shutdown()
        print("✅ CPU-GPU协同训练器已关闭")
        
        print(f"\n" + "="*50)
        print(f"阶段 {current_stage} 训练完成！开始自动量化模型...")
        print("="*50)
        try:
            import subprocess
            import sys
            # 确保最佳模型存在
            if not os.path.exists(best_model_path):
                print(f"警告：最佳模型文件 {best_model_path} 不存在，跳过量化")
                return
            # 调用量化脚本
            quantize_script = os.path.join("inference", "quantize_model.py")
            if os.path.exists(quantize_script):
                print("开始量化模型...")
                cmd = [
                    sys.executable, quantize_script,
                    "--model_path", best_model_path,
                    "--quant_path", f"models/quantized_seg_model_stage{current_stage}.pt",
                    "--data_root", "data/combined_dataset"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ 模型量化成功！")
                    print("量化输出:")
                    print(result.stdout)
                    # 检查量化模型文件
                    quantized_model_path = f"models/quantized_seg_model_stage{current_stage}.pt"
                    if os.path.exists(quantized_model_path):
                        file_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
                        print(f"量化模型文件大小: {file_size:.2f} MB")
                        # 对比原始模型大小
                        original_size = os.path.getsize(best_model_path) / (1024 * 1024)  # MB
                        compression_ratio = original_size / file_size
                        print(f"原始模型大小: {original_size:.2f} MB")
                        print(f"压缩比: {compression_ratio:.2f}x")
                    else:
                        print("❌ 量化模型文件未生成")
                else:
                    print("❌ 模型量化失败！")
                    print("错误输出:")
                    print(result.stderr)
            else:
                print(f"❌ 量化脚本 {quantize_script} 不存在")
        except Exception as e:
            print(f"❌ 量化过程中发生错误: {e}")
        print(f"\n" + "="*50)
        print(f"阶段 {current_stage} 训练和量化流程完成！")
        print("="*50)
        # 提示下一阶段
        next_stage = current_stage + 1
        next_stage_start = next_stage * 20 - 19
        next_stage_end = next_stage * 20
        print(f"\n下一阶段: 阶段 {next_stage} (epoch {next_stage_start}-{next_stage_end})")
        print("重新运行脚本继续下一阶段训练...")
    else:
        print(f"\n当前阶段 {current_stage} 未完成，请继续训练...")

# ====== 训练与验证流程优化 ======
def train_multi_class_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler):
    """
    多类别分类训练函数
    """
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_damage_iou = 0
    num_batches = 0
    
    for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        sim_feats = sim_feats.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images, sim_feats)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 计算性能指标
        with torch.no_grad():
            performance = evaluate_multi_class_performance(outputs, masks)
            total_loss += loss.item()
            total_accuracy += performance['accuracy']
            total_damage_iou += performance['damage_iou']
            num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {performance["accuracy"]:.4f}, '
                  f'Damage IoU: {performance["damage_iou"]:.4f}')
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_damage_iou = total_damage_iou / num_batches
    
    return avg_loss, avg_accuracy, avg_damage_iou

def val_multi_class_epoch(model, val_loader, criterion, device, scaler):
    """
    多类别分类验证函数
    """
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_damage_iou = 0
    total_severe_iou = 0
    class_ious = [0] * 5
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks, sim_feats) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sim_feats = sim_feats.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(images, sim_feats)
                loss = criterion(outputs, masks)
            
            # 计算性能指标
            performance = evaluate_multi_class_performance(outputs, masks)
            total_loss += loss.item()
            total_accuracy += performance['accuracy']
            total_damage_iou += performance['damage_iou']
            total_severe_iou += performance['severe_damage_iou']
            
            # 累积每个类别的IoU
            for i, iou in enumerate(performance['class_ious']):
                class_ious[i] += iou
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_damage_iou = total_damage_iou / num_batches
    avg_severe_iou = total_severe_iou / num_batches
    avg_class_ious = [iou / num_batches for iou in class_ious]
    
    return avg_loss, avg_accuracy, avg_damage_iou, avg_severe_iou, avg_class_ious

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss = 0
    total_iou = 0
    for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        sim_feats = sim_feats.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, sim_feats)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        # 后处理优化IoU - 使用简化版本
        with torch.no_grad():
            # 使用简化的后处理函数，提高训练速度
            processed_outputs = torch.stack([
                simple_postprocess(out) for out in outputs.detach().cpu()
            ]).to(device)
            iou = iou_score(processed_outputs, masks)
        total_loss += loss.item()
        total_iou += iou.item()
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f} IoU: {iou.item():.4f}')
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    return avg_loss, avg_iou

def val_epoch(model, val_loader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for batch_idx, (images, masks, sim_feats) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sim_feats = sim_feats.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, sim_feats)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
            # 验证时使用简化后处理
            processed_outputs = torch.stack([
                simple_postprocess(out) for out in outputs.detach().cpu()
            ]).to(device)
            iou = iou_score(processed_outputs, masks)
            total_loss += loss.item()
            total_iou += iou.item()
    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    return avg_loss, avg_iou

# ====== 模型集成结构 ======
class ModelEnsemble(nn.Module):
    def __init__(self, model_paths, device):
        super().__init__()
        self.models = []
        for path in model_paths:
            model = EnhancedDeepLab().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            self.models.append(model)
    def forward(self, x, sim_feat):
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(x, sim_feat)
                outputs.append(torch.sigmoid(output))
        weights = torch.linspace(0.5, 1.5, len(outputs)).to(x.device)
        weights = F.softmax(weights, dim=0)
        ensemble_output = torch.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            ensemble_output += weights[i] * out
        return ensemble_output

# 多尺度特征融合模型 - 修复版本
class MultiScaleDeepLab(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, sim_feat_dim=11):
        super().__init__()
        # 使用现有的EnhancedDeepLab作为基础
        self.base_model = smp.DeepLabV3Plus(
            encoder_name="resnext101_32x8d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
        
        # 多尺度分支 - 增强版本
        self.scale_branches = nn.ModuleList([
            # 下采样分支 (1/4分辨率)
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # 原始分辨率分支
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # 上采样分支 (2x分辨率)
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 仿真特征融合模块 - 增强版本
        self.sim_fusion = nn.Sequential(
            nn.Linear(sim_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 注意力门控机制 - 重新添加
        self.attention_gate = nn.Sequential(
            nn.Conv2d(2048 + 128 + 64 + 64 + 2048, 1024, kernel_size=1),  # 基础特征 + 3个尺度特征 + sim特征
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合卷积 - 增强版本
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 + 128 + 64 + 64, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, img, sim_feat):
        # 输入检查
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"[警告] 模型输入图像包含NaN/Inf!")
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(sim_feat).any() or torch.isinf(sim_feat).any():
            print(f"[警告] 模型输入sim特征包含NaN/Inf!")
            sim_feat = torch.nan_to_num(sim_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 基础模型特征提取
        base_features = self.base_model.encoder(img)
        base_feature = base_features[-1]  # [B, 2048, H/32, W/32]
        B, C, H, W = base_feature.shape
        
        # 多尺度特征提取
        scale_features = []
        for branch in self.scale_branches:
            scale_feat = branch(img)
            # 调整到与基础特征相同的尺寸
            if scale_feat.shape[-2:] != base_feature.shape[-2:]:
                scale_feat = F.interpolate(scale_feat, size=base_feature.shape[-2:], 
                                          mode='bilinear', align_corners=True)
            scale_features.append(scale_feat)
        
        # 仿真特征融合
        sim_proj = self.sim_fusion(sim_feat)  # [B, 2048]
        sim_proj = sim_proj.view(B, -1, 1, 1)  # [B, 2048, 1, 1]
        sim_proj = sim_proj.expand(-1, -1, H, W)  # [B, 2048, H, W]
        
        # 特征拼接
        combined_features = torch.cat([base_feature] + scale_features, dim=1)  # [B, 2048+128+64+64, H, W]
        
        # 注意力融合 - 重新添加
        attention_input = torch.cat([combined_features, sim_proj], dim=1)
        attention = self.attention_gate(attention_input)
        attended_features = base_feature * attention + sim_proj * (1 - attention)
        
        # 最终特征融合
        fused_features = self.fusion_conv(combined_features)
        
        # 空间注意力
        spatial_weights = self.spatial_attention(fused_features)
        fused_features = fused_features * spatial_weights
        
        # 通道注意力
        channel_weights = self.channel_attention(fused_features)
        fused_features = fused_features * channel_weights
        
        # 检查融合后的特征
        if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
            print(f"[警告] 融合特征包含NaN/Inf，使用原始特征!")
            fused_features = base_feature
        
        # 通过解码器
        features = list(base_features)
        features[-1] = fused_features
        out = self.base_model.decoder(features)
        out = self.base_model.segmentation_head(out)
        
        # 检查最终输出
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[警告] 模型输出包含NaN/Inf，返回零张量!")
            out = torch.zeros_like(out)
        
        return out

# 添加IoU-based难样本识别
def identify_hard_samples(self, pred, target):
    pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > 0.5).float()
    
    # 计算每个像素的IoU贡献
    intersection = pred_binary * target
    union = pred_binary + target - intersection
    
    # 低IoU区域作为难样本
    pixel_iou = intersection / (union + 1e-8)
    hard_mask = pixel_iou < 0.5
    
    return hard_mask

# 集成训练策略 (Ensemble Training Strategy) - 改进版本

# 集成训练策略 (Ensemble Training Strategy) - 改进版本
class EnsembleTrainer:
    def __init__(self, models, train_loader, val_loader, device, 
                 use_enhanced_loss=True, use_mixed_precision=True):
        self.models = models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_enhanced_loss = use_enhanced_loss
        self.use_mixed_precision = use_mixed_precision
        
        # 将模型移动到设备
        for model in self.models:
            model.to(device)
        
        # 优化器
        self.optimizers = []
        for model in self.models:
            # 分层学习率
            encoder_params = []
            decoder_params = []
            fusion_params = []
            
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    encoder_params.append(param)
                elif 'decoder' in name or 'segmentation_head' in name:
                    decoder_params.append(param)
                else:
                    fusion_params.append(param)
            
            optimizer = optim.AdamW([
                {'params': encoder_params, 'lr': 1e-5},
                {'params': decoder_params, 'lr': 1e-4},
                {'params': fusion_params, 'lr': 5e-4}
            ], weight_decay=1e-4)
            self.optimizers.append(optimizer)
        
        # 损失函数
        if use_enhanced_loss:
            base_criterion = HybridLoss()
            self.criterion = BoundaryAwareLoss(base_criterion, alpha=0.3, beta=0.2)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 学习率调度器
        self.schedulers = []
        for optimizer in self.optimizers:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            self.schedulers.append(scheduler)
        
        # 混合精度训练
        if use_mixed_precision:
            self.scalers = [GradScaler() for _ in range(len(self.models))]
        else:
            self.scalers = [None] * len(self.models)
        
        # 早停机制
        self.best_ensemble_iou = 0.0
        self.patience = 15
        self.patience_counter = 0
        
        # 模型保存路径
        self.save_dir = "models/ensemble"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        total_losses = [0.0] * len(self.models)
        total_ious = [0.0] * len(self.models)
        
        # 设置模型为训练模式
        for model in self.models:
            model.train()
        
        # 循环训练每个模型
        for batch_idx, (images, masks, sim_feats) in enumerate(tqdm(self.train_loader, 
                                                                   desc=f"Ensemble Epoch {epoch}")):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            sim_feats = sim_feats.to(self.device, non_blocking=True)
            
            # 训练每个模型
            for i, (model, optimizer, scaler) in enumerate(zip(self.models, self.optimizers, self.scalers)):
                optimizer.zero_grad()
                
                # 前向传播
                if self.use_mixed_precision and scaler is not None:
                    with autocast('cuda'):
                        outputs = model(images, sim_feats)
                        
                        # 调整尺寸
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                                  mode='bilinear', align_corners=False)
                        
                        # 计算损失
                        loss = self.criterion(outputs, masks)
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images, sim_feats)
                    
                    # 调整尺寸
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                              mode='bilinear', align_corners=False)
                    
                    # 计算损失
                    loss = self.criterion(outputs, masks)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                
                # 计算IoU
                with torch.no_grad():
                    pred_prob = torch.sigmoid(outputs)
                    iou = iou_score(pred_prob, masks).item()
                
                total_losses[i] += loss.item()
                total_ious[i] += iou
        
        # 计算平均损失和IoU
        avg_losses = [loss / len(self.train_loader) for loss in total_losses]
        avg_ious = [iou / len(self.train_loader) for iou in total_ious]
        
        # 更新学习率调度器
        for scheduler in self.schedulers:
            scheduler.step()
        
        return avg_losses, avg_ious
    
    def validate_ensemble(self):
        """验证集成模型"""
        ensemble_iou = 0.0
        ensemble_dice = 0.0
        
        # 设置模型为评估模式
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            for images, masks, sim_feats in tqdm(self.val_loader, desc="Ensemble Validation"):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                sim_feats = sim_feats.to(self.device, non_blocking=True)
                
                # 模型预测
                predictions = []
                for model in self.models:
                    outputs = model(images, sim_feats)
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                              mode='bilinear', align_corners=False)
                    predictions.append(torch.sigmoid(outputs))
                
                # 集成预测 - 加权平均
                weights = torch.linspace(0.8, 1.2, len(predictions)).to(self.device)
                weights = F.softmax(weights, dim=0)
                
                ensemble_pred = torch.zeros_like(predictions[0])
                for i, pred in enumerate(predictions):
                    ensemble_pred += weights[i] * pred
                
                # 计算指标
                iou = iou_score(ensemble_pred, masks).item()
                dice = dice_score(ensemble_pred, masks).item()
                
                ensemble_iou += iou
                ensemble_dice += dice
        
        ensemble_iou /= len(self.val_loader)
        ensemble_dice /= len(self.val_loader)
        
        return ensemble_iou, ensemble_dice
    
    def save_models(self, epoch, ensemble_iou):
        """保存模型"""
        for i, model in enumerate(self.models):
            model_path = os.path.join(self.save_dir, f"ensemble_model_{i}_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[i].state_dict(),
                'scheduler_state_dict': self.schedulers[i].state_dict(),
                'ensemble_iou': ensemble_iou
            }, model_path)
        
        print(f"✅ 集成模型已保存，Ensemble IoU: {ensemble_iou:.4f}")
    
    def train(self, epochs=50):
        """完整训练流程"""
        print("🚀 开始集成训练...")
        print(f"模型数量: {len(self.models)}")
        print(f"使用增强损失: {self.use_enhanced_loss}")
        print(f"使用混合精度: {self.use_mixed_precision}")
        
        for epoch in range(1, epochs + 1):
            # 训练阶段
            avg_losses, avg_ious = self.train_epoch(epoch)
            
            # 验证阶段
            ensemble_iou, ensemble_dice = self.validate_ensemble()
            
            # 打印训练信息
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"训练损失: {[f'{loss:.4f}' for loss in avg_losses]}")
            print(f"训练IoU: {[f'{iou:.4f}' for iou in avg_ious]}")
            print(f"集成验证 IoU: {ensemble_iou:.4f}, Dice: {ensemble_dice:.4f}")
            
            # 早停检查
            if ensemble_iou > self.best_ensemble_iou:
                self.best_ensemble_iou = ensemble_iou
                self.patience_counter = 0
                # 保存最佳模型
                self.save_models(epoch, ensemble_iou)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"早停触发！{self.patience} 个epoch没有改善")
                    break
        
        print(f"🎉 集成训练完成！最佳Ensemble IoU: {self.best_ensemble_iou:.4f}")
        return self.models

def ensemble_training(models, train_loader, val_loader, device, epochs=50):
    """简化的集成训练接口"""
    trainer = EnsembleTrainer(models, train_loader, val_loader, device)
    return trainer.train(epochs)

def validate_ensemble(models, val_loader, device):
    """简化的集成验证接口"""
    ensemble_iou = 0
    for images, masks, sim_feats in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        sim_feats = sim_feats.to(device)
        
        # 模型预测
        predictions = []
        for model in models:
            with torch.no_grad():
                outputs = model(images, sim_feats)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                predictions.append(torch.sigmoid(outputs))
        
        # 集成预测
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        # 计算IoU
        iou = iou_score(ensemble_pred, masks)
        ensemble_iou += iou.item()
    
    return ensemble_iou / len(val_loader)

# 为了向后兼容性，添加DeepLabWithSimFeature作为EnhancedDeepLab的别名
DeepLabWithSimFeature = EnhancedDeepLab

class DualModelEnsemble(nn.Module):
    """双模型集成：DeepLab + 分割版LandslideDetector"""
    def __init__(self, deeplab_model, landslide_model, fusion_weight=0.5):
        super().__init__()
        self.deeplab_model = deeplab_model
        self.landslide_model = landslide_model
        self.fusion_weight = fusion_weight
        
        # 可学习的融合权重
        self.learnable_weight = nn.Parameter(torch.tensor([fusion_weight]))
        
    def forward(self, img, sim_feat=None):
        # DeepLab前向传播
        deeplab_output = self.deeplab_model(img, sim_feat)
        
        # LandslideDetector前向传播（不需要sim_feat）
        landslide_output = self.landslide_model(img)
        
        # 检查输出是否包含NaN
        if torch.isnan(deeplab_output).any() or torch.isinf(deeplab_output).any():
            print("[警告] DeepLab输出包含NaN/Inf，使用零张量")
            deeplab_output = torch.zeros_like(deeplab_output)
        
        if torch.isnan(landslide_output).any() or torch.isinf(landslide_output).any():
            print("[警告] LandslideDetector输出包含NaN/Inf，使用零张量")
            landslide_output = torch.zeros_like(landslide_output)
        
        # 确保输出尺寸一致
        if deeplab_output.shape != landslide_output.shape:
            landslide_output = F.interpolate(landslide_output, size=deeplab_output.shape[2:], mode='bilinear', align_corners=False)
        
        # 加权融合
        weight = torch.sigmoid(self.learnable_weight)  # 确保权重在[0,1]范围内
        
        # 检查权重是否包含NaN
        if torch.isnan(weight) or torch.isinf(weight):
            print("[警告] 融合权重包含NaN/Inf，使用默认权重0.5")
            weight = torch.tensor(0.5, device=img.device)
        
        ensemble_output = weight * deeplab_output + (1 - weight) * landslide_output
        
        # 最终检查
        if torch.isnan(ensemble_output).any() or torch.isinf(ensemble_output).any():
            print("[警告] 融合输出包含NaN/Inf，使用零张量")
            ensemble_output = torch.zeros_like(ensemble_output)
        
        return ensemble_output
    
    def get_fusion_weight(self):
        """获取当前融合权重"""
        return torch.sigmoid(self.learnable_weight).item()

def train_dual_model_epoch(ensemble_model, train_loader, optimizer, criterion, device, epoch, scaler):
    """训练双模型集成的一个epoch"""
    ensemble_model.train()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, masks, sim_feats) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        sim_feats = sim_feats.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            # 前向传播
            outputs = ensemble_model(images, sim_feats)
            
                    # 计算损失
        loss = criterion(outputs, masks)
        
        # 检查损失是否包含NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[警告] 损失包含NaN/Inf，跳过batch {batch_idx}")
            continue
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 检查梯度是否包含NaN/Inf（在unscale之前）
        grad_norm = 0
        has_nan_grad = False
        has_inf_grad = False
        
        # 使用scaler检查梯度
        scaler.unscale_(optimizer)
        
        for param in ensemble_model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                if torch.isinf(param.grad).any():
                    has_inf_grad = True
                grad_norm += param.grad.data.norm(2).item() ** 2
        
        grad_norm = grad_norm ** (1. / 2)
        
        if has_nan_grad or has_inf_grad:
            print(f"[警告] 梯度包含NaN/Inf，进行梯度修复")
            # 将NaN/Inf梯度替换为0
            for param in ensemble_model.parameters():
                if param.grad is not None:
                    param.grad.data = torch.where(
                        torch.isnan(param.grad.data) | torch.isinf(param.grad.data),
                        torch.zeros_like(param.grad.data),
                        param.grad.data
                    )
        
        # 更严格的梯度裁剪和自适应学习率调整
        if grad_norm > 10:
            print(f"[警告] 梯度范数过大 ({grad_norm:.2f})，进行梯度裁剪并降低学习率")
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=5.0)
            # 临时降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        elif grad_norm > 5:
            print(f"[警告] 梯度范数较大 ({grad_norm:.2f})，进行梯度裁剪")
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 检查权重是否变为NaN/Inf
        has_nan_weight = False
        has_inf_weight = False
        for param in ensemble_model.parameters():
            if torch.isnan(param).any():
                has_nan_weight = True
            if torch.isinf(param).any():
                has_inf_weight = True
        
        if has_nan_weight or has_inf_weight:
            print(f"[警告] 权重包含NaN/Inf，进行权重修复")
            # 将NaN/Inf权重替换为小的随机值
            for param in ensemble_model.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    param.data = torch.where(
                        torch.isnan(param.data) | torch.isinf(param.data),
                        torch.randn_like(param.data) * 0.01,
                        param.data
                    )
        
        # 计算IoU
        with torch.no_grad():
            pred_masks = torch.sigmoid(outputs) > 0.5
            iou = iou_score(pred_masks, masks)
        
        total_loss += loss.item()
        total_iou += iou.item()
        num_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{iou.item():.4f}',
            'Fusion Weight': f'{ensemble_model.get_fusion_weight():.3f}'
        })
    
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou

def val_dual_model_epoch(ensemble_model, val_loader, criterion, device, scaler):
    """验证双模型集成"""
    ensemble_model.eval()
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks, sim_feats in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            sim_feats = sim_feats.to(device)
            
            with autocast(device_type='cuda'):
                outputs = ensemble_model(images, sim_feats)
                loss = criterion(outputs, masks)
            
            # 计算IoU
            pred_masks = torch.sigmoid(outputs) > 0.5
            iou = iou_score(pred_masks, masks)
            
            total_loss += loss.item()
            total_iou += iou.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_iou

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
