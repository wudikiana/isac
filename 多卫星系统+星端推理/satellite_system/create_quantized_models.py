#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化模型生成脚本
用于生成多卫星系统所需的量化模型
"""

import torch
import os
import argparse
import logging
from train_model import EnhancedDeepLab

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_quantized_model(model_path: str, output_path: str, model_name: str = "model"):
    """创建量化模型"""
    try:
        logger.info(f"开始量化模型: {model_name}")
        
        # 创建模型架构
        model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        
        # 加载模型权重
        logger.info(f"加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 处理不同的模型状态字典格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理状态字典键名
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除各种前缀
            new_key = key
            prefixes_to_remove = [
                'deeplab_model.',
                'landslide_model.',
                'model.',
                'module.'
            ]
            
            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            
            new_state_dict[new_key] = value
        
        # 加载模型权重
        try:
            model.load_state_dict(new_state_dict, strict=True)
            logger.info("模型严格加载成功")
        except Exception as e:
            logger.warning(f"严格加载失败，尝试非严格加载: {e}")
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("模型非严格加载成功")
        
        model.eval()
        
        # 动态量化
        logger.info("开始动态量化...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8
        )
        logger.info("动态量化完成")
        
        # 准备示例输入进行跟踪
        dummy_image = torch.randn(1, 3, 256, 256)
        dummy_sim_feat = torch.randn(1, 11)
        
        # 跟踪量化模型
        logger.info("开始模型跟踪...")
        traced_model = torch.jit.trace(quantized_model, (dummy_image, dummy_sim_feat))
        
        # 保存量化模型
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.jit.save(traced_model, output_path)
        logger.info(f"量化模型已保存到: {output_path}")
        
        # 验证量化模型
        logger.info("验证量化模型...")
        loaded_model = torch.jit.load(output_path, map_location='cpu')
        with torch.no_grad():
            test_output = loaded_model(dummy_image, dummy_sim_feat)
            logger.info(f"量化模型验证通过，输出形状: {test_output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"量化模型失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='创建量化模型')
    parser.add_argument('--local_model', type=str, 
                       default='models/best_multimodal_patch_model.pth',
                       help='本地模型路径')
    parser.add_argument('--local_quantized', type=str,
                       default='models/quantized_seg_model.pt',
                       help='本地量化模型输出路径')
    parser.add_argument('--federated_quantized', type=str,
                       default='models/quantized_federated_model.pt',
                       help='联邦学习量化模型输出路径')
    parser.add_argument('--federated_model', type=str,
                       default='models/best_multimodal_patch_model.pth',
                       help='联邦学习模型路径（初始使用本地模型）')
    
    args = parser.parse_args()
    
    logger.info("开始创建量化模型...")
    
    # 创建本地量化模型
    if os.path.exists(args.local_model):
        success = create_quantized_model(
            args.local_model, 
            args.local_quantized, 
            "本地模型"
        )
        if success:
            logger.info("✅ 本地量化模型创建成功")
        else:
            logger.error("❌ 本地量化模型创建失败")
    else:
        logger.warning(f"本地模型文件不存在: {args.local_model}")
    
    # 创建联邦学习量化模型
    if os.path.exists(args.federated_model):
        success = create_quantized_model(
            args.federated_model,
            args.federated_quantized,
            "联邦学习模型"
        )
        if success:
            logger.info("✅ 联邦学习量化模型创建成功")
        else:
            logger.error("❌ 联邦学习量化模型创建失败")
    else:
        logger.warning(f"联邦学习模型文件不存在: {args.federated_model}")
    
    logger.info("量化模型创建完成")

if __name__ == "__main__":
    main() 