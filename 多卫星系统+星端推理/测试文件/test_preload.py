#!/usr/bin/env python3
"""
测试数据预加载功能
"""

import torch
import time
import psutil
from train_model import get_multimodal_patch_dataloaders, MemoryCachedDataset, GPUPreloadedDataset

def test_data_loading_speed():
    """测试不同数据加载方式的速度"""
    print("="*60)
    print("🚀 数据加载速度测试")
    print("="*60)
    
    # 系统信息
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    print(f"系统内存: {total_memory_gb:.1f} GB")
    print(f"GPU显存: {gpu_memory_gb:.1f} GB")
    
    # 测试标准加载
    print("\n📁 测试标准数据加载...")
    start_time = time.time()
    train_loader, val_loader = get_multimodal_patch_dataloaders(
        data_root="D:/patch_dataset",
        sim_feature_csv="data/sim_features.csv",
        batch_size=32,
        num_workers=4,
        damage_boost=2,
        normal_ratio=0.05,
        preload_to_memory=False,
        preload_to_gpu=False
    )
    standard_time = time.time() - start_time
    print(f"标准加载耗时: {standard_time:.2f}秒")
    
    # 测试内存预加载
    if total_memory_gb >= 16:
        print("\n💾 测试内存预加载...")
        start_time = time.time()
        train_loader_mem, val_loader_mem = get_multimodal_patch_dataloaders(
            data_root="D:/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=32,
            num_workers=2,
            damage_boost=2,
            normal_ratio=0.05,
            preload_to_memory=True,
            preload_to_gpu=False
        )
        memory_time = time.time() - start_time
        print(f"内存预加载耗时: {memory_time:.2f}秒")
        print(f"速度提升: {standard_time/memory_time:.2f}x")
    
    # 测试GPU预加载
    if gpu_memory_gb >= 8:
        print("\n🚀 测试GPU预加载...")
        start_time = time.time()
        train_loader_gpu, val_loader_gpu = get_multimodal_patch_dataloaders(
            data_root="D:/patch_dataset",
            sim_feature_csv="data/sim_features.csv",
            batch_size=32,
            num_workers=0,
            damage_boost=2,
            normal_ratio=0.05,
            preload_to_memory=False,
            preload_to_gpu=True
        )
        gpu_time = time.time() - start_time
        print(f"GPU预加载耗时: {gpu_time:.2f}秒")
        print(f"速度提升: {standard_time/gpu_time:.2f}x")
    
    # 测试训练速度
    print("\n" + "="*60)
    print("🏃 训练速度测试")
    print("="*60)
    
    # 使用标准加载器测试训练速度
    print("测试标准加载器的训练速度...")
    start_time = time.time()
    batch_count = 0
    for batch_idx, (images, masks, sim_feats) in enumerate(train_loader):
        if batch_idx >= 10:  # 只测试前10个batch
            break
        batch_count += 1
        # 模拟训练步骤
        time.sleep(0.01)  # 模拟计算时间
    
    standard_train_time = time.time() - start_time
    print(f"标准加载训练10个batch耗时: {standard_train_time:.2f}秒")
    print(f"平均每batch: {standard_train_time/batch_count:.3f}秒")
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    test_data_loading_speed() 