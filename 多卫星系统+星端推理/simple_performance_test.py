#!/usr/bin/env python3
"""
简单的CPU-GPU协同优化性能测试
"""

import torch
import time
import numpy as np

def test_standard_training():
    """测试标准训练性能"""
    print("🧪 测试标准训练性能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 1, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    batch_size = 8
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 64, 64)).float().to(device)
    
    # 预热
    for _ in range(3):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # 性能测试
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(20):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    std_time = time.time() - start_time
    
    print(f"✅ 标准训练完成: {std_time:.3f}秒 (20步)")
    return std_time

def test_cpu_assisted_training():
    """测试CPU辅助训练性能"""
    print("🧪 测试CPU辅助训练性能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建一个兼容的测试模型
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 1, 1)
        
        def forward(self, img, sim_feat):
            # 忽略sim_feat，只处理图像
            x = self.conv1(img)
            x = self.relu(x)
            x = self.conv2(x)
            return x
    
    model = TestModel().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 导入混合精度训练器
    try:
        from train_model import HybridPrecisionTrainer
        hybrid_trainer = HybridPrecisionTrainer(model, optimizer, device)
        
        batch_size = 8
        images = torch.randn(batch_size, 3, 64, 64).to(device)
        masks = torch.randint(0, 2, (batch_size, 1, 64, 64)).float().to(device)
        sim_feats = torch.randn(batch_size, 11).to(device)
        
        # 预热
        for _ in range(3):
            loss, outputs = hybrid_trainer.train_step(images, masks, sim_feats)
        
        # 性能测试
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(20):
            loss, outputs = hybrid_trainer.train_step(images, masks, sim_feats)
        
        torch.cuda.synchronize()
        hybrid_time = time.time() - start_time
        
        # 获取性能统计
        stats = hybrid_trainer.get_performance_stats()
        
        print(f"✅ CPU辅助训练完成: {hybrid_time:.3f}秒 (20步)")
        
        if stats:
            print(f"  GPU利用率: {stats['gpu_utilization']:.1f}%")
            print(f"  CPU分担比例: {100-stats['gpu_utilization']:.1f}%")
        
        hybrid_trainer.shutdown()
        return hybrid_time, stats
        
    except Exception as e:
        print(f"❌ CPU辅助训练测试失败: {e}")
        return None, None

def main():
    print("="*60)
    print("🚀 CPU-GPU协同优化性能测试")
    print("="*60)
    
    # 测试标准训练
    std_time = test_standard_training()
    
    # 测试CPU辅助训练
    result = test_cpu_assisted_training()
    if result[0] is not None:
        hybrid_time, stats = result
        
        # 性能对比
        print(f"\n📊 性能对比结果:")
        print(f"  标准训练时间: {std_time:.3f}秒")
        print(f"  CPU辅助训练时间: {hybrid_time:.3f}秒")
        print(f"  性能提升: {std_time/hybrid_time:.2f}x")
        
        if stats:
            print(f"  GPU利用率: {stats['gpu_utilization']:.1f}%")
            print(f"  CPU分担比例: {100-stats['gpu_utilization']:.1f}%")
        
        print("\n🎉 CPU-GPU协同优化测试完成！")
    else:
        print("\n❌ CPU辅助训练测试失败，请检查代码")

if __name__ == "__main__":
    main()
