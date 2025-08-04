#!/usr/bin/env python3
"""
简化的CPU-GPU协同测试
"""

import torch
import time
import threading

def test_cpu_gpu_sync():
    """测试CPU-GPU同步工作"""
    print("🧪 测试CPU-GPU同步工作...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 64, 64)).float().to(device)
    
    # 测试标准GPU训练
    print("  测试标准GPU训练...")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 1, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    gpu_only_time = time.time() - start_time
    
    # 测试CPU-GPU协同
    print("  测试CPU-GPU协同...")
    
    def cpu_worker(outputs, masks):
        """CPU工作线程"""
        with torch.no_grad():
            outputs_cpu = outputs.cpu().detach()
            masks_cpu = masks.cpu().detach()
            
            # 计算指标
            preds = (torch.sigmoid(outputs_cpu) > 0.5).float()
            accuracy = (preds == masks_cpu).float().mean()
            
            # 模拟额外计算
            time.sleep(0.001)
            
            return accuracy.item()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        
        # 启动CPU线程
        cpu_thread = threading.Thread(target=cpu_worker, args=(outputs, masks))
        cpu_thread.start()
        
        # GPU反向传播
        loss.backward()
        optimizer.step()
        
        # 等待CPU完成
        cpu_thread.join()
    
    torch.cuda.synchronize()
    hybrid_time = time.time() - start_time
    
    print(f"✅ 测试完成:")
    print(f"  纯GPU时间: {gpu_only_time:.3f}秒")
    print(f"  CPU-GPU协同时间: {hybrid_time:.3f}秒")
    print(f"  性能提升: {gpu_only_time/hybrid_time:.2f}x")
    
    return gpu_only_time, hybrid_time

def test_cpu_intensive():
    """测试CPU密集型任务"""
    print("\n🧪 测试CPU密集型任务...")
    
    def cpu_compute(data):
        """CPU计算任务"""
        result = 0
        for i in range(10000):
            result += i * data
        return result
    
    # 测试串行执行
    print("  测试串行执行...")
    start_time = time.time()
    for i in range(10):
        result = cpu_compute(i)
    serial_time = time.time() - start_time
    
    # 测试并行执行
    print("  测试并行执行...")
    start_time = time.time()
    threads = []
    for i in range(10):
        thread = threading.Thread(target=cpu_compute, args=(i,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    parallel_time = time.time() - start_time
    
    print(f"✅ CPU任务测试完成:")
    print(f"  串行时间: {serial_time:.3f}秒")
    print(f"  并行时间: {parallel_time:.3f}秒")
    print(f"  加速比: {serial_time/parallel_time:.2f}x")
    
    return serial_time, parallel_time

def main():
    print("="*50)
    print("🚀 简化CPU-GPU协同测试")
    print("="*50)
    
    # 测试CPU-GPU协同
    gpu_time, hybrid_time = test_cpu_gpu_sync()
    
    # 测试CPU密集型任务
    serial_time, parallel_time = test_cpu_intensive()
    
    # 总结
    print(f"\n📊 测试总结:")
    print(f"  GPU训练加速: {gpu_time/hybrid_time:.2f}x")
    print(f"  CPU并行加速: {serial_time/parallel_time:.2f}x")
    
    print("\n🎉 简化测试完成！")

if __name__ == "__main__":
    main() 