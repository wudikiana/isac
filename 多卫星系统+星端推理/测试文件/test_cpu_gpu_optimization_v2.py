#!/usr/bin/env python3
"""
增强版CPU-GPU协同优化测试
"""

import torch
import time
import numpy as np
import threading
import queue

def test_cpu_intensive_task():
    """测试CPU密集型任务"""
    print("🧪 测试CPU密集型任务...")
    
    def cpu_worker(task_queue, result_queue, worker_id):
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:
                    break
                
                # 执行CPU密集型计算
                start_time = time.time()
                
                # 模拟复杂的CPU计算
                data = task['data']
                result = 0
                for i in range(1000):  # 模拟1000次计算
                    result += np.sin(i) * np.cos(i) * data
                
                compute_time = time.time() - start_time
                result_queue.put({
                    'worker_id': worker_id,
                    'result': result,
                    'compute_time': compute_time
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"CPU工作线程 {worker_id} 错误: {e}")
    
    # 创建任务队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # 启动CPU工作线程
    num_workers = 4
    workers = []
    for i in range(num_workers):
        worker = threading.Thread(target=cpu_worker, args=(task_queue, result_queue, i))
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    # 提交任务
    num_tasks = 20
    for i in range(num_tasks):
        task_queue.put({'data': i * 0.1})
    
    # 等待结果
    results = []
    start_time = time.time()
    
    for _ in range(num_tasks):
        result = result_queue.get()
        results.append(result)
    
    total_time = time.time() - start_time
    
    # 关闭工作线程
    for _ in range(num_workers):
        task_queue.put(None)
    
    for worker in workers:
        worker.join()
    
    print(f"✅ CPU密集型任务完成: {total_time:.3f}秒")
    print(f"  任务数: {num_tasks}, 工作线程: {num_workers}")
    print(f"  平均任务时间: {sum(r['compute_time'] for r in results) / len(results):.3f}秒")
    
    return total_time

def test_gpu_training_with_cpu_assist():
    """测试GPU训练与CPU辅助"""
    print("\n🧪 测试GPU训练与CPU辅助...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试模型
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 1, 1)
        
        def forward(self, img, sim_feat):
            x = self.conv1(img)
            x = self.relu(x)
            x = self.conv2(x)
            return x
    
    model = TestModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建数据
    batch_size = 8
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 64, 64)).float().to(device)
    sim_feats = torch.randn(batch_size, 11).to(device)
    
    # 测试标准训练
    print("  测试标准训练...")
    scaler = torch.amp.GradScaler()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(10):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images, sim_feats)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    std_time = time.time() - start_time
    
    # 测试CPU辅助训练
    print("  测试CPU辅助训练...")
    
    def cpu_metrics_worker(outputs, masks):
        """CPU指标计算工作线程"""
        with torch.no_grad():
            outputs_cpu = outputs.cpu().detach()
            masks_cpu = masks.cpu().detach()
            
            # 计算各种指标
            preds = (torch.sigmoid(outputs_cpu) > 0.5).float()
            
            # IoU计算
            intersection = (preds * masks_cpu).sum()
            union = (preds + masks_cpu).sum() - intersection
            iou = (intersection + 1e-5) / (union + 1e-5)
            
            # Dice计算
            dice = (2. * intersection + 1e-5) / (preds.sum() + masks_cpu.sum() + 1e-5)
            
            # 准确率计算
            accuracy = (preds == masks_cpu).float().mean()
            
            # 模拟额外计算
            time.sleep(0.001)
            
            return {'iou': iou.item(), 'dice': dice.item(), 'accuracy': accuracy.item()}
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(10):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images, sim_feats)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        
        # 启动CPU指标计算线程
        cpu_thread = threading.Thread(target=cpu_metrics_worker, args=(outputs, masks))
        cpu_thread.start()
        
        # GPU反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 等待CPU计算完成
        cpu_thread.join()
    
    torch.cuda.synchronize()
    hybrid_time = time.time() - start_time
    
    print(f"✅ GPU训练测试完成:")
    print(f"  标准训练时间: {std_time:.3f}秒")
    print(f"  CPU辅助训练时间: {hybrid_time:.3f}秒")
    print(f"  性能提升: {std_time/hybrid_time:.2f}x")
    
    return std_time, hybrid_time

def test_memory_bandwidth():
    """测试内存带宽优化"""
    print("\n🧪 测试内存带宽优化...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试数据大小
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        print(f"  测试数据大小: {size}x{size}")
        
        # 创建测试数据
        data = torch.randn(size, size).to(device)
        
        # 测试GPU计算
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            result = torch.mm(data, data)
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        # 测试CPU计算
        data_cpu = data.cpu()
        start_time = time.time()
        
        for _ in range(10):
            result = torch.mm(data_cpu, data_cpu)
        
        cpu_time = time.time() - start_time
        
        print(f"    GPU时间: {gpu_time:.3f}秒, CPU时间: {cpu_time:.3f}秒")
        print(f"    加速比: {cpu_time/gpu_time:.2f}x")

def main():
    print("="*60)
    print("🚀 增强版CPU-GPU协同优化测试")
    print("="*60)
    
    # 测试CPU密集型任务
    cpu_time = test_cpu_intensive_task()
    
    # 测试GPU训练与CPU辅助
    gpu_std_time, gpu_hybrid_time = test_gpu_training_with_cpu_assist()
    
    # 测试内存带宽
    test_memory_bandwidth()
    
    # 总结
    print(f"\n📊 测试总结:")
    print(f"  CPU密集型任务: {cpu_time:.3f}秒")
    print(f"  GPU标准训练: {gpu_std_time:.3f}秒")
    print(f"  GPU+CPU辅助: {gpu_hybrid_time:.3f}秒")
    print(f"  训练性能提升: {gpu_std_time/gpu_hybrid_time:.2f}x")
    
    print("\n🎉 增强版CPU-GPU协同优化测试完成！")

if __name__ == "__main__":
    main() 