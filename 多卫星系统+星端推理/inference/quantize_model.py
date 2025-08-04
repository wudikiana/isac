import torch
import os
import argparse
import time
import psutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_model import EnhancedDeepLab  # 导入自定义模型

def quantize_model():
    parser = argparse.ArgumentParser(description='Quantize segmentation model')
    parser.add_argument('--model_path', type=str, default='models/best_multimodal_patch_model.pth',
                      help='Path to the trained model')
    parser.add_argument('--quant_path', type=str, default='models/quantized_seg_model.pt',
                      help='Path to save the quantized model')
    parser.add_argument('--data_root', type=str, default='data/combined_dataset',
                      help='Root directory of the dataset')
    args = parser.parse_args()
    
    device = "cpu"
    torch.backends.quantized.engine = 'fbgemm'
    print(f"Quantizing model on {device} (engine: {torch.backends.quantized.engine})")
    os.makedirs(os.path.dirname(args.quant_path), exist_ok=True)
    
    try:
        # 使用与训练完全相同的自定义模型
        model = EnhancedDeepLab(
            in_channels=3,
            num_classes=1,
            sim_feat_dim=11
        )
        
        # 智能加载模型 - 支持检查点格式和纯权重格式
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 检查点格式
            print("检测到检查点格式，加载模型状态字典...")
            state_dict = checkpoint['model_state_dict']
            if 'best_val_iou' in checkpoint:
                print(f"模型最佳IoU: {checkpoint['best_val_iou']:.4f}")
        else:
            # 纯权重格式
            print("检测到纯权重格式，直接加载...")
            state_dict = checkpoint
        
        # 处理模型结构兼容性问题
        print("检查模型结构兼容性...")
        model_state_dict = model.state_dict()
        
        # 处理键名映射问题
        updated_state_dict = {}
        for key, value in state_dict.items():
            # 处理各种可能的键名变化
            new_key = key
            
            # 处理从sim_fc到sim_fusion的键名变化
            if key.startswith('sim_fc.') and key not in model_state_dict:
                new_key = key.replace('sim_fc.', 'sim_fusion.')
            
            # 处理其他可能的键名变化
            if new_key not in model_state_dict:
                # 尝试移除各种前缀
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
            
            if new_key in model_state_dict:
                updated_state_dict[new_key] = value
                if new_key != key:
                    print(f"键名映射: {key} -> {new_key}")
            else:
                print(f"警告: 无法映射键 {key}")
        
        # 加载兼容后的状态字典
        try:
            model.load_state_dict(updated_state_dict, strict=False)
            print("模型加载成功（使用非严格模式）")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return
            
        model.eval()
        print("模型加载成功")
        
        # 验证模型结构
        dummy_image = torch.randn(1, 3, 256, 256)
        dummy_sim_feat = torch.randn(1, 11)
        with torch.no_grad():
            test_output = model(dummy_image, dummy_sim_feat)
            print(f"模型前向传播测试通过，输出形状: {test_output.shape}")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 动态量化（注意量化层范围）
    print("开始动态量化...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},  # 量化卷积和线性层
            dtype=torch.qint8
        )
        print("动态量化完成")
    except Exception as e:
        print(f"动态量化失败: {e}")
        return
    
    try:
        # 准备示例输入（图像+模拟特征）
        dummy_image = torch.randn(1, 3, 256, 256)
        dummy_sim_feat = torch.randn(1, 11)
        
        # 跟踪量化模型
        print("开始模型跟踪...")
        traced_model = torch.jit.trace(quantized_model, (dummy_image, dummy_sim_feat))
        torch.jit.save(traced_model, args.quant_path)
        print(f"量化模型已保存到 {args.quant_path}")
        
        # 验证量化模型可加载
        print("验证量化模型...")
        loaded_quant = torch.jit.load(args.quant_path)
        test_output = loaded_quant(dummy_image, dummy_sim_feat)  # 测试推理
        print(f"量化模型验证通过，输出形状: {test_output.shape}")
        
    except Exception as e:
        print(f"量化模型保存/验证失败: {e}")
        return
    
    # 性能测试
    print("\n" + "="*40)
    print("开始量化模型性能测试...")
    print("="*40)
    test_quantized_model(args.quant_path, device)

def test_quantized_model(model_path, device, repetitions=100):
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print("量化模型加载成功")
    except Exception as e:
        print(f"量化模型加载失败: {e}")
        return
    
    # 准备测试输入
    input_shape = (1, 3, 256, 256)
    dummy_image = torch.randn(input_shape).to(device)
    dummy_sim_feat = torch.randn(1, 11).to(device)
    
    # Warmup
    print("预热模型...")
    for _ in range(10):
        _ = model(dummy_image, dummy_sim_feat)
    
    # 延迟测试
    print(f"开始延迟测试 ({repetitions} 次推理)...")
    latencies = []
    try:
        with torch.no_grad():
            for i in range(repetitions):
                start_time = time.perf_counter()
                _ = model(dummy_image, dummy_sim_feat)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # 毫秒
                if (i+1) % 20 == 0:
                    print(f"  已处理 {i+1}/{repetitions} 个样本")
    except Exception as e:
        print(f"延迟测试失败: {e}")
        return
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_latency = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5
        
        print(f"\n延迟测试结果 (CPU):")
        print(f"  平均延迟: {avg_latency:.2f} ms")
        print(f"  最小延迟: {min_latency:.2f} ms")
        print(f"  最大延迟: {max_latency:.2f} ms")
        print(f"  标准差: {std_latency:.2f} ms")
        print(f"  吞吐量: {1000/avg_latency:.1f} FPS")
    
    # 内存测试
    print("\n内存使用测试...")
    try:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 多次推理以稳定内存使用
        for _ in range(10):
            _ = model(dummy_image, dummy_sim_feat)
            
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_usage = mem_after - mem_before
        print(f"  推理内存使用: {memory_usage:.2f} MB")
        
        # 保存性能报告
        performance_report = {
            'model_path': model_path,
            'avg_latency_ms': avg_latency if latencies else 0,
            'min_latency_ms': min_latency if latencies else 0,
            'max_latency_ms': max_latency if latencies else 0,
            'std_latency_ms': std_latency if latencies else 0,
            'fps': 1000/avg_latency if latencies else 0,
            'memory_usage_mb': memory_usage,
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存到CSV
        import csv
        csv_path = os.path.join(os.path.dirname(model_path), 'quantization_performance.csv')
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=performance_report.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(performance_report)
        
        print(f"性能报告已保存到: {csv_path}")
        
    except Exception as e:
        print(f"内存测试失败: {e}")

if __name__ == "__main__":
    quantize_model()