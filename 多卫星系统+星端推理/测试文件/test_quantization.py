#!/usr/bin/env python3
"""
测试量化功能的脚本
"""

import os
import sys
import torch
import subprocess

def test_quantization():
    print("="*60)
    print("开始测试量化功能...")
    print("="*60)
    
    # 检查必要文件
    required_files = [
        "inference/quantize_model.py",
        "train_model.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            return False
        else:
            print(f"✅ 找到文件: {file_path}")
    
    # 检查是否有训练好的模型
    model_paths = [
        "models/best_multimodal_patch_model.pth",
        "models/checkpoint.pth"
    ]
    
    model_found = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ 找到模型文件: {model_path}")
            model_found = True
            break
    
    if not model_found:
        print("❌ 未找到训练好的模型文件")
        print("请先运行训练脚本生成模型")
        return False
    
    # 测试量化脚本
    print("\n开始测试量化脚本...")
    try:
        cmd = [
            sys.executable, "inference/quantize_model.py",
            "--model_path", model_paths[0] if os.path.exists(model_paths[0]) else model_paths[1],
            "--quant_path", "models/test_quantized_model.pt"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 量化脚本执行成功")
            print("量化输出:")
            print(result.stdout)
            
            # 检查量化模型文件
            if os.path.exists("models/test_quantized_model.pt"):
                file_size = os.path.getsize("models/test_quantized_model.pt") / (1024 * 1024)
                print(f"✅ 量化模型文件生成成功，大小: {file_size:.2f} MB")
                
                # 测试加载量化模型
                try:
                    quantized_model = torch.jit.load("models/test_quantized_model.pt")
                    dummy_image = torch.randn(1, 3, 256, 256)
                    dummy_sim_feat = torch.randn(1, 11)
                    
                    with torch.no_grad():
                        output = quantized_model(dummy_image, dummy_sim_feat)
                    
                    print(f"✅ 量化模型推理测试通过，输出形状: {output.shape}")
                    return True
                    
                except Exception as e:
                    print(f"❌ 量化模型推理测试失败: {e}")
                    return False
            else:
                print("❌ 量化模型文件未生成")
                return False
        else:
            print("❌ 量化脚本执行失败")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 量化脚本执行超时")
        return False
    except Exception as e:
        print(f"❌ 量化脚本执行异常: {e}")
        return False

def test_training_quantization_integration():
    print("\n" + "="*60)
    print("测试训练脚本中的量化集成...")
    print("="*60)
    
    # 检查训练脚本中是否包含量化代码
    try:
        with open("train_model.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        if "自动量化" in content and "quantize_model.py" in content:
            print("✅ 训练脚本包含量化集成代码")
            return True
        else:
            print("❌ 训练脚本缺少量化集成代码")
            return False
            
    except Exception as e:
        print(f"❌ 读取训练脚本失败: {e}")
        return False

if __name__ == "__main__":
    print("量化功能测试")
    print("="*60)
    
    # 测试量化脚本
    quantization_ok = test_quantization()
    
    # 测试训练集成
    integration_ok = test_training_quantization_integration()
    
    print("\n" + "="*60)
    print("测试结果总结:")
    print(f"量化脚本测试: {'✅ 通过' if quantization_ok else '❌ 失败'}")
    print(f"训练集成测试: {'✅ 通过' if integration_ok else '❌ 失败'}")
    
    if quantization_ok and integration_ok:
        print("\n🎉 所有测试通过！量化功能已准备就绪")
        print("训练完成后将自动进行模型量化")
    else:
        print("\n⚠️  部分测试失败，请检查相关配置")
    print("="*60) 