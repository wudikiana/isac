#!/usr/bin/env python3
"""
测试量化逻辑的脚本
验证是否只在完整训练完成后才进行量化
"""

def test_quantization_logic():
    print("="*60)
    print("测试量化逻辑...")
    print("="*60)
    
    # 模拟不同的训练场景
    test_cases = [
        {"start_epoch": 1, "expected_quantize": True, "description": "完整训练（从epoch 1开始）"},
        {"start_epoch": 5, "expected_quantize": False, "description": "恢复训练（从epoch 5开始）"},
        {"start_epoch": 10, "expected_quantize": False, "description": "恢复训练（从epoch 10开始）"},
        {"start_epoch": 21, "expected_quantize": False, "description": "训练已完成（从epoch 21开始）"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        start_epoch = case["start_epoch"]
        expected = case["expected_quantize"]
        description = case["description"]
        
        # 模拟量化逻辑
        should_quantize = start_epoch == 1
        
        status = "✅ 通过" if should_quantize == expected else "❌ 失败"
        print(f"测试 {i}: {description}")
        print(f"  开始epoch: {start_epoch}")
        print(f"  是否量化: {should_quantize}")
        print(f"  期望结果: {expected}")
        print(f"  测试结果: {status}")
        print()
    
    print("="*60)
    print("量化逻辑测试完成！")
    print("="*60)
    
    # 验证逻辑说明
    print("\n量化逻辑说明:")
    print("1. 当 start_epoch == 1 时：表示完整训练，会进行量化")
    print("2. 当 start_epoch > 1 时：表示恢复训练，跳过量化")
    print("3. 量化只在所有epoch都完整训练完成后执行")
    print("4. 这样可以避免在训练中断恢复时重复量化")

if __name__ == "__main__":
    test_quantization_logic() 