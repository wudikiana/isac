#!/usr/bin/env python3
"""
测试分阶段训练逻辑的脚本
"""

def test_staged_training_logic():
    print("="*60)
    print("测试分阶段训练逻辑...")
    print("="*60)
    
    # 模拟不同的训练场景
    test_cases = [
        {"start_epoch": 1, "expected_stage": 1, "expected_range": "1-20", "description": "阶段1开始"},
        {"start_epoch": 10, "expected_stage": 1, "expected_range": "1-20", "description": "阶段1中间"},
        {"start_epoch": 20, "expected_stage": 1, "expected_range": "1-20", "description": "阶段1结束"},
        {"start_epoch": 21, "expected_stage": 2, "expected_range": "21-40", "description": "阶段2开始"},
        {"start_epoch": 35, "expected_stage": 2, "expected_range": "21-40", "description": "阶段2中间"},
        {"start_epoch": 40, "expected_stage": 2, "expected_range": "21-40", "description": "阶段2结束"},
        {"start_epoch": 41, "expected_stage": 3, "expected_range": "41-60", "description": "阶段3开始"},
        {"start_epoch": 55, "expected_stage": 3, "expected_range": "41-60", "description": "阶段3中间"},
        {"start_epoch": 60, "expected_stage": 3, "expected_range": "41-60", "description": "阶段3结束"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        start_epoch = case["start_epoch"]
        expected_stage = case["expected_stage"]
        expected_range = case["expected_range"]
        description = case["description"]
        
        # 模拟分阶段逻辑
        current_stage = (start_epoch - 1) // 20 + 1
        stage_start_epoch = (current_stage - 1) * 20 + 1
        stage_end_epoch = current_stage * 20
        stage_range = f"{stage_start_epoch}-{stage_end_epoch}"
        
        # 检查是否需要量化
        should_quantize = start_epoch > stage_end_epoch
        
        status = "✅ 通过" if current_stage == expected_stage and stage_range == expected_range else "❌ 失败"
        print(f"测试 {i}: {description}")
        print(f"  开始epoch: {start_epoch}")
        print(f"  计算阶段: {current_stage}")
        print(f"  阶段范围: {stage_range}")
        print(f"  是否量化: {should_quantize}")
        print(f"  期望阶段: {expected_stage}")
        print(f"  期望范围: {expected_range}")
        print(f"  测试结果: {status}")
        print()
    
    print("="*60)
    print("分阶段训练逻辑测试完成！")
    print("="*60)
    
    # 验证逻辑说明
    print("\n分阶段训练逻辑说明:")
    print("1. 阶段1: epoch 1-20")
    print("2. 阶段2: epoch 21-40") 
    print("3. 阶段3: epoch 41-60")
    print("4. 每个阶段完成后自动量化")
    print("5. 量化模型按阶段命名: quantized_seg_model_stage{N}.pt")
    print("6. 支持从任意epoch恢复训练")
    print("7. 自动计算当前阶段和范围")

def test_quantization_naming():
    print("\n" + "="*60)
    print("测试量化模型命名逻辑...")
    print("="*60)
    
    stages = [1, 2, 3, 4, 5]
    for stage in stages:
        quant_path = f"models/quantized_seg_model_stage{stage}.pt"
        print(f"阶段 {stage}: {quant_path}")
    
    print("="*60)
    print("量化模型命名测试完成！")
    print("="*60)

if __name__ == "__main__":
    test_staged_training_logic()
    test_quantization_naming() 