#!/usr/bin/env python3
"""
对比测试：展示选项1的效果
"""

import os
import sys
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_comparison():
    """对比测试：展示选项1的效果"""
    print("=== 对比测试：选项1效果展示 ===")
    
    try:
        from data_utils.data_loader import get_segmentation_dataloaders
        
        # 测试1：显示警告（原始行为）
        print("\n--- 测试1：显示警告（原始行为） ---")
        start_time = time.time()
        train_loader_with_warnings, val_loader_with_warnings, test_loader_with_warnings = get_segmentation_dataloaders(
            data_root="data/combined_dataset",
            batch_size=2,
            num_workers=0,
            show_warnings=True,  # 显示警告
            skip_problematic_samples=False  # 保留所有数据
        )
        time_with_warnings = time.time() - start_time
        print(f"✓ 显示警告时训练集大小: {len(train_loader_with_warnings.dataset)}")
        print(f"✓ 显示警告时耗时: {time_with_warnings:.2f}秒")
        
        # 测试2：隐藏警告（选项1）
        print("\n--- 测试2：隐藏警告（选项1） ---")
        start_time = time.time()
        train_loader_no_warnings, val_loader_no_warnings, test_loader_no_warnings = get_segmentation_dataloaders(
            data_root="data/combined_dataset",
            batch_size=2,
            num_workers=0,
            show_warnings=False,  # 隐藏警告
            skip_problematic_samples=False  # 保留所有数据
        )
        time_no_warnings = time.time() - start_time
        print(f"✓ 隐藏警告时训练集大小: {len(train_loader_no_warnings.dataset)}")
        print(f"✓ 隐藏警告时耗时: {time_no_warnings:.2f}秒")
        
        # 测试3：跳过问题样本
        print("\n--- 测试3：跳过问题样本 ---")
        start_time = time.time()
        train_loader_skip, val_loader_skip, test_loader_skip = get_segmentation_dataloaders(
            data_root="data/combined_dataset",
            batch_size=2,
            num_workers=0,
            show_warnings=True,  # 显示警告
            skip_problematic_samples=True  # 跳过问题样本
        )
        time_skip = time.time() - start_time
        print(f"✓ 跳过问题样本时训练集大小: {len(train_loader_skip.dataset)}")
        print(f"✓ 跳过问题样本时耗时: {time_skip:.2f}秒")
        
        # 结果对比
        print(f"\n=== 结果对比 ===")
        print(f"显示警告时数据集大小: {len(train_loader_with_warnings.dataset)}")
        print(f"隐藏警告时数据集大小: {len(train_loader_no_warnings.dataset)}")
        print(f"跳过问题样本时数据集大小: {len(train_loader_skip.dataset)}")
        
        if len(train_loader_with_warnings.dataset) == len(train_loader_no_warnings.dataset):
            print("✓ 选项1成功：隐藏警告但保留了所有数据")
        else:
            print("✗ 选项1失败：数据集大小不一致")
        
        skipped_count = len(train_loader_with_warnings.dataset) - len(train_loader_skip.dataset)
        print(f"跳过的样本数量: {skipped_count}")
        
        print(f"\n=== 性能对比 ===")
        print(f"显示警告耗时: {time_with_warnings:.2f}秒")
        print(f"隐藏警告耗时: {time_no_warnings:.2f}秒")
        print(f"跳过问题样本耗时: {time_skip:.2f}秒")
        
        if time_no_warnings < time_with_warnings:
            print("✓ 隐藏警告提高了性能")
        else:
            print("⚠ 隐藏警告没有显著提高性能")
        
        print(f"\n=== 推荐使用方式 ===")
        print("选项1（推荐用于训练）:")
        print("  show_warnings=False, skip_problematic_samples=False")
        print("  - 隐藏警告信息")
        print("  - 保留所有数据")
        print("  - 适合训练阶段")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comparison() 