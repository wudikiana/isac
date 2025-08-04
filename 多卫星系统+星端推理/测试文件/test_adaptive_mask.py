import torch
import torch.nn as nn
from train_model import process_xview2_mask

def test_adaptive_mask():
    """测试动态掩码调整功能"""
    print("=== 测试动态掩码调整功能 ===")
    
    # 创建测试掩码
    batch_size = 2
    height, width = 224, 224
    
    # 创建不同损坏比例的掩码
    test_cases = [
        ("低损坏比例", 0.05),  # 5%损坏
        ("中等损坏比例", 0.15),  # 15%损坏
        ("高损坏比例", 0.25),  # 25%损坏
        ("很高损坏比例", 0.45),  # 45%损坏
        ("极高损坏比例", 0.75),  # 75%损坏
    ]
    
    for case_name, damage_ratio in test_cases:
        print(f"\n测试: {case_name} (损坏比例: {damage_ratio:.1%})")
        
        # 创建掩码
        mask = torch.zeros(batch_size, 1, height, width)
        
        # 随机设置损坏区域
        num_damage_pixels = int(height * width * damage_ratio)
        damage_indices = torch.randperm(height * width)[:num_damage_pixels]
        
        for idx in damage_indices:
            h, w = idx // width, idx % width
            # 随机分配损坏级别 (2, 3, 4)
            damage_level = torch.randint(2, 5, (1,)).item()
            mask[0, 0, h, w] = damage_level
        
        # 计算实际损坏比例
        actual_damage_ratio = (mask >= 2).sum().float() / mask.numel()
        print(f"  实际损坏比例: {actual_damage_ratio:.4f}")
        
        # 测试不同的处理方式
        processing_methods = ['all', 'light', 'adaptive']
        
        for method in processing_methods:
            if method == 'adaptive':
                # 使用动态调整
                processed_mask = process_xview2_mask(mask, method, actual_damage_ratio)
            else:
                # 使用固定方式
                processed_mask = process_xview2_mask(mask, method)
            
            # 统计结果
            unique_values = torch.unique(processed_mask)
            print(f"  {method}: 唯一值 {unique_values.tolist()}, 范围 [{processed_mask.min():.3f}, {processed_mask.max():.3f}]")
            
            # 对于adaptive方式，显示权重调整
            if method == 'adaptive':
                if actual_damage_ratio < 0.1:
                    print(f"    → 低损坏比例，增加权重")
                elif actual_damage_ratio < 0.3:
                    print(f"    → 中等损坏比例，适度调整")
                elif actual_damage_ratio > 0.7:
                    print(f"    → 高损坏比例，降低权重")
                else:
                    print(f"    → 适中损坏比例，标准权重")

def test_sample_ratio_calculation():
    """测试样本比例计算"""
    print("\n=== 测试样本比例计算 ===")
    
    # 创建测试数据
    batch_size = 4
    height, width = 224, 224
    
    # 模拟不同batch的损坏情况
    test_batches = [
        ("正常样本", 0.02),  # 2%损坏
        ("轻微损坏", 0.08),  # 8%损坏
        ("中等损坏", 0.18),  # 18%损坏
        ("严重损坏", 0.35),  # 35%损坏
    ]
    
    for batch_name, target_ratio in test_batches:
        print(f"\n{batch_name} (目标损坏比例: {target_ratio:.1%})")
        
        # 创建掩码
        masks = torch.zeros(batch_size, 1, height, width)
        
        # 设置损坏区域
        num_damage_pixels = int(batch_size * height * width * target_ratio)
        damage_indices = torch.randperm(batch_size * height * width)[:num_damage_pixels]
        
        for idx in damage_indices:
            b, h, w = idx // (height * width), (idx % (height * width)) // width, idx % width
            damage_level = torch.randint(2, 5, (1,)).item()
            masks[b, 0, h, w] = damage_level
        
        # 计算损坏样本比例
        damage_pixels = (masks >= 2).sum().float()
        total_pixels = masks.numel()
        current_sample_ratio = damage_pixels / total_pixels
        
        print(f"  计算得到的损坏比例: {current_sample_ratio:.4f}")
        print(f"  目标比例: {target_ratio:.4f}")
        print(f"  误差: {abs(current_sample_ratio - target_ratio):.4f}")
        
        # 应用动态调整
        processed_masks = process_xview2_mask(masks, 'adaptive', current_sample_ratio)
        
        # 统计处理后的权重分布
        unique_values = torch.unique(processed_masks)
        print(f"  处理后权重: {unique_values.tolist()}")

def test_weight_adjustment():
    """测试权重调整策略"""
    print("\n=== 测试权重调整策略 ===")
    
    # 创建标准掩码
    mask = torch.zeros(1, 1, 100, 100)
    
    # 设置不同损坏级别
    mask[0, 0, 10:20, 10:20] = 2  # 轻微损坏
    mask[0, 0, 30:40, 30:40] = 3  # 中等损坏
    mask[0, 0, 50:60, 50:60] = 4  # 严重损坏
    
    # 测试不同样本比例下的权重调整
    test_ratios = [0.05, 0.15, 0.25, 0.45, 0.75]
    
    for ratio in test_ratios:
        print(f"\n样本比例: {ratio:.1%}")
        
        processed_mask = process_xview2_mask(mask, 'adaptive', ratio)
        
        # 统计各损坏级别的权重
        light_damage_weight = processed_mask[mask == 2].mean().item()
        medium_damage_weight = processed_mask[mask == 3].mean().item()
        severe_damage_weight = processed_mask[mask == 4].mean().item()
        
        print(f"  轻微损坏权重: {light_damage_weight:.3f}")
        print(f"  中等损坏权重: {medium_damage_weight:.3f}")
        print(f"  严重损坏权重: {severe_damage_weight:.3f}")
        
        # 分析调整策略
        if ratio < 0.1:
            print(f"  → 低比例策略: 增加权重以平衡")
        elif ratio < 0.3:
            print(f"  → 中等比例策略: 适度调整")
        elif ratio > 0.7:
            print(f"  → 高比例策略: 降低权重避免过拟合")
        else:
            print(f"  → 标准策略: 使用默认权重")

if __name__ == "__main__":
    test_adaptive_mask()
    test_sample_ratio_calculation()
    test_weight_adjustment() 