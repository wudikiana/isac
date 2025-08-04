import torch
from train_model import get_multimodal_patch_dataloaders

def test_normal_ratio():
    """测试正常样本比例是否生效"""
    print("🧪 测试正常样本比例...")
    
    # 获取数据加载器
    train_loader, val_loader = get_multimodal_patch_dataloaders(
        data_root="data/patch_dataset",
        sim_feature_csv="data/sim_features.csv",
        batch_size=4,
        num_workers=0,
        damage_boost=2,
        normal_ratio=0.05
    )
    
    print(f"训练集大小: {len(train_loader)}")
    
    # 统计前几个batch的样本分布
    normal_count = 0
    damage_count = 0
    
    for i, (images, masks, sim_feats) in enumerate(train_loader):
        if i >= 5:  # 只测试前5个batch
            break
            
        # 统计每个batch中的正常和损坏样本
        batch_normal = (masks.sum(dim=[1, 2]) == 0).sum().item()
        batch_damage = (masks.sum(dim=[1, 2]) > 0).sum().item()
        
        normal_count += batch_normal
        damage_count += batch_damage
        
        print(f"Batch {i}: 正常={batch_normal}, 损坏={batch_damage}")
    
    total_samples = normal_count + damage_count
    actual_ratio = normal_count / total_samples if total_samples > 0 else 0
    
    print(f"\n📊 统计结果:")
    print(f"  总样本数: {total_samples}")
    print(f"  正常样本数: {normal_count}")
    print(f"  损坏样本数: {damage_count}")
    print(f"  正常样本比例: {actual_ratio * 100:.1f}%")
    
    if actual_ratio > 0:
        print("✅ 正常样本添加成功!")
        return True
    else:
        print("❌ 没有检测到正常样本!")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("测试正常样本比例")
    print("=" * 50)
    
    success = test_normal_ratio()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 正常样本比例测试通过!")
    else:
        print("❌ 正常样本比例测试失败!")
    print("=" * 50)