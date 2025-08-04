import torch
import torch.nn as nn
import torch.nn.functional as F
from models.starlite_cnn import create_segmentation_landslide_model
from train_model import EnhancedDeepLab, DualModelEnsemble, DiceLoss

def debug_nan_root_cause():
    """诊断NaN的根本原因"""
    print("=== 诊断NaN根本原因 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    deeplab_model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11).to(device)
    landslide_model = create_segmentation_landslide_model(
        num_classes=1,
        use_attention=True,
        use_fpn=True,
        use_dynamic_attention=True,
        use_multi_scale=True
    ).to(device)
    
    ensemble_model = DualModelEnsemble(deeplab_model, landslide_model, fusion_weight=0.5).to(device)
    
    # 创建测试数据
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    sim_feat = torch.randn(batch_size, 11).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 224, 224)).float().to(device)
    
    print("1. 检查模型权重初始化...")
    
    # 检查模型权重
    def check_weights(model, name):
        for param_name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ {name} - {param_name} 包含NaN")
                return False
            if torch.isinf(param).any():
                print(f"❌ {name} - {param_name} 包含Inf")
                return False
        print(f"✅ {name} 权重正常")
        return True
    
    check_weights(deeplab_model, "DeepLab")
    check_weights(landslide_model, "LandslideDetector")
    check_weights(ensemble_model, "Ensemble")
    
    print("\n2. 逐步检查前向传播...")
    
    # 检查DeepLab前向传播
    print("检查DeepLab前向传播...")
    deeplab_model.eval()
    with torch.no_grad():
        try:
            deeplab_output = deeplab_model(x, sim_feat)
            print(f"✅ DeepLab输出形状: {deeplab_output.shape}")
            print(f"   DeepLab输出范围: [{deeplab_output.min().item():.4f}, {deeplab_output.max().item():.4f}]")
            print(f"   DeepLab输出是否包含NaN: {torch.isnan(deeplab_output).any()}")
            print(f"   DeepLab输出是否包含Inf: {torch.isinf(deeplab_output).any()}")
        except Exception as e:
            print(f"❌ DeepLab前向传播失败: {e}")
    
    # 检查LandslideDetector前向传播
    print("\n检查LandslideDetector前向传播...")
    landslide_model.eval()
    with torch.no_grad():
        try:
            landslide_output = landslide_model(x)
            print(f"✅ LandslideDetector输出形状: {landslide_output.shape}")
            print(f"   LandslideDetector输出范围: [{landslide_output.min().item():.4f}, {landslide_output.max().item():.4f}]")
            print(f"   LandslideDetector输出是否包含NaN: {torch.isnan(landslide_output).any()}")
            print(f"   LandslideDetector输出是否包含Inf: {torch.isinf(landslide_output).any()}")
        except Exception as e:
            print(f"❌ LandslideDetector前向传播失败: {e}")
    
    print("\n3. 检查集成模型前向传播...")
    
    ensemble_model.eval()
    with torch.no_grad():
        try:
            ensemble_output = ensemble_model(x, sim_feat)
            print(f"✅ 集成模型输出形状: {ensemble_output.shape}")
            print(f"   集成模型输出范围: [{ensemble_output.min().item():.4f}, {ensemble_output.max().item():.4f}]")
            print(f"   集成模型输出是否包含NaN: {torch.isnan(ensemble_output).any()}")
            print(f"   集成模型输出是否包含Inf: {torch.isinf(ensemble_output).any()}")
        except Exception as e:
            print(f"❌ 集成模型前向传播失败: {e}")
    
    print("\n4. 检查损失函数...")
    
    criterion = DiceLoss()
    
    # 检查损失函数
    ensemble_model.train()
    try:
        outputs = ensemble_model(x, sim_feat)
        loss = criterion(outputs, masks)
        print(f"✅ 损失值: {loss.item():.4f}")
        print(f"   损失是否包含NaN: {torch.isnan(loss)}")
        print(f"   损失是否包含Inf: {torch.isinf(loss)}")
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
    
    print("\n5. 检查梯度...")
    
    try:
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        nan_grad_count = 0
        inf_grad_count = 0
        
        for name, param in ensemble_model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"❌ 梯度包含NaN: {name}")
                    nan_grad_count += 1
                if torch.isinf(param.grad).any():
                    print(f"❌ 梯度包含Inf: {name}")
                    inf_grad_count += 1
                
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        print(f"✅ 梯度范数: {total_norm:.4f}")
        print(f"   包含NaN的梯度数量: {nan_grad_count}")
        print(f"   包含Inf的梯度数量: {inf_grad_count}")
        
    except Exception as e:
        print(f"❌ 梯度计算失败: {e}")
    
    print("\n6. 检查学习率设置...")
    
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=1e-5, weight_decay=1e-3)
    print(f"✅ 学习率: {optimizer.param_groups[0]['lr']}")
    print(f"   权重衰减: {optimizer.param_groups[0]['weight_decay']}")
    
    print("\n7. 检查数据...")
    
    print(f"✅ 输入数据范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"   输入数据是否包含NaN: {torch.isnan(x).any()}")
    print(f"   输入数据是否包含Inf: {torch.isinf(x).any()}")
    
    print(f"✅ 仿真特征范围: [{sim_feat.min().item():.4f}, {sim_feat.max().item():.4f}]")
    print(f"   仿真特征是否包含NaN: {torch.isnan(sim_feat).any()}")
    print(f"   仿真特征是否包含Inf: {torch.isinf(sim_feat).any()}")
    
    print(f"✅ 掩码数据范围: [{masks.min().item():.4f}, {masks.max().item():.4f}]")
    print(f"   掩码数据是否包含NaN: {torch.isnan(masks).any()}")
    print(f"   掩码数据是否包含Inf: {torch.isinf(masks).any()}")

if __name__ == "__main__":
    debug_nan_root_cause() 