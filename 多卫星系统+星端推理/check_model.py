import torch
import os

def check_model():
    model_path = "models/best_multimodal_patch_model.pth"
    print(f"检查模型文件: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"State dict keys (前10个): {list(state_dict.keys())[:10]}")
                print(f"State dict keys (后10个): {list(state_dict.keys())[-10:]}")
                
                # 检查是否有deeplab相关的键
                deeplab_keys = [k for k in state_dict.keys() if 'deeplab' in k.lower()]
                print(f"Deeplab相关键数量: {len(deeplab_keys)}")
                if deeplab_keys:
                    print(f"Deeplab相关键 (前5个): {deeplab_keys[:5]}")
                
                # 检查权重值的统计信息
                weight_values = []
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        weight_values.append(value.mean().item())
                
                if weight_values:
                    print(f"权重均值统计: min={min(weight_values):.4f}, max={max(weight_values):.4f}, mean={sum(weight_values)/len(weight_values):.4f}")
                
                if 'epoch' in checkpoint:
                    print(f"训练轮次: {checkpoint['epoch']}")
                if 'best_val_iou' in checkpoint:
                    print(f"最佳IoU: {checkpoint['best_val_iou']}")
        else:
            print("Checkpoint不是字典格式")
            
    except Exception as e:
        print(f"加载模型失败: {e}")

if __name__ == "__main__":
    check_model() 