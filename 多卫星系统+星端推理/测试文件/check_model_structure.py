 import torch
import sys
import os
sys.path.append(os.path.abspath('.'))
from train_model import EnhancedDeepLab

def check_model_structure():
    # 加载checkpoint
    checkpoint = torch.load('models/best_multimodal_patch_model.pth', map_location='cpu')
    print("Checkpoint keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict")
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Model state dict has {len(state_dict)} keys")
        print("First 10 keys:", list(state_dict.keys())[:10])
        
        # 创建模型
        model = EnhancedDeepLab(in_channels=3, num_classes=1, sim_feat_dim=11)
        model_dict = model.state_dict()
        print(f"Current model has {len(model_dict)} keys")
        print("First 10 keys:", list(model_dict.keys())[:10])
        
        # 检查匹配的键
        matched_keys = []
        for key in state_dict.keys():
            if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                matched_keys.append(key)
        
        print(f"Matched keys: {len(matched_keys)}")
        if len(matched_keys) > 0:
            print("First 5 matched keys:", matched_keys[:5])
        
        # 检查不匹配的键
        unmatched_keys = []
        for key in state_dict.keys():
            if key not in model_dict:
                unmatched_keys.append(key)
        
        print(f"Unmatched keys: {len(unmatched_keys)}")
        if len(unmatched_keys) > 0:
            print("First 5 unmatched keys:", unmatched_keys[:5])

if __name__ == "__main__":
    check_model_structure()