import numpy as np
import torch
from train_model import postprocess
from skimage import measure
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk, binary_opening, binary_closing

def test_postprocess_step_by_step():
    """逐步调试后处理函数"""
    print("=== 逐步调试后处理函数 ===")
    
    # 测试场景：全一输入
    print("\n测试全一输入:")
    ones_output = torch.ones(1, 1, 64, 64)
    
    # 步骤1：应用sigmoid
    prob = torch.sigmoid(ones_output).cpu().numpy()
    print("1. 应用sigmoid后:")
    print(f"   形状: {prob.shape}")
    print(f"   值范围: {prob.min():.4f} - {prob.max():.4f}")
    
    # 步骤2：提取2D数组
    if prob.ndim == 4:
        prob_2d = prob[0, 0] if prob.shape[1] == 1 else prob[0, :, :, 0]
    print("2. 提取2D数组后:")
    print(f"   形状: {prob_2d.shape}")
    print(f"   值范围: {prob_2d.min():.4f} - {prob_2d.max():.4f}")
    
    # 步骤3：二值化
    prob_range = prob_2d.max() - prob_2d.min()
    if prob_range < 0.01:
        adaptive_thresh = prob_2d.mean()
        binary = (prob_2d > adaptive_thresh).astype(np.uint8)
        print(f"3. 使用自适应阈值 {adaptive_thresh:.4f}")
    else:
        binary = (prob_2d > 0.5).astype(np.uint8)
        print("3. 使用固定阈值 0.5")
    
    print(f"   二值化结果: min={binary.min()}, max={binary.max()}, mean={binary.mean():.4f}")
    
    # 步骤4：保守过滤
    if binary.mean() > 0.95:
        high_conf_thresh = max(0.8, np.percentile(prob_2d, 90))
        binary = (prob_2d > high_conf_thresh).astype(np.uint8)
        print(f"4. 保守过滤后: min={binary.min()}, max={binary.max()}, mean={binary.mean():.4f}")
    
    # 步骤5：连通域分析
    print("5. 连通域分析:")
    labels = measure.label(binary)
    print(f"   标签数量: {labels.max()}")
    print(f"   标签形状: {labels.shape}")
    
    # 步骤6：移除小区域
    properties = measure.regionprops(labels)
    print(f"   区域数量: {len(properties)}")
    for i, prop in enumerate(properties):
        print(f"   区域{i+1}: 标签={prop.label}, 面积={prop.area}")
        if prop.area < 100:  # min_area
            labels[labels == prop.label] = 0
            print(f"      -> 移除小区域")
    
    # 步骤7：合并邻近区域
    distance = distance_transform_edt(labels == 0)
    close_mask = distance < 10  # merge_distance
    labels[close_mask] = 1
    print(f"   合并后标签数量: {labels.max()}")
    
    # 步骤8：边界优化
    refined = np.zeros_like(binary)
    print(f"   开始边界优化，最大标签: {labels.max()}")
    
    for i in range(1, labels.max() + 1):
        region = (labels == i)
        print(f"   处理区域{i}: 大小={region.sum()}")
        
        # 形态学优化
        try:
            if region.ndim > 2:
                region = region.squeeze()
            
            if region.ndim == 2:
                selem_open = disk(1)
                selem_close = disk(2)
                
                if selem_open.shape[0] > region.shape[0] or selem_open.shape[1] > region.shape[1]:
                    print(f"      -> 结构元素过大，跳过形态学优化")
                    refined[region] = 1
                    continue
                
                region = binary_opening(region, footprint=selem_open)
                region = binary_closing(region, footprint=selem_close)
                print(f"      -> 形态学优化完成")
            else:
                print(f"      -> 不支持的维度: {region.ndim}")
                refined[region] = 1
                continue
                
        except Exception as morph_error:
            print(f"      -> 形态学操作失败: {morph_error}")
            pass
        
        refined[region] = 1
    
    print(f"   最终refined: min={refined.min()}, max={refined.max()}, mean={refined.mean():.4f}")
    
    # 步骤9：运行完整函数
    print("\n6. 运行完整后处理函数:")
    result = postprocess(ones_output, debug_mode=True)
    print(f"   完整函数结果: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")

if __name__ == "__main__":
    test_postprocess_step_by_step() 