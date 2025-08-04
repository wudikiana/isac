import numpy as np
import os

def check_npy_nan_inf(folder_path):
    """
    检查指定文件夹中所有.npy文件是否包含NaN或Inf值
    :param folder_path: 包含.npy文件的文件夹路径
    """
    print(f"\n检查目录: {folder_path}")
    total_files = 0
    problem_files = 0
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            total_files += 1
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 加载numpy数组
                arr = np.load(file_path)
                
                # 检查NaN和Inf
                has_nan = np.isnan(arr).any()
                has_inf = np.isinf(arr).any()
                
                if has_nan or has_inf:
                    problem_files += 1
                    print(f"⚠️ 发现异常文件: {filename}")
                    if has_nan:
                        print(f"   包含 NaN 值: {np.isnan(arr).sum()}个")
                    if has_inf:
                        print(f"   包含 Inf 值: {np.isinf(arr).sum()}个")
            
            except Exception as e:
                print(f"❌ 加载文件失败: {filename} - {str(e)}")
    
    print(f"检查完成! 共扫描 {total_files} 个文件, 发现 {problem_files} 个问题文件")

if __name__ == "__main__":
    # 检查图像和掩码目录
    base_dir = 'data/patch_dataset/train'
    check_npy_nan_inf(os.path.join(base_dir, 'images'))
    check_npy_nan_inf(os.path.join(base_dir, 'masks'))
