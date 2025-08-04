import pandas as pd
import numpy as np

print("测试CSV文件修复...")
print("="*50)

try:
    # 读取CSV文件
    df = pd.read_csv('data/sim_features.csv')
    print(f"CSV文件形状: {df.shape}")
    print(f"CSV列名: {df.columns.tolist()}")
    
    # 查找文件名列
    filename_col = 'img_path'  # 根据图片显示，列名是img_path
    
    # 检查数值型列
    numeric_cols = []
    for col in df.columns:
        if col != filename_col:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                print(f"跳过非数值列: {col} (包含字符串值)")
    
    print(f"\n数值特征列: {numeric_cols}")
    print(f"特征维度: {len(numeric_cols)}")
    
    # 测试几行数据
    print(f"\n测试前3行数据:")
    for i in range(min(3, len(df))):
        filename = str(df.iloc[i][filename_col])
        features = df.iloc[i][numeric_cols].values.astype(np.float32)
        print(f"行 {i+1}: {filename}")
        print(f"  特征形状: {features.shape}")
        print(f"  特征范围: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  特征均值: {features.mean():.3f}")
    
    print("\n✅ CSV文件修复成功！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}") 