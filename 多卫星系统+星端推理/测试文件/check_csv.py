import pandas as pd

try:
    df = pd.read_csv('data/sim_features.csv')
    print("CSV文件列名:", df.columns.tolist())
    print("前3行:")
    print(df.head(3))
    print("文件形状:", df.shape)
except Exception as e:
    print("读取CSV文件失败:", e) 