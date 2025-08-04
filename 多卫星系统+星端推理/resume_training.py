#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练恢复脚本
从第20个epoch继续训练
"""

import os
import sys

def main():
    print("🚀 开始恢复训练...")
    
    # 检查检查点文件
    checkpoint_path = "models/checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return
    
    print(f"✅ 发现检查点文件: {checkpoint_path}")
    
    # 运行训练脚本
    print("🔄 启动训练脚本...")
    os.system("python train_model.py")

if __name__ == "__main__":
    main()
