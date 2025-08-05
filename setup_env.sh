#!/bin/bash
# Satellite ISAC Project Environment Setup Script
# Author: DeepSeek Chat
# Date: 2025-07-05
# Usage: bash setup_env.sh

set -e  # Exit immediately if any command fails

echo -e "\033[1;36m==============================="
echo "卫星ISAC项目环境安装脚本"
echo "===============================\033[0m"

# 1. 自动安装Miniconda（如果未检测到conda）
if ! command -v conda &> /dev/null; then
    echo -e "\033[1;33m[警告] 未检测到Anaconda/Miniconda，即将自动安装Miniconda\033[0m"
    
    # 选择适合操作系统的安装包
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            CONDA_INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
        else
            CONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        echo -e "\033[1;31m[错误] 不支持的操作系统: $OSTYPE\033[0m"
        exit 1
    fi

    # 从清华镜像站下载
    CONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$CONDA_INSTALLER"
    echo -e "\033[1;32m[1/5] 正在从清华镜像下载Miniconda...\033[0m"
    wget -q $CONDA_URL -O /tmp/$CONDA_INSTALLER

    # 安装Miniconda
    echo -e "\033[1;32m[2/5] 正在安装Miniconda...\033[0m"
    bash /tmp/$CONDA_INSTALLER -b -p $HOME/miniconda
    rm /tmp/$CONDA_INSTALLER

    # 初始化conda
    echo -e "\033[1;32m[3/5] 初始化conda...\033[0m"
    source $HOME/miniconda/bin/activate
    conda init bash
    if [[ -f ~/.bashrc ]]; then
        source ~/.bashrc
    fi
    
    echo -e "\033[1;36mMiniconda安装完成！需要重新打开终端使配置生效\033[0m"
    echo -e "或者运行以下命令立即生效：\nsource ~/.bashrc"
    exit 0
fi

# 2. 创建conda环境
ENV_NAME="isac"
PYTHON_VERSION="3.11"

if conda env list | grep -q "$ENV_NAME"; then
    echo -e "\033[1;33m[注意] 已存在 '$ENV_NAME' 环境，跳过创建\033[0m"
else
    echo -e "\033[1;32m[4/5] 正在创建conda环境 '$ENV_NAME' (Python $PYTHON_VERSION)...\033[0m"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# 3. 激活环境并配置清华PyPI源
echo -e "\033[1;32m[5/5] 激活环境并配置清华PyPI镜像...\033[0m"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 4. 安装依赖
REQ_FILE="requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo -e "\033[1;33m[注意] 未检测到requirements.txt，将安装核心依赖...\033[0m"
    pip install --upgrade pip
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
                gymnasium==1.2.0 stable-baselines3==2.3.0 \
                numpy==2.3.1 pandas==2.3.1 matplotlib==3.10.3 \
                wandb==0.21.0 optuna==3.6.1 pyyaml==6.0.2
else
    echo -e "\033[1;32m检测到requirements.txt，正在安装全部依赖...\033[0m"
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
fi

# 5. 验证安装
echo -e "\033[1;35m\n验证安装结果：\033[0m"
python -c "import torch; print(f'PyTorch版本: {torch.__version__} (CUDA可用: {torch.cuda.is_available()})')"
python -c "import gymnasium; print(f'Gymnasium版本: {gymnasium.__version__}')"
python -c "import pandas as pd; print(f'Pandas版本: {pd.__version__}')"

echo -e "\033[1;36m\n环境安装完成！请执行以下命令激活环境："
echo -e "conda activate $ENV_NAME\033[0m"