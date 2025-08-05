#!/bin/bash

# =============================================
# Satellite ISAC Project Setup Tool
# =============================================
# Supports: Linux (Ubuntu, Debian, CentOS, Fedora) and macOS
# =============================================

# 定义颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 重置颜色

# 检查当前用户是否为root
check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo -e "${YELLOW}需要管理员权限，请输入密码${NC}"
        sudo "$0" "$@"
        exit $?
    fi
}

# 显示菜单
show_menu() {
    clear
    echo -e "${GREEN}============================================="
    echo -e "      Satellite ISAC 项目环境安装工具"
    echo -e "=============================================${NC}"
    echo -e "1. 使用 Miniconda 安装 (推荐)"
    echo -e "2. 使用 Python 虚拟环境安装"
    echo -e "3. 直接安装到系统 Python 环境"
    echo -e "4. 退出"
    echo -e ""
    read -p "请选择安装方式 [1-4]: " choice
}

# 安装系统依赖
install_system_deps() {
    echo -e "${BLUE}==> 安装系统依赖...${NC}"
    
    # 检测系统类型
    if [ -f /etc/os-release ]; then
        # Linux 系统
        source /etc/os-release
        case $ID in
            ubuntu|debian)
                sudo apt-get update
                sudo apt-get install -y curl wget python3 python3-pip python3-venv build-essential
                ;;
            centos|rhel|fedora)
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y curl wget python3 python3-pip
                ;;
            *)
                echo -e "${RED}不支持的操作系统: $ID${NC}"
                exit 1
                ;;
        esac
    elif [ "$(uname)" == "Darwin" ]; then
        # macOS 系统
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}未找到 Homebrew，正在安装...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew update
        brew install python3 wget
    else
        echo -e "${RED}不支持的操作系统${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}系统依赖安装完成!${NC}"
}

# 安装 Miniconda
install_miniconda() {
    echo -e "${BLUE}==> 安装 Miniconda...${NC}"
    
    # 检测系统架构
    if [ "$(uname -m)" == "x86_64" ]; then
        arch="x86_64"
    elif [ "$(uname -m)" == "arm64" ]; then
        arch="arm64"
    else
        arch="x86_64"
    fi
    
    # 下载 Miniconda
    if [ "$(uname)" == "Darwin" ]; then
        wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-${arch}.sh
    else
        wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${arch}.sh
    fi
    
    # 检查下载是否成功
    if [ ! -f "miniconda.sh" ]; then
        echo -e "${RED}Miniconda 下载失败${NC}"
        exit 1
    fi
    
    # 安装 Miniconda
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    
    # 设置环境变量
    export PATH="$HOME/miniconda/bin:$PATH"
    source $HOME/miniconda/etc/profile.d/conda.sh
    conda init bash
    
    echo -e "${GREEN}Miniconda 安装成功!${NC}"
}

# 使用 Miniconda 安装
install_with_miniconda() {
    echo -e "${BLUE}==> 使用 Miniconda 安装${NC}"
    
    # 检查是否已安装 Miniconda
    if ! command -v conda &> /dev/null; then
        install_miniconda
        # 重新加载环境变量
        export PATH="$HOME/miniconda/bin:$PATH"
        source $HOME/miniconda/etc/profile.d/conda.sh
    fi
    
    # 创建环境
    ENV_NAME="isac"
    PYTHON_VERSION="3.11"
    
    if conda env list | grep -q "$ENV_NAME"; then
        echo -e "${YELLOW}环境 '$ENV_NAME' 已存在${NC}"
        read -p "是否重新创建环境? [y/N]: " recreate
        if [[ "$recreate" == "y" || "$recreate" == "Y" ]]; then
            conda env remove -n $ENV_NAME -y
            conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        fi
    else
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    fi
    
    # 检查环境是否创建成功
    if ! conda env list | grep -q "$ENV_NAME"; then
        echo -e "${RED}环境创建失败${NC}"
        exit 1
    fi
    
    # 激活环境并安装依赖
    conda activate $ENV_NAME
    install_dependencies
    
    echo -e "${GREEN}============================================="
    echo -e "环境安装完成!"
    echo -e "激活环境命令:"
    echo -e "  conda activate $ENV_NAME"
    echo -e "=============================================${NC}"
}

# 使用虚拟环境安装
install_with_venv() {
    echo -e "${BLUE}==> 使用 Python 虚拟环境安装${NC}"
    
    # 检查 Python 版本
    python3 -c "import sys; sys.exit(0) if sys.version_info >= (3,7) else sys.exit(1)"
    if [ $? -ne 0 ]; then
        echo -e "${RED}需要 Python 3.7 或更高版本${NC}"
        exit 1
    fi
    
    # 创建虚拟环境
    VENV_NAME="isac-venv"
    VENV_DIR="./$VENV_NAME"
    
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}虚拟环境已存在: $VENV_DIR${NC}"
        read -p "是否重新创建环境? [y/N]: " recreate
        if [[ "$recreate" == "y" || "$recreate" == "Y" ]]; then
            rm -rf "$VENV_DIR"
            python3 -m venv "$VENV_DIR"
        fi
    else
        python3 -m venv "$VENV_DIR"
    fi
    
    # 检查环境是否创建成功
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${RED}虚拟环境创建失败${NC}"
        exit 1
    fi
    
    # 激活环境并安装依赖
    source "$VENV_DIR/bin/activate"
    install_dependencies
    
    echo -e "${GREEN}============================================="
    echo -e "环境安装完成!"
    echo -e "激活环境命令:"
    echo -e "  source $VENV_DIR/bin/activate"
    echo -e "=============================================${NC}"
}

# 直接安装到系统 Python
install_to_system() {
    echo -e "${YELLOW}警告: 这将修改系统 Python 环境"
    echo -e "       可能会影响其他项目${NC}"
    read -p "确定要继续吗? [y/N]: " confirm
    
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        return
    fi
    
    install_dependencies
    
    echo -e "${GREEN}============================================="
    echo -e "依赖安装完成!"
    echo -e "=============================================${NC}"
}

# 安装依赖
install_dependencies() {
    echo -e "${BLUE}==> 配置清华 PyPI 镜像...${NC}"
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
    
    echo -e "${BLUE}==> 升级 pip...${NC}"
    pip install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
        echo -e "${BLUE}==> 从 requirements.txt 安装依赖...${NC}"
        
        # 创建修复版 requirements
        echo "numpy==1.26.4" > requirements_fixed.txt
        echo "scipy==1.13.0" >> requirements_fixed.txt
        cat requirements.txt >> requirements_fixed.txt
        
        pip install -r requirements_fixed.txt
        rm requirements_fixed.txt
    else
        echo -e "${BLUE}==> 安装核心依赖...${NC}"
        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
        pip install gymnasium==1.2.0 stable-baselines3==2.3.0
        pip install numpy==1.26.4 pandas==2.3.1 matplotlib==3.10.3
        pip install wandb==0.21.0 optuna==3.6.1 pyyaml==6.0.2
    fi
    
    # 验证安装
    echo -e "${BLUE}==> 验证安装...${NC}"
    python -c "import torch; print(f'PyTorch 版本: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}')"
    python -c "import gymnasium; print(f'Gymnasium 版本: {gymnasium.__version__}')"
    python -c "import pandas as pd; print(f'Pandas 版本: {pd.__version__}')"
    python -c "try: import scipy; print(f'SciPy 版本: {scipy.__version__}'); except: print('SciPy 未安装')"
}

# 主函数
main() {
    # 检查是否需要 root 权限
    if [ "$1" != "--no-root" ]; then
        check_root "$@"
    fi
    
    # 安装系统依赖
    install_system_deps
    
    while true; do
        show_menu
        
        case $choice in
            1)
                install_with_miniconda
                ;;
            2)
                install_with_venv
                ;;
            3)
                install_to_system
                ;;
            4)
                echo -e "${GREEN}退出...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选择，请重试${NC}"
                sleep 2
                continue
                ;;
        esac
        
        read -p "按 Enter 键返回菜单..."
    done
}

# 启动主函数
main "$@"
