@echo off
:: Satellite ISAC Project Environment Setup Script for Windows
:: Author: DeepSeek Chat
:: Date: 2025-08-05
:: Usage: Double-click this file or run from Command Prompt

echo ===============================
echo 卫星ISAC项目环境安装脚本 (Windows版)
echo ===============================

:: 1. Check for Conda installation
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [警告] 未检测到Anaconda/Miniconda，即将自动安装Miniconda
    
    :: Download Miniconda
    echo [1/5] 正在从清华镜像下载Miniconda...
    curl -o %TEMP%\Miniconda3-latest-Windows-x86_64.exe https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
    
    :: Install Miniconda
    echo [2/5] 正在安装Miniconda...
    start /wait "" %TEMP%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=%USERPROFILE%\Miniconda3
    
    :: Clean up
    del %TEMP%\Miniconda3-latest-Windows-x86_64.exe
    
    echo [3/5] Miniconda安装完成!
    echo 请关闭并重新打开命令提示符使配置生效
    pause
    exit /b
)

:: 2. Create conda environment
set ENV_NAME=isac
set PYTHON_VERSION=3.11

conda env list | findstr "%ENV_NAME%" >nul
if %ERRORLEVEL% equ 0 (
    echo [注意] 已存在 '%ENV_NAME%' 环境，跳过创建
) else (
    echo [4/5] 正在创建conda环境 '%ENV_NAME%' (Python %PYTHON_VERSION%)...
    conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
)

:: 3. Activate environment and configure Tsinghua PyPI mirror
echo [5/5] 激活环境并配置清华PyPI镜像...
call conda activate %ENV_NAME%

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

:: 4. Install dependencies
if not exist "requirements.txt" (
    echo [注意] 未检测到requirements.txt，将安装核心依赖...
    python -m pip install --upgrade pip
    python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 ^
                gymnasium==1.2.0 stable-baselines3==2.3.0 ^
                numpy==2.3.1 pandas==2.3.1 matplotlib==3.10.3 ^
                wandb==0.21.0 optuna==3.6.1 pyyaml==6.0.2
) else (
    echo 检测到requirements.txt，正在安装全部依赖...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

:: 5. Verify installation
echo.
echo 验证安装结果：
python -c "import torch; print(f'PyTorch版本: {torch.__version__} (CUDA可用: {torch.cuda.is_available()}')"
python -c "import gymnasium; print(f'Gymnasium版本: {gymnasium.__version__}')"
python -c "import pandas as pd; print(f'Pandas版本: {pd.__version__}')"

echo.
echo 环境安装完成！请执行以下命令激活环境：
echo conda activate %ENV_NAME%
pause
