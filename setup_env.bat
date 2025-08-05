@echo off
:: Satellite ISAC Project Environment Setup Script for Windows
:: Author: DeepSeek Chat
:: Revised Date: 2025-08-05
:: Version: 2.0
:: Features:
:: - 自动检测并安装Miniconda（清华镜像）
:: - 可靠的环境创建和依赖安装
:: - 全面的错误检查和用户指导
:: Usage: 右键"以管理员身份运行"（推荐）

echo ===============================
echo 卫星ISAC项目环境安装脚本 (Windows版)
echo ===============================
echo 注意: 推荐右键本脚本选择"以管理员身份运行"
echo.

:: 初始化错误标志
set ERR_FLAG=0

:: 1. 检查Conda安装
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [1/7] 未检测到Anaconda/Miniconda，准备安装Miniconda...
    
    :: 下载Miniconda
    echo [2/7] 正在从清华镜像下载Miniconda...
    curl -f -o %TEMP%\Miniconda3-latest-Windows-x86_64.exe https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe || (
        echo [错误] Miniconda下载失败，请检查网络连接
        set ERR_FLAG=1
        goto :ERROR_HANDLE
    )
    
    :: 安装Miniconda
    echo [3/7] 正在安装Miniconda（静默模式）...
    start /wait "" %TEMP%\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=%USERPROFILE%\Miniconda3 || (
        echo [错误] Miniconda安装失败
        set ERR_FLAG=1
        goto :ERROR_HANDLE
    )
    
    :: 清理安装包
    del %TEMP%\Miniconda3-latest-Windows-x86_64.exe
    
    echo [4/7] Miniconda安装成功！
    echo.
    echo 请执行以下操作：
    echo 1. 关闭所有命令提示符窗口
    echo 2. 重新打开新的命令提示符
    echo 3. 再次运行此脚本以完成环境配置
    pause
    exit /b 0
)

:: 2. 配置Conda环境
set ENV_NAME=isac
set PYTHON_VERSION=3.11
set CONDA_BASE=%USERPROFILE%\Miniconda3
set CONDA_EXE=%CONDA_BASE%\Scripts\conda.exe

:: 检查是否在正确的环境中运行
if "%CONDA_DEFAULT_ENV%" == "%ENV_NAME%" (
    echo [注意] 检测到当前已在 %ENV_NAME% 环境中运行
    echo 建议在base环境中运行此脚本
    pause
)

:: 3. 创建Conda环境
echo [5/7] 检查环境 %ENV_NAME%...
"%CONDA_EXE%" env list | findstr "%ENV_NAME%" >nul
if %ERRORLEVEL% equ 0 (
    echo [注意] 环境 %ENV_NAME% 已存在，跳过创建
) else (
    echo [6/7] 正在创建环境 %ENV_NAME% (Python %PYTHON_VERSION%)...
    "%CONDA_EXE%" create -n %ENV_NAME% python=%PYTHON_VERSION% -y || (
        echo [错误] 环境创建失败
        set ERR_FLAG=1
        goto :ERROR_HANDLE
    )
)

:: 4. 在新环境中安装依赖
echo [7/7] 正在配置依赖项...
echo 将使用清华PyPI镜像加速安装...
"%CONDA_EXE%" run -n %ENV_NAME% pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
"%CONDA_EXE%" run -n %ENV_NAME% pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

if not exist "requirements.txt" (
    echo 未找到requirements.txt，安装核心依赖...
    "%CONDA_EXE%" run -n %ENV_NAME% python -m pip install --upgrade pip || (
        echo [错误] pip升级失败
        set ERR_FLAG=1
        goto :ERROR_HANDLE
    )
    
    :: 使用更稳定的依赖版本
    "%CONDA_EXE%" run -n %ENV_NAME% pip install ^
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 ^
        gymnasium==1.2.0 stable-baselines3==2.3.0 ^
        numpy==1.26.4 pandas==2.1.4 matplotlib==3.8.0 ^
        wandb==0.16.0 optuna==3.4.0 pyyaml==6.0.1
) else (
    echo 检测到requirements.txt，正在安装全部依赖...
    "%CONDA_EXE%" run -n %ENV_NAME% pip install -r requirements.txt
)

if %ERRORLEVEL% neq 0 (
    echo [警告] 某些依赖可能安装失败
    set ERR_FLAG=1
)

:: 5. 验证安装
echo.
echo ===== 验证安装结果 =====
"%CONDA_EXE%" run -n %ENV_NAME% python -c "import sys; print(f'Python路径: {sys.executable}')"
"%CONDA_EXE%" run -n %ENV_NAME% python -c "import torch; print(f'PyTorch版本: {torch.__version__} | CUDA可用: {torch.cuda.is_available()}')"
"%CONDA_EXE%" run -n %ENV_NAME% python -c "import gymnasium; print(f'Gymnasium版本: {gymnasium.__version__}')"
"%CONDA_EXE%" run -n %ENV_NAME% python -c "import pandas as pd; print(f'Pandas版本: {pd.__version__}')"

:ERROR_HANDLE
if %ERR_FLAG% equ 0 (
    echo.
    echo ===== 环境配置成功完成! =====
    echo 使用以下命令激活环境:
    echo conda activate %ENV_NAME%
) else (
    echo.
    echo ===== 遇到错误 =====
    echo 某些步骤未能完成，请检查:
    echo 1. 是否以管理员身份运行
    echo 2. 网络连接是否正常
    echo 3. 磁盘空间是否充足
    echo 4. 查看上方的具体错误信息
)

pause
exit /b %ERR_FLAG%
