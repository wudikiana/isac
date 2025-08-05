@echo off
setlocal enabledelayedexpansion

:: =============================================
:: Satellite ISAC Project Setup Tool (Windows)
:: =============================================
title Satellite ISAC Environment Setup

:: =============================================
:: Set working directory
:: =============================================
set "SCRIPT_DIR=%~dp0"
cd /d "!SCRIPT_DIR!"
echo Working directory: %cd%
echo.

:: =============================================
:: Main Menu
:: =============================================
:menu
cls
echo =============================================
echo      Satellite ISAC Project Setup Tool
echo =============================================
echo 1. Install with Miniconda (Recommended)
echo 2. Install with Python Virtual Environment
echo 3. Install directly to system Python
echo 4. Exit
echo.
set /p choice="Select installation method [1-4]: "

if "%choice%"=="1" goto miniconda
if "%choice%"=="2" goto venv
if "%choice%"=="3" goto system
if "%choice%"=="4" exit /b

echo Invalid selection, please try again
goto menu

:: =============================================
:: Option 1: Install with Miniconda
:: =============================================
:miniconda
cls
echo =============================================
echo       Installing with Miniconda
echo =============================================

:: Check if Conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Conda is already installed
    goto conda_env
)

echo [INFO] Miniconda will be installed
echo.

:: Download Miniconda
echo [Step 1/4] Downloading Miniconda...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe', '%TEMP%\Miniconda3.exe')"
if not exist "%TEMP%\Miniconda3.exe" (
    echo [ERROR] Miniconda download failed
    pause
    goto menu
)

:: Install Miniconda
echo [Step 2/4] Installing Miniconda...
echo Installation path: %USERPROFILE%\Miniconda3
start /wait "" "%TEMP%\Miniconda3.exe" /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=%USERPROFILE%\Miniconda3
del "%TEMP%\Miniconda3.exe"

echo [Step 3/4] Miniconda installed successfully!
echo Please restart your command prompt or computer
echo Then run this script again to complete setup
pause
exit /b

:: Create Conda environment
:conda_env
set ENV_NAME=isac
set PYTHON_VERSION=3.11

echo [Step 4/4] Checking environment...
conda env list | findstr /b /c:"%ENV_NAME%" >nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Environment '%ENV_NAME%' already exists
    set /p RECREATE="Recreate environment? [y/N]: "
    if /i "!RECREATE!"=="y" (
        conda env remove -n %ENV_NAME% -y
        conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    )
) else (
    echo [Step 4/4] Creating environment '%ENV_NAME%'...
    conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Environment creation failed
    pause
    goto menu
)

:: Install dependencies
echo Installing project dependencies...
call :install_dependencies "conda run -n %ENV_NAME%"

echo.
echo =============================================
echo Environment setup complete!
echo To activate the environment, run:
echo conda activate %ENV_NAME%
echo =============================================
pause
goto menu

:: =============================================
:: Option 2: Install with Python Virtual Environment
:: =============================================
:venv
cls
echo =============================================
echo     Installing with Python Virtual Environment
echo =============================================

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.7+
    pause
    goto menu
)

:: Check Python version
python --version
python -c "import sys; sys.exit(0) if sys.version_info >= (3,7) else sys.exit(1)"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.7 or higher is required
    pause
    goto menu
)

:: Create virtual environment
set VENV_NAME=isac-venv
set VENV_DIR="%cd%\%VENV_NAME%"

if exist %VENV_DIR% (
    echo [INFO] Virtual environment already exists: %VENV_DIR%
    set /p RECREATE="Recreate environment? [y/N]: "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q %VENV_DIR% 2>nul
        python -m venv %VENV_DIR%
    )
) else (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Virtual environment creation failed
    pause
    goto menu
)

:: Install dependencies
call :install_dependencies "call %VENV_DIR%\Scripts\activate.bat &&"

echo.
echo =============================================
echo Environment setup complete!
echo To activate the environment, run:
echo %VENV_DIR%\Scripts\activate
echo =============================================
pause
goto menu

:: =============================================
:: Option 3: Install directly to system Python
:: =============================================
:system
cls
echo =============================================
echo   Install directly to system Python
echo =============================================
echo [WARNING] This will modify your system Python
echo          and may affect other projects
echo.
set /p CONFIRM="Are you sure? [y/N]: "
if /i not "!CONFIRM!"=="y" goto menu

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found
    pause
    goto menu
)

:: Install dependencies
call :install_dependencies ""

echo.
echo =============================================
echo Dependencies installed successfully!
echo =============================================
pause
goto menu

:: =============================================
:: Common function to install dependencies
:: =============================================
:install_dependencies
set ACTIVATE_CMD=%1

echo.
echo Configuring Tsinghua PyPI mirror...
%ACTIVATE_CMD% pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
%ACTIVATE_CMD% pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

echo Upgrading pip...
%ACTIVATE_CMD% python -m pip install --upgrade pip

if exist "requirements.txt" (
    echo Installing from requirements.txt...
    
    :: Create fixed requirements
    echo numpy==1.26.4 > requirements_fixed.txt
    echo scipy==1.13.0 >> requirements_fixed.txt
    type requirements.txt >> requirements_fixed.txt
    
    %ACTIVATE_CMD% python -m pip install -r requirements_fixed.txt
    del requirements_fixed.txt
) else (
    echo Installing core dependencies...
    %ACTIVATE_CMD% python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
    %ACTIVATE_CMD% python -m pip install gymnasium==1.2.0 stable-baselines3==2.3.0
    %ACTIVATE_CMD% python -m pip install numpy==1.26.4 pandas==2.3.1 matplotlib==3.10.3
    %ACTIVATE_CMD% python -m pip install wandb==0.21.0 optuna==3.6.1 pyyaml==6.0.2
)

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation failed
    pause
    goto menu
)

echo.
echo Verifying installation:
%ACTIVATE_CMD% python -c "import torch; print('PyTorch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
%ACTIVATE_CMD% python -c "import gymnasium; print('Gymnasium version:', gymnasium.__version__)"
%ACTIVATE_CMD% python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
%ACTIVATE_CMD% python -c "try: import scipy; print('SciPy version:', scipy.__version__); except: print('SciPy not installed')"

exit /b
