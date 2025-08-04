@echo off
rem 设置环境变量
set PYTHONPATH=..;%PYTHONPATH%
set CUDA_VISIBLE_DEVICES=0

rem 检查量化模型是否存在
if not exist "models\quantized_seg_model.pt" (
    echo Quantized model not found, creating it now...
    python quantize_model.py
)

echo.
echo 请选择推理模式：
echo 1 - 批量推理整个文件夹
echo 2 - 随机推理文件夹中的一张图片
set /p mode_choice=请输入选项数字（1或2）：
if "%mode_choice%"=="1" (
    set mode=all
) else if "%mode_choice%"=="2" (
    set mode=random
) else (
    echo 输入无效，默认选择批量推理整个文件夹。
    set mode=all
)

set img_path=data/combined_dataset/images/tier3/
set model_type=original
set csv_path=inference/perf_report.csv

python run_inference.py --img_path %img_path% --model_type %model_type% --csv_path %csv_path% --mode %mode%

echo 推理流程已完成。
