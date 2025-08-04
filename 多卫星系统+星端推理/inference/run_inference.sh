#!/bin/bash

# 设置环境
export PYTHONPATH=$PWD/..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 检查量化模型是否存在，不存在则创建
if [ ! -f "models/quantized_seg_model.pt" ]; then
    echo "Quantized model not found, creating it now..."
    python quantize_model.py
fi

echo
echo "请选择推理模式："
echo "1 - 批量推理整个文件夹"
echo "2 - 随机推理文件夹中的一张图片"
read -p "请输入选项数字（1或2）：" mode_choice
if [ "$mode_choice" = "1" ]; then
    mode=all
elif [ "$mode_choice" = "2" ]; then
    mode=random
else
    echo "输入无效，默认选择批量推理整个文件夹。"
    mode=all
fi

img_path=data/combined_dataset/images/tier3/
model_type=original
csv_path=inference/perf_report.csv

python run_inference.py --img_path $img_path --model_type $model_type --csv_path $csv_path --mode $mode

echo "推理流程已完成。"