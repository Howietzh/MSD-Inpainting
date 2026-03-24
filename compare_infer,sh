#!/bin/bash

set -euo pipefail

# 定义基础路径
BASE_LORA_DIR="/home/doctor/tzh/MSD-Inpainting/defectfill_lora_weights"
BASE_OUTPUT_DIR="/home/doctor/tzh/MSD-Inpainting/data/CCM-Defect/generated_defect_dataset"

# 循环从 0 开始，到 200 结束，步长为 20
for epoch in {0..200..20}; do
    echo "=========================================================="
    echo "  🚀 开始进行推理验证 - Epoch: ${epoch} "
    echo "=========================================================="

    # 构建当前 epoch 的权重路径和输出路径
    CURRENT_LORA_WEIGHTS="${BASE_LORA_DIR}/checkpoint-epoch-${epoch}"
    # 为了方便对比，将结果保存在单独的子文件夹中，例如 /generated_defect_dataset/epoch_20
    CURRENT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/epoch_${epoch}"

    # 检查权重文件夹是否存在，如果不存在则跳过 (比如某次训练中断没生成对应的 epoch)
    if [ ! -d "$CURRENT_LORA_WEIGHTS" ]; then
        echo "⚠️  警告: 找不到权重文件夹 ${CURRENT_LORA_WEIGHTS}，正在跳过..."
        continue
    fi

    CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
        --multi_gpu \
        --num_processes=2 \
        --num_machines=1 \
        --mixed_precision="fp16" \
        --dynamo_backend="no" \
        inference.py \
        --lora_weights "$CURRENT_LORA_WEIGHTS" \
        --output_dir "$CURRENT_OUTPUT_DIR"

    echo "✅ Epoch ${epoch} 推理完成! 结果保存在: ${CURRENT_OUTPUT_DIR}"
    echo ""
done

echo "🎉 所有推理任务执行完毕，可以前往 ${BASE_OUTPUT_DIR} 对比不同 epoch 的生成效果了。"