#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_config.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
PYTHON_BIN="${PYTHON_BIN:-python}"

read_yaml_value() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    value = yaml.safe_load(stream)
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

WEIGHTS_ROOT="${WEIGHTS_ROOT:-$(read_yaml_value "$TRAIN_CONFIG" paths.output_dir)}"
LOG_ROOT="${LOG_ROOT:-$(read_yaml_value "$TRAIN_CONFIG" paths.logging_dir)}"

train_variant() {
    local name="$1"
    shift
    local output_dir="${WEIGHTS_ROOT%/}/${name}"
    local log_dir="${LOG_ROOT%/}/${name}"

    echo "=========================================================="
    echo "Training component ablation: $name"
    echo "Weights: $output_dir"
    echo "Logs   : $log_dir"
    echo "=========================================================="

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$ACCELERATE_BIN" launch \
        --multi_gpu \
        --num_processes="$NUM_PROCESSES" \
        --num_machines=1 \
        --mixed_precision=fp16 \
        --dynamo_backend=no \
        train.py \
        --config "$TRAIN_CONFIG" \
        --set "paths.output_dir=${output_dir}" \
        --set "paths.logging_dir=${log_dir}" \
        "$@"
}

train_variant component_ablation_full
train_variant component_ablation_no_dsl \
    --set ablation.use_defect_sensitive_loss=false
train_variant component_ablation_no_dmaa \
    --set ablation.use_dual_mask_attention=false \
    --set loss_weights.lambda_attn_def=0.0 \
    --set loss_weights.lambda_attn_comp=0.0
train_variant component_ablation_no_ti \
    --set ablation.use_textual_inversion=false

echo "w/o CDME is inference-only and reuses component_ablation_full weights."
