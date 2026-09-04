#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_config.yaml}"
INFER_CONFIG="${INFER_CONFIG:-configs/inference_config.yaml}"
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
GENERATED_ROOT="${GENERATED_ROOT:-$(read_yaml_value "$INFER_CONFIG" paths.output_dir)}"
FULL_WEIGHTS_NAME="${FULL_WEIGHTS_NAME:-component_ablation_full}"

infer_variant() {
    local output_name="$1"
    local weights_name="$2"
    local mask_strategy="$3"
    local require_resolved_config="${4:-0}"
    local weights_dir="${WEIGHTS_ROOT%/}/${weights_name}"
    local output_dir="${GENERATED_ROOT%/}/${output_name}"
    local resolved_config="${weights_dir%/}/resolved_train_config.yaml"

    if [[ ! -f "$resolved_config" ]]; then
        if [[ "$require_resolved_config" -eq 1 ]]; then
            echo "Missing resolved training config required by $output_name: $resolved_config" >&2
            echo "Retrain this variant with run_component_ablation_train.sh." >&2
            exit 2
        fi
        resolved_config="$TRAIN_CONFIG"
        echo "Warning: legacy weights have no resolved_train_config.yaml; using $TRAIN_CONFIG"
    fi

    echo "=========================================================="
    echo "Inference component ablation: $output_name"
    echo "Weights      : $weights_dir"
    echo "Mask strategy: $mask_strategy"
    echo "Output       : $output_dir"
    echo "=========================================================="

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$ACCELERATE_BIN" launch \
        --multi_gpu \
        --num_processes="$NUM_PROCESSES" \
        --num_machines=1 \
        --mixed_precision=fp16 \
        --dynamo_backend=no \
        inference.py \
        --config "$INFER_CONFIG" \
        --lora_weights "$weights_dir" \
        --output_dir "$output_dir" \
        --set "paths.model_config=${resolved_config}" \
        --set "inference.mask_strategy=${mask_strategy}"
}

infer_variant component_ablation_full "$FULL_WEIGHTS_NAME" cdme 0
infer_variant component_ablation_no_dsl component_ablation_no_dsl cdme 0
infer_variant component_ablation_no_dmaa component_ablation_no_dmaa cdme 0
infer_variant component_ablation_no_cdme "$FULL_WEIGHTS_NAME" reference_elastic 0
infer_variant component_ablation_no_ti component_ablation_no_ti cdme 1
