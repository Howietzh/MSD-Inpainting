#!/bin/bash

set -euo pipefail

TRAIN_CONFIG="configs/train_config.yaml"
INFER_CONFIG="configs/inference_config.yaml"
WEIGHTS_ROOT=""
OUTPUT_ROOT=""
CHECKPOINT_REL=""
RUN_ALL_EXPERIMENTS=0
EXPERIMENT_NAMES=("defectfill_origin" "increase_text_encoder_learning_rates")
INFER_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash run_infer.sh [options] [inference overrides...]

Modes:
  1. Single experiment:
     bash run_infer.sh --exp-name exp_a
  2. Multiple experiments:
     bash run_infer.sh --exp-name exp_a --exp-name exp_b
  3. All experiments under the weights root:
     bash run_infer.sh --all-experiments

Options:
  --exp-name NAME        Experiment subdirectory name under the weights root. Repeatable.
  --all-experiments      Run inference for every experiment directory under the weights root.
  --checkpoint RELPATH   Optional checkpoint subdirectory under each experiment, e.g. checkpoint-epoch-90.
  --train-config PATH    Training config path. Default: configs/train_config.yaml
  --infer-config PATH    Inference config path. Default: configs/inference_config.yaml
  --weights-root PATH    Override the base directory containing experiment weight folders.
  --output-root PATH     Override the base directory for batched inference outputs.
  --help                 Show this help message.

Additional arguments are passed through to inference.py. For example:
  bash run_infer.sh --exp-name exp_a \
    --set inference.batch_size=4 \
    --set inference.lpips_backbone=alex
EOF
}

read_yaml_value() {
    python - "$1" "$2" <<'PY'
import sys
import yaml

config_path, dotted_key = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

value = data
for part in dotted_key.split("."):
    value = value[part]
print(value)
PY
}

while (($#)); do
    case "$1" in
        --exp-name)
            EXPERIMENT_NAMES+=("$2")
            shift 2
            ;;
        --all-experiments)
            RUN_ALL_EXPERIMENTS=1
            shift
            ;;
        --checkpoint)
            CHECKPOINT_REL="$2"
            shift 2
            ;;
        --train-config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --infer-config)
            INFER_CONFIG="$2"
            shift 2
            ;;
        --weights-root)
            WEIGHTS_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            INFER_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$WEIGHTS_ROOT" ]]; then
    WEIGHTS_ROOT="$(read_yaml_value "$TRAIN_CONFIG" "paths.output_dir")"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="$(read_yaml_value "$INFER_CONFIG" "paths.output_dir")"
fi

if [[ "$RUN_ALL_EXPERIMENTS" -eq 1 ]]; then
    while IFS= read -r exp_dir; do
        EXPERIMENT_NAMES+=("$(basename "$exp_dir")")
    done < <(find "$WEIGHTS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#EXPERIMENT_NAMES[@]} -eq 0 ]]; then
    default_lora_path="$(read_yaml_value "$INFER_CONFIG" "paths.lora_weights")"
    default_output_path="$(read_yaml_value "$INFER_CONFIG" "paths.output_dir")"

    echo "=========================================================="
    echo "Single inference run from config"
    echo "Infer cfg : ${INFER_CONFIG}"
    echo "Weights   : ${default_lora_path}"
    echo "Output    : ${default_output_path}"
    echo "=========================================================="

    CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
        --multi_gpu \
        --num_processes=2 \
        --num_machines=1 \
        --mixed_precision="fp16" \
        --dynamo_backend="no" \
        inference.py \
        --config "$INFER_CONFIG" \
        "${INFER_EXTRA_ARGS[@]}"
    exit 0
fi

for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    lora_dir="${WEIGHTS_ROOT%/}/${exp_name}"
    output_dir="${OUTPUT_ROOT%/}/${exp_name}"

    if [[ -n "$CHECKPOINT_REL" ]]; then
        lora_dir="${lora_dir%/}/${CHECKPOINT_REL}"
        output_dir="${output_dir%/}/${CHECKPOINT_REL}"
    fi

    if [[ ! -d "$lora_dir" ]]; then
        echo "⚠️  跳过 ${exp_name}: 找不到权重目录 ${lora_dir}"
        continue
    fi

    echo "=========================================================="
    echo "Experiment : ${exp_name}"
    echo "Infer cfg  : ${INFER_CONFIG}"
    echo "Weights    : ${lora_dir}"
    echo "Output     : ${output_dir}"
    echo "=========================================================="

    CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
        --multi_gpu \
        --num_processes=2 \
        --num_machines=1 \
        --mixed_precision="fp16" \
        --dynamo_backend="no" \
        inference.py \
        --config "$INFER_CONFIG" \
        --lora_weights "$lora_dir" \
        --output_dir "$output_dir" \
        --set "paths.model_config=${TRAIN_CONFIG}" \
        "${INFER_EXTRA_ARGS[@]}"
done
