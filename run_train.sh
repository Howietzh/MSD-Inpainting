#!/bin/bash

set -euo pipefail

TRAIN_CONFIG="configs/train_config.yaml"
INFER_CONFIG="configs/inference_config.yaml"
RUN_INFERENCE=1
EXP_NAME="defect_focused_denoising_loss_and_defect_focused_attention_loss_and_componet_attention_loss"
TRAIN_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash run_train.sh [options] [train overrides...]

Options:
  --exp-name NAME        Experiment name. Defaults to a timestamp.
  --train-config PATH    Training config path. Default: configs/train_config.yaml
  --infer-config PATH    Inference config path. Default: configs/inference_config.yaml
  --skip-infer           Skip the post-training inference run.
  --help                 Show this help message.

Additional arguments are passed through to train.py. For example:
  bash run_train.sh --exp-name micro_a \
    --set loss_weights.lambda_rec=1.2 \
    --set 'loss_weights.defect_class_weights.<foreign_particle>=2.5'
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
            EXP_NAME="$2"
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
        --skip-infer)
            RUN_INFERENCE=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            TRAIN_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="$(date +%Y%m%d_%H%M%S)"
fi

TRAIN_OUTPUT_BASE="$(read_yaml_value "$TRAIN_CONFIG" "paths.output_dir")"
TRAIN_LOG_BASE="$(read_yaml_value "$TRAIN_CONFIG" "paths.logging_dir")"
INFER_OUTPUT_BASE="$(read_yaml_value "$INFER_CONFIG" "paths.output_dir")"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_BASE%/}/${EXP_NAME}"
TRAIN_LOG_DIR="${TRAIN_LOG_BASE%/}/${EXP_NAME}"
INFER_OUTPUT_DIR="${INFER_OUTPUT_BASE%/}/${EXP_NAME}"

echo "=========================================================="
echo "Experiment : ${EXP_NAME}"
echo "Train cfg  : ${TRAIN_CONFIG}"
echo "Infer cfg  : ${INFER_CONFIG}"
echo "Weights    : ${TRAIN_OUTPUT_DIR}"
echo "Logs       : ${TRAIN_LOG_DIR}"
if [[ "$RUN_INFERENCE" -eq 1 ]]; then
    echo "Inference  : ${INFER_OUTPUT_DIR}"
else
    echo "Inference  : skipped"
fi
echo "=========================================================="

CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision="fp16" \
    --dynamo_backend="no" \
    train.py \
    --config "$TRAIN_CONFIG" \
    --set "paths.output_dir=${TRAIN_OUTPUT_DIR}" \
    --set "paths.logging_dir=${TRAIN_LOG_DIR}" \
    "${TRAIN_EXTRA_ARGS[@]}"

if [[ "$RUN_INFERENCE" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
        --multi_gpu \
        --num_processes=2 \
        --num_machines=1 \
        --mixed_precision="fp16" \
        --dynamo_backend="no" \
        inference.py \
        --config "$INFER_CONFIG" \
        --lora_weights "$TRAIN_OUTPUT_DIR" \
        --output_dir "$INFER_OUTPUT_DIR" \
        --set "paths.model_config=${TRAIN_CONFIG}"
fi
