#!/bin/bash

set -euo pipefail

TRAIN_CONFIG="configs/train_config.yaml"
INFER_CONFIG="configs/inference_config.yaml"
LORA_WEIGHTS=""
NORMAL_DIR=""
STATS_CACHE=""
SERVER_NAME="0.0.0.0"
SERVER_PORT="7860"
GRADIO_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash run_gradio.sh [options] [gradio extra args...]

Options:
  --train-config PATH    Training config path. Default: configs/train_config.yaml
  --infer-config PATH    Inference config path. Default: configs/inference_config.yaml
  --lora-weights PATH    Override LoRA weights directory.
  --normal-dir PATH      Override normal component directory.
  --stats-cache PATH     Override defect stats cache path.
  --server-name HOST     Gradio server host. Default: 0.0.0.0
  --server-port PORT     Gradio server port. Default: 7860
  --help                 Show this help message.

Examples:
  bash run_gradio.sh
  bash run_gradio.sh --lora-weights /path/to/checkpoint-epoch-10
  bash run_gradio.sh --server-port 7861 --device cpu
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
        --train-config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --infer-config)
            INFER_CONFIG="$2"
            shift 2
            ;;
        --lora-weights)
            LORA_WEIGHTS="$2"
            shift 2
            ;;
        --normal-dir)
            NORMAL_DIR="$2"
            shift 2
            ;;
        --stats-cache)
            STATS_CACHE="$2"
            shift 2
            ;;
        --server-name)
            SERVER_NAME="$2"
            shift 2
            ;;
        --server-port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            GRADIO_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$LORA_WEIGHTS" ]]; then
    LORA_WEIGHTS="$(read_yaml_value "$INFER_CONFIG" "paths.lora_weights")"
fi

if [[ -z "$NORMAL_DIR" ]]; then
    NORMAL_DIR="$(read_yaml_value "$INFER_CONFIG" "paths.normal_dir")"
fi

if [[ -z "$STATS_CACHE" ]]; then
    STATS_CACHE="$(read_yaml_value "$INFER_CONFIG" "paths.stats_cache")"
fi

echo "=========================================================="
echo "Gradio Demo"
echo "Train cfg   : ${TRAIN_CONFIG}"
echo "Infer cfg   : ${INFER_CONFIG}"
echo "LoRA        : ${LORA_WEIGHTS}"
echo "Normal dir  : ${NORMAL_DIR}"
echo "Stats cache : ${STATS_CACHE}"
echo "Server      : ${SERVER_NAME}:${SERVER_PORT}"
echo "=========================================================="

python app_gradio.py \
    --train-config "$TRAIN_CONFIG" \
    --infer-config "$INFER_CONFIG" \
    --lora-weights "$LORA_WEIGHTS" \
    --normal-dir "$NORMAL_DIR" \
    --stats-cache "$STATS_CACHE" \
    --server-name "$SERVER_NAME" \
    --server-port "$SERVER_PORT" \
    "${GRADIO_EXTRA_ARGS[@]}"
