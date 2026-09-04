#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/component_ablation_segmentation_config.yaml}"
if [[ $# -ge 1 ]]; then shift; fi
NUM_GPUS="${NUM_GPUS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
    train_segmentation_compare.py \
    --config "$CONFIG" \
    "$@"
