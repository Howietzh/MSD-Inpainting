#!/bin/bash

set -euo pipefail

CONFIG="${1:-configs/segmentation_ablation_config.yaml}"
shift || true

NUM_GPUS="${NUM_GPUS:-2}"

python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
    train_segmentation_ablation.py \
    --config "$CONFIG" \
    "$@"
