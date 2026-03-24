#!/bin/bash

set -euo pipefail

TRAIN_EXTRA_ARGS=("$@")

CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision="fp16" \
    --dynamo_backend="no" \
    train.py \
    --config configs/train_config.yaml \
    "${TRAIN_EXTRA_ARGS[@]}"

CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision="fp16" \
    --dynamo_backend="no" \
    inference.py
