#!/bin/bash

set -euo pipefail

CONFIG="${1:-configs/segmentation_config.yaml}"
shift || true

python train_segmentation_compare.py --config "$CONFIG" "$@"
