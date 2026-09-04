#!/bin/bash

set -euo pipefail

CONFIG="${1:-configs/measurement_eval_config.yaml}"
shift || true

python evaluate_measurements.py \
    --config "$CONFIG" \
    "$@"
