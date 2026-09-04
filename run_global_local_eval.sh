#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    cat <<'EOF'
Usage: bash run_global_local_eval.sh GENERATED_ROOT REPORT_ROOT [run_eval options]

GENERATED_ROOT must contain one subdirectory per method or ablation experiment.
Each experiment directory must contain images/, defect_masks/, and metadata_gpu*.jsonl.

Example:
  bash run_global_local_eval.sh \
    /data/CCM-Defect/generation_experiments \
    /data/CCM-Defect/eval_reports/global_local \
    --set generation.feature_batch_size=16 \
    --set generation.local_padding_ratio=0.25
EOF
    exit 2
fi

GENERATED_ROOT="$1"
REPORT_ROOT="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$REPORT_ROOT"

bash "$SCRIPT_DIR/run_eval.sh" \
    --all-experiments \
    --generated-root "$GENERATED_ROOT" \
    --report-root "$REPORT_ROOT" \
    "$@"

python "$SCRIPT_DIR/summarize_evaluation.py" \
    --report-dir "$REPORT_ROOT" \
    --output "$REPORT_ROOT/global_local_metrics.csv"

echo "JSON reports: $REPORT_ROOT"
echo "Paper-ready CSV: $REPORT_ROOT/global_local_metrics.csv"
