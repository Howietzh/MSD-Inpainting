#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$SCRIPT_DIR/data/CCM-Defect"
GENERATED_ROOT="${1:-$DATA_ROOT/generated_defect_dataset}"
REPORT_ROOT="${2:-$DATA_ROOT/eval_reports/global_local_four_methods}"
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

cd "$SCRIPT_DIR"
mkdir -p "$REPORT_ROOT"

methods=(
    "dfmgan_ccms"
    "defectfill_origin"
    "seas"
    "increase_text_encoder_learning_rates"
)
for method in "${methods[@]}"; do
    method_dir="$GENERATED_ROOT/$method"
    if [[ ! -d "$method_dir/images" || ! -d "$method_dir/defect_masks" ]]; then
        echo "Missing images/ or defect_masks/ for $method: $method_dir"
        exit 2
    fi
    if ! compgen -G "$method_dir/metadata_gpu*.jsonl" >/dev/null; then
        echo "Missing metadata_gpu*.jsonl for $method: $method_dir"
        exit 2
    fi
done

eval_args=()
for method in "${methods[@]}"; do
    eval_args+=(--exp-name "$method")
done

bash "$SCRIPT_DIR/run_eval.sh" \
    "${eval_args[@]}" \
    --generated-root "$GENERATED_ROOT" \
    --report-root "$REPORT_ROOT" \
    --set "paths.real_holdout_dir=$DATA_ROOT/defect_test_holdout" \
    "$@"

python "$SCRIPT_DIR/summarize_evaluation.py" \
    --report-dir "$REPORT_ROOT" \
    --experiments "${methods[@]}" \
    --output "$REPORT_ROOT/global_local_metrics.csv"

echo "Compared methods: ${methods[*]}"
echo "JSON reports: $REPORT_ROOT"
echo "Paper-ready CSV: $REPORT_ROOT/global_local_metrics.csv"
