#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data/CCM-Defect}"
GENERATED_ROOT="${GENERATED_ROOT:-$DATA_ROOT/generated_defect_dataset}"
REPORT_ROOT="${REPORT_ROOT:-$DATA_ROOT/eval_reports/component_ablation}"
SEGMENTATION_RESULTS="${SEGMENTATION_RESULTS:-$SCRIPT_DIR/segmentation_results/component_ablation/comparison.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

experiments=(
    component_ablation_full
    component_ablation_no_dsl
    component_ablation_no_dmaa
    component_ablation_no_cdme
    component_ablation_no_ti
)

for experiment in "${experiments[@]}"; do
    generated_dir="${GENERATED_ROOT%/}/${experiment}"
    if [[ ! -d "$generated_dir/images" || ! -d "$generated_dir/defect_masks" ]]; then
        echo "Missing generated images or masks: $generated_dir" >&2
        exit 2
    fi
done

mkdir -p "$REPORT_ROOT"
eval_args=()
for experiment in "${experiments[@]}"; do
    eval_args+=(--exp-name "$experiment")
done

bash "$SCRIPT_DIR/run_eval.sh" \
    "${eval_args[@]}" \
    --generated-root "$GENERATED_ROOT" \
    --report-root "$REPORT_ROOT" \
    --set "paths.real_holdout_dir=$DATA_ROOT/defect_test_holdout" \
    "$@"

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_evaluation.py" \
    --report-dir "$REPORT_ROOT" \
    --experiments "${experiments[@]}" \
    --output "$REPORT_ROOT/generation_metrics.csv"

if [[ -f "$SEGMENTATION_RESULTS" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/summarize_component_ablation.py" \
        --generation-report-dir "$REPORT_ROOT" \
        --segmentation-report "$SEGMENTATION_RESULTS" \
        --output-csv "$REPORT_ROOT/table4_component_ablation.csv" \
        --output-tex "$REPORT_ROOT/table4_component_ablation_rows.tex"
else
    echo "Generation metrics completed."
    echo "Run run_component_ablation_segmentation.sh, then rerun this script to build Table 4."
fi
