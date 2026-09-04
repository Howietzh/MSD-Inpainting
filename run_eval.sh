#!/bin/bash

set -euo pipefail

EVAL_CONFIG="configs/eval_config.yaml"
INFER_CONFIG="configs/inference_config.yaml"
GENERATED_ROOT=""
REPORT_ROOT=""
CHECKPOINT_REL=""
RUN_ALL_EXPERIMENTS=0
EXPERIMENT_NAMES=()
EVAL_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash run_eval.sh [options] [evaluation overrides...]

Modes:
  1. Single experiment:
     bash run_eval.sh --exp-name exp_a
  2. Multiple experiments:
     bash run_eval.sh --exp-name exp_a --exp-name exp_b
  3. All experiments under the generated root:
     bash run_eval.sh --all-experiments
  4. Single evaluation run from config:
     bash run_eval.sh

Options:
  --exp-name NAME        Experiment subdirectory name under the generated root. Repeatable.
  --all-experiments      Run evaluation for every experiment directory under the generated root.
  --checkpoint RELPATH   Optional checkpoint subdirectory under each experiment, e.g. checkpoint-epoch-90.
  --eval-config PATH     Evaluation config path. Default: configs/eval_config.yaml
  --infer-config PATH    Inference config path. Default: configs/inference_config.yaml
  --generated-root PATH  Override the base directory containing generated experiment outputs.
  --report-root PATH     Override the base directory for evaluation reports.
  --help                 Show this help message.

Every run reports both full-image (global) and defect-ROI (local) KID/IC-LPIPS.
Additional arguments are passed through to evaluate.py. For example:
  bash run_eval.sh --exp-name exp_a \
    --set generation.feature_batch_size=8 \
    --set generation.lpips_batch_size=16 \
    --set generation.local_padding_ratio=0.25
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
            EXPERIMENT_NAMES+=("$2")
            shift 2
            ;;
        --all-experiments)
            RUN_ALL_EXPERIMENTS=1
            shift
            ;;
        --checkpoint)
            CHECKPOINT_REL="$2"
            shift 2
            ;;
        --eval-config)
            EVAL_CONFIG="$2"
            shift 2
            ;;
        --infer-config)
            INFER_CONFIG="$2"
            shift 2
            ;;
        --generated-root)
            GENERATED_ROOT="$2"
            shift 2
            ;;
        --report-root)
            REPORT_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            EVAL_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$GENERATED_ROOT" ]]; then
    GENERATED_ROOT="$(read_yaml_value "$INFER_CONFIG" "paths.output_dir")"
fi

if [[ -z "$REPORT_ROOT" ]]; then
    default_report_path="$(read_yaml_value "$EVAL_CONFIG" "paths.report_path")"
    REPORT_ROOT="$(dirname "$default_report_path")"
fi

if [[ "$RUN_ALL_EXPERIMENTS" -eq 1 ]]; then
    while IFS= read -r exp_dir; do
        EXPERIMENT_NAMES+=("$(basename "$exp_dir")")
    done < <(find "$GENERATED_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#EXPERIMENT_NAMES[@]} -eq 0 ]]; then
    default_generated_dir="$(read_yaml_value "$EVAL_CONFIG" "paths.generated_dir")"
    default_report_path="$(read_yaml_value "$EVAL_CONFIG" "paths.report_path")"

    echo "=========================================================="
    echo "Single evaluation run from config"
    echo "Eval cfg   : ${EVAL_CONFIG}"
    echo "Generated  : ${default_generated_dir}"
    echo "Report     : ${default_report_path}"
    echo "=========================================================="

    python evaluate.py \
        --config "$EVAL_CONFIG" \
        "${EVAL_EXTRA_ARGS[@]}"
    exit 0
fi

for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    generated_dir="${GENERATED_ROOT%/}/${exp_name}"
    report_path="${REPORT_ROOT%/}/${exp_name}.json"

    if [[ -n "$CHECKPOINT_REL" ]]; then
        generated_dir="${generated_dir%/}/${CHECKPOINT_REL}"
        report_suffix="${CHECKPOINT_REL//\//__}"
        report_path="${REPORT_ROOT%/}/${exp_name}__${report_suffix}.json"
    fi

    if [[ ! -d "$generated_dir" ]]; then
        echo "⚠️  跳过 ${exp_name}: 找不到生成结果目录 ${generated_dir}"
        continue
    fi

    echo "=========================================================="
    echo "Experiment : ${exp_name}"
    echo "Eval cfg   : ${EVAL_CONFIG}"
    echo "Generated  : ${generated_dir}"
    echo "Report     : ${report_path}"
    echo "=========================================================="

    python evaluate.py \
        --config "$EVAL_CONFIG" \
        --set "paths.generated_dir=${generated_dir}" \
        --set "paths.report_path=${report_path}" \
        "${EVAL_EXTRA_ARGS[@]}"
done
