import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("component_ablation_full", "Full MSD-Inpainting"),
    ("component_ablation_no_dsl", "w/o Defect-Sensitive Loss (DSL)"),
    ("component_ablation_no_dmaa", "w/o Dual Mask-Guided Attention (DMAA)"),
    ("component_ablation_no_cdme", "w/o Component-Aware Defect Mask Engine (CDME)"),
    ("component_ablation_no_ti", "w/o Textual Inversion (TI)"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine generation and segmentation metrics into paper Table 4."
    )
    parser.add_argument("--generation-report-dir", type=Path, required=True)
    parser.add_argument("--segmentation-report", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    return parser.parse_args()


def load_generation_summary(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)["generation"]["summary"]


def load_segmentation_summary(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)["summary"]


def main():
    args = parse_args()
    segmentation = load_segmentation_summary(args.segmentation_report)
    rows = []
    for experiment, label in EXPERIMENTS:
        generation_path = args.generation_report_dir / f"{experiment}.json"
        if not generation_path.exists():
            raise FileNotFoundError(f"Missing generation report: {generation_path}")
        if experiment not in segmentation:
            raise KeyError(f"Missing segmentation results for {experiment}")

        generation = load_generation_summary(generation_path)
        segment = segmentation[experiment]
        rows.append(
            {
                "experiment": experiment,
                "variant": label,
                "global_kid": generation["global_kid_mean"],
                "local_kid": generation["local_kid_mean"],
                "global_ic_lpips": generation["global_ic_lpips_mean"],
                "local_ic_lpips": generation["local_ic_lpips_mean"],
                "foreground_miou_mean": segment["mIoU_foreground"]["mean"],
                "foreground_miou_std": segment["mIoU_foreground"]["std"],
                "recall_mean": segment["Recall"]["mean"],
                "recall_std": segment["Recall"]["std"],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latex_rows = []
    for row in rows:
        latex_rows.append(
            f"{row['variant']} & {row['global_kid']:.2f} & {row['local_kid']:.2f} & "
            f"{row['global_ic_lpips']:.3f} & {row['local_ic_lpips']:.3f} & "
            f"{row['foreground_miou_mean']:.4f} $\\pm$ {row['foreground_miou_std']:.4f} & "
            f"{row['recall_mean']:.4f} $\\pm$ {row['recall_std']:.4f} \\\\"
        )
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
