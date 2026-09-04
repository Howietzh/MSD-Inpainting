import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = (
    "global_kid",
    "local_kid",
    "global_ic_lpips",
    "local_ic_lpips",
)


def display_experiment_name(name: str):
    aliases = {
        "defectfill_origin": "DefectFill",
        "dfmgan_ccms": "DFMGAN",
        "seas": "SeaS",
    }
    if name.startswith("increase_text_encoder_learning_r"):
        return "MSD-Inpainting"
    return aliases.get(name, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect global/local generation metrics from evaluation JSON reports."
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Only summarize these report stems, in the given order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(report_path: Path):
    with open(report_path, "r", encoding="utf-8") as f:
        generation = json.load(f)["generation"]

    rows = []
    for task_key, task in sorted(generation["tasks"].items()):
        rows.append(
            {
                "experiment": display_experiment_name(report_path.stem),
                "task": task_key,
                **{field: task[field] for field in METRIC_FIELDS},
                "num_real": task["num_real"],
                "num_fake": task["num_fake"],
            }
        )

    summary = generation["summary"]
    rows.append(
        {
            "experiment": display_experiment_name(report_path.stem),
            "task": "MEAN",
            **{field: summary[f"{field}_mean"] for field in METRIC_FIELDS},
            "num_real": summary["num_real_total"],
            "num_fake": summary["num_fake_total"],
        }
    )
    return rows


def main():
    args = parse_args()
    if args.experiments:
        report_paths = [args.report_dir / f"{name}.json" for name in args.experiments]
        missing = [path for path in report_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing evaluation reports: {missing}")
    else:
        report_paths = sorted(args.report_dir.glob("*.json"))
    if not report_paths:
        raise FileNotFoundError(f"No evaluation JSON reports found under {args.report_dir}")

    rows = []
    for report_path in report_paths:
        rows.extend(load_rows(report_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "task", *METRIC_FIELDS, "num_real", "num_fake"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
