from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.segmentation_ablation_dataset import AblationSegmentationDataset
from dataset.segmentation_dataset import (
    COMPONENT_DEFECT_CLASS_MAP,
    NUM_CLASSES,
    SegmentationTransform,
    read_metadata,
    task_key,
)
from models.mobilevit_unet import MobileViTUNet
from utils.config_overrides import apply_config_overrides
from utils.defect_measurements import measure_defect_mask, quantities_for_task, summarize_errors


TASK_LABELS = {
    "<flexible_printed_circuit_crack>::<flexible_printed_circuit>": "FPC crack",
    "<end_face_scratch>::<end_face>": "End-face scratch",
    "<lens_scratch>::<lens>": "Lens scratch",
    "<foreign_particle>::<end_face>": "End-face foreign particle",
    "<foreign_particle>::<lens>": "Lens foreign particle",
}
TASK_ORDER = list(TASK_LABELS)
QUANTITY_UNITS = {
    "length": "px",
    "area": "px^2",
    "mean_width": "px",
    "count": "count",
    "total_area": "px^2",
    "mean_equivalent_radius": "px",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate physical defect measurements from existing segmentation checkpoints."
    )
    parser.add_argument("--config", default="configs/measurement_eval_config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--max-samples-per-task", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def load_configs(path: str, overrides: list[str]):
    with open(path, "r", encoding="utf-8") as file:
        measurement_config = yaml.safe_load(file)
    apply_config_overrides(measurement_config, overrides)
    segmentation_path = Path(measurement_config["segmentation_config"])
    with open(segmentation_path, "r", encoding="utf-8") as file:
        segmentation_config = yaml.safe_load(file)
    return measurement_config, segmentation_config


def limit_records(records: list[dict], maximum: int | None) -> list[dict]:
    if maximum is None:
        return records
    counts = defaultdict(int)
    selected = []
    for record in records:
        key = task_key(record)
        if counts[key] < maximum:
            selected.append(record)
            counts[key] += 1
    return selected


def checkpoint_paths(segmentation_config: dict) -> dict[str, dict[int, Path]]:
    root = Path(segmentation_config["paths"]["output_dir"]) / "checkpoints"
    paths = {}
    missing = []
    for experiment in segmentation_config["experiments"]:
        paths[experiment] = {}
        for raw_seed in segmentation_config["training"]["seeds"]:
            seed = int(raw_seed)
            path = root / experiment / f"seed_{seed}" / "best.pt"
            paths[experiment][seed] = path
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing segmentation checkpoints:\n  " + "\n  ".join(missing))
    return paths


def make_loader(records: list[dict], segmentation_config: dict, measurement_config: dict):
    image_size = int(segmentation_config["training"]["image_size"])
    dataset = AblationSegmentationDataset(
        records,
        SegmentationTransform(size=image_size, training=False),
        copy_paste=False,
    )
    evaluation = measurement_config["evaluation"]
    loader = DataLoader(
        dataset,
        batch_size=int(evaluation["batch_size"]),
        shuffle=False,
        num_workers=int(evaluation["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def measure_ground_truth(loader, records, evaluation):
    measured = defaultdict(dict)
    offset = 0
    for batch in loader:
        labels = batch["labels"].numpy()
        for batch_index in range(labels.shape[0]):
            record_index = offset + batch_index
            key = task_key(records[record_index])
            class_id = COMPONENT_DEFECT_CLASS_MAP[key]
            measured[key][record_index] = measure_defect_mask(
                labels[batch_index] == class_id,
                key,
                particle_minimum_area=int(evaluation["particle_minimum_area"]),
                linear_minimum_area=int(evaluation["linear_minimum_area"]),
            )
        offset += labels.shape[0]
    if offset != len(records):
        raise RuntimeError(f"Ground-truth pass returned {offset} of {len(records)} samples")
    return measured


@torch.inference_mode()
def measure_predictions(model, loader, records, device, evaluation):
    measured = defaultdict(dict)
    offset = 0
    use_amp = bool(evaluation.get("mixed_precision", True)) and device.type == "cuda"
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(pixel_values).argmax(dim=1)
        predictions = predictions.cpu().numpy()
        for batch_index in range(predictions.shape[0]):
            record_index = offset + batch_index
            key = task_key(records[record_index])
            class_id = COMPONENT_DEFECT_CLASS_MAP[key]
            measured[key][record_index] = measure_defect_mask(
                predictions[batch_index] == class_id,
                key,
                particle_minimum_area=int(evaluation["particle_minimum_area"]),
                linear_minimum_area=int(evaluation["linear_minimum_area"]),
            )
        offset += predictions.shape[0]
    if offset != len(records):
        raise RuntimeError(f"Prediction pass returned {offset} of {len(records)} samples")
    return measured


def summarize_all(ground_truth, predictions, experiments, seeds, evaluation):
    rows = []
    nested = {}
    for experiment in experiments:
        nested[experiment] = {}
        for key in sorted(ground_truth, key=lambda item: TASK_ORDER.index(item)):
            record_indices = sorted(ground_truth[key])
            nested[experiment][key] = {}
            for quantity in quantities_for_task(key):
                gt_values = np.asarray([ground_truth[key][index][quantity] for index in record_indices])
                predicted_values = np.asarray(
                    [
                        [predictions[experiment][seed][key][index][quantity] for index in record_indices]
                        for seed in seeds
                    ]
                )
                seed_offset = sum(map(ord, f"{experiment}:{key}:{quantity}"))
                summary = summarize_errors(
                    gt_values,
                    predicted_values,
                    bootstrap_iterations=int(evaluation["bootstrap_iterations"]),
                    bootstrap_seed=int(evaluation["bootstrap_seed"]) + seed_offset,
                    epsilon=float(evaluation["epsilon"]),
                )
                nested[experiment][key][quantity] = summary
                rows.append(
                    {
                        "experiment": experiment,
                        "task": key,
                        "task_label": TASK_LABELS.get(key, key),
                        "quantity": quantity,
                        "unit": QUANTITY_UNITS[quantity],
                        "ground_truth_mean": summary["ground_truth_mean"],
                        "ground_truth_std": summary["ground_truth_std"],
                        "mae": summary["mae"],
                        "mae_ci_95_low": summary["mae_ci_95"][0],
                        "mae_ci_95_high": summary["mae_ci_95"][1],
                        "nmae_percent": summary["nmae_percent"],
                        "max_percentage_error_mean": summary["max_percentage_error_mean"],
                        "max_percentage_error_std": summary["max_percentage_error_std"],
                        "max_percentage_error_per_seed": json.dumps(
                            summary["max_percentage_error_per_seed"]
                        ),
                        "max_percentage_error_worst_seed": summary[
                            "max_percentage_error_worst_seed"
                        ],
                        "num_images": summary["num_images"],
                        "num_seeds": summary["num_seeds"],
                    }
                )
    return nested, rows


def build_detail_rows(ground_truth, predictions, records, experiments, seeds):
    rows = []
    for experiment in experiments:
        for seed in seeds:
            for key in sorted(ground_truth, key=lambda item: TASK_ORDER.index(item)):
                for record_index in sorted(ground_truth[key]):
                    for quantity in quantities_for_task(key):
                        reference = ground_truth[key][record_index][quantity]
                        estimate = predictions[experiment][seed][key][record_index][quantity]
                        rows.append(
                            {
                                "experiment": experiment,
                                "seed": seed,
                                "record_index": record_index,
                                "image_path": records[record_index]["image_path"],
                                "task": key,
                                "task_label": TASK_LABELS.get(key, key),
                                "quantity": quantity,
                                "unit": QUANTITY_UNITS[quantity],
                                "ground_truth": reference,
                                "prediction": estimate,
                                "absolute_error": abs(estimate - reference),
                            }
                        )
    return rows


def write_outputs(
    output_dir: Path,
    report: dict,
    rows: list[dict],
    detail_rows: list[dict],
    experiments: list[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "measurement_metrics.json"
    csv_path = output_dir / "measurement_metrics.csv"
    detail_path = output_dir / "measurement_per_image.csv"
    latex_path = output_dir / "measurement_table_rows.tex"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(detail_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    lookup = {(row["experiment"], row["task"], row["quantity"]): row for row in rows}
    lines = []
    for key in sorted(report["metrics"][experiments[0]], key=lambda item: TASK_ORDER.index(item)):
        quantities = quantities_for_task(key)
        for quantity_index, quantity in enumerate(quantities):
            gt = lookup[(experiments[0], key, quantity)]
            cells = []
            for experiment in experiments:
                row = lookup[(experiment, key, quantity)]
                cells.append(
                    f"{row['max_percentage_error_mean']:.2f} $\\pm$ "
                    f"{row['max_percentage_error_std']:.2f}\\%"
                )
            task_cell = TASK_LABELS.get(key, key) if quantity_index == 0 else ""
            lines.append(
                f"{task_cell} & {quantity.replace('_', ' ')} & "
                f"{gt['ground_truth_mean']:.3f} $\\pm$ {gt['ground_truth_std']:.3f} & "
                + " & ".join(cells)
                + " \\\\"
            )
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, detail_path, latex_path


def main():
    args = parse_args()
    measurement_config, segmentation_config = load_configs(args.config, args.overrides)
    requested_device = str(measurement_config.get("device", segmentation_config.get("device", "cuda")))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(requested_device)
    evaluation = measurement_config["evaluation"]
    seeds = [int(seed) for seed in segmentation_config["training"]["seeds"]]
    experiments = list(segmentation_config["experiments"])
    checkpoints = checkpoint_paths(segmentation_config)
    records = limit_records(
        read_metadata(Path(segmentation_config["paths"]["real_test_dir"]), generated=False),
        args.max_samples_per_task,
    )
    loader = make_loader(records, segmentation_config, measurement_config)
    ground_truth = measure_ground_truth(loader, records, evaluation)

    model = MobileViTUNet(
        num_classes=NUM_CLASSES,
        pretrained_model_name=segmentation_config["pretrained_model_name"],
        local_files_only=bool(segmentation_config.get("local_files_only", False)),
    ).to(device)
    model.eval()
    predictions = defaultdict(dict)
    for experiment in experiments:
        for seed in seeds:
            checkpoint_path = checkpoints[experiment][seed]
            print(f"Evaluating measurements: experiment={experiment} seed={seed} checkpoint={checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            predictions[experiment][seed] = measure_predictions(
                model, loader, records, device, evaluation
            )

    metrics, rows = summarize_all(ground_truth, predictions, experiments, seeds, evaluation)
    detail_rows = build_detail_rows(ground_truth, predictions, records, experiments, seeds)
    output_dir = Path(measurement_config["paths"]["output_dir"])
    report = {
        "protocol": {
            "coordinate_system": f"{segmentation_config['training']['image_size']}x{segmentation_config['training']['image_size']} pixels",
            "holdout": str(Path(segmentation_config["paths"]["real_test_dir"])),
            "experiments": experiments,
            "seeds": seeds,
            "linear_measurements": [
                "Zhang-Suen skeletonization",
                "8-connected weighted skeleton length",
                "area",
                "mean width=2*EDT on skeleton",
            ],
            "particle_measurements": ["component count", "total area", "mean equivalent radius"],
            "particle_minimum_area": int(evaluation["particle_minimum_area"]),
            "mae_aggregation": "mean over holdout images and model seeds",
            "nmae_definition": "100 * sum absolute error / (num_seeds * sum ground truth + epsilon)",
            "confidence_interval": "paired image bootstrap of seed-averaged absolute errors",
            "bootstrap_iterations": int(evaluation["bootstrap_iterations"]),
            "bootstrap_seed": int(evaluation["bootstrap_seed"]),
            "primary_measurement_error": (
                "For each seed, maximum absolute percentage error across test images; "
                "reported as mean and sample standard deviation across seeds"
            ),
        },
        "metrics": metrics,
    }
    paths = write_outputs(output_dir, report, rows, detail_rows, experiments)
    print("Wrote measurement evaluation outputs:")
    for path in paths:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
