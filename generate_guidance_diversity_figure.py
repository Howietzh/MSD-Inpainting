import argparse
import gc
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dataset.normal_dataset import NormalComponentDataset
from generate_qualitative_figure import (
    TARGET_SIZE,
    build_pipe,
    build_shared_mask,
    choose_existing_path,
    load_yaml,
    resolve_best_lora_dir,
    run_generation,
    serialize_path,
    tensor_mask_to_uint8,
    tensor_to_rgb_image,
)
from utils.mask_ops import DefectMaskEngine
from utils.runtime import resolve_model_source, resolve_weight_dtype


EXPERIMENT = "increase_text_encoder_learning_rates"
DEFAULT_GUIDANCE_SCALES = "12,11,10,9,8,7,6,5,4,3"
MASK_INSET_RATIO = 0.22
CELL_GAP = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a label-free guidance-scale diversity figure using MSD-Inpainting."
    )
    parser.add_argument("--train-config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--infer-config", type=str, default="configs/inference_config.yaml")
    parser.add_argument("--weights-root", type=str, default="defectfill_lora_weights")
    parser.add_argument("--experiment", type=str, default=EXPERIMENT)
    parser.add_argument("--normal-dir", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--stats-cache", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="qualitative_figure")
    parser.add_argument("--guidance-scales", type=str, default=DEFAULT_GUIDANCE_SCALES)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--cell-size", type=int, default=192)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser.parse_args()


def parse_guidance_scales(value: str) -> list[float]:
    scales = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not scales:
        raise ValueError("At least one guidance scale must be provided.")
    return scales


def select_task_samples(
    normal_dir: Path,
    component_token: str,
    count: int,
    seed: int,
    used_indices: set[int],
):
    dataset = NormalComponentDataset(data_dir=str(normal_dir), size=TARGET_SIZE, target_comp=component_token)
    if len(dataset) == 0:
        raise ValueError(f"No normal samples found for component token {component_token}.")

    indices = list(range(len(dataset)))
    random.Random(int(seed)).shuffle(indices)
    unused_indices = [index for index in indices if index not in used_indices]
    if count > len(unused_indices):
        print(
            f"Warning: {component_token} has only {len(unused_indices)} unused normal images; "
            "some references must be reused across the figure."
        )
        selected_indices = unused_indices + [
            indices[index % len(indices)] for index in range(count - len(unused_indices))
        ]
    else:
        selected_indices = unused_indices[:count]
    used_indices.update(selected_indices)

    samples = []
    for dataset_index in selected_indices:
        sample = dataset[dataset_index]
        samples.append(
            {
                "dataset_index": int(dataset_index),
                "image_path": sample["image_path"],
                "component_mask_path": sample["component_mask_path"],
                "image_np": tensor_to_rgb_image(sample["pixel_values"]),
                "component_mask_np": tensor_mask_to_uint8(sample["mask_values"]),
            }
        )
    return samples


def add_bottom_left_mask_inset(
    image: Image.Image,
    defect_mask_np: np.ndarray,
    ratio: float = MASK_INSET_RATIO,
) -> Image.Image:
    output = image.convert("RGB").copy()
    width, height = output.size
    inset_size = max(24, int(round(width * ratio)))
    border = max(2, inset_size // 32)
    margin = max(6, inset_size // 12)

    mask_rgb = Image.fromarray(defect_mask_np, mode="L").resize(
        (inset_size, inset_size),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")
    tile = Image.new("RGB", (inset_size + 2 * border, inset_size + 2 * border), "white")
    tile.paste(mask_rgb, (border, border))
    output.paste(tile, (margin, height - tile.height - margin))
    return output


def make_grid(cells: list[list[Image.Image]], cell_size: int) -> Image.Image:
    rows = len(cells)
    columns = len(cells[0])
    width = columns * cell_size + (columns - 1) * CELL_GAP
    height = rows * cell_size + (rows - 1) * CELL_GAP
    grid = Image.new("RGB", (width, height), "white")

    for row_idx, row in enumerate(cells):
        if len(row) != columns:
            raise ValueError("All figure rows must contain the same number of cells.")
        for column_idx, image in enumerate(row):
            resized = image.convert("RGB").resize(
                (cell_size, cell_size),
                resample=Image.Resampling.LANCZOS,
            )
            x = column_idx * (cell_size + CELL_GAP)
            y = row_idx * (cell_size + CELL_GAP)
            grid.paste(resized, (x, y))
    return grid


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return serialize_path(value)
    return value


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    train_config = load_yaml(args.train_config)
    infer_config = load_yaml(args.infer_config)
    infer_options = infer_config.get("inference", {})
    guidance_scales = parse_guidance_scales(args.guidance_scales)

    if args.cell_size <= 0:
        raise ValueError("cell-size must be a positive integer.")
    if args.dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_root = choose_existing_path(
        args.weights_root,
        [
            repo_root / "defectfill_lora_weights",
            repo_root.parent / "defectfill_lora_weights",
            train_config["paths"].get("output_dir"),
        ],
    )
    normal_dir = choose_existing_path(
        args.normal_dir,
        [
            infer_config["paths"].get("normal_dir"),
            repo_root / "data" / "CCM-Defect" / "normal_components",
            repo_root.parent / "data" / "CCM-Defect" / "normal_components",
        ],
    )
    train_dir = choose_existing_path(
        args.train_dir,
        [
            infer_config["paths"].get("train_dir"),
            train_config["paths"].get("data_dir"),
            repo_root / "data" / "CCM-Defect" / "defect_train_concept",
            repo_root.parent / "data" / "CCM-Defect" / "defect_train_concept",
        ],
    )
    stats_cache = choose_existing_path(
        args.stats_cache,
        [
            output_dir / "guidance_diversity_defect_stats_cache.json",
            infer_config["paths"].get("stats_cache"),
        ],
        create_parent=True,
    )

    tasks = list(infer_config.get("tasks", []))
    if args.max_tasks is not None:
        tasks = tasks[: max(0, int(args.max_tasks))]
    if not tasks:
        raise ValueError("No qualitative tasks were selected.")

    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else infer_options.get("num_inference_steps", 30)
    )
    negative_prompt = (
        args.negative_prompt
        if args.negative_prompt is not None
        else infer_options.get("negative_prompt", "")
    )
    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    weight_dtype = resolve_weight_dtype(train_config.get("training", {}).get("mixed_precision", "no"))
    if device.type == "cpu":
        weight_dtype = torch.float32

    lora_dir, weight_source = resolve_best_lora_dir(weights_root / args.experiment)
    model_source = resolve_model_source(train_config["paths"])
    mask_engine = DefectMaskEngine(train_dir=train_dir, cache_file=stats_cache, target_size=TARGET_SIZE)
    mask_engine.load_or_compute_stats(tasks)
    if stats_cache.exists():
        with open(stats_cache, "r", encoding="utf-8") as file:
            mask_engine.stats_cache = json.load(file)

    manifest = {
        "train_config": serialize_path(Path(args.train_config)),
        "infer_config": serialize_path(Path(args.infer_config)),
        "weights_root": serialize_path(weights_root),
        "experiment": args.experiment,
        "lora_dir": serialize_path(lora_dir),
        "weight_source": weight_source,
        "normal_dir": serialize_path(normal_dir),
        "train_dir": serialize_path(train_dir),
        "stats_cache": serialize_path(stats_cache),
        "base_seed": int(args.base_seed),
        "guidance_scales": guidance_scales,
        "num_inference_steps": num_inference_steps,
        "negative_prompt": negative_prompt,
        "cell_size": int(args.cell_size),
        "cell_gap": CELL_GAP,
        "dpi": int(args.dpi),
        "mask_inset_position": "bottom_left",
        "mask_inset_ratio": MASK_INSET_RATIO,
        "labels_in_figure": False,
        "tasks": [],
    }
    figure_rows = []
    used_normal_indices = {}

    print(f"Loading {args.experiment} weights from {lora_dir} ({weight_source})")
    pipe = build_pipe(model_source, weight_dtype, device, lora_dir)
    try:
        for task_idx, task in enumerate(tasks):
            defect_token = task["defect"]
            component_token = task["comp"]
            prompt = f"a photo of {component_token} with {defect_token}"
            sample_seed = int(args.base_seed) + task_idx * 10000
            samples = select_task_samples(
                normal_dir,
                component_token,
                len(guidance_scales),
                sample_seed,
                used_normal_indices.setdefault(component_token, set()),
            )
            task_manifest = {
                "row_index": task_idx,
                "defect_token": defect_token,
                "component_token": component_token,
                "prompt": prompt,
                "cells": [],
            }
            row = []

            print(f"[Row {task_idx}] {defect_token} on {component_token}")
            for column_idx, (guidance_scale, sample) in enumerate(zip(guidance_scales, samples)):
                mask_seed = int(args.base_seed) + task_idx * 100000 + column_idx * 100
                generation_seed = int(args.base_seed) + task_idx * 1000000 + column_idx * 1000
                defect_mask_np, mask_params, mask_details = build_shared_mask(
                    mask_engine,
                    sample["component_mask_np"],
                    defect_token,
                    mask_seed,
                )
                generated = run_generation(
                    pipe=pipe,
                    prompt=prompt,
                    image_np=sample["image_np"],
                    defect_mask_np=defect_mask_np,
                    seed=generation_seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                )
                row.append(add_bottom_left_mask_inset(generated, defect_mask_np))
                task_manifest["cells"].append(
                    {
                        "row_index": task_idx,
                        "column_index": column_idx,
                        "guidance_scale": guidance_scale,
                        "sample_seed": sample_seed,
                        "mask_seed": mask_seed,
                        "generation_seed": generation_seed,
                        "normal_dataset_index": sample["dataset_index"],
                        "normal_image_path": sample["image_path"],
                        "component_mask_path": sample["component_mask_path"],
                        "mask_area": int(cv2.countNonZero(defect_mask_np)),
                        "mask_params": mask_params,
                        "mask_details": mask_details,
                    }
                )
                print(
                    f"  [Column {column_idx}] guidance={guidance_scale:g} "
                    f"normal_index={sample['dataset_index']} mask_seed={mask_seed}"
                )

            figure_rows.append(row)
            manifest["tasks"].append(task_manifest)
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    figure = make_grid(figure_rows, int(args.cell_size))
    png_path = output_dir / "msd_guidance_diversity_qualitative.png"
    pdf_path = output_dir / "msd_guidance_diversity_qualitative.pdf"
    manifest_path = output_dir / "msd_guidance_diversity_manifest.json"
    figure.save(png_path, dpi=(int(args.dpi), int(args.dpi)))
    figure.save(pdf_path, resolution=int(args.dpi))
    manifest["figure_size_pixels"] = list(figure.size)
    manifest["png_path"] = serialize_path(png_path)
    manifest["pdf_path"] = serialize_path(pdf_path)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(to_jsonable(manifest), file, indent=2, ensure_ascii=False)

    print(f"Saved label-free qualitative figure to {png_path}")
    print(f"Saved PDF figure to {pdf_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
