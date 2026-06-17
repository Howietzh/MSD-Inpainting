import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset.segmentation_dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    DefectSegmentationDataset,
    SegmentationTransform,
    build_balanced_splits,
    read_metadata,
)
from models.mobilevit_unet import MobileViTUNet
from utils.config_overrides import apply_config_overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Compare generated datasets with MobileViT-UNet.")
    parser.add_argument("--config", default="configs/segmentation_config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--max-samples-per-task", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DistributedContext:
    def __init__(self, requested_device: str):
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed MobileViT-UNet training requires CUDA.")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            dist.init_process_group(backend="nccl", device_id=self.device)
        else:
            if requested_device == "cuda" and not torch.cuda.is_available():
                requested_device = "cpu"
            self.device = torch.device(requested_device)

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier(device_ids=[self.local_rank])

    def reduce_totals(self, totals: np.ndarray) -> np.ndarray:
        if not self.distributed:
            return totals
        tensor = torch.as_tensor(totals, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().numpy()

    def close(self):
        if self.distributed:
            dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.task_matrices = defaultdict(
            lambda: torch.zeros((num_classes, num_classes), dtype=torch.int64)
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, task_keys: list[str]):
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()
        for prediction, target, key in zip(predictions, targets, task_keys):
            valid = (target >= 0) & (target < self.num_classes)
            indices = self.num_classes * target[valid].to(torch.int64) + prediction[valid].to(torch.int64)
            matrix = torch.bincount(indices, minlength=self.num_classes**2).reshape(
                self.num_classes, self.num_classes
            )
            self.matrix += matrix
            self.task_matrices[key] += matrix


def metrics_from_matrix(matrix: torch.Tensor) -> dict:
    matrix = matrix.to(torch.float64)
    true_positive = matrix.diag()
    target_count = matrix.sum(dim=1)
    prediction_count = matrix.sum(dim=0)
    union = target_count + prediction_count - true_positive

    iou = torch.where(union > 0, true_positive / union, torch.nan)
    precision = torch.where(prediction_count > 0, true_positive / prediction_count, torch.nan)
    recall = torch.where(target_count > 0, true_positive / target_count, torch.nan)

    return {
        "mIoU": float(torch.nanmean(iou)),
        "mIoU_foreground": float(torch.nanmean(iou[1:])),
        "Precision": float(torch.nanmean(precision[1:])),
        "Recall": float(torch.nanmean(recall[1:])),
        "per_class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
        "per_class_precision": {name: float(value) for name, value in zip(CLASS_NAMES, precision)},
        "per_class_recall": {name: float(value) for name, value in zip(CLASS_NAMES, recall)},
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }


def multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def compute_loss(logits, labels, ce_weight: float, dice_weight: float):
    ce_loss = F.cross_entropy(logits, labels)
    dice_loss = multiclass_dice_loss(logits, labels)
    total_loss = ce_weight * ce_loss + dice_weight * dice_loss
    return total_loss, ce_loss, dice_loss


def create_loaders(train_records, validation_records, test_records, config, seed, distributed_context):
    training = config["training"]
    augmentation = config["augmentation"]
    size = int(training["image_size"])
    train_transform = SegmentationTransform(
        size=size,
        training=True,
        resize_min=float(augmentation["resize_min"]),
        resize_max=float(augmentation["resize_max"]),
        clahe_probability=float(augmentation["clahe_probability"]),
        clahe_clip_limit=float(augmentation["clahe_clip_limit"]),
        clahe_grid_size=int(augmentation["clahe_grid_size"]),
    )
    eval_transform = SegmentationTransform(size=size, training=False)
    train_dataset = DefectSegmentationDataset(train_records, train_transform)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=True,
            seed=seed,
        )
        if distributed_context.distributed
        else None
    )
    generator = torch.Generator().manual_seed(seed + distributed_context.rank)
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        DefectSegmentationDataset(validation_records, eval_transform),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        DefectSegmentationDataset(test_records, eval_transform),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader, test_loader, train_sampler


def train_epoch(model, loader, optimizer, scaler, device, ce_weight, dice_weight, use_amp, distributed_context):
    model.train()
    totals = np.zeros(4, dtype=np.float64)
    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, ce_loss, dice_loss = compute_loss(logits, labels, ce_weight, dice_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        totals += [
            loss.detach().item() * batch_size,
            ce_loss.detach().item() * batch_size,
            dice_loss.detach().item() * batch_size,
            batch_size,
        ]
    totals = distributed_context.reduce_totals(totals)
    return {"loss": totals[0] / totals[3], "ce_loss": totals[1] / totals[3], "dice_loss": totals[2] / totals[3]}


@torch.no_grad()
def evaluate(model, loader, device, ce_weight, dice_weight, use_amp):
    model.eval()
    totals = np.zeros(4, dtype=np.float64)
    confusion = ConfusionMatrix(NUM_CLASSES)
    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, ce_loss, dice_loss = compute_loss(logits, labels, ce_weight, dice_weight)
        batch_size = images.shape[0]
        totals += [
            loss.detach().item() * batch_size,
            ce_loss.detach().item() * batch_size,
            dice_loss.detach().item() * batch_size,
            batch_size,
        ]
        confusion.update(logits.argmax(dim=1), labels, list(batch["task_key"]))

    metrics = metrics_from_matrix(confusion.matrix)
    metrics.update({"loss": totals[0] / totals[3], "ce_loss": totals[1] / totals[3], "dice_loss": totals[2] / totals[3]})
    metrics["tasks"] = {key: metrics_from_matrix(matrix) for key, matrix in sorted(confusion.task_matrices.items())}
    return metrics


def save_checkpoint(path: Path, model, epoch: int, score: float, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "epoch": epoch,
            "validation_mIoU_foreground": score,
            "config": config,
        },
        path,
    )


def train_run(experiment, seed, splits, test_records, config, output_dir, distributed_context):
    seed_everything(seed + distributed_context.rank)
    device = distributed_context.device
    training = config["training"]
    run_dir = output_dir / experiment / f"seed_{seed}"
    checkpoint_path = output_dir / "checkpoints" / experiment / f"seed_{seed}" / "best.pt"
    tensorboard_dir = output_dir / "tensorboard" / experiment / f"seed_{seed}"
    if distributed_context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    distributed_context.barrier()
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if distributed_context.is_main else None

    train_loader, validation_loader, test_loader, train_sampler = create_loaders(
        splits["train"], splits["validation"], test_records, config, seed, distributed_context
    )
    model = MobileViTUNet(
        num_classes=NUM_CLASSES,
        pretrained_model_name=config["pretrained_model_name"],
        local_files_only=bool(config.get("local_files_only", False)),
    ).to(device)
    if distributed_context.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            float(training["encoder_learning_rate"]),
            float(training["decoder_learning_rate"]),
        ),
        weight_decay=float(training["weight_decay"]),
    )
    if distributed_context.distributed:
        model = DistributedDataParallel(model, device_ids=[distributed_context.local_rank])
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce_weight = float(training["ce_weight"])
    dice_weight = float(training["dice_weight"])
    validation_every = int(training["validation_every"])
    if validation_every <= 0:
        raise ValueError("training.validation_every must be positive.")
    if distributed_context.is_main and checkpoint_path.exists():
        checkpoint_path.unlink()
    distributed_context.barrier()
    best_score = -math.inf
    best_epoch = None
    best_validation = None
    history = []

    try:
        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                ce_weight,
                dice_weight,
                use_amp,
                distributed_context,
            )
            scheduler.step()
            if distributed_context.is_main:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch)
                writer.add_scalar("learning_rate/encoder", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("learning_rate/decoder", optimizer.param_groups[1]["lr"], epoch)

            entry = {"epoch": epoch, "train": train_metrics}
            should_validate = epoch % validation_every == 0 or epoch == epochs
            if should_validate:
                distributed_context.barrier()
                if distributed_context.is_main:
                    validation = evaluate(
                        unwrap_model(model), validation_loader, device, ce_weight, dice_weight, use_amp
                    )
                    entry["validation"] = validation
                    for name in ("loss", "ce_loss", "dice_loss", "mIoU", "mIoU_foreground", "Precision", "Recall"):
                        writer.add_scalar(f"validation/{name}", validation[name], epoch)
                    score = validation["mIoU_foreground"]
                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                        best_validation = validation
                        save_checkpoint(checkpoint_path, model, epoch, score, config)
                    print(
                        f"[{experiment} seed={seed} epoch={epoch}] "
                        f"train_loss={train_metrics['loss']:.4f} val_fg_mIoU={score:.4f}"
                    )
                distributed_context.barrier()
            if distributed_context.is_main:
                history.append(entry)
    finally:
        if writer is not None:
            writer.close()

    distributed_context.barrier()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    distributed_context.barrier()
    result = None
    if distributed_context.is_main:
        test_metrics = evaluate(unwrap_model(model), test_loader, device, ce_weight, dice_weight, use_amp)
        result = {
            "experiment": experiment,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "test": test_metrics,
            "num_train": len(splits["train"]),
            "num_validation": len(splits["validation"]),
            "num_test": len(test_records),
            "world_size": distributed_context.world_size,
            "per_gpu_batch_size": int(training["batch_size"]),
            "effective_batch_size": int(training["batch_size"]) * distributed_context.world_size,
            "checkpoint": str(checkpoint_path.resolve()),
            "tensorboard": str(tensorboard_dir.resolve()),
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as file:
            json.dump(to_jsonable(result), file, indent=2, ensure_ascii=False)
        with open(run_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(to_jsonable(history), file, indent=2, ensure_ascii=False)
    distributed_context.barrier()
    return result


def summarize(results: list[dict], experiments: list[str]):
    metric_names = ["mIoU", "mIoU_foreground", "Precision", "Recall"]

    def statistics(values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        return {
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "values": values.tolist(),
        }

    summary = {}
    for experiment in experiments:
        experiment_results = [result for result in results if result["experiment"] == experiment]
        summary[experiment] = {}
        for metric in metric_names:
            summary[experiment][metric] = statistics(
                [result["test"][metric] for result in experiment_results]
            )

        summary[experiment]["per_class"] = {}
        for class_name in CLASS_NAMES:
            summary[experiment]["per_class"][class_name] = {
                metric: statistics(
                    [result["test"][f"per_class_{metric}"][class_name] for result in experiment_results]
                )
                for metric in ("iou", "precision", "recall")
            }

        task_names = sorted(
            {
                task_name
                for result in experiment_results
                for task_name in result["test"]["tasks"]
            }
        )
        summary[experiment]["tasks"] = {
            task_name: {
                metric: statistics(
                    [
                        result["test"]["tasks"][task_name][metric]
                        for result in experiment_results
                        if task_name in result["test"]["tasks"]
                    ]
                )
                for metric in metric_names
            }
            for task_name in task_names
        }
    return summary


def write_summary(output_dir: Path, summary: dict, results: list[dict], common_counts: dict, config: dict):
    report = {"summary": summary, "common_task_counts": common_counts, "runs": results, "config": config}
    with open(output_dir / "comparison.json", "w", encoding="utf-8") as file:
        json.dump(to_jsonable(report), file, indent=2, ensure_ascii=False)

    metric_names = ["mIoU", "mIoU_foreground", "Precision", "Recall"]
    with open(output_dir / "comparison.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", *[f"{metric}_mean" for metric in metric_names], *[f"{metric}_std" for metric in metric_names]])
        for experiment, metrics in summary.items():
            writer.writerow(
                [experiment]
                + [metrics[metric]["mean"] for metric in metric_names]
                + [metrics[metric]["std"] for metric in metric_names]
            )

    lines = [
        "| Experiment | mIoU | mIoU foreground | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment, metrics in summary.items():
        values = [f"{metrics[metric]['mean']:.4f} ± {metrics[metric]['std']:.4f}" for metric in metric_names]
        lines.append(f"| {experiment} | {' | '.join(values)} |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def limit_records_per_task(records: list[dict], limit: int | None):
    if limit is None:
        return records
    grouped = defaultdict(list)
    for record in records:
        grouped[f"{record['defect_token']}::{record['object_token']}"].append(record)
    return [record for key in sorted(grouped) for record in grouped[key][:limit]]


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    apply_config_overrides(config, args.overrides)

    distributed_context = DistributedContext(str(config.get("device", "cuda")))
    output_dir = Path(config["paths"]["output_dir"])
    if distributed_context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_context.barrier()
    experiments = list(config["experiments"])
    generated_root = Path(config["paths"]["generated_root"])
    experiment_records = {
        experiment: limit_records_per_task(
            read_metadata(generated_root / experiment, generated=True),
            args.max_samples_per_task,
        )
        for experiment in experiments
    }
    splits, common_counts = build_balanced_splits(
        experiment_records,
        validation_ratio=float(config["training"]["validation_ratio"]),
        seed=int(config["training"]["seeds"][0]),
    )
    test_records = read_metadata(Path(config["paths"]["real_test_dir"]), generated=False)
    results = []
    try:
        for experiment in experiments:
            for seed in config["training"]["seeds"]:
                result = train_run(
                    experiment,
                    int(seed),
                    splits[experiment],
                    test_records,
                    config,
                    output_dir,
                    distributed_context,
                )
                if distributed_context.is_main:
                    results.append(result)
        if distributed_context.is_main:
            summary = summarize(results, experiments)
            write_summary(output_dir, summary, results, common_counts, config)
            print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))
    finally:
        distributed_context.close()


if __name__ == "__main__":
    main()
