import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset.segmentation_ablation_dataset import (
    AblationSegmentationDataset,
    limit_records_per_task,
    split_real_records_by_task,
)
from dataset.segmentation_dataset import NUM_CLASSES, SegmentationTransform, read_metadata
from models.mobilevit_unet import MobileViTUNet
from train_segmentation_compare import (
    DistributedContext,
    evaluate,
    save_checkpoint,
    seed_everything,
    summarize,
    to_jsonable,
    train_epoch,
    unwrap_model,
    worker_init_fn,
    write_summary,
)
from utils.config_overrides import apply_config_overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Run MobileViT-UNet segmentation ablations.")
    parser.add_argument("--config", default="configs/segmentation_ablation_config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--max-samples-per-task", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def make_transforms(config):
    training = config["training"]
    augmentation = config["augmentation"]
    image_size = int(training["image_size"])
    train_transform = SegmentationTransform(
        size=image_size,
        training=True,
        resize_min=float(augmentation["resize_min"]),
        resize_max=float(augmentation["resize_max"]),
        clahe_probability=float(augmentation["clahe_probability"]),
        clahe_clip_limit=float(augmentation["clahe_clip_limit"]),
        clahe_grid_size=int(augmentation["clahe_grid_size"]),
    )
    eval_transform = SegmentationTransform(size=image_size, training=False)
    return train_transform, eval_transform


def make_loader(dataset, config, seed, distributed_context, shuffle: bool):
    training = config["training"]
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=shuffle,
            seed=seed,
        )
        if distributed_context.distributed and shuffle
        else None
    )
    return (
        DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            generator=torch.Generator().manual_seed(seed + distributed_context.rank),
            num_workers=int(training["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        ),
        sampler,
    )


def build_experiment_dataset(experiment, real_train_records, generated_records, config):
    train_transform, _ = make_transforms(config)
    copy_paste_probability = float(config["augmentation"].get("copy_paste_probability", 1.0))
    real_plain = AblationSegmentationDataset(real_train_records, train_transform, copy_paste=False)

    if experiment == "original":
        return real_plain, {
            "real_train_samples": len(real_train_records),
            "generated_train_samples": 0,
            "copy_paste": False,
        }
    if experiment == "original_copy_paste":
        return (
            AblationSegmentationDataset(
                real_train_records,
                train_transform,
                copy_paste=True,
                copy_paste_probability=copy_paste_probability,
            ),
            {
                "real_train_samples": len(real_train_records),
                "generated_train_samples": 0,
                "copy_paste": True,
                "copy_paste_probability": copy_paste_probability,
            },
        )
    if experiment == "original_generated":
        generated_dataset = AblationSegmentationDataset(generated_records, train_transform, copy_paste=False)
        return (
            ConcatDataset([real_plain, generated_dataset]),
            {
                "real_train_samples": len(real_train_records),
                "generated_train_samples": len(generated_records),
                "copy_paste": False,
            },
        )
    raise ValueError(f"Unknown ablation experiment: {experiment}")


def build_eval_dataset(records, config):
    _, eval_transform = make_transforms(config)
    return AblationSegmentationDataset(records, eval_transform, copy_paste=False)


def train_ablation_run(
    experiment,
    seed,
    train_dataset,
    validation_dataset,
    test_dataset,
    dataset_info,
    config,
    output_dir,
    distributed_context,
):
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

    train_loader, train_sampler = make_loader(train_dataset, config, seed, distributed_context, shuffle=True)
    validation_loader, _ = make_loader(validation_dataset, config, seed, distributed_context, shuffle=False)
    test_loader, _ = make_loader(test_dataset, config, seed, distributed_context, shuffle=False)

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
                        unwrap_model(model),
                        validation_loader,
                        device,
                        ce_weight,
                        dice_weight,
                        use_amp,
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
            "num_train": len(train_dataset),
            "num_validation": len(validation_dataset),
            "num_test": len(test_dataset),
            "world_size": distributed_context.world_size,
            "per_gpu_batch_size": int(training["batch_size"]),
            "effective_batch_size": int(training["batch_size"]) * distributed_context.world_size,
            "checkpoint": str(checkpoint_path.resolve()),
            "tensorboard": str(tensorboard_dir.resolve()),
            **dataset_info,
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as file:
            json.dump(to_jsonable(result), file, indent=2, ensure_ascii=False)
        with open(run_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(to_jsonable(history), file, indent=2, ensure_ascii=False)
    distributed_context.barrier()
    return result


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

    real_records = limit_records_per_task(
        read_metadata(Path(config["paths"]["real_train_dir"]), generated=False),
        args.max_samples_per_task,
    )
    generated_records = limit_records_per_task(
        read_metadata(Path(config["paths"]["generated_dir"]), generated=True),
        args.max_samples_per_task,
    )
    test_records = read_metadata(Path(config["paths"]["real_test_dir"]), generated=False)
    real_train_records, validation_records, split_manifest = split_real_records_by_task(
        real_records,
        validation_ratio=float(config["training"]["validation_ratio"]),
        seed=int(config["training"]["seeds"][0]),
    )
    validation_dataset = build_eval_dataset(validation_records, config)
    test_dataset = build_eval_dataset(test_records, config)

    results = []
    try:
        for experiment in config["experiments"]:
            train_dataset, dataset_info = build_experiment_dataset(
                experiment,
                real_train_records,
                generated_records,
                config,
            )
            for seed in config["training"]["seeds"]:
                result = train_ablation_run(
                    experiment,
                    int(seed),
                    train_dataset,
                    validation_dataset,
                    test_dataset,
                    dataset_info,
                    config,
                    output_dir,
                    distributed_context,
                )
                if distributed_context.is_main:
                    results.append(result)
        if distributed_context.is_main:
            summary = summarize(results, list(config["experiments"]))
            report_counts = {
                "real_split": split_manifest,
                "real_train_total": len(real_train_records),
                "real_validation_total": len(validation_records),
                "generated_train_total": len(generated_records),
            }
            write_summary(output_dir, summary, results, report_counts, config)
            print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))
    finally:
        distributed_context.close()


if __name__ == "__main__":
    main()
