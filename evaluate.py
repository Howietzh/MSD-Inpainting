import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from utils.config_overrides import apply_config_overrides
from utils.evaluation_regions import partition_records_by_mask_validity, square_defect_roi_bounds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "generation"],
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted paths, e.g. --set generation.feature_batch_size=8",
    )
    return parser.parse_args()


def make_task_key(object_token: str, defect_token: str):
    return f"{defect_token}::{object_token}"


def read_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_idx}: {exc}") from exc
    return records


def validate_record(record: dict, data_dir: Path, source_name: str, index: int):
    required_fields = ["image_path", "defect_mask_path", "object_token", "defect_token"]
    for field in required_fields:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Missing required field '{field}' in {source_name} record #{index} under {data_dir}"
            )

    image_path = data_dir / record["image_path"]
    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing image file for {source_name} record #{index}: {image_path}"
        )

    defect_mask_path = data_dir / record["defect_mask_path"]
    if not defect_mask_path.exists():
        raise FileNotFoundError(
            f"Missing defect mask for {source_name} record #{index}: {defect_mask_path}"
        )


def load_real_metadata(data_dir: Path):
    metadata_path = data_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing holdout metadata file: {metadata_path}")

    records = read_jsonl(metadata_path)
    for idx, record in enumerate(records):
        validate_record(record, data_dir, metadata_path.name, idx)
    return records


def load_generated_metadata(data_dir: Path):
    shard_paths = sorted(data_dir.glob("metadata_gpu*.jsonl"))
    if not shard_paths:
        hint = ""
        if (data_dir / "metadata.jsonl").exists():
            hint = " Found metadata.jsonl only; this evaluator expects the new inference output format with metadata_gpu*.jsonl."
        elif any(path.is_dir() for path in data_dir.iterdir()) if data_dir.exists() else False:
            hint = " Make sure paths.generated_dir points to a single experiment or checkpoint output directory, not the experiments root."
        raise FileNotFoundError(f"Missing metadata_gpu*.jsonl under {data_dir}.{hint}")

    records = []
    for shard_path in shard_paths:
        shard_records = read_jsonl(shard_path)
        for idx, record in enumerate(shard_records):
            validate_record(record, data_dir, shard_path.name, idx)
        records.extend(shard_records)
    return records


def group_records_by_task(records):
    grouped = defaultdict(list)
    for record in records:
        key = make_task_key(record["object_token"], record["defect_token"])
        grouped[key].append(record)
    return grouped


def read_rgb_image(path: Path, size: int):
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), resample=Image.BILINEAR)


def read_local_rgb_image(
    image_path: Path,
    mask_path: Path,
    size: int,
    padding_ratio: float,
):
    """Crop a square defect ROI around a non-empty mask and resize it."""
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size:
        raise ValueError(
            f"Image and defect mask dimensions differ: image={image_path} {image.size}, "
            f"mask={mask_path} {mask.size}"
        )

    try:
        bounds = square_defect_roi_bounds(
            np.asarray(mask),
            image_width=image.width,
            image_height=image.height,
            padding_ratio=padding_ratio,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot compute local metrics for {mask_path}: {exc}") from exc
    roi = image.crop(bounds)
    return roi.resize((size, size), resample=Image.BILINEAR)


def image_to_tensor(image: Image.Image):
    return transforms.ToTensor()(image)


def imagenet_normalize(tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


class ImageOnlyDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        records: list[dict],
        image_size: int,
        region: str = "global",
        local_padding_ratio: float = 0.25,
    ):
        self.data_dir = data_dir
        self.records = records
        self.image_size = image_size
        self.region = region
        self.local_padding_ratio = local_padding_ratio
        if region not in {"global", "local"}:
            raise ValueError(f"Unsupported evaluation region: {region}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.region == "local":
            image = read_local_rgb_image(
                self.data_dir / record["image_path"],
                self.data_dir / record["defect_mask_path"],
                self.image_size,
                self.local_padding_ratio,
            )
        else:
            image = read_rgb_image(self.data_dir / record["image_path"], self.image_size)
        return image_to_tensor(image)


class LPIPSPairDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        pairs: list[tuple[dict, dict]],
        image_size: int,
        region: str = "global",
        local_padding_ratio: float = 0.25,
    ):
        self.data_dir = data_dir
        self.pairs = pairs
        self.image_size = image_size
        self.region = region
        self.local_padding_ratio = local_padding_ratio
        if region not in {"global", "local"}:
            raise ValueError(f"Unsupported evaluation region: {region}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        rec_a, rec_b = self.pairs[index]
        if self.region == "local":
            img_a = read_local_rgb_image(
                self.data_dir / rec_a["image_path"],
                self.data_dir / rec_a["defect_mask_path"],
                self.image_size,
                self.local_padding_ratio,
            )
            img_b = read_local_rgb_image(
                self.data_dir / rec_b["image_path"],
                self.data_dir / rec_b["defect_mask_path"],
                self.image_size,
                self.local_padding_ratio,
            )
        else:
            img_a = Image.open(self.data_dir / rec_a["image_path"]).convert("RGB")
            img_b = Image.open(self.data_dir / rec_b["image_path"]).convert("RGB")
        return self.transform(img_a), self.transform(img_b)


def polynomial_mmd(x, y):
    dim = x.shape[1]
    gamma = 1.0 / dim

    k_xx = (gamma * (x @ x.T) + 1.0).pow(3)
    k_yy = (gamma * (y @ y.T) + 1.0).pow(3)
    k_xy = (gamma * (x @ y.T) + 1.0).pow(3)

    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        return torch.tensor(float("nan"))

    sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
    sum_xy = k_xy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


def build_inception(device):
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_inception_features(model, loader, device):
    features = []
    for images in loader:
        images = images.to(device)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = imagenet_normalize(images)
        feats = model(images)
        if isinstance(feats, tuple):
            feats = feats[0]
        features.append(feats.detach().cpu())
    return torch.cat(features, dim=0) if features else torch.empty((0, 2048))


def compute_kid(real_features, fake_features, subset_size=100, num_subsets=50, seed=42):
    real_count = real_features.shape[0]
    fake_count = fake_features.shape[0]
    subset_size = min(subset_size, real_count, fake_count)
    if subset_size < 2:
        return float("nan")

    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(num_subsets):
        real_idx = generator.choice(real_count, size=subset_size, replace=False)
        fake_idx = generator.choice(fake_count, size=subset_size, replace=False)
        score = polynomial_mmd(real_features[real_idx], fake_features[fake_idx])
        scores.append(float(score))
    return float(np.mean(scores) * 1000.0)


def build_pairs(records: list[dict], max_pairs: int):
    num_records = len(records)
    if num_records < 2:
        return []

    total_pairs = num_records * (num_records - 1) // 2
    if total_pairs <= max_pairs:
        return [
            (records[i], records[j])
            for i in range(num_records)
            for j in range(i + 1, num_records)
        ]

    rng = np.random.default_rng(42)
    chosen = set()
    pairs = []
    while len(pairs) < max_pairs:
        i, j = sorted(rng.choice(num_records, size=2, replace=False).tolist())
        if (i, j) in chosen:
            continue
        chosen.add((i, j))
        pairs.append((records[i], records[j]))
    return pairs


@torch.no_grad()
def compute_ic_lpips(
    records,
    data_dir: Path,
    image_size: int,
    max_pairs: int,
    batch_size: int,
    metric,
    device,
    region: str = "global",
    local_padding_ratio: float = 0.25,
):
    pairs = build_pairs(records, max_pairs)
    if not pairs:
        return float("nan")

    loader = DataLoader(
        LPIPSPairDataset(
            data_dir,
            pairs,
            image_size,
            region=region,
            local_padding_ratio=local_padding_ratio,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    scores = []
    for img_a, img_b in loader:
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        batch_scores = metric(img_a, img_b).view(-1).detach().cpu().tolist()
        scores.extend(float(score) for score in batch_scores)

    return float(np.mean(scores)) if scores else float("nan")


def nanmean(values: list[float]):
    finite_values = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("nan")


def evaluate_generation(config, device):
    real_dir = Path(config["paths"]["real_holdout_dir"])
    fake_dir = Path(config["paths"]["generated_dir"])
    image_size = int(config["generation"]["image_size"])
    feature_batch_size = int(config["generation"]["feature_batch_size"])
    subset_size = int(config["generation"]["kid_subset_size"])
    num_subsets = int(config["generation"]["kid_num_subsets"])
    max_pairs = int(config["generation"]["ic_lpips_max_pairs_per_category"])
    lpips_backbone = str(config["generation"].get("ic_lpips_backbone", "alex"))
    lpips_batch_size = int(config["generation"].get("lpips_batch_size", 32))
    local_image_size = int(config["generation"].get("local_image_size", image_size))
    local_padding_ratio = float(config["generation"].get("local_padding_ratio", 0.25))
    if local_image_size <= 0:
        raise ValueError("generation.local_image_size must be positive")
    if not np.isfinite(local_padding_ratio) or local_padding_ratio < 0:
        raise ValueError("generation.local_padding_ratio must be finite and non-negative")

    real_records = load_real_metadata(real_dir)
    fake_records = load_generated_metadata(fake_dir)

    real_grouped = group_records_by_task(real_records)
    fake_grouped = group_records_by_task(fake_records)

    all_task_keys = sorted(set(real_grouped) | set(fake_grouped))
    inception = build_inception(device)
    import lpips

    lpips_metric = lpips.LPIPS(net=lpips_backbone).to(device)
    lpips_metric.eval()
    task_results = {}
    skipped_tasks = {}

    for task_key in all_task_keys:
        real_task_records = real_grouped.get(task_key, [])
        fake_task_records = fake_grouped.get(task_key, [])

        if not real_task_records:
            skipped_tasks[task_key] = {"reason": "missing real samples"}
            continue
        if not fake_task_records:
            skipped_tasks[task_key] = {"reason": "missing generated samples"}
            continue

        real_local_records, real_empty_mask_records = partition_records_by_mask_validity(
            real_dir, real_task_records
        )
        fake_local_records, fake_empty_mask_records = partition_records_by_mask_validity(
            fake_dir, fake_task_records
        )
        if real_empty_mask_records or fake_empty_mask_records:
            print(
                f"Warning: {task_key} local metrics exclude empty masks "
                f"(real={len(real_empty_mask_records)}, fake={len(fake_empty_mask_records)}). "
                "Global metrics still include every image."
            )

        real_global_loader = DataLoader(
            ImageOnlyDataset(real_dir, real_task_records, image_size, region="global"),
            batch_size=feature_batch_size,
            shuffle=False,
            num_workers=0,
        )
        fake_global_loader = DataLoader(
            ImageOnlyDataset(fake_dir, fake_task_records, image_size, region="global"),
            batch_size=feature_batch_size,
            shuffle=False,
            num_workers=0,
        )

        real_local_loader = DataLoader(
            ImageOnlyDataset(
                real_dir,
                real_local_records,
                local_image_size,
                region="local",
                local_padding_ratio=local_padding_ratio,
            ),
            batch_size=feature_batch_size,
            shuffle=False,
            num_workers=0,
        )
        fake_local_loader = DataLoader(
            ImageOnlyDataset(
                fake_dir,
                fake_local_records,
                local_image_size,
                region="local",
                local_padding_ratio=local_padding_ratio,
            ),
            batch_size=feature_batch_size,
            shuffle=False,
            num_workers=0,
        )

        real_global_features = extract_inception_features(inception, real_global_loader, device)
        fake_global_features = extract_inception_features(inception, fake_global_loader, device)
        real_local_features = extract_inception_features(inception, real_local_loader, device)
        fake_local_features = extract_inception_features(inception, fake_local_loader, device)
        global_kid = compute_kid(
            real_global_features,
            fake_global_features,
            subset_size=subset_size,
            num_subsets=num_subsets,
        )
        local_kid = compute_kid(
            real_local_features,
            fake_local_features,
            subset_size=subset_size,
            num_subsets=num_subsets,
        )
        global_ic_lpips = compute_ic_lpips(
            fake_task_records,
            fake_dir,
            image_size=image_size,
            max_pairs=max_pairs,
            batch_size=lpips_batch_size,
            metric=lpips_metric,
            device=device,
            region="global",
        )
        local_ic_lpips = compute_ic_lpips(
            fake_local_records,
            fake_dir,
            image_size=local_image_size,
            max_pairs=max_pairs,
            batch_size=lpips_batch_size,
            metric=lpips_metric,
            device=device,
            region="local",
            local_padding_ratio=local_padding_ratio,
        )

        object_token = fake_task_records[0]["object_token"]
        defect_token = fake_task_records[0]["defect_token"]
        task_results[task_key] = {
            "object_token": object_token,
            "defect_token": defect_token,
            "global_kid": global_kid,
            "local_kid": local_kid,
            "global_ic_lpips": global_ic_lpips,
            "local_ic_lpips": local_ic_lpips,
            # Backward-compatible aliases. These retain the historical full-image semantics.
            "kid": global_kid,
            "ic_lpips": global_ic_lpips,
            "num_real": len(real_task_records),
            "num_fake": len(fake_task_records),
            "num_real_local": len(real_local_records),
            "num_fake_local": len(fake_local_records),
            "num_real_empty_masks": len(real_empty_mask_records),
            "num_fake_empty_masks": len(fake_empty_mask_records),
            "real_mask_valid_rate": len(real_local_records) / len(real_task_records),
            "fake_mask_valid_rate": len(fake_local_records) / len(fake_task_records),
        }

    summary = {
        "global_kid_mean": nanmean([result["global_kid"] for result in task_results.values()]),
        "local_kid_mean": nanmean([result["local_kid"] for result in task_results.values()]),
        "global_ic_lpips_mean": nanmean(
            [result["global_ic_lpips"] for result in task_results.values()]
        ),
        "local_ic_lpips_mean": nanmean(
            [result["local_ic_lpips"] for result in task_results.values()]
        ),
        # Backward-compatible aliases. These retain the historical full-image semantics.
        "kid_mean": nanmean([result["global_kid"] for result in task_results.values()]),
        "ic_lpips_mean": nanmean(
            [result["global_ic_lpips"] for result in task_results.values()]
        ),
        "num_tasks_total": len(all_task_keys),
        "num_tasks_evaluated": len(task_results),
        "num_real_total": int(sum(result["num_real"] for result in task_results.values())),
        "num_fake_total": int(sum(result["num_fake"] for result in task_results.values())),
        "num_real_local_total": int(
            sum(result["num_real_local"] for result in task_results.values())
        ),
        "num_fake_local_total": int(
            sum(result["num_fake_local"] for result in task_results.values())
        ),
        "num_real_empty_masks_total": int(
            sum(result["num_real_empty_masks"] for result in task_results.values())
        ),
        "num_fake_empty_masks_total": int(
            sum(result["num_fake_empty_masks"] for result in task_results.values())
        ),
    }
    summary["real_mask_valid_rate"] = (
        summary["num_real_local_total"] / summary["num_real_total"]
        if summary["num_real_total"]
        else float("nan")
    )
    summary["fake_mask_valid_rate"] = (
        summary["num_fake_local_total"] / summary["num_fake_total"]
        if summary["num_fake_total"]
        else float("nan")
    )

    return {
        "metric_protocol": {
            "global": "full image resized to generation.image_size",
            "local": "square defect-mask bounding-box ROI with configurable context padding",
            "empty_mask_policy": (
                "included in global metrics; excluded from local metrics and reported via "
                "mask-validity counts/rates"
            ),
            "local_image_size": local_image_size,
            "local_padding_ratio": local_padding_ratio,
            "kid_scale": "polynomial MMD multiplied by 1000",
        },
        "summary": summary,
        "tasks": task_results,
        "skipped_tasks": skipped_tasks,
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    apply_config_overrides(config, args.overrides)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    report = {"generation": evaluate_generation(config, device)}

    output_path = Path(config["paths"]["report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
