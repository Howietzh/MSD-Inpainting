import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from utils.config_overrides import apply_config_overrides


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "generation", "classification", "localization"],
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted paths, e.g. --set generation.feature_batch_size=8",
    )
    return parser.parse_args()


def load_metadata(data_dir: Path):
    metadata_path = data_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def group_records_by_object(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("object_token", "unknown")].append(record)
    return grouped


def read_rgb_image(path: Path, size: int):
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), resample=Image.BILINEAR)


def read_mask(path: Path, size: int):
    mask = Image.open(path).convert("L")
    mask = mask.resize((size, size), resample=Image.NEAREST)
    mask_np = np.array(mask, dtype=np.uint8)
    return (mask_np > 0).astype(np.uint8)


def image_to_tensor(image: Image.Image):
    tensor = transforms.ToTensor()(image)
    return tensor


def imagenet_normalize(tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


class ImageOnlyDataset(Dataset):
    def __init__(self, data_dir: Path, records: list[dict], image_size: int):
        self.data_dir = data_dir
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_rgb_image(self.data_dir / record["image_path"], self.image_size)
        return image_to_tensor(image), record


class ClassificationDataset(Dataset):
    def __init__(self, data_dir: Path, records: list[dict], label_to_idx: dict[str, int], image_size: int):
        self.data_dir = data_dir
        self.records = records
        self.label_to_idx = label_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.data_dir / record["image_path"]).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[record["defect_token"]]
        return image, label


class LocalizationDataset(Dataset):
    def __init__(self, data_dir: Path, records: list[dict], image_size: int, include_masks: bool = True):
        self.data_dir = data_dir
        self.records = records
        self.include_masks = include_masks
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.data_dir / record["image_path"]).convert("RGB")
        image = self.image_transform(image)
        if not self.include_masks:
            return image

        mask = read_mask(self.data_dir / record["defect_mask_path"], image.shape[-1])
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._block(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce_loss
    return loss.mean()


def polynomial_mmd(x, y):
    dim = x.shape[1]
    gamma = 1.0 / dim

    k_xx = (gamma * (x @ x.T) + 1.0).pow(3)
    k_yy = (gamma * (y @ y.T) + 1.0).pow(3)
    k_xy = (gamma * (x @ y.T) + 1.0).pow(3)

    m = x.shape[0]
    n = y.shape[0]

    sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
    sum_xy = k_xy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


def binary_clf_curve(y_true, y_score):
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    return fps.astype(np.float64), tps.astype(np.float64), thresholds.astype(np.float64)


def roc_auc_score_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score).astype(np.float64)
    positives = y_true.sum()
    negatives = y_true.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    fps, tps, _ = binary_clf_curve(y_true, y_score)
    fpr = np.r_[0.0, fps / negatives, 1.0]
    tpr = np.r_[0.0, tps / positives, 1.0]
    return float(np.trapz(tpr, fpr))


def average_precision_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score).astype(np.float64)
    positives = y_true.sum()
    if positives == 0:
        return float("nan")

    fps, tps, _ = binary_clf_curve(y_true, y_score)
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def f1_max_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score).astype(np.float64)
    positives = y_true.sum()
    if positives == 0:
        return float("nan")

    fps, tps, _ = binary_clf_curve(y_true, y_score)
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / positives
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.max(f1))


def compute_pro(masks: list[np.ndarray], score_maps: list[np.ndarray], max_fpr: float = 0.3, num_thresholds: int = 200):
    all_scores = np.concatenate([score.ravel() for score in score_maps])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    pros = []
    fprs = []
    total_negatives = sum((mask == 0).sum() for mask in masks)

    for threshold in thresholds:
        fp = 0
        region_overlaps = []

        for mask, score_map in zip(masks, score_maps):
            pred = (score_map >= threshold).astype(np.uint8)
            fp += ((pred == 1) & (mask == 0)).sum()

            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            for label_idx in range(1, num_labels):
                region = labels == label_idx
                denom = region.sum()
                if denom > 0:
                    region_overlaps.append(float((pred[region] == 1).sum() / denom))

        if not region_overlaps:
            continue

        fpr = fp / max(total_negatives, 1)
        if fpr <= max_fpr:
            fprs.append(fpr)
            pros.append(float(np.mean(region_overlaps)))

    if len(pros) < 2:
        return float("nan")

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    scaled_fprs = fprs / max_fpr
    return float(np.trapz(pros, scaled_fprs))


def build_inception(device):
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_inception_features(model, loader, device):
    features = []
    for images, _ in loader:
        images = images.to(device)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = imagenet_normalize(images)
        feats = model(images)
        if isinstance(feats, tuple):
            feats = feats[0]
        features.append(feats.detach().cpu())
    return torch.cat(features, dim=0)


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


@torch.no_grad()
def compute_ic_lpips(records, data_dir: Path, image_size: int, max_pairs: int, device):
    metric = lpips.LPIPS(net="vgg").to(device)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["defect_token"]].append(record)

    category_scores = []
    for defect_token, defect_records in grouped.items():
        if len(defect_records) < 2:
            continue

        pairs = []
        total_pairs = len(defect_records) * (len(defect_records) - 1) // 2
        if total_pairs <= max_pairs:
            for i in range(len(defect_records)):
                for j in range(i + 1, len(defect_records)):
                    pairs.append((defect_records[i], defect_records[j]))
        else:
            rng = np.random.default_rng(42)
            chosen = set()
            while len(pairs) < max_pairs:
                i, j = sorted(rng.choice(len(defect_records), size=2, replace=False).tolist())
                if (i, j) not in chosen:
                    chosen.add((i, j))
                    pairs.append((defect_records[i], defect_records[j]))

        pair_scores = []
        for rec_a, rec_b in pairs:
            img_a = read_rgb_image(data_dir / rec_a["image_path"], image_size)
            img_b = read_rgb_image(data_dir / rec_b["image_path"], image_size)
            ten_a = image_to_tensor(img_a).unsqueeze(0).to(device) * 2 - 1
            ten_b = image_to_tensor(img_b).unsqueeze(0).to(device) * 2 - 1
            score = metric(ten_a, ten_b)
            pair_scores.append(float(score.item()))

        if pair_scores:
            category_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(category_scores)) if category_scores else float("nan")


def evaluate_generation(config, device):
    real_dir = Path(config["paths"]["real_holdout_dir"])
    fake_dir = Path(config["paths"]["generated_dir"])
    image_size = int(config["generation"]["image_size"])
    batch_size = int(config["generation"]["feature_batch_size"])
    subset_size = int(config["generation"]["kid_subset_size"])
    num_subsets = int(config["generation"]["kid_num_subsets"])
    max_pairs = int(config["generation"]["ic_lpips_max_pairs_per_category"])

    real_records = load_metadata(real_dir)
    fake_records = load_metadata(fake_dir)

    real_grouped = group_records_by_object(real_records)
    fake_grouped = group_records_by_object(fake_records)

    inception = build_inception(device)
    results = {}

    for object_token in sorted(set(real_grouped.keys()) & set(fake_grouped.keys())):
        real_loader = DataLoader(
            ImageOnlyDataset(real_dir, real_grouped[object_token], image_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        fake_loader = DataLoader(
            ImageOnlyDataset(fake_dir, fake_grouped[object_token], image_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        real_features = extract_inception_features(inception, real_loader, device)
        fake_features = extract_inception_features(inception, fake_loader, device)

        kid = compute_kid(real_features, fake_features, subset_size=subset_size, num_subsets=num_subsets)
        ic_lpips = compute_ic_lpips(fake_grouped[object_token], fake_dir, image_size, max_pairs, device)
        results[object_token] = {
            "kid": kid,
            "ic_lpips": ic_lpips,
            "num_real": len(real_grouped[object_token]),
            "num_fake": len(fake_grouped[object_token]),
        }

    return results


def build_resnet34(num_classes, pretrained):
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_classifier(model, train_loader, device, epochs, lr):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_classifier(model, eval_loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def evaluate_classification(config, device):
    train_dir = Path(config["paths"]["generated_dir"])
    test_dir = Path(config["paths"]["real_holdout_dir"])
    image_size = int(config["classification"]["image_size"])
    batch_size = int(config["classification"]["batch_size"])
    epochs = int(config["classification"]["epochs"])
    lr = float(config["classification"]["learning_rate"])
    pretrained = bool(config["classification"].get("pretrained_backbone", False))

    train_records = load_metadata(train_dir)
    test_records = load_metadata(test_dir)

    train_grouped = group_records_by_object(train_records)
    test_grouped = group_records_by_object(test_records)

    results = {}
    for object_token in sorted(set(train_grouped.keys()) & set(test_grouped.keys())):
        train_defects = sorted({record["defect_token"] for record in train_grouped[object_token]})
        test_defects = sorted({record["defect_token"] for record in test_grouped[object_token]})
        label_space = sorted(set(train_defects) & set(test_defects))
        if len(label_space) < 2:
            results[object_token] = {
                "accuracy": float("nan"),
                "num_classes": len(label_space),
                "note": "Need at least 2 shared defect categories for classification evaluation.",
            }
            continue

        label_to_idx = {label: idx for idx, label in enumerate(label_space)}
        train_subset = [record for record in train_grouped[object_token] if record["defect_token"] in label_to_idx]
        test_subset = [record for record in test_grouped[object_token] if record["defect_token"] in label_to_idx]

        train_loader = DataLoader(
            ClassificationDataset(train_dir, train_subset, label_to_idx, image_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            ClassificationDataset(test_dir, test_subset, label_to_idx, image_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = build_resnet34(len(label_to_idx), pretrained)
        train_classifier(model, train_loader, device, epochs, lr)
        accuracy = evaluate_classifier(model, test_loader, device)
        results[object_token] = {
            "accuracy": accuracy * 100.0,
            "num_classes": len(label_to_idx),
            "num_train": len(train_subset),
            "num_test": len(test_subset),
        }

    return results


def train_localizer(model, train_loader, device, epochs, lr, focal_alpha, focal_gamma):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = sigmoid_focal_loss(logits, masks, alpha=focal_alpha, gamma=focal_gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_localizer(model, loader, device):
    model.eval()
    score_maps = []
    gt_masks = []
    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        score_maps.extend([prob[0] for prob in probs])
        gt_masks.extend([mask[0].numpy().astype(np.uint8) for mask in masks])
    return score_maps, gt_masks


def evaluate_localization(config, device):
    train_dir = Path(config["paths"]["generated_dir"])
    test_dir = Path(config["paths"]["real_holdout_dir"])
    normal_dir = Path(config["paths"]["normal_dir"])
    image_size = int(config["localization"]["image_size"])
    batch_size = int(config["localization"]["batch_size"])
    epochs = int(config["localization"]["epochs"])
    lr = float(config["localization"]["learning_rate"])
    focal_alpha = float(config["localization"].get("focal_alpha", 0.25))
    focal_gamma = float(config["localization"].get("focal_gamma", 2.0))

    train_records = load_metadata(train_dir)
    test_records = load_metadata(test_dir)
    normal_records = load_metadata(normal_dir)

    train_grouped = group_records_by_object(train_records)
    test_grouped = group_records_by_object(test_records)
    normal_grouped = group_records_by_object(normal_records)

    results = {}
    for object_token in sorted(set(train_grouped.keys()) & set(test_grouped.keys())):
        train_records_object = train_grouped[object_token]
        normal_records_object = normal_grouped.get(object_token, [])
        test_records_object = test_grouped[object_token]

        if not normal_records_object:
            results[object_token] = {
                "auroc": float("nan"),
                "ap": float("nan"),
                "f1_max": float("nan"),
                "pro": float("nan"),
                "note": "No normal samples available for localization training.",
            }
            continue

        zero_mask_records = []
        for record in normal_records_object:
            cloned = dict(record)
            zero_mask_records.append(cloned)

        train_dataset_records = train_records_object + zero_mask_records

        class LocalizationTrainDataset(Dataset):
            def __init__(self, generated_dir, normal_dir, generated_records, normal_records, image_size):
                self.generated_dir = generated_dir
                self.normal_dir = normal_dir
                self.generated_records = generated_records
                self.normal_records = normal_records
                self.image_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                self.image_size = image_size

            def __len__(self):
                return len(self.generated_records) + len(self.normal_records)

            def __getitem__(self, index):
                if index < len(self.generated_records):
                    record = self.generated_records[index]
                    data_dir = self.generated_dir
                    mask = read_mask(data_dir / record["defect_mask_path"], self.image_size)
                else:
                    record = self.normal_records[index - len(self.generated_records)]
                    data_dir = self.normal_dir
                    mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

                image = Image.open(data_dir / record["image_path"]).convert("RGB")
                image = self.image_transform(image)
                mask = torch.from_numpy(mask).float().unsqueeze(0)
                return image, mask

        train_loader = DataLoader(
            LocalizationTrainDataset(train_dir, normal_dir, train_records_object, normal_records_object, image_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            LocalizationDataset(test_dir, test_records_object, image_size, include_masks=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = SimpleUNet()
        train_localizer(model, train_loader, device, epochs, lr, focal_alpha, focal_gamma)
        score_maps, gt_masks = predict_localizer(model, test_loader, device)

        y_true = np.concatenate([mask.ravel() for mask in gt_masks])
        y_score = np.concatenate([score_map.ravel() for score_map in score_maps])

        results[object_token] = {
            "auroc": roc_auc_score_binary(y_true, y_score),
            "ap": average_precision_binary(y_true, y_score),
            "f1_max": f1_max_binary(y_true, y_score),
            "pro": compute_pro(gt_masks, score_maps),
            "num_train": len(train_records_object) + len(normal_records_object),
            "num_test": len(test_records_object),
        }

    return results


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    apply_config_overrides(config, args.overrides)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    report = {}

    if args.task in {"all", "generation"}:
        report["generation"] = evaluate_generation(config, device)
    if args.task in {"all", "classification"}:
        report["classification"] = evaluate_classification(config, device)
    if args.task in {"all", "localization"}:
        report["localization"] = evaluate_localization(config, device)

    output_path = Path(config["paths"]["report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
