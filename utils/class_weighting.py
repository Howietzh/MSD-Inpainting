import json
from pathlib import Path

import numpy as np
from PIL import Image


def compute_defect_class_weights(data_dir, defect_tokens, max_weight: float = 6.0):
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot compute defect class weights because metadata is missing: {metadata_path}")

    defect_ratios_by_token = {token: [] for token in defect_tokens}

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            defect_token = item.get("defect_token")
            defect_mask_path = item.get("defect_mask_path")
            if defect_token not in defect_ratios_by_token or defect_mask_path is None:
                continue

            mask = np.array(Image.open(data_dir / defect_mask_path).convert("L"))
            defect_mask = mask > 127
            defect_pixels = int(defect_mask.sum())
            total_pixels = int(defect_mask.size)

            if total_pixels <= 0:
                continue

            defect_ratios_by_token[defect_token].append(defect_pixels / total_pixels)

    if not any(defect_ratios_by_token.values()):
        return {token: 1.0 for token in defect_tokens}

    raw_weights = {}
    for token in defect_tokens:
        defect_ratios = defect_ratios_by_token[token]
        if not defect_ratios:
            raw_weights[token] = 1.0
            continue

        mean_ratio = float(np.mean(defect_ratios))
        # raw_weights[token] = float(np.sqrt(1.0 / max(mean_ratio, 1e-6)))
        raw_weights[token] = float(np.log10(1.0 / max(mean_ratio, 1e-6)))

    normalized_weights = {}
    for token, raw_weight in raw_weights.items():
        normalized = raw_weight
        normalized_weights[token] = float(normalized)

    return normalized_weights


def resolve_defect_class_weights(config):
    configured_weights = config.get("loss_weights", {}).get("defect_class_weights", {}) or {}
    defect_tokens = config.get("defect_tokens", [])

    if configured_weights:
        return {token: float(configured_weights[token]) for token in defect_tokens}

    if not defect_tokens:
        return {}

    return compute_defect_class_weights(config["paths"]["data_dir"], defect_tokens)
