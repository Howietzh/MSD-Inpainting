import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.segmentation_dataset import (
    COMPONENT_DEFECT_CLASS_MAP,
    SegmentationTransform,
    task_key,
)


def split_real_records_by_task(records: list[dict], validation_ratio: float, seed: int):
    train_records = []
    validation_records = []
    split_manifest = {}
    grouped = defaultdict(list)
    for record in records:
        grouped[task_key(record)].append(record)

    for key in sorted(grouped):
        task_records = list(grouped[key])
        random.Random(seed + sum(map(ord, key))).shuffle(task_records)
        validation_count = max(1, int(round(len(task_records) * validation_ratio)))
        validation_count = min(validation_count, len(task_records) - 1)
        validation = task_records[:validation_count]
        train = task_records[validation_count:]
        validation_records.extend(validation)
        train_records.extend(train)
        split_manifest[key] = {
            "train": len(train),
            "validation": len(validation),
            "train_paths": [record["image_path"] for record in train],
            "validation_paths": [record["image_path"] for record in validation],
        }
    return train_records, validation_records, split_manifest


def limit_records_per_task(records: list[dict], limit: int | None):
    if limit is None:
        return records
    grouped = defaultdict(list)
    for record in records:
        grouped[task_key(record)].append(record)
    return [record for key in sorted(grouped) for record in grouped[key][:limit]]


def semantic_class_id(record: dict) -> int:
    return COMPONENT_DEFECT_CLASS_MAP[task_key(record)]


def load_image_and_semantic_mask(record: dict):
    data_dir = Path(record["_data_dir"])
    image = np.asarray(Image.open(data_dir / record["image_path"]).convert("RGB"))
    source_mask = np.asarray(Image.open(data_dir / record["defect_mask_path"]).convert("L"))
    if image.shape[:2] != source_mask.shape:
        raise ValueError(
            f"Image and mask dimensions differ for {record['image_path']}: "
            f"image={image.shape[:2]}, mask={source_mask.shape}"
        )
    semantic_mask = np.zeros_like(source_mask, dtype=np.uint8)
    semantic_mask[source_mask > 0] = semantic_class_id(record)
    return image, semantic_mask


def paste_defect(receiver_image, receiver_mask, donor_image, donor_mask, class_id: int):
    donor_foreground = donor_mask == class_id
    coords = cv2.findNonZero(donor_foreground.astype(np.uint8))
    if coords is None:
        return receiver_image, receiver_mask

    x, y, width, height = cv2.boundingRect(coords)
    donor_patch = donor_image[y : y + height, x : x + width]
    donor_mask_patch = donor_foreground[y : y + height, x : x + width]
    if donor_patch.size == 0 or not np.any(donor_mask_patch):
        return receiver_image, receiver_mask

    image_height, image_width = receiver_mask.shape
    max_scale = min(image_width / max(width, 1), image_height / max(height, 1), 1.5)
    min_scale = min(0.7, max_scale)
    scale = random.uniform(min_scale, max_scale) if max_scale > min_scale else max_scale
    target_width = max(1, min(image_width, int(round(width * scale))))
    target_height = max(1, min(image_height, int(round(height * scale))))

    donor_patch = cv2.resize(donor_patch, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    donor_mask_patch = cv2.resize(
        donor_mask_patch.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    if target_width >= image_width or target_height >= image_height:
        paste_x = max(0, (image_width - target_width) // 2)
        paste_y = max(0, (image_height - target_height) // 2)
    else:
        paste_x = random.randint(0, image_width - target_width)
        paste_y = random.randint(0, image_height - target_height)

    output_image = receiver_image.copy()
    output_mask = receiver_mask.copy()
    region = output_image[paste_y : paste_y + target_height, paste_x : paste_x + target_width]
    mask_region = output_mask[paste_y : paste_y + target_height, paste_x : paste_x + target_width]
    alpha = donor_mask_patch[..., None]
    region[:] = np.where(alpha, donor_patch, region)
    mask_region[donor_mask_patch] = class_id
    return output_image, output_mask


class AblationSegmentationDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        transform: SegmentationTransform,
        copy_paste: bool = False,
        copy_paste_probability: float = 1.0,
    ):
        self.records = records
        self.transform = transform
        self.copy_paste = copy_paste
        self.copy_paste_probability = copy_paste_probability
        self.records_by_task = defaultdict(list)
        for index, record in enumerate(records):
            self.records_by_task[task_key(record)].append(index)

    def __len__(self):
        return len(self.records)

    def _sample_donor_index(self, receiver_index: int, key: str):
        candidates = self.records_by_task[key]
        if len(candidates) <= 1:
            return receiver_index
        donor_index = receiver_index
        while donor_index == receiver_index:
            donor_index = random.choice(candidates)
        return donor_index

    def __getitem__(self, index):
        record = self.records[index]
        image, mask = load_image_and_semantic_mask(record)
        if self.copy_paste and random.random() < self.copy_paste_probability:
            donor_index = self._sample_donor_index(index, task_key(record))
            donor_record = self.records[donor_index]
            donor_image, donor_mask = load_image_and_semantic_mask(donor_record)
            image, mask = paste_defect(
                image,
                mask,
                donor_image,
                donor_mask,
                semantic_class_id(record),
            )

        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
            "task_key": task_key(record),
        }

