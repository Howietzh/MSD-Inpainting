import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


COMPONENT_DEFECT_CLASS_MAP = {
    "<flexible_printed_circuit_crack>::<flexible_printed_circuit>": 1,
    "<end_face_scratch>::<end_face>": 2,
    "<lens_scratch>::<lens>": 3,
    "<foreign_particle>::<end_face>": 4,
    "<foreign_particle>::<lens>": 5,
}
CLASS_NAMES = [
    "background",
    "flexible_printed_circuit_crack_on_flexible_printed_circuit",
    "end_face_scratch_on_end_face",
    "lens_scratch_on_lens",
    "foreign_particle_on_end_face",
    "foreign_particle_on_lens",
]
NUM_CLASSES = len(CLASS_NAMES)
if sorted(COMPONENT_DEFECT_CLASS_MAP.values()) != list(range(1, NUM_CLASSES)):
    raise ValueError("COMPONENT_DEFECT_CLASS_MAP values must be contiguous and start at 1.")

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def task_key(record: dict) -> str:
    return f"{record['defect_token']}::{record['object_token']}"


def read_metadata(data_dir: Path, generated: bool) -> list[dict]:
    metadata_paths = sorted(data_dir.glob("metadata_gpu*.jsonl")) if generated else [data_dir / "metadata.jsonl"]
    if not metadata_paths or any(not path.exists() for path in metadata_paths):
        pattern = "metadata_gpu*.jsonl" if generated else "metadata.jsonl"
        raise FileNotFoundError(f"Missing {pattern} under {data_dir}")

    records = []
    for metadata_path in metadata_paths:
        with open(metadata_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                for field in ("image_path", "defect_mask_path", "object_token", "defect_token"):
                    if not record.get(field):
                        raise ValueError(f"Missing {field} in {metadata_path}:{line_number}")
                if task_key(record) not in COMPONENT_DEFECT_CLASS_MAP:
                    raise ValueError(
                        f"Unknown component-defect pair {task_key(record)} in {metadata_path}:{line_number}"
                    )
                for path_field in ("image_path", "defect_mask_path"):
                    referenced_path = data_dir / record[path_field]
                    if not referenced_path.is_file():
                        raise FileNotFoundError(
                            f"Missing {path_field} referenced by {metadata_path}:{line_number}: "
                            f"{referenced_path}"
                        )
                record = dict(record)
                record["_data_dir"] = str(data_dir.resolve())
                records.append(record)
    return records


def build_balanced_splits(
    experiment_records: dict[str, list[dict]],
    validation_ratio: float,
    seed: int,
) -> tuple[dict[str, dict[str, list[dict]]], dict[str, int]]:
    grouped = {}
    all_tasks = set()
    for experiment, records in experiment_records.items():
        by_task = defaultdict(list)
        for record in records:
            by_task[task_key(record)].append(record)
        grouped[experiment] = by_task
        all_tasks.update(by_task)

    result = {experiment: {"train": [], "validation": []} for experiment in experiment_records}
    common_counts = {}
    for key in sorted(all_tasks):
        counts = [len(grouped[experiment].get(key, [])) for experiment in experiment_records]
        common_count = min(counts)
        if common_count < 2:
            raise ValueError(f"Task {key} needs at least two records in every experiment; counts={counts}")
        common_counts[key] = common_count
        validation_count = max(1, int(round(common_count * validation_ratio)))
        validation_count = min(validation_count, common_count - 1)

        for experiment_index, experiment in enumerate(experiment_records):
            records = list(grouped[experiment][key])
            random.Random(seed + experiment_index * 100000 + sum(map(ord, key))).shuffle(records)
            selected = records[:common_count]
            result[experiment]["validation"].extend(selected[:validation_count])
            result[experiment]["train"].extend(selected[validation_count:])
    return result, common_counts


class SegmentationTransform:
    def __init__(
        self,
        size: int,
        training: bool,
        resize_min: float = 0.75,
        resize_max: float = 1.25,
        clahe_probability: float = 0.5,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
    ):
        self.size = size
        self.training = training
        self.resize_min = resize_min
        self.resize_max = resize_max
        self.clahe_probability = clahe_probability
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_grid_size, clahe_grid_size),
        )

    def _random_resize(self, image: np.ndarray, mask: np.ndarray):
        scale = random.uniform(self.resize_min, self.resize_max)
        resized_size = max(1, int(round(self.size * scale)))
        image = cv2.resize(image, (resized_size, resized_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (resized_size, resized_size), interpolation=cv2.INTER_NEAREST)

        if resized_size >= self.size:
            top = random.randint(0, resized_size - self.size)
            left = random.randint(0, resized_size - self.size)
            return (
                image[top : top + self.size, left : left + self.size],
                mask[top : top + self.size, left : left + self.size],
            )

        pad_height = self.size - resized_size
        pad_width = self.size - resized_size
        top = random.randint(0, pad_height)
        left = random.randint(0, pad_width)
        image = cv2.copyMakeBorder(
            image,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv2.BORDER_REFLECT_101,
        )
        mask = cv2.copyMakeBorder(
            mask,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        return image, mask

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if self.training:
            image, mask = self._random_resize(image, mask)
            if random.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if random.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)
            rotations = random.randint(0, 3)
            if rotations:
                image, mask = np.rot90(image, rotations), np.rot90(mask, rotations)
            if random.random() < self.clahe_probability:
                image = self._apply_clahe(np.ascontiguousarray(image))
        else:
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        mask = np.ascontiguousarray(mask, dtype=np.int64)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask)


class DefectSegmentationDataset(Dataset):
    def __init__(self, records: list[dict], transform: SegmentationTransform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data_dir = Path(record["_data_dir"])
        image = np.asarray(Image.open(data_dir / record["image_path"]).convert("RGB"))
        source_mask = np.asarray(Image.open(data_dir / record["defect_mask_path"]).convert("L"))
        if image.shape[:2] != source_mask.shape:
            raise ValueError(
                f"Image and mask dimensions differ for {record['image_path']}: "
                f"image={image.shape[:2]}, mask={source_mask.shape}"
            )
        # Empty generated masks stay all-background so generation failures affect downstream quality.
        semantic_mask = np.zeros_like(source_mask, dtype=np.uint8)
        semantic_mask[source_mask > 0] = COMPONENT_DEFECT_CLASS_MAP[task_key(record)]
        image_tensor, mask_tensor = self.transform(image, semantic_mask)
        return {
            "pixel_values": image_tensor,
            "labels": mask_tensor,
            "task_key": task_key(record),
        }
