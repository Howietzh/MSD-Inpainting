from __future__ import annotations

import math

import cv2
import numpy as np


LINEAR_QUANTITIES = ("length", "area", "mean_width")
PARTICLE_QUANTITIES = ("count", "total_area", "mean_equivalent_radius")


def measurement_kind(task: str) -> str:
    """Return the measurement family for a component-defect task key."""
    defect_token = task.split("::", maxsplit=1)[0]
    if defect_token in {
        "<flexible_printed_circuit_crack>",
        "<end_face_scratch>",
        "<lens_scratch>",
    }:
        return "linear"
    if defect_token == "<foreign_particle>":
        return "particle"
    raise ValueError(f"No measurement protocol is defined for task {task!r}")


def quantities_for_task(task: str) -> tuple[str, ...]:
    return LINEAR_QUANTITIES if measurement_kind(task) == "linear" else PARTICLE_QUANTITIES


def _filter_components(mask: np.ndarray, minimum_area: int) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    if minimum_area < 1:
        raise ValueError("minimum_area must be at least 1")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = np.flatnonzero(areas >= minimum_area) + 1
    if keep.size == 0:
        return np.zeros_like(binary), np.empty(0, dtype=np.float64)
    return np.isin(labels, keep).astype(np.uint8), areas[keep - 1]


def _clockwise_neighbours(image: np.ndarray) -> tuple[np.ndarray, ...]:
    padded = np.pad(image, 1, mode="constant", constant_values=False)
    return (
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    )


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    """Compute a deterministic one-pixel skeleton with Zhang-Suen thinning."""
    skeleton = np.asarray(mask, dtype=bool).copy()
    changed = True
    while changed:
        changed = False
        for first_subiteration in (True, False):
            neighbours = _clockwise_neighbours(skeleton)
            neighbour_count = sum(neighbours)
            transitions = sum(
                (~neighbours[index]) & neighbours[(index + 1) % 8]
                for index in range(8)
            )
            p2, _, p4, _, p6, _, p8, _ = neighbours
            if first_subiteration:
                triplet_a = p2 & p4 & p6
                triplet_b = p4 & p6 & p8
            else:
                triplet_a = p2 & p4 & p8
                triplet_b = p2 & p6 & p8
            remove = (
                skeleton
                & (neighbour_count >= 2)
                & (neighbour_count <= 6)
                & (transitions == 1)
                & ~triplet_a
                & ~triplet_b
            )
            if np.any(remove):
                skeleton[remove] = False
                changed = True
    return skeleton


def skeleton_graph_length(skeleton: np.ndarray) -> float:
    """Sum unique 8-neighbour graph edges with unit/diagonal Euclidean weights."""
    pixels = np.asarray(skeleton, dtype=bool)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2-D skeleton, got shape={pixels.shape}")
    horizontal = np.count_nonzero(pixels[:, :-1] & pixels[:, 1:])
    vertical = np.count_nonzero(pixels[:-1, :] & pixels[1:, :])
    diagonal_down_right = np.count_nonzero(pixels[:-1, :-1] & pixels[1:, 1:])
    diagonal_down_left = np.count_nonzero(pixels[:-1, 1:] & pixels[1:, :-1])
    return float(horizontal + vertical + math.sqrt(2.0) * (diagonal_down_right + diagonal_down_left))


def measure_linear_mask(mask: np.ndarray, minimum_area: int = 1) -> dict[str, float]:
    retained, _ = _filter_components(mask, minimum_area)
    area = float(np.count_nonzero(retained))
    if area == 0:
        return {"length": 0.0, "area": 0.0, "mean_width": 0.0}

    skeleton = skeletonize_binary(retained)
    length = skeleton_graph_length(skeleton)
    distances = cv2.distanceTransform(retained, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    mean_width = float((2.0 * distances[skeleton]).mean()) if np.any(skeleton) else 0.0
    return {"length": length, "area": area, "mean_width": mean_width}


def measure_particle_mask(mask: np.ndarray, minimum_area: int = 4) -> dict[str, float]:
    _, areas = _filter_components(mask, minimum_area)
    if areas.size == 0:
        return {"count": 0.0, "total_area": 0.0, "mean_equivalent_radius": 0.0}
    radii = np.sqrt(areas / math.pi)
    return {
        "count": float(areas.size),
        "total_area": float(areas.sum()),
        "mean_equivalent_radius": float(radii.mean()),
    }


def measure_defect_mask(
    mask: np.ndarray,
    task: str,
    particle_minimum_area: int = 4,
    linear_minimum_area: int = 1,
) -> dict[str, float]:
    if measurement_kind(task) == "linear":
        return measure_linear_mask(mask, minimum_area=linear_minimum_area)
    return measure_particle_mask(mask, minimum_area=particle_minimum_area)


def summarize_errors(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    epsilon: float = 1.0e-8,
) -> dict[str, float | list[float]]:
    """Summarize seed-by-image predictions with paired image bootstrap for MAE."""
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if ground_truth.ndim != 1 or predictions.ndim != 2:
        raise ValueError("ground_truth must be [images] and predictions must be [seeds, images]")
    if predictions.shape[1] != ground_truth.size or ground_truth.size == 0:
        raise ValueError(
            f"Incompatible or empty arrays: ground_truth={ground_truth.shape}, predictions={predictions.shape}"
        )
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")

    absolute_errors = np.abs(predictions - ground_truth[None, :])
    per_image_error = absolute_errors.mean(axis=0)
    mae = float(per_image_error.mean())
    denominator = predictions.shape[0] * ground_truth.sum() + epsilon
    nmae = float(100.0 * absolute_errors.sum() / denominator)

    rng = np.random.default_rng(bootstrap_seed)
    sample_count = ground_truth.size
    bootstrap_mae = np.empty(bootstrap_iterations, dtype=np.float64)
    for index in range(bootstrap_iterations):
        sampled_images = rng.integers(0, sample_count, size=sample_count)
        bootstrap_mae[index] = per_image_error[sampled_images].mean()
    ci_low, ci_high = np.percentile(bootstrap_mae, [2.5, 97.5])

    return {
        "ground_truth_mean": float(ground_truth.mean()),
        "ground_truth_std": float(ground_truth.std(ddof=1)) if sample_count > 1 else 0.0,
        "mae": mae,
        "mae_ci_95": [float(ci_low), float(ci_high)],
        "nmae_percent": nmae,
        "num_images": int(sample_count),
        "num_seeds": int(predictions.shape[0]),
    }
