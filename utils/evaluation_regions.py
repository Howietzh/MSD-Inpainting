import numpy as np


def square_defect_roi_bounds(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Return PIL-style bounds for a padded square ROI enclosing a defect mask."""
    if mask.ndim != 2:
        raise ValueError(f"defect mask must be 2-D, got shape {mask.shape}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(padding_ratio) or padding_ratio < 0:
        raise ValueError("padding_ratio must be finite and non-negative")

    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("defect mask is empty")

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    box_width = x_max - x_min
    box_height = y_max - y_min
    side = max(1, int(np.ceil(max(box_width, box_height) * (1.0 + 2.0 * padding_ratio))))
    side = min(side, image_width, image_height)

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(left, 0), image_width - side)
    top = min(max(top, 0), image_height - side)
    return left, top, left + side, top + side
