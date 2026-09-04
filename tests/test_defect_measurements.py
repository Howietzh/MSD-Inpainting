import math

import numpy as np

from utils.defect_measurements import (
    measure_defect_mask,
    measure_linear_mask,
    measure_particle_mask,
    skeletonize_binary,
    skeleton_graph_length,
    summarize_errors,
)


def test_skeleton_graph_length_uses_euclidean_edge_weights():
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[1, 1] = True
    skeleton[1, 2] = True
    skeleton[2, 3] = True
    assert math.isclose(skeleton_graph_length(skeleton), 1.0 + math.sqrt(2.0))


def test_linear_measurement_for_single_pixel_wide_line():
    mask = np.zeros((9, 12), dtype=np.uint8)
    mask[4, 2:9] = 1
    result = measure_linear_mask(mask)
    assert result["area"] == 7.0
    assert math.isclose(result["length"], 6.0)
    assert math.isclose(result["mean_width"], 2.0)


def test_thick_rectangle_is_thinned_to_a_one_pixel_skeleton():
    mask = np.zeros((11, 15), dtype=np.uint8)
    mask[3:8, 3:12] = 1
    skeleton = skeletonize_binary(mask)
    assert np.any(skeleton)
    assert not np.any(skeleton[:-1] & skeleton[1:])


def test_particle_measurement_filters_components_smaller_than_four_pixels():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1, 1:4] = 1
    mask[5:7, 5:7] = 1
    mask[8:10, 8:11] = 1
    result = measure_particle_mask(mask, minimum_area=4)
    assert result["count"] == 2.0
    assert result["total_area"] == 10.0
    assert math.isclose(
        result["mean_equivalent_radius"],
        (math.sqrt(4.0 / math.pi) + math.sqrt(6.0 / math.pi)) / 2.0,
    )


def test_empty_particle_mask_has_zero_measurements():
    result = measure_defect_mask(
        np.zeros((8, 8), dtype=np.uint8),
        "<foreign_particle>::<lens>",
    )
    assert result == {"count": 0.0, "total_area": 0.0, "mean_equivalent_radius": 0.0}


def test_summary_averages_errors_across_seeds_and_images():
    ground_truth = np.asarray([2.0, 4.0])
    predictions = np.asarray([[1.0, 5.0], [3.0, 7.0]])
    result = summarize_errors(
        ground_truth,
        predictions,
        bootstrap_iterations=100,
        bootstrap_seed=7,
    )
    assert result["mae"] == 1.5
    assert math.isclose(result["nmae_percent"], 50.0)
    assert result["ground_truth_mean"] == 3.0
    assert result["max_percentage_error_per_seed"] == [50.0, 75.0]
    assert result["max_percentage_error_mean"] == 62.5
    assert math.isclose(result["max_percentage_error_std"], 12.5 * math.sqrt(2.0))
    assert result["max_percentage_error_worst_seed"] == 75.0
    assert result["num_images"] == 2
    assert result["num_seeds"] == 2
