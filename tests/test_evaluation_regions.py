import unittest

import numpy as np

from utils.evaluation_regions import square_defect_roi_bounds


class SquareDefectRoiBoundsTest(unittest.TestCase):
    def test_centered_mask_gets_square_context(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:5, 4:6] = 255

        bounds = square_defect_roi_bounds(mask, 10, 10, padding_ratio=0.5)

        self.assertEqual(bounds, (3, 2, 7, 6))

    def test_boundary_mask_shifts_roi_inside_image(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1

        bounds = square_defect_roi_bounds(mask, 10, 10, padding_ratio=0.5)

        self.assertEqual(bounds, (0, 0, 4, 4))

    def test_empty_mask_is_rejected(self):
        mask = np.zeros((10, 10), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "empty"):
            square_defect_roi_bounds(mask, 10, 10, padding_ratio=0.25)

    def test_invalid_padding_is_rejected(self):
        mask = np.ones((2, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "padding_ratio"):
            square_defect_roi_bounds(mask, 2, 2, padding_ratio=-0.1)


if __name__ == "__main__":
    unittest.main()
