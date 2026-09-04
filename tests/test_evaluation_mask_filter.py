import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class EvaluationMaskFilterTest(unittest.TestCase):
    def test_partitions_empty_masks_without_dropping_valid_masks(self):
        from utils.evaluation_regions import partition_records_by_mask_validity

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            masks_dir = root / "defect_masks"
            masks_dir.mkdir()
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(masks_dir / "empty.png")
            valid_mask = np.zeros((8, 8), dtype=np.uint8)
            valid_mask[2:4, 3:5] = 255
            Image.fromarray(valid_mask).save(masks_dir / "valid.png")
            records = [
                {"defect_mask_path": "defect_masks/empty.png"},
                {"defect_mask_path": "defect_masks/valid.png"},
            ]

            valid, empty = partition_records_by_mask_validity(root, records)

            self.assertEqual(valid, [records[1]])
            self.assertEqual(empty, [records[0]])


if __name__ == "__main__":
    unittest.main()
