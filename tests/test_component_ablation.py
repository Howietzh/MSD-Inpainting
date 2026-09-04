import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from utils.ablation import build_conditioning_prompt

try:
    import torch
    from losses.defectfill_loss import DefectFillLoss
except ImportError:
    torch = None
    DefectFillLoss = None

try:
    import cv2  # noqa: F401
    from utils.mask_ops import DefectMaskEngine
except ImportError:
    DefectMaskEngine = None


class ComponentAblationTest(unittest.TestCase):
    def test_no_textual_inversion_uses_natural_language_prompt(self):
        config = {
            "ablation": {"use_textual_inversion": False},
            "token_init_phrases": {
                "<lens>": "lens",
                "<lens_scratch>": "lens scratch",
            },
        }
        self.assertEqual(
            build_conditioning_prompt(config, "<lens>", "<lens_scratch>"),
            "a photo of lens with lens scratch",
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_no_dsl_reduces_to_uniform_mse(self):
        criterion = DefectFillLoss(
            lambda_rec=1.0,
            lambda_attn_def=0.0,
            lambda_attn_comp=0.0,
            defect_class_weights={"<lens_scratch>": 2.0},
            use_defect_sensitive_weighting=False,
        )
        prediction = torch.ones((1, 4, 4, 4))
        target = torch.zeros_like(prediction)
        defect_mask = torch.zeros((1, 1, 8, 8))
        defect_mask[:, :, 2:6, 2:6] = 1
        component_mask = torch.ones_like(defect_mask)

        loss, details = criterion(
            prediction,
            target,
            defect_mask,
            component_mask,
            ["<lens_scratch>"],
        )

        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(details["mean_defect_weight"], 2.0)

    @unittest.skipIf(DefectMaskEngine is None, "OpenCV is not installed")
    def test_no_cdme_uses_reference_mask_and_component_constraint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mask_dir = root / "defect_masks"
            mask_dir.mkdir()
            reference = np.zeros((16, 16), dtype=np.uint8)
            reference[4:8, 5:9] = 255
            Image.fromarray(reference).save(mask_dir / "reference.png")
            record = {
                "defect_token": "<lens_scratch>",
                "object_token": "<lens>",
                "defect_mask_path": "defect_masks/reference.png",
            }
            (root / "metadata.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            component = np.zeros((16, 16), dtype=np.uint8)
            component[:, :7] = 255
            engine = DefectMaskEngine(root, root / "stats.json", target_size=16)
            warped = engine.generate_reference_elastic_mask(
                component,
                "<lens_scratch>",
                "<lens>",
                alpha=0.0,
                sigma=1.0,
            )

            self.assertGreater(np.count_nonzero(warped), 0)
            self.assertEqual(np.count_nonzero(warped[:, 7:]), 0)


if __name__ == "__main__":
    unittest.main()
