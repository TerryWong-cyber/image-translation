import unittest

import numpy as np

from image_translation.components.ocr.paddleocr3_adapter import (
    paddleocr_detection_boxes,
    paddleocr_results_to_legacy,
)


class PaddleOcr3AdapterTest(unittest.TestCase):
    def test_converts_pipeline_result_to_legacy_contract(self):
        results = [
            {
                "rec_polys": np.array(
                    [
                        [[1, 2], [11, 2], [11, 8], [1, 8]],
                        [[20, 3], [30, 3], [30, 9], [20, 9]],
                    ],
                    dtype=np.int16,
                ),
                "rec_texts": ["hello", "world"],
                "rec_scores": np.array([0.95, 0.75], dtype=np.float32),
            }
        ]

        converted = paddleocr_results_to_legacy(results)

        self.assertEqual(converted[0][0][0][0], [1.0, 2.0])
        self.assertEqual(converted[0][0][1][0], "hello")
        self.assertAlmostEqual(converted[0][0][1][1], 0.95, places=5)
        self.assertEqual(converted[0][1][1][0], "world")

    def test_extracts_unfiltered_detection_polygons(self):
        results = [
            {
                "dt_polys": np.array(
                    [[[1, 2], [11, 2], [11, 8], [1, 8]]],
                    dtype=np.int16,
                ),
                "rec_polys": [],
            }
        ]

        self.assertEqual(
            paddleocr_detection_boxes(results),
            [[[1.0, 2.0], [11.0, 2.0], [11.0, 8.0], [1.0, 8.0]]],
        )

    def test_empty_results_preserve_one_image_wrapper(self):
        self.assertEqual(paddleocr_results_to_legacy([]), [[]])

    def test_rejects_misaligned_recognition_fields(self):
        with self.assertRaisesRegex(ValueError, "must have the same length"):
            paddleocr_results_to_legacy(
                [{"rec_polys": [np.zeros((4, 2))], "rec_texts": [], "rec_scores": []}]
            )

    def test_rejects_unexpected_result_type(self):
        with self.assertRaisesRegex(TypeError, "Unexpected PaddleOCR result type"):
            paddleocr_results_to_legacy([object()])


if __name__ == "__main__":
    unittest.main()
