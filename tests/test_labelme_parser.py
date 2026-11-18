from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_manager.infrastructure.parsers.labelme_parser import LabelMeParser


class LabelMeParserTests(unittest.TestCase):
    def test_preserves_known_and_unknown_fields(self) -> None:
        raw = {
            "version": "5.4.1",
            "flags": {"reviewed": True},
            "shapes": [{
                "label": "scratch",
                "points": [[1, 2], [10, 20]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "customShapeValue": 7,
            }],
            "imagePath": "sample.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 200,
            "customRootValue": {"source": "lab"},
        }
        result = LabelMeParser().parse_data(raw)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.document.extra["customRootValue"], {"source": "lab"})
        self.assertEqual(result.document.shapes[0].extra["customShapeValue"], 7)
        self.assertEqual(result.document.to_dict()["shapes"][0]["label"], "scratch")
        self.assertEqual(result.document.category_names, {"scratch"})

    def test_reports_out_of_bounds_and_unsupported_shape(self) -> None:
        raw = {
            "flags": {},
            "shapes": [{
                "label": "point-label",
                "points": [[500, -1]],
                "shape_type": "point",
                "flags": {},
            }],
            "imagePath": "sample.jpg",
            "imageHeight": 100,
            "imageWidth": 200,
        }
        result = LabelMeParser().parse_data(raw)
        codes = {issue.code for issue in result.issues}
        self.assertIn("unsupported_shape", codes)
        self.assertIn("point_out_of_bounds", codes)
        self.assertFalse(result.is_valid)

    def test_empty_annotation_is_informational(self) -> None:
        result = LabelMeParser().parse_data({
            "flags": {}, "shapes": [], "imagePath": "sample.jpg",
            "imageHeight": 10, "imageWidth": 10,
        })
        self.assertTrue(result.is_valid)
        self.assertIn("empty_annotation", {issue.code for issue in result.issues})

    def test_invalid_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            result = LabelMeParser().parse(path)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "json_read_failed")

    def test_image_size_probe_reports_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            image_path.write_bytes(b"placeholder")
            parser = LabelMeParser(image_size_probe=lambda _: (20, 30))
            result = parser.parse_data({
                "flags": {}, "shapes": [], "imagePath": "sample.jpg",
                "imageHeight": 10, "imageWidth": 10,
            }, image_path=image_path)
        self.assertIn("image_size_mismatch", {issue.code for issue in result.issues})


if __name__ == "__main__":
    unittest.main()

