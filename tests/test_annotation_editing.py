from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_manager.application.services.annotation_service import AnnotationService
from dataset_manager.infrastructure.parsers.labelme_parser import LabelMeParser
from dataset_manager.infrastructure.storage.labelme_annotation_store import LabelMeAnnotationStore


class AnnotationEditingTests(unittest.TestCase):
    def test_updates_one_label_and_preserves_all_other_fields(self) -> None:
        original = {
            "version": "3.3.10",
            "flags": {"reviewed": True},
            "shapes": [{
                "label": "old-label",
                "score": None,
                "points": [[1.25, 2.5], [30.0, 40.0]],
                "group_id": None,
                "description": "keep me",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {"station": 7},
                "kie_linking": [],
            }],
            "imagePath": "sample.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 200,
            "customRoot": {"source": "factory"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            service = AnnotationService(LabelMeParser(), LabelMeAnnotationStore())

            result = service.update_shape_label(str(path), 0, "new-label")
            updated = json.loads(path.read_text(encoding="utf-8"))

        expected = json.loads(json.dumps(original))
        expected["shapes"][0]["label"] = "new-label"
        self.assertEqual(updated, expected)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.document.shapes[0].label, "new-label")

    def test_rejects_blank_label_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text('{"shapes": []}', encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                LabelMeAnnotationStore().update_shape_label(path, 0, "   ")
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
