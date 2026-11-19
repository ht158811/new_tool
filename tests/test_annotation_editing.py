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

    def test_deletes_one_shape_and_preserves_other_json_fields(self) -> None:
        original = {
            "version": "5.0",
            "flags": {"reviewed": True},
            "shapes": [
                {"label": "first", "points": [[1, 2], [3, 4]], "shape_type": "rectangle"},
                {"label": "second", "points": [[5, 6], [7, 8]], "shape_type": "rectangle"},
            ],
            "imagePath": "sample.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 200,
            "customRoot": {"keep": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            service = AnnotationService(LabelMeParser(), LabelMeAnnotationStore())

            result = service.delete_shape(str(path), 0)
            updated = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(updated["shapes"], [original["shapes"][1]])
        self.assertEqual(updated["customRoot"], {"keep": True})
        self.assertEqual(result.document.shapes[0].label, "second")

    def test_deletes_json_without_deleting_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            annotation_path = Path(directory) / "sample.json"
            image_path.write_bytes(b"image-bytes")
            annotation_path.write_text('{"shapes": []}', encoding="utf-8")
            service = AnnotationService(LabelMeParser(), LabelMeAnnotationStore())

            service.delete_annotation(str(annotation_path))

            self.assertFalse(annotation_path.exists())
            self.assertEqual(image_path.read_bytes(), b"image-bytes")

    def test_moves_and_adds_rectangles_atomically(self) -> None:
        original = {
            "shapes": [{
                "label": "old",
                "points": [[1, 2], [30, 40]],
                "group_id": 7,
                "shape_type": "rectangle",
                "flags": {"keep": True},
                "custom": "preserved",
            }],
            "imagePath": "sample.jpg",
            "imageData": None,
            "imageWidth": 200,
            "imageHeight": 100,
            "rootExtra": {"keep": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            service = AnnotationService(LabelMeParser(), LabelMeAnnotationStore())

            service.update_rectangle_points(str(path), 0, [[11, 12], [40, 50]])
            service.add_rectangle(str(path), "new", [[60, 20], [90, 45]])
            updated = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(updated["shapes"][0]["points"], [[11.0, 12.0], [40.0, 50.0]])
        self.assertEqual(updated["shapes"][0]["custom"], "preserved")
        self.assertEqual(updated["shapes"][0]["group_id"], 7)
        self.assertEqual(updated["shapes"][1], {
            "label": "new",
            "points": [[60.0, 20.0], [90.0, 45.0]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        })
        self.assertEqual(updated["rootExtra"], {"keep": True})

    def test_creates_labelme_json_for_unannotated_image_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            annotation_path = Path(directory) / "sample.json"
            image_path.write_bytes(b"image")
            service = AnnotationService(LabelMeParser(), LabelMeAnnotationStore())

            result = service.create_annotation_with_rectangle(
                str(annotation_path), str(image_path), 200, 100,
                "new", [[10, 20], [80, 70]],
            )
            created = json.loads(annotation_path.read_text(encoding="utf-8"))

            self.assertEqual(created["imagePath"], "sample.jpg")
            self.assertEqual(created["imageWidth"], 200)
            self.assertEqual(created["imageHeight"], 100)
            self.assertEqual(created["shapes"][0]["label"], "new")
            self.assertTrue(result.is_valid)
            before = annotation_path.read_bytes()
            with self.assertRaises(FileExistsError):
                service.create_annotation_with_rectangle(
                    str(annotation_path), str(image_path), 200, 100,
                    "other", [[1, 1], [2, 2]],
                )
            self.assertEqual(annotation_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
