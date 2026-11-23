from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset_manager.application.model_testing.models import ModelTestConfig, ModelType
from dataset_manager.application.model_testing.service import ModelTestService
from dataset_manager.domain.models.project import Dataset, Project, TaskType
from dataset_manager.infrastructure.storage.project_repository import JsonProjectRepository


class _Cache:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def read(self, dataset_id: str, scan_id: str) -> dict[str, object]:
        return {"records": self.records}


class _Runner:
    def __init__(self, *, segmentation: bool = False) -> None:
        self.loaded = False
        self.predict_calls = 0
        self.segmentation = segmentation

    def load(self, config: ModelTestConfig) -> dict[str, object]:
        self.loaded = True
        return {"backend": "fake", "task": "detect", "classes": {"0": "defect"}}

    def predict(self, image_path: str, config: ModelTestConfig) -> dict[str, object]:
        self.predict_calls += 1
        result: dict[str, object] = {
            "original_shape": [24, 32],
            "detections": [{
                "class_id": 0,
                "class_name": "defect",
                "confidence": 0.9,
                "box_xyxy": [1, 2, 10, 12],
            }],
            "speed_ms": {"inference": 1.5},
        }
        if self.segmentation:
            result["segments"] = [[[1, 2], [10, 2], [10, 12], [1, 12]]]
        return result


class ModelTestServiceTests(unittest.TestCase):
    def _context(
        self, root: Path, image_names: tuple[str, ...] = ("sample.jpg",)
    ) -> tuple[JsonProjectRepository, Project, Dataset, Path, list[dict[str, object]]]:
        records: list[dict[str, object]] = []
        for name in image_names:
            image = root / name
            Image.new("RGB", (32, 24), "white").save(image)
            records.append({"image_path": str(image), "annotation_path": None})
        model = root / "model.pt"
        model.write_bytes(b"test-model")
        project = Project.create("P")
        dataset = Dataset.create(
            "D", str(root), str(root), str(root), TaskType.OBJECT_DETECTION
        )
        dataset.active_scan_id = "a" * 32
        project.datasets.append(dataset)
        repository = JsonProjectRepository(root / "Projects")
        repository.save(project)
        return repository, project, dataset, model, records

    @staticmethod
    def _config(
        project: Project,
        dataset: Dataset,
        model: Path,
        model_type: ModelType = ModelType.YOLO_DETECTION,
        overwrite: bool = False,
    ) -> ModelTestConfig:
        return ModelTestConfig(
            project.id, dataset.id, str(model), model_type, 640, 640,
            0.25, 0.7, 300, "cpu", overwrite,
        )

    def test_generates_labelme_json_beside_image_without_modifying_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, project, dataset, model, records = self._context(root)
            image = root / "sample.jpg"
            before = image.read_bytes()
            runner = _Runner()
            progress = []

            result = ModelTestService(repository, _Cache(records), runner).execute(
                self._config(project, dataset, model), progress.append
            )

            annotation = json.loads((root / "sample.json").read_text(encoding="utf-8"))
            self.assertTrue(runner.loaded)
            self.assertEqual((result.succeeded, result.failed), (1, 0))
            self.assertEqual(result.generated_annotations, 1)
            self.assertEqual(annotation["imagePath"], "sample.jpg")
            self.assertEqual((annotation["imageWidth"], annotation["imageHeight"]), (32, 24))
            self.assertEqual(annotation["shapes"][0]["label"], "defect")
            self.assertEqual(annotation["shapes"][0]["shape_type"], "rectangle")
            self.assertEqual(annotation["shapes"][0]["points"], [[1.0, 2.0], [10.0, 12.0]])
            self.assertEqual(progress[0].processed, 1)
            self.assertEqual(image.read_bytes(), before)

    def test_existing_json_is_skipped_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, project, dataset, model, records = self._context(root)
            annotation = root / "sample.json"
            annotation.write_text('{"sentinel": true}', encoding="utf-8")
            runner = _Runner()

            result = ModelTestService(repository, _Cache(records), runner).execute(
                self._config(project, dataset, model)
            )

            self.assertEqual(runner.predict_calls, 0)
            self.assertEqual(result.skipped_existing, 1)
            self.assertEqual(result.generated_annotations, 0)
            self.assertEqual(json.loads(annotation.read_text(encoding="utf-8")), {"sentinel": True})

    def test_overwrite_replaces_existing_labelme_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, project, dataset, model, records = self._context(root)
            annotation = root / "sample.json"
            annotation.write_text('{"sentinel": true}', encoding="utf-8")

            result = ModelTestService(repository, _Cache(records), _Runner()).execute(
                self._config(project, dataset, model, overwrite=True)
            )

            document = json.loads(annotation.read_text(encoding="utf-8"))
            self.assertNotIn("sentinel", document)
            self.assertEqual(document["shapes"][0]["label"], "defect")
            self.assertEqual(result.generated_annotations, 1)

    def test_segmentation_generates_labelme_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, project, dataset, model, records = self._context(root)

            ModelTestService(repository, _Cache(records), _Runner(segmentation=True)).execute(
                self._config(project, dataset, model, ModelType.YOLO_SEGMENTATION)
            )

            document = json.loads((root / "sample.json").read_text(encoding="utf-8"))
            self.assertEqual(document["shapes"][0]["shape_type"], "polygon")
            self.assertEqual(len(document["shapes"][0]["points"]), 4)

    def test_cancellation_discards_all_pending_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, project, dataset, model, records = self._context(
                root, ("one.jpg", "two.jpg")
            )
            runner = _Runner()

            result = ModelTestService(repository, _Cache(records), runner).execute(
                self._config(project, dataset, model),
                is_cancelled=lambda: runner.predict_calls >= 1,
            )

            self.assertTrue(result.cancelled)
            self.assertEqual(result.generated_annotations, 0)
            self.assertFalse((root / "one.json").exists())
            self.assertFalse((root / "two.json").exists())


if __name__ == "__main__":
    unittest.main()
