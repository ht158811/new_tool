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
    def __init__(self) -> None:
        self.loaded = False

    def load(self, config: ModelTestConfig) -> dict[str, object]:
        self.loaded = True
        return {"backend": "fake", "task": "detect", "classes": {"0": "defect"}}

    def predict(self, image_path: str, config: ModelTestConfig) -> dict[str, object]:
        return {
            "detections": [{
                "class_id": 0, "class_name": "defect", "confidence": 0.9,
                "box_xyxy": [1, 2, 10, 12],
            }],
            "speed_ms": {"inference": 1.5},
        }


class ModelTestServiceTests(unittest.TestCase):
    def test_writes_readable_prediction_report_without_modifying_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            Image.new("RGB", (32, 24), "white").save(image)
            before = image.read_bytes()
            model = root / "model.pt"
            model.write_bytes(b"test-model")
            output = root / "result.json"
            project = Project.create("P")
            dataset = Dataset.create(
                "D", str(root), str(root), str(root), TaskType.OBJECT_DETECTION
            )
            dataset.active_scan_id = "a" * 32
            project.datasets.append(dataset)
            repository = JsonProjectRepository(root / "Projects")
            repository.save(project)
            runner = _Runner()
            service = ModelTestService(
                repository, _Cache([{"image_path": str(image)}]), runner
            )
            progress = []
            result = service.execute(
                ModelTestConfig(
                    project.id, dataset.id, str(model), ModelType.YOLO_DETECTION,
                    640, 640, 0.25, 0.7, 300, "cpu", str(output),
                ),
                progress.append,
            )

            self.assertTrue(runner.loaded)
            self.assertEqual((result.succeeded, result.failed), (1, 0))
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["configuration"]["model_type"], "yolo_detection")
            self.assertEqual(report["predictions"][0]["detections"][0]["class_name"], "defect")
            self.assertEqual(len(report["model"]["sha256"]), 64)
            self.assertEqual(progress[0].processed, 1)
            self.assertEqual(image.read_bytes(), before)

    def test_refuses_to_overwrite_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            Image.new("RGB", (32, 24), "white").save(image)
            model = root / "model.pt"
            model.write_bytes(b"model")
            output = root / "result.json"
            output.write_text("{}", encoding="utf-8")
            project = Project.create("P")
            dataset = Dataset.create("D", str(root), str(root), str(root), TaskType.OBJECT_DETECTION)
            dataset.active_scan_id = "b" * 32
            project.datasets.append(dataset)
            repository = JsonProjectRepository(root / "Projects")
            repository.save(project)
            service = ModelTestService(repository, _Cache([{"image_path": str(image)}]), _Runner())
            config = ModelTestConfig(
                project.id, dataset.id, str(model), ModelType.YOLO_DETECTION,
                640, 640, 0.25, 0.7, 300, "", str(output),
            )
            with self.assertRaises(FileExistsError):
                service.execute(config)
            self.assertEqual(output.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
