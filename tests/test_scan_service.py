from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image

from dataset_manager.application.errors import ScanCancelledError
from dataset_manager.application.errors import ScanAlreadyRunningError
from dataset_manager.application.dto.scan_result import ScanResult
from dataset_manager.application.dto.image_record import ImageFilter
from dataset_manager.application.services.image_service import ImageService
from dataset_manager.application.services.dataset_service import DatasetService
from dataset_manager.domain.models.project import utc_now_iso
from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.application.services.scan_service import ScanService
from dataset_manager.domain.models.project import Dataset, TaskType
from dataset_manager.infrastructure.parsers.labelme_parser import LabelMeParser
from dataset_manager.infrastructure.scanning.labelme_scanner import LabelMeDatasetScanner
from dataset_manager.infrastructure.scanning.scan_cache import ScanCacheStore
from dataset_manager.infrastructure.storage.project_repository import JsonProjectRepository


def write_label(path: Path, image_name: str, label: str = "scratch") -> None:
    path.write_text(json.dumps({
        "version": "5.0",
        "flags": {},
        "shapes": [{
            "label": label,
            "points": [[1, 1], [5, 5]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": 10,
        "imageWidth": 10,
    }), encoding="utf-8")


class ScanServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.projects = self.root / "management" / "Projects"
        self.cache = self.root / "management" / "Cache"
        self.images.mkdir()
        self.labels.mkdir()
        self.repository = JsonProjectRepository(self.projects)
        self.project_service = ProjectService(self.repository)
        self.project = self.project_service.create_project("PCB")
        scanner = LabelMeDatasetScanner(
            LabelMeParser(), image_size_probe=lambda _: (10, 10)
        )
        self.cache_store = ScanCacheStore(self.cache)
        self.scan_service = ScanService(self.repository, scanner, self.cache_store)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def dataset(self, name: str = "Line A") -> Dataset:
        return Dataset.create(
            name=name,
            root_path=str(self.root.resolve()),
            image_directory=str(self.images.resolve()),
            annotation_directory=str(self.labels.resolve()),
            task_type=TaskType.OBJECT_DETECTION,
        )

    def test_scan_pairs_files_and_reports_missing_and_orphan(self) -> None:
        (self.images / "paired.jpg").write_bytes(b"image")
        (self.images / "missing.png").write_bytes(b"image")
        write_label(self.labels / "paired.json", "paired.jpg")
        write_label(self.labels / "orphan.json", "orphan.jpg", "dent")

        progress = []
        result = self.scan_service.add_dataset(
            self.project.id, self.dataset(), progress=progress.append
        )

        self.assertEqual(result.image_count, 2)
        self.assertEqual(result.annotation_count, 2)
        self.assertEqual(result.paired_count, 1)
        self.assertEqual(result.categories, ["dent", "scratch"])
        codes = {issue.code for record in result.records for issue in record.issues}
        self.assertIn("missing_annotation", codes)
        self.assertIn("orphan_annotation", codes)
        paired_record = next(record for record in result.records if record.annotation_path and record.image_path)
        missing_record = next(record for record in result.records if record.image_path and not record.annotation_path)
        self.assertTrue(paired_record.annotation_valid)
        self.assertFalse(missing_record.annotation_valid)
        self.assertEqual(progress[-1].processed, progress[-1].total)

        stored = self.repository.get(self.project.id).datasets[0]
        self.assertEqual(stored.active_scan_id, result.scan_id)
        self.assertEqual(stored.scan_summary.image_count, 2)
        self.assertEqual(stored.scan_summary.valid_annotation_image_count, 1)
        self.assertTrue(self.cache_store.path_for(stored.id, result.scan_id).is_file())

    def test_single_root_recursively_pairs_colocated_images_and_json(self) -> None:
        dataset_root = self.root / "colocated"
        first = dataset_root / "line_a" / "batch_01"
        second = dataset_root / "line_b"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / "sample_1.jpg").write_bytes(b"image")
        write_label(first / "sample_1.json", "sample_1.jpg")
        (second / "sample_2.png").write_bytes(b"image")
        write_label(second / "sample_2.json", "sample_2.png", "dent")
        dataset = Dataset.create(
            name="Colocated",
            root_path=str(dataset_root),
            image_directory=str(dataset_root),
            annotation_directory=str(dataset_root),
            task_type=TaskType.OBJECT_DETECTION,
        )

        result = self.scan_service.add_dataset(self.project.id, dataset)

        self.assertEqual(result.image_count, 2)
        self.assertEqual(result.annotation_count, 2)
        self.assertEqual(result.paired_count, 2)
        self.assertEqual(result.categories, ["dent", "scratch"])
        self.assertFalse(any(
            issue.code in {"missing_annotation", "orphan_annotation"}
            for record in result.records for issue in record.issues
        ))

    def test_json_image_path_pairs_when_json_filename_is_different(self) -> None:
        dataset_root = self.root / "image_path_pairing"
        nested = dataset_root / "camera_07"
        nested.mkdir(parents=True)
        (nested / "frame_000008.jpg").write_bytes(b"image")
        write_label(nested / "manual_annotation.json", "frame_000008.jpg", "scratch")
        dataset = Dataset.create(
            name="ImagePath fallback",
            root_path=str(dataset_root),
            image_directory=str(dataset_root),
            annotation_directory=str(dataset_root),
            task_type=TaskType.OBJECT_DETECTION,
        )

        result = self.scan_service.add_dataset(self.project.id, dataset)

        self.assertEqual(result.paired_count, 1)
        self.assertEqual(result.records[0].annotation_path, str((nested / "manual_annotation.json").resolve()))
        self.assertNotIn(
            "missing_annotation", {issue.code for issue in result.records[0].issues}
        )

    def test_same_display_name_is_allowed_for_different_dataset_paths(self) -> None:
        roots = [self.root / "session_a" / "camera_07", self.root / "session_b" / "camera_07"]
        datasets = []
        for index, root in enumerate(roots):
            root.mkdir(parents=True)
            (root / f"frame_{index}.jpg").write_bytes(b"image")
            write_label(root / f"frame_{index}.json", f"frame_{index}.jpg")
            datasets.append(Dataset.create(
                name="camera_07",
                root_path=str(root),
                image_directory=str(root),
                annotation_directory=str(root),
                task_type=TaskType.OBJECT_DETECTION,
            ))

        for dataset in datasets:
            self.scan_service.add_dataset(self.project.id, dataset)

        stored = self.repository.get(self.project.id).datasets
        self.assertEqual([dataset.name for dataset in stored], ["camera_07", "camera_07"])
        self.assertNotEqual(stored[0].id, stored[1].id)

    def test_delete_dataset_metadata_keeps_external_files(self) -> None:
        image_path = self.images / "keep.jpg"
        annotation_path = self.labels / "keep.json"
        image_path.write_bytes(b"image")
        write_label(annotation_path, "keep.jpg")
        dataset = self.dataset("Remove metadata")
        result = self.scan_service.add_dataset(self.project.id, dataset)
        cache_path = self.cache_store.path_for(dataset.id, result.scan_id)
        self.assertTrue(cache_path.exists())

        deletion = DatasetService(
            self.repository, self.cache_store
        ).delete_dataset_management_info(self.project.id, dataset.id)

        self.assertEqual(deletion.dataset.id, dataset.id)
        self.assertIsNone(deletion.cache_cleanup_warning)
        self.assertEqual(self.repository.get(self.project.id).datasets, [])
        self.assertFalse(cache_path.exists())
        self.assertTrue(image_path.exists())
        self.assertTrue(annotation_path.exists())

    def test_default_pillow_probe_reads_real_image_dimensions(self) -> None:
        image_path = self.images / "real.png"
        Image.new("RGB", (23, 17), color=(20, 30, 40)).save(image_path)
        write_label(self.labels / "real.json", "real.png")
        scanner = LabelMeDatasetScanner(LabelMeParser())
        result = scanner.scan(self.dataset("Real image"))
        record = result.records[0]
        self.assertEqual((record.width, record.height), (23, 17))
        self.assertIn("image_size_mismatch", {issue.code for issue in record.issues})

    def test_image_service_filters_by_name_category_and_dimensions(self) -> None:
        (self.images / "scratch_a.jpg").write_bytes(b"image")
        (self.images / "dent_b.jpg").write_bytes(b"image")
        write_label(self.labels / "scratch_a.json", "scratch_a.jpg", "scratch")
        write_label(self.labels / "dent_b.json", "dent_b.jpg", "dent")
        dataset = self.dataset("Filters")
        self.scan_service.add_dataset(self.project.id, dataset)
        service = ImageService(self.repository, self.cache_store)

        by_name = service.list_images(
            self.project.id, dataset.id, ImageFilter(filename="DENT")
        )
        by_category = service.list_images(
            self.project.id, dataset.id, ImageFilter(category="scratch")
        )
        by_dimensions = service.list_images(
            self.project.id, dataset.id, ImageFilter(min_width=11)
        )

        self.assertEqual(Path(by_name[0].image_path).name, "dent_b.jpg")
        self.assertEqual(Path(by_category[0].image_path).name, "scratch_a.jpg")
        self.assertEqual(by_dimensions, [])
        self.assertTrue(by_name[0].annotation_valid)

    def test_malformed_paired_json_is_not_marked_valid(self) -> None:
        (self.images / "bad.jpg").write_bytes(b"image")
        (self.labels / "bad.json").write_text("{not-json", encoding="utf-8")
        dataset = self.dataset("Invalid JSON")

        result = self.scan_service.add_dataset(self.project.id, dataset)
        image = ImageService(self.repository, self.cache_store).list_images(
            self.project.id, dataset.id
        )[0]

        self.assertFalse(result.records[0].annotation_valid)
        self.assertFalse(image.annotation_valid)
        stored = self.repository.get(self.project.id).find_dataset(dataset.id)
        self.assertEqual(stored.scan_summary.valid_annotation_image_count, 0)

    def test_cancelled_first_scan_does_not_commit_dataset_or_cache(self) -> None:
        (self.images / "one.jpg").write_bytes(b"image")
        write_label(self.labels / "one.json", "one.jpg")
        dataset = self.dataset()

        with self.assertRaises(ScanCancelledError):
            self.scan_service.add_dataset(
                self.project.id, dataset, is_cancelled=lambda: True
            )

        self.assertEqual(self.repository.get(self.project.id).datasets, [])
        self.assertFalse((self.cache / "scans" / dataset.id).exists())

    def test_refresh_replaces_active_cache_only_after_success(self) -> None:
        (self.images / "one.jpg").write_bytes(b"image")
        write_label(self.labels / "one.json", "one.jpg")
        dataset = self.dataset()
        first = self.scan_service.add_dataset(self.project.id, dataset)
        first_cache = self.cache_store.path_for(dataset.id, first.scan_id)
        self.assertTrue(first_cache.exists())

        (self.images / "two.jpg").write_bytes(b"image")
        second = self.scan_service.refresh_dataset(self.project.id, dataset.id)
        stored = self.repository.get(self.project.id).find_dataset(dataset.id)

        self.assertEqual(stored.active_scan_id, second.scan_id)
        self.assertEqual(stored.scan_summary.image_count, 2)
        self.assertFalse(first_cache.exists())
        self.assertTrue(self.cache_store.path_for(dataset.id, second.scan_id).exists())

    def test_cancelled_refresh_preserves_previous_snapshot(self) -> None:
        (self.images / "one.jpg").write_bytes(b"image")
        write_label(self.labels / "one.json", "one.jpg")
        dataset = self.dataset()
        first = self.scan_service.add_dataset(self.project.id, dataset)

        with self.assertRaises(ScanCancelledError):
            self.scan_service.refresh_dataset(
                self.project.id, dataset.id, is_cancelled=lambda: True
            )

        stored = self.repository.get(self.project.id).find_dataset(dataset.id)
        self.assertEqual(stored.active_scan_id, first.scan_id)
        self.assertTrue(self.cache_store.path_for(dataset.id, first.scan_id).exists())

    def test_configuration_error_prevents_add(self) -> None:
        dataset = self.dataset()
        dataset.image_directory = str(self.root / "not-found")
        with self.assertRaises(ValueError):
            self.scan_service.add_dataset(self.project.id, dataset)
        self.assertEqual(self.repository.get(self.project.id).datasets, [])

    def test_classification_does_not_infer_image_class_from_shapes(self) -> None:
        (self.images / "one.jpg").write_bytes(b"image")
        write_label(self.labels / "one.json", "one.jpg", "shape-label")
        dataset = self.dataset("Classification")
        dataset.task_type = TaskType.IMAGE_CLASSIFICATION

        result = self.scan_service.add_dataset(self.project.id, dataset)

        self.assertEqual(result.categories, [])
        self.assertEqual(result.records[0].categories, [])
        self.assertIn("classification_mapping_required", {issue.code for issue in result.issues})

    def test_same_dataset_cannot_be_scanned_twice_concurrently(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingScanner:
            def scan(inner_self, dataset, progress=None, is_cancelled=None):
                started.set()
                release.wait(timeout=5)
                return ScanResult(
                    scan_id="a" * 32,
                    dataset_id=dataset.id,
                    completed_at=utc_now_iso(),
                    records=[],
                    issues=[],
                    image_count=0,
                    annotation_count=0,
                    paired_count=0,
                    categories=[],
                )

        service = ScanService(self.repository, BlockingScanner(), self.cache_store)
        dataset = self.dataset("Concurrent")
        first_error: list[Exception] = []

        def first_scan() -> None:
            try:
                service.add_dataset(self.project.id, dataset)
            except Exception as error:
                first_error.append(error)

        thread = threading.Thread(target=first_scan)
        thread.start()
        self.assertTrue(started.wait(timeout=2))
        try:
            with self.assertRaises(ScanAlreadyRunningError):
                service.add_dataset(self.project.id, dataset)
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertFalse(first_error)


if __name__ == "__main__":
    unittest.main()
