from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from dataset_manager.application.preprocessing.models import (
    AugmentationConfig, ConversionConfig, ConversionTarget, CropConfig,
    LabelRenameConfig, LabelRetentionMode, LabelSplitConfig, OperationType, ProcessingConfig,
    ProcessingRecord, ProcessingScope,
)
from dataset_manager.application.preprocessing.preprocessing_service import PreprocessingService
from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.application.services.scan_service import ScanService
from dataset_manager.domain.models.project import Dataset, TaskType
from dataset_manager.infrastructure.scanning.labelme_scanner import LabelMeDatasetScanner
from dataset_manager.infrastructure.scanning.scan_cache import ScanCacheStore
from dataset_manager.infrastructure.preprocessing.common import flattened_output_stem
from dataset_manager.infrastructure.preprocessing.label_rename_engine import LabelRenameEngine
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore
from dataset_manager.infrastructure.storage.project_repository import JsonProjectRepository


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_label(path: Path, image_name: str) -> None:
    path.write_text(json.dumps({
        "version": "3.3.10",
        "flags": {},
        "shapes": [
            {
                "label": "defect",
                "points": [[20, 10], [60, 10], [60, 40], [20, 40]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "custom": "preserved",
            },
            {
                "label": "region",
                "points": [[40, 20], [80, 20], [80, 60], [40, 60]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            },
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": 80,
        "imageWidth": 100,
        "rootExtra": {"keep": True},
    }), encoding="utf-8")


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.valid_image = self.source / "valid.jpg"
        self.valid_json = self.source / "valid.json"
        Image.new("RGB", (100, 80), (50, 80, 100)).save(self.valid_image)
        write_label(self.valid_json, self.valid_image.name)
        Image.new("RGB", (100, 80), (10, 20, 30)).save(self.source / "without.jpg")
        Image.new("RGB", (100, 80), (30, 20, 10)).save(self.source / "invalid.jpg")
        (self.source / "invalid.json").write_text("{broken", encoding="utf-8")

        self.repository = JsonProjectRepository(self.base / "management" / "Projects")
        self.cache = ScanCacheStore(self.base / "management" / "Cache")
        project = ProjectService(self.repository).create_project("P")
        self.project_id = project.id
        dataset = Dataset.create(
            "D", str(self.source), str(self.source), str(self.source),
            TaskType.OBJECT_DETECTION,
        )
        self.scanner = LabelMeDatasetScanner()
        ScanService(
            self.repository, self.scanner, self.cache
        ).add_dataset(project.id, dataset)
        self.dataset_id = dataset.id
        self.service = PreprocessingService(
            self.repository, self.cache, source_scanner=self.scanner
        )
        self.output = self.base / "output"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, operation, scope=ProcessingScope.VALID_ANNOTATIONS, **kwargs):
        return ProcessingConfig(
            self.project_id, self.dataset_id, str(self.output), scope, operation, **kwargs
        )

    def test_scope_preview_distinguishes_invalid_json_from_missing_json(self) -> None:
        valid = self.service.preview(
            self.project_id, self.dataset_id, ProcessingScope.VALID_ANNOTATIONS
        )
        missing = self.service.preview(
            self.project_id, self.dataset_id, ProcessingScope.WITHOUT_JSON
        )
        all_images = self.service.preview(
            self.project_id, self.dataset_id, ProcessingScope.ALL_IMAGES
        )
        self.assertEqual(
            (valid.total_images, valid.valid_json, valid.without_json, valid.invalid_json),
            (3, 1, 1, 1),
        )
        self.assertEqual(valid.selected_total, 1)
        self.assertEqual(missing.selected_total, 1)
        self.assertEqual(all_images.selected_total, 3)

    def test_source_root_defaults_to_dataset_and_can_limit_processing_to_subfolder(self) -> None:
        nested = self.source / "selected_subfolder"
        nested.mkdir()
        nested_image = nested / "nested.jpg"
        nested_json = nested / "nested.json"
        Image.new("RGB", (100, 80), (1, 2, 3)).save(nested_image)
        write_label(nested_json, nested_image.name)
        ScanService(
            self.repository, LabelMeDatasetScanner(), self.cache
        ).refresh_dataset(self.project_id, self.dataset_id)

        default_preview = self.service.preview(
            self.project_id, self.dataset_id, ProcessingScope.ALL_IMAGES
        )
        nested_preview = self.service.preview(
            self.project_id, self.dataset_id, ProcessingScope.ALL_IMAGES, str(nested)
        )
        self.assertEqual(default_preview.total_images, 4)
        self.assertEqual(nested_preview.total_images, 1)

        result = self.service.execute(ProcessingConfig(
            self.project_id,
            self.dataset_id,
            str(self.output),
            ProcessingScope.VALID_ANNOTATIONS,
            OperationType.CROP,
            source_root=str(nested),
            crop=CropConfig(rows=1, columns=1, keep_empty=True),
        ))
        self.assertEqual(result.input_images, 1)
        self.assertEqual(result.selected_images, 1)
        self.assertTrue((Path(result.output_directory) / "data" / "nested_r1c1.jpg").is_file())

        outside = self.base / "not_sibling" / "outside"
        outside.mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.service.preview(
                self.project_id, self.dataset_id,
                ProcessingScope.ALL_IMAGES, str(outside),
            )

    def test_source_root_can_use_sibling_directory_with_temporary_scan(self) -> None:
        sibling = self.base / "sibling_source"
        sibling.mkdir()
        image = sibling / "sibling.jpg"
        annotation = sibling / "sibling.json"
        Image.new("RGB", (100, 80), (4, 5, 6)).save(image)
        write_label(annotation, image.name)

        preview = self.service.preview(
            self.project_id, self.dataset_id,
            ProcessingScope.VALID_ANNOTATIONS, str(sibling),
        )
        self.assertEqual(preview.total_images, 1)
        self.assertEqual(preview.valid_json, 1)

        result = self.service.execute(ProcessingConfig(
            self.project_id,
            self.dataset_id,
            str(self.output),
            ProcessingScope.VALID_ANNOTATIONS,
            OperationType.CROP,
            source_root=str(sibling),
            crop=CropConfig(rows=1, columns=1, keep_empty=True),
        ))
        self.assertEqual(result.input_images, 1)
        self.assertTrue(
            (Path(result.output_directory) / "data" / "sibling_r1c1.jpg").is_file()
        )

    def test_preview_reuses_records_for_the_same_scan_snapshot(self) -> None:
        class CountingCache:
            def __init__(self, delegate) -> None:
                self.delegate = delegate
                self.read_count = 0

            def read(self, dataset_id: str, scan_id: str):
                self.read_count += 1
                return self.delegate.read(dataset_id, scan_id)

        counting_cache = CountingCache(self.cache)
        service = PreprocessingService(self.repository, counting_cache)
        service.preview(
            self.project_id, self.dataset_id, ProcessingScope.VALID_ANNOTATIONS
        )
        service.preview(
            self.project_id, self.dataset_id, ProcessingScope.ALL_IMAGES
        )
        self.assertEqual(counting_cache.read_count, 1)

    def test_crop_preserves_sources_and_clips_rectangle_and_polygon(self) -> None:
        before_image, before_json = digest(self.valid_image), digest(self.valid_json)
        result = self.service.execute(self.config(
            OperationType.CROP,
            crop=CropConfig(rows=2, columns=2, keep_empty=True),
        ))
        output = Path(result.output_directory)
        labels = sorted((output / "data").glob("*.json"))
        images = sorted((output / "data").glob("*.jpg"))
        self.assertEqual(len(labels), len(images))
        self.assertTrue(all(path.parent == output / "data" for path in labels + images))
        self.assertEqual({path.stem for path in labels}, {path.stem for path in images})
        first = json.loads(next(path for path in labels if "r1c1" in path.name).read_text())
        first_label_path = next(path for path in labels if "r1c1" in path.name)
        self.assertEqual(first_label_path.name, "valid_r1c1.json")
        self.assertEqual(first["imagePath"], first_label_path.with_suffix(".jpg").name)
        rectangle = first["shapes"][0]
        self.assertEqual(rectangle["points"], [[20.0, 10.0], [50, 10.0], [50, 40], [20.0, 40]])
        self.assertEqual(first["imageWidth"], 50)
        self.assertEqual(first["imageHeight"], 40)
        self.assertIsNone(first["imageData"])
        self.assertEqual(first["rootExtra"], {"keep": True})
        manifest = json.loads((output / "result_manifest.json").read_text())
        self.assertNotIn("input_dataset_root", manifest)
        self.assertNotIn(str(self.source), json.dumps(manifest))
        self.assertEqual(digest(self.valid_image), before_image)
        self.assertEqual(digest(self.valid_json), before_json)

    def test_grid_crop_covers_uneven_image_without_gaps_or_overlap(self) -> None:
        config = CropConfig(rows=3, columns=3, keep_empty=True)
        windows = self.service._crop._grid_windows(100, 80, config)
        self.assertEqual(len(windows), 9)
        self.assertEqual(sum(width * height for _, _, _, _, width, height in windows), 8000)
        self.assertEqual(
            [(x, width) for row, _, x, _, width, _ in windows if row == 0],
            [(0, 33), (33, 33), (66, 34)],
        )
        self.assertEqual(
            [(y, height) for _, column, _, y, _, height in windows if column == 0],
            [(0, 26), (26, 27), (53, 27)],
        )

    def test_grid_crop_maps_point_line_and_legacy_shape_coordinates(self) -> None:
        raw = json.loads(self.valid_json.read_text())
        raw["shapes"] = [
            {"label": "legacy_box", "points": [[10, 10], [60, 30]], "flags": {}},
            {
                "label": "mark", "points": [[10, 10]],
                "shape_type": "point", "flags": {},
            },
            {
                "label": "trace", "points": [[10, 10], [90, 70]],
                "shape_type": "line", "flags": {},
            },
        ]
        self.valid_json.write_text(json.dumps(raw), encoding="utf-8")
        result = self.service.execute(self.config(
            OperationType.CROP,
            crop=CropConfig(rows=2, columns=2, keep_empty=True),
        ))
        labels = Path(result.output_directory) / "data"
        top_left = json.loads(next(labels.rglob("*r1c1.json")).read_text())
        self.assertEqual(top_left["shapes"][0]["points"], [[10.0, 10.0], [50, 30.0]])
        self.assertEqual(top_left["shapes"][1]["points"], [[10.0, 10.0]])
        bottom_right = json.loads(next(labels.rglob("*r2c2.json")).read_text())
        line = next(shape for shape in bottom_right["shapes"] if shape["label"] == "trace")
        self.assertEqual(line["points"], [[0.0, 0.0], [40.0, 30.0]])

    def test_geometric_and_appearance_augmentation_coordinates(self) -> None:
        before_image, before_json = digest(self.valid_image), digest(self.valid_json)
        horizontal = self.service.execute(self.config(
            OperationType.AUGMENTATION,
            augmentation=AugmentationConfig(horizontal_flip=True, outputs_per_image=1),
        ))
        label = json.loads(next((Path(horizontal.output_directory) / "data").glob("*.json")).read_text())
        self.assertEqual(label["shapes"][0]["points"][0], [80.0, 10.0])

        brightness = self.service.execute(self.config(
            OperationType.AUGMENTATION,
            augmentation=AugmentationConfig(brightness=1.2, outputs_per_image=1),
        ))
        bright_label = json.loads(next((Path(brightness.output_directory) / "data").glob("*.json")).read_text())
        bright_image = next((Path(brightness.output_directory) / "data").glob("*.jpg"))
        self.assertEqual(bright_image.name, "valid_a1.jpg")
        with Image.open(bright_image) as saved:
            self.assertTrue(saved.quantization)
            self.assertLessEqual(max(max(table) for table in saved.quantization.values()), 2)
        original = json.loads(self.valid_json.read_text())
        self.assertEqual(bright_label["shapes"], original["shapes"])
        self.assertNotEqual(horizontal.output_directory, brightness.output_directory)
        self.assertEqual(digest(self.valid_image), before_image)
        self.assertEqual(digest(self.valid_json), before_json)

    def test_rotation_matches_reference_script_and_combination_is_reproducible(self) -> None:
        rotated = self.service.execute(self.config(
            OperationType.AUGMENTATION,
            augmentation=AugmentationConfig(rotate_90=True, outputs_per_image=1),
        ))
        rotated_label = json.loads(
            next((Path(rotated.output_directory) / "data").glob("*.json")).read_text()
        )
        self.assertEqual(rotated_label["imageWidth"], 80)
        self.assertEqual(rotated_label["imageHeight"], 100)
        self.assertEqual(rotated_label["shapes"][0]["points"][0], [10.0, 80.0])

        combination = AugmentationConfig(
            horizontal_flip=True,
            brightness=1.1,
            outputs_per_image=3,
            max_operations_per_output=2,
            random_seed=123,
        )
        first = self.service.execute(self.config(
            OperationType.AUGMENTATION, augmentation=combination,
        ))
        second = self.service.execute(self.config(
            OperationType.AUGMENTATION, augmentation=combination,
        ))
        first_files = sorted((Path(first.output_directory) / "data").glob("*.jpg"))
        second_files = sorted((Path(second.output_directory) / "data").glob("*.jpg"))
        self.assertEqual([path.name for path in first_files], [path.name for path in second_files])
        self.assertEqual([digest(path) for path in first_files], [digest(path) for path in second_files])

    def test_flat_output_names_do_not_recreate_paths_or_collide(self) -> None:
        first = flattened_output_stem(
            str(self.source / "line_a" / "same.jpg"), str(self.source)
        )
        second = flattened_output_stem(
            str(self.source / "line_b" / "same.jpg"), str(self.source)
        )
        self.assertNotEqual(first, second)
        self.assertNotIn("/", first)
        self.assertNotIn("\\", first)

        records = [
            ProcessingRecord(str(self.source / "line_b" / "same.jpg"), None, False),
            ProcessingRecord(str(self.source / "line_a" / "same.jpg"), None, False),
            ProcessingRecord(str(self.source / "unique.jpg"), None, False),
        ]
        assigned = self.service._assign_output_stems(records)
        stems = {Path(record.image_path).parent.name: record.output_stem for record in assigned}
        self.assertEqual(stems["line_a"], "same")
        self.assertEqual(stems["line_b"], "same_2")
        self.assertEqual(stems[self.source.name], "unique")

    def test_label_split_uses_separate_before_after_rules_and_label_folders(self) -> None:
        raw = json.loads(self.valid_json.read_text())
        raw["shapes"] = [
            {
                "label": "car;damaged", "points": [[1, 1], [20, 20]],
                "shape_type": "rectangle", "flags": {},
            },
            {
                "label": "car;dirty", "points": [[30, 10], [50, 30]],
                "shape_type": "rectangle", "flags": {},
            },
            {
                "label": "board", "points": [[60, 20], [80, 40]],
                "shape_type": "rectangle", "flags": {},
            },
        ]
        self.valid_json.write_text(json.dumps(raw), encoding="utf-8")
        before_image, before_json = digest(self.valid_image), digest(self.valid_json)
        result = self.service.execute(self.config(
            OperationType.LABEL_SPLIT,
            label_split=LabelSplitConfig(
                separator=";",
                create_before=True,
                create_after=True,
                before_retention=LabelRetentionMode.BEFORE,
                after_retention=LabelRetentionMode.AFTER,
                before_group_by_label=False,
                after_group_by_label=True,
            ),
        ))
        output = Path(result.output_directory)
        before_labels = list((output / "before").glob("*.json"))
        self.assertEqual(len(before_labels), 1)
        before_values = sorted(
            shape["label"] for shape in json.loads(before_labels[0].read_text())["shapes"]
        )
        self.assertEqual(before_values, ["board", "car", "car"])
        after_labels = list((output / "after").glob("*/*.json"))
        self.assertEqual(len(after_labels), 3)
        after_values = []
        for label_path in after_labels:
            after_data = json.loads(label_path.read_text())
            after_values.extend(shape["label"] for shape in after_data["shapes"])
            paired_image = label_path.with_suffix(self.valid_image.suffix)
            self.assertTrue(paired_image.is_file())
            self.assertEqual(after_data["imagePath"], paired_image.name)
            self.assertEqual(digest(paired_image), before_image)
            self.assertEqual(paired_image.name, "valid_post.jpg")
        self.assertEqual(sorted(after_values), ["board", "damaged", "dirty"])
        self.assertEqual(
            {path.parent.name for path in after_labels},
            {"board", "damaged", "dirty"},
        )
        self.assertEqual(digest(self.valid_image), before_image)
        self.assertEqual(digest(self.valid_json), before_json)
        manifest = json.loads((output / "result_manifest.json").read_text())
        self.assertEqual(manifest["operation"], "label_split")
        self.assertEqual(manifest["parameters"]["separator"], ";")
        self.assertEqual(manifest["parameters"]["before_retention"], "before")
        self.assertEqual(manifest["parameters"]["after_retention"], "after")

    def test_label_split_outputs_have_independent_retention_modes(self) -> None:
        self.assertIs(
            LabelSplitConfig(before_retention="both").before_retention,
            LabelRetentionMode.BOTH,
        )
        raw = json.loads(self.valid_json.read_text())
        raw["shapes"] = [{
            "label": "car;damaged", "points": [[1, 1], [20, 20]],
            "shape_type": "rectangle", "flags": {},
        }]
        self.valid_json.write_text(json.dumps(raw), encoding="utf-8")

        before = self.service.execute(self.config(
            OperationType.LABEL_SPLIT,
            label_split=LabelSplitConfig(
                separator=";",
                create_before=True,
                create_after=False,
                before_retention=LabelRetentionMode.BOTH,
                before_group_by_label=True,
            ),
        ))
        before_root = Path(before.output_directory)
        self.assertTrue((before_root / "before").is_dir())
        self.assertFalse((before_root / "after").exists())
        before_json = next((before_root / "before").glob("*/*.json"))
        self.assertEqual(before_json.parent.name, "car;damaged")
        self.assertEqual(
            json.loads(before_json.read_text())["shapes"][0]["label"],
            "car;damaged",
        )

        after = self.service.execute(self.config(
            OperationType.LABEL_SPLIT,
            label_split=LabelSplitConfig(
                separator=";",
                create_before=False,
                create_after=True,
                after_retention=LabelRetentionMode.BEFORE,
                after_group_by_label=True,
            ),
        ))
        after_root = Path(after.output_directory)
        self.assertFalse((after_root / "before").exists())
        self.assertTrue((after_root / "after").is_dir())
        after_json = next((after_root / "after").glob("*/*.json"))
        self.assertEqual(
            json.loads(after_json.read_text())["shapes"][0]["label"], "car"
        )
        self.assertEqual(after_json.parent.name, "car")

        self.assertEqual(
            self.service._label_split._safe_folder_name('a/b:c?d'),
            "a_b_c_d",
        )

    def test_batch_label_rename_preserves_labelme_and_source_files(self) -> None:
        before_image = digest(self.valid_image)
        before_json = digest(self.valid_json)
        result = self.service.execute(ProcessingConfig(
            self.project_id,
            self.dataset_id,
            "",
            ProcessingScope.VALID_ANNOTATIONS,
            OperationType.LABEL_RENAME,
            label_rename=LabelRenameConfig("defect", "scratch"),
        ))
        data = json.loads(self.valid_json.read_text(encoding="utf-8"))
        self.assertEqual(
            [shape["label"] for shape in data["shapes"]],
            ["scratch", "region"],
        )
        self.assertEqual(data["shapes"][0]["custom"], "preserved")
        self.assertEqual(data["rootExtra"], {"keep": True})
        self.assertEqual(data["imagePath"], "valid.jpg")
        self.assertEqual(digest(self.valid_image), before_image)
        self.assertNotEqual(digest(self.valid_json), before_json)
        self.assertEqual(Path(result.output_directory), self.source)
        self.assertEqual(result.generated_images, 0)
        self.assertEqual(result.generated_annotations, 1)
        self.assertEqual(result.modified_shapes, 1)
        self.assertFalse(self.output.exists())

        with self.assertRaises(ValueError):
            LabelRenameConfig("defect", " defect ")

    def test_cancelled_in_place_label_rename_does_not_write(self) -> None:
        before = digest(self.valid_json)
        result = self.service.execute(
            ProcessingConfig(
                self.project_id, self.dataset_id, "",
                ProcessingScope.VALID_ANNOTATIONS, OperationType.LABEL_RENAME,
                label_rename=LabelRenameConfig("defect", "scratch"),
            ),
            is_cancelled=lambda: True,
        )
        self.assertTrue(result.cancelled)
        self.assertEqual(digest(self.valid_json), before)

    def test_in_place_label_rename_rolls_back_committed_json_on_failure(self) -> None:
        second_json = self.source / "second.json"
        second_json.write_bytes(self.valid_json.read_bytes())
        originals = {
            self.valid_json: self.valid_json.read_bytes(),
            second_json: second_json.read_bytes(),
        }

        class FailingStore:
            def __init__(self) -> None:
                self.calls = 0
                self.real = AtomicJsonStore()

            def write(self, path, data) -> None:
                self.calls += 1
                if self.calls == 2:
                    raise OSError("simulated write failure")
                self.real.write(path, data)

        engine = LabelRenameEngine(FailingStore())
        records = [
            ProcessingRecord(str(self.valid_image), str(self.valid_json), True),
            ProcessingRecord(str(self.source / "second.jpg"), str(second_json), True),
        ]
        with self.assertRaises(OSError):
            engine.rename_in_place(records, LabelRenameConfig("defect", "scratch"))
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_yolo_conversion_is_parseable_and_does_not_overwrite(self) -> None:
        raw = json.loads(self.valid_json.read_text())
        raw["shapes"] = [raw["shapes"][0]]
        self.valid_json.write_text(json.dumps(raw), encoding="utf-8")
        config = self.config(
            OperationType.CONVERSION,
            conversion=ConversionConfig(ConversionTarget.YOLO),
        )
        first = self.service.execute(config)
        second = self.service.execute(config)
        self.assertNotEqual(first.output_directory, second.output_directory)
        line = next((Path(first.output_directory) / "labels").rglob("*.txt")).read_text().strip()
        values = line.split()
        self.assertEqual(len(values), 5)
        self.assertTrue(all(0 <= float(value) <= 1 for value in values[1:]))
        self.assertTrue((Path(first.output_directory) / "result_manifest.json").is_file())
        self.assertEqual(
            digest(Path(first.output_directory) / "images" / self.valid_image.name),
            digest(self.valid_image),
        )

        custom = ProcessingConfig(
            self.project_id, self.dataset_id, str(self.output),
            ProcessingScope.VALID_ANNOTATIONS, OperationType.CONVERSION,
            result_name="my_yolo_export",
            conversion=ConversionConfig(ConversionTarget.YOLO),
        )
        custom_result = self.service.execute(custom)
        self.assertEqual(Path(custom_result.output_directory).name, "my_yolo_export")
        with self.assertRaises(FileExistsError):
            self.service.execute(custom)

    def test_conversion_splits_train_validation_and_test_deterministically(self) -> None:
        raw = json.loads(self.valid_json.read_text())
        raw["shapes"] = [raw["shapes"][0]]
        self.valid_json.write_text(json.dumps(raw), encoding="utf-8")
        for index in range(1, 10):
            image = self.source / f"sample_{index}.jpg"
            annotation = image.with_suffix(".json")
            Image.new("RGB", (100, 80), (index, 20, 30)).save(image)
            write_label(annotation, image.name)
            document = json.loads(annotation.read_text())
            document["shapes"] = [document["shapes"][0]]
            annotation.write_text(json.dumps(document), encoding="utf-8")
        ScanService(
            self.repository, LabelMeDatasetScanner(), self.cache
        ).refresh_dataset(self.project_id, self.dataset_id)
        split_config = ConversionConfig(
            ConversionTarget.YOLO, validation_ratio=0.2, test_ratio=0.1, split_seed=123
        )

        first = self.service.execute(self.config(
            OperationType.CONVERSION,
            conversion=split_config,
        ))
        second = self.service.execute(self.config(
            OperationType.CONVERSION,
            conversion=split_config,
        ))

        first_root, second_root = Path(first.output_directory), Path(second.output_directory)
        expected = {"train": 7, "val": 2, "test": 1}
        for split, count in expected.items():
            first_names = {
                path.name for path in (first_root / "images" / split).rglob("*.jpg")
            }
            second_names = {
                path.name for path in (second_root / "images" / split).rglob("*.jpg")
            }
            self.assertEqual(len(first_names), count)
            self.assertEqual(first_names, second_names)
            self.assertEqual(
                len(list((first_root / "labels" / split).rglob("*.txt"))), count
            )

        coco = self.service.execute(self.config(
            OperationType.CONVERSION,
            conversion=ConversionConfig(
                ConversionTarget.COCO, validation_ratio=0.2,
                test_ratio=0.1, split_seed=123,
            ),
        ))
        coco_root = Path(coco.output_directory)
        for split, count in expected.items():
            annotation_file = coco_root / "annotations" / f"instances_{split}.json"
            document = json.loads(annotation_file.read_text())
            self.assertEqual(len(document["images"]), count)
            self.assertEqual(len(list((coco_root / "images" / split).rglob("*.jpg"))), count)

    def test_delete_unannotated_images_in_place_and_rechecks_late_json(self) -> None:
        unannotated = self.source / "without.jpg"
        late_json = self.source / "manual_annotation.json"
        write_label(late_json, unannotated.name)
        config = ProcessingConfig(
            self.project_id, self.dataset_id, "",
            ProcessingScope.WITHOUT_JSON, OperationType.DELETE_UNANNOTATED,
        )

        protected = self.service.execute(config)
        self.assertTrue(unannotated.exists())
        self.assertEqual(protected.deleted_images, 0)
        self.assertEqual(protected.succeeded, 0)

        late_json.unlink()
        deleted = self.service.execute(config)
        self.assertFalse(unannotated.exists())
        self.assertEqual(deleted.deleted_images, 1)
        self.assertTrue(self.valid_image.exists())
        self.assertTrue((self.source / "invalid.jpg").exists())

    def test_cancel_delete_unannotated_restores_staged_images(self) -> None:
        second = self.source / "without_second.jpg"
        Image.new("RGB", (100, 80), (1, 2, 3)).save(second)
        ScanService(
            self.repository, LabelMeDatasetScanner(), self.cache
        ).refresh_dataset(self.project_id, self.dataset_id)
        checks = 0

        def cancel_after_first_stage() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 2

        result = self.service.execute(
            ProcessingConfig(
                self.project_id, self.dataset_id, "",
                ProcessingScope.WITHOUT_JSON, OperationType.DELETE_UNANNOTATED,
            ),
            is_cancelled=cancel_after_first_stage,
        )

        self.assertTrue(result.cancelled)
        self.assertTrue((self.source / "without.jpg").exists())
        self.assertTrue(second.exists())
        self.assertFalse(any(self.source.glob(".dataset_manager_delete_*")))

    def test_coco_includes_unannotated_images_and_voc_is_parseable(self) -> None:
        raw = json.loads(self.valid_json.read_text())
        raw["shapes"] = [raw["shapes"][0]]
        self.valid_json.write_text(json.dumps(raw), encoding="utf-8")
        coco_result = self.service.execute(self.config(
            OperationType.CONVERSION,
            scope=ProcessingScope.ALL_IMAGES,
            conversion=ConversionConfig(ConversionTarget.COCO),
        ))
        coco = json.loads((Path(coco_result.output_directory) / "annotations.json").read_text())
        self.assertEqual(len(coco["images"]), 3)
        self.assertEqual(len(coco["annotations"]), 1)
        self.assertEqual({image["file_name"] for image in coco["images"]}, {
            "valid.jpg", "without.jpg", "invalid.jpg",
        })

        voc_result = self.service.execute(self.config(
            OperationType.CONVERSION,
            conversion=ConversionConfig(ConversionTarget.VOC),
        ))
        root = ET.parse(next((Path(voc_result.output_directory) / "labels").rglob("*.xml"))).getroot()
        self.assertEqual(root.findtext("filename"), "valid.jpg")
        self.assertEqual(root.findtext("object/name"), "defect")

    def test_cancel_and_unsafe_output_do_not_touch_source(self) -> None:
        before = digest(self.valid_json)
        result = self.service.execute(
            self.config(OperationType.CROP, crop=CropConfig()),
            is_cancelled=lambda: True,
        )
        self.assertTrue(result.cancelled)
        self.assertFalse(Path(result.output_directory).exists())
        self.assertEqual(digest(self.valid_json), before)

        checks = 0

        def cancel_during_conversion() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 3

        conversion = self.service.execute(
            self.config(
                OperationType.CONVERSION,
                scope=ProcessingScope.ALL_IMAGES,
                conversion=ConversionConfig(ConversionTarget.COCO),
            ),
            is_cancelled=cancel_during_conversion,
        )
        self.assertTrue(conversion.cancelled)
        self.assertFalse(Path(conversion.output_directory).exists())
        self.assertFalse(any(self.output.glob(".*.tmp")))

        with self.assertRaises(ValueError):
            self.service.execute(ProcessingConfig(
                self.project_id, self.dataset_id, str(self.source / "output"),
                ProcessingScope.VALID_ANNOTATIONS, OperationType.CROP,
                crop=CropConfig(),
            ))


if __name__ == "__main__":
    unittest.main()
