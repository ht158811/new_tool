from __future__ import annotations

import copy
import random
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from dataset_manager.application.preprocessing.models import ConversionConfig, ConversionTarget, ProcessingRecord
from dataset_manager.domain.models.project import TaskType
from dataset_manager.infrastructure.preprocessing.common import (
    copy_image_exact, load_json_copy, rectangle_bounds, relative_output_path,
)
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class ConversionEngine:
    def __init__(self, json_store: AtomicJsonStore | None = None) -> None:
        self._json_store = json_store or AtomicJsonStore()

    def process_all(
        self,
        records: list[ProcessingRecord],
        dataset_root: str,
        output_directory: Path,
        task_type: TaskType,
        categories: list[str],
        config: ConversionConfig,
        is_cancelled: Callable[[], bool] | None = None,
        on_record: Callable[[int, str], None] | None = None,
    ) -> tuple[int, int, list[str]]:
        if task_type is not TaskType.OBJECT_DETECTION:
            raise ValueError("当前仅实现目标检测格式转换；分类映射和语义分割格式待确认")
        category_to_id = {name: index for index, name in enumerate(categories)}
        if not category_to_id:
            raise ValueError("没有可用于转换的类别")
        if config.target is ConversionTarget.COCO:
            return self._to_coco_partitions(
                self._partition_records(records, config), dataset_root,
                output_directory, category_to_id, is_cancelled, on_record,
            )
        generated_images = generated_annotations = 0
        warnings: list[str] = []
        if config.target is ConversionTarget.YOLO:
            (output_directory / "classes.txt").write_text(
                "\n".join(categories) + "\n", encoding="utf-8"
            )
        processed = 0
        for partition, partition_records in self._partition_records(records, config).items():
            partition_path = Path(partition) if partition else Path()
            for record in partition_records:
                self._check_cancelled(is_cancelled)
                relative = relative_output_path(record.image_path, dataset_root)
                image_output = output_directory / "images" / partition_path / relative
                copy_image_exact(record.image_path, image_output)
                generated_images += 1
                if not record.annotation_valid or not record.annotation_path:
                    warnings.append(f"无有效标注，仅复制图片: {Path(record.image_path).name}")
                else:
                    raw = load_json_copy(record.annotation_path)
                    suffix = ".txt" if config.target is ConversionTarget.YOLO else ".xml"
                    label_output = (
                        output_directory / "labels" / partition_path /
                        relative.with_suffix(suffix)
                    )
                    if config.target is ConversionTarget.YOLO:
                        self._write_yolo(raw, label_output, category_to_id)
                    else:
                        self._write_voc(raw, label_output, image_output.name, category_to_id)
                    generated_annotations += 1
                processed += 1
                if on_record is not None:
                    on_record(processed, Path(record.image_path).name)
        return generated_images, generated_annotations, warnings

    @staticmethod
    def _partition_records(
        records: list[ProcessingRecord], config: ConversionConfig
    ) -> dict[str, list[ProcessingRecord]]:
        """Return a deterministic train/val/test split without changing source order."""
        if config.validation_ratio == 0.0 and config.test_ratio == 0.0:
            return {"": list(records)}
        shuffled = list(records)
        random.Random(config.split_seed).shuffle(shuffled)
        ratios = {
            "train": 1.0 - config.validation_ratio - config.test_ratio,
            "val": config.validation_ratio,
            "test": config.test_ratio,
        }
        required_partitions = [name for name, ratio in ratios.items() if ratio > 0]
        if len(shuffled) < len(required_partitions):
            raise ValueError(
                f"当前仅有 {len(shuffled)} 张图片，无法同时划分到 "
                f"{len(required_partitions)} 个非空数据集"
            )
        raw_counts = {name: len(shuffled) * ratio for name, ratio in ratios.items()}
        counts = {name: int(value) for name, value in raw_counts.items()}
        remainder = len(shuffled) - sum(counts.values())
        for name in sorted(
            ratios, key=lambda item: (raw_counts[item] - counts[item], ratios[item]),
            reverse=True,
        )[:remainder]:
            counts[name] += 1
        for name in required_partitions:
            if counts[name] > 0:
                continue
            donors = [
                candidate for candidate in required_partitions
                if counts[candidate] > 1
            ]
            if not donors:
                raise ValueError("图片数量不足，无法保证训练集、验证集和测试集均非空")
            donor = max(donors, key=lambda candidate: counts[candidate])
            counts[donor] -= 1
            counts[name] += 1
        train_count = counts["train"]
        validation_count = counts["val"]
        return {
            "train": shuffled[:train_count],
            "val": shuffled[train_count:train_count + validation_count],
            "test": shuffled[train_count + validation_count:],
        }

    def _to_coco_partitions(
        self,
        partitions: dict[str, list[ProcessingRecord]],
        dataset_root: str,
        output_directory: Path,
        category_to_id: dict[str, int],
        is_cancelled: Callable[[], bool] | None,
        on_record: Callable[[int, str], None] | None,
    ) -> tuple[int, int, list[str]]:
        generated_images = generated_annotations = processed = 0
        warnings: list[str] = []
        split_enabled = "" not in partitions
        for partition, records in partitions.items():
            def report(local_processed: int, current: str, *, offset: int = processed) -> None:
                if on_record is not None:
                    on_record(offset + local_processed, current)

            images, annotations, partition_warnings = self._to_coco(
                records, dataset_root, output_directory, category_to_id,
                is_cancelled, report, partition if split_enabled else "",
            )
            generated_images += images
            generated_annotations += annotations
            processed += len(records)
            warnings.extend(partition_warnings)
        return generated_images, generated_annotations, warnings

    def _write_yolo(self, raw: dict[str, Any], path: Path, category_to_id: dict[str, int]) -> None:
        width, height = int(raw["imageWidth"]), int(raw["imageHeight"])
        lines = []
        for shape in self._rectangle_shapes(raw):
            label = str(shape["label"])
            if label not in category_to_id:
                raise ValueError(f"类别不在映射中: {label}")
            x1, y1, x2, y2 = rectangle_bounds(shape["points"])
            lines.append(
                f"{category_to_id[label]} {((x1+x2)/2)/width:.8f} {((y1+y2)/2)/height:.8f} "
                f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_voc(
        self, raw: dict[str, Any], path: Path, image_name: str, category_to_id: dict[str, int]
    ) -> None:
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = image_name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(raw["imageWidth"])
        ET.SubElement(size, "height").text = str(raw["imageHeight"])
        ET.SubElement(size, "depth").text = "3"
        for shape in self._rectangle_shapes(raw):
            label = str(shape["label"])
            if label not in category_to_id:
                raise ValueError(f"类别不在映射中: {label}")
            x1, y1, x2, y2 = rectangle_bounds(shape["points"])
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            box = ET.SubElement(obj, "bndbox")
            for key, value in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
                ET.SubElement(box, key).text = str(round(value))
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def _to_coco(
        self,
        records: list[ProcessingRecord],
        dataset_root: str,
        output_directory: Path,
        category_to_id: dict[str, int],
        is_cancelled: Callable[[], bool] | None,
        on_record: Callable[[int, str], None] | None,
        partition: str = "",
    ) -> tuple[int, int, list[str]]:
        images, annotations, warnings = [], [], []
        annotation_id = 1
        for image_id, record in enumerate(records, start=1):
            self._check_cancelled(is_cancelled)
            relative = relative_output_path(record.image_path, dataset_root)
            partition_path = Path(partition) if partition else Path()
            image_output = output_directory / "images" / partition_path / relative
            copy_image_exact(record.image_path, image_output)
            if not record.annotation_valid or not record.annotation_path:
                with Image.open(record.image_path) as source:
                    width, height = source.size
                images.append({
                    "id": image_id, "file_name": relative.as_posix(),
                    "width": width, "height": height,
                })
                warnings.append(f"无有效标注，仅复制图片: {Path(record.image_path).name}")
                if on_record is not None:
                    on_record(image_id, Path(record.image_path).name)
                continue
            raw = load_json_copy(record.annotation_path)
            images.append({
                "id": image_id, "file_name": relative.as_posix(),
                "width": raw["imageWidth"], "height": raw["imageHeight"],
            })
            for shape in self._rectangle_shapes(raw):
                label = str(shape["label"])
                if label not in category_to_id:
                    raise ValueError(f"类别不在映射中: {label}")
                x1, y1, x2, y2 = rectangle_bounds(shape["points"])
                annotations.append({
                    "id": annotation_id, "image_id": image_id,
                    "category_id": category_to_id[label] + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                })
                annotation_id += 1
            if on_record is not None:
                on_record(image_id, Path(record.image_path).name)
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": class_id + 1, "name": name}
                for name, class_id in category_to_id.items()
            ],
        }
        annotation_output = (
            output_directory / "annotations" / f"instances_{partition}.json"
            if partition else output_directory / "annotations.json"
        )
        self._json_store.write(annotation_output, coco)
        return len(records), sum(record.annotation_valid for record in records), warnings

    @staticmethod
    def _check_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("格式转换任务已取消")

    @staticmethod
    def _rectangle_shapes(raw: dict[str, Any]) -> list[dict[str, Any]]:
        shapes = raw.get("shapes", [])
        if not isinstance(shapes, list):
            raise ValueError("shapes 必须是数组")
        unsupported = [shape.get("shape_type") for shape in shapes if shape.get("shape_type") != "rectangle"]
        if unsupported:
            raise ValueError(f"目标检测转换不允许静默丢弃 shape: {sorted(set(map(str, unsupported)))}")
        return [copy.deepcopy(shape) for shape in shapes]
