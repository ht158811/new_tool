from __future__ import annotations

from pathlib import Path
from typing import Any
import math

from dataset_manager.application.ports.annotation_writer import AnnotationWriter
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class LabelMeAnnotationStore(AnnotationWriter):
    def __init__(self, store: AtomicJsonStore | None = None) -> None:
        self._store = store or AtomicJsonStore()

    def update_shape_label(self, annotation_path: Path, shape_index: int, label: str) -> None:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("类别名称不能为空")
        data = self._store.read(annotation_path)
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError("LabelMe JSON 缺少有效的 shapes 数组")
        if shape_index < 0 or shape_index >= len(shapes):
            raise IndexError(f"shape 索引超出范围: {shape_index}")
        shape: Any = shapes[shape_index]
        if not isinstance(shape, dict):
            raise ValueError(f"shapes[{shape_index}] 不是对象")
        shape["label"] = clean_label
        self._store.write(annotation_path, data)

    def delete_shape(self, annotation_path: Path, shape_index: int) -> None:
        data = self._store.read(annotation_path)
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError("LabelMe JSON 缺少有效的 shapes 数组")
        if shape_index < 0 or shape_index >= len(shapes):
            raise IndexError(f"shape 索引超出范围: {shape_index}")
        del shapes[shape_index]
        self._store.write(annotation_path, data)

    @staticmethod
    def delete_annotation(annotation_path: Path) -> None:
        if annotation_path.suffix.casefold() != ".json":
            raise ValueError("只允许删除 JSON 标注文件")
        if annotation_path.is_symlink():
            raise ValueError("为避免误删，不允许删除符号链接标注文件")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"JSON 标注文件不存在: {annotation_path.name}")
        annotation_path.unlink()

    def update_rectangle_points(
        self, annotation_path: Path, shape_index: int, points: list[list[float]]
    ) -> None:
        normalized = self._validated_rectangle_points(points)
        data = self._store.read(annotation_path)
        self._validate_image_bounds(normalized, data)
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError("LabelMe JSON 缺少有效的 shapes 数组")
        if shape_index < 0 or shape_index >= len(shapes):
            raise IndexError(f"shape 索引超出范围: {shape_index}")
        shape: Any = shapes[shape_index]
        if not isinstance(shape, dict) or shape.get("shape_type") != "rectangle":
            raise ValueError("只能拖动 rectangle 类型标注")
        shape["points"] = normalized
        self._store.write(annotation_path, data)

    def add_rectangle(
        self, annotation_path: Path, label: str, points: list[list[float]]
    ) -> None:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("类别名称不能为空")
        normalized = self._validated_rectangle_points(points)
        data = self._store.read(annotation_path)
        self._validate_image_bounds(normalized, data)
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError("LabelMe JSON 缺少有效的 shapes 数组")
        shapes.append({
            "label": clean_label,
            "points": normalized,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        })
        self._store.write(annotation_path, data)

    def create_annotation_with_rectangle(
        self,
        annotation_path: Path,
        image_name: str,
        width: int,
        height: int,
        label: str,
        points: list[list[float]],
    ) -> None:
        if annotation_path.exists():
            raise FileExistsError(f"JSON 文件已存在，拒绝覆盖: {annotation_path.name}")
        if width <= 0 or height <= 0:
            raise ValueError("图片尺寸必须大于 0")
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("类别名称不能为空")
        normalized = self._validated_rectangle_points(points)
        self._validate_image_bounds(normalized, {"imageWidth": width, "imageHeight": height})
        self._store.write(annotation_path, {
            "version": "5.0.1",
            "flags": {},
            "shapes": [{
                "label": clean_label,
                "points": normalized,
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }],
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        })

    @staticmethod
    def _validated_rectangle_points(points: list[list[float]]) -> list[list[float]]:
        if len(points) not in {2, 4}:
            raise ValueError("rectangle 坐标必须包含 2 个或 4 个点")
        normalized: list[list[float]] = []
        for point in points:
            if len(point) != 2:
                raise ValueError("每个坐标点必须包含 x 和 y")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0:
                raise ValueError("矩形框坐标必须是非负有限数值")
            normalized.append([x, y])
        xs = [point[0] for point in normalized]
        ys = [point[1] for point in normalized]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            raise ValueError("矩形框必须具有有效宽度和高度")
        return normalized

    @staticmethod
    def _validate_image_bounds(points: list[list[float]], data: dict[str, Any]) -> None:
        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if isinstance(width, int) and isinstance(height, int):
            if any(x > width or y > height for x, y in points):
                raise ValueError("矩形框坐标超出图片尺寸")

