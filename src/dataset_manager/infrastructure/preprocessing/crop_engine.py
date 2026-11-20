from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from dataset_manager.application.preprocessing.models import CropConfig, ProcessingRecord
from dataset_manager.infrastructure.preprocessing.common import (
    flattened_output_stem, load_json_copy, prepare_labelme, rectangle_bounds,
    relative_output_path, save_image,
)
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class CropEngine:
    def __init__(self, json_store: AtomicJsonStore | None = None) -> None:
        self._json_store = json_store or AtomicJsonStore()

    def process(
        self,
        record: ProcessingRecord,
        dataset_root: str,
        output_directory: Path,
        config: CropConfig,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> tuple[int, int, int]:
        self._validate(config)
        relative = relative_output_path(record.image_path, dataset_root)
        flat_stem = record.output_stem or flattened_output_stem(record.image_path, dataset_root)
        raw = load_json_copy(record.annotation_path) if record.annotation_valid and record.annotation_path else None
        with Image.open(record.image_path) as source_image:
            source_image.load()
            width, height = source_image.size
            windows = self._grid_windows(width, height, config)
            generated_images = 0
            generated_json = 0
            skipped = 0
            for row, column, x, y, crop_width, crop_height in windows:
                if is_cancelled is not None and is_cancelled():
                    raise InterruptedError("裁剪任务已取消")
                shapes = self._crop_shapes(raw.get("shapes", []), x, y, crop_width, crop_height) if raw else []
                if raw and not shapes and not config.keep_empty:
                    skipped += 1
                    continue
                stem = f"{flat_stem}_r{row + 1}c{column + 1}"
                image_output = output_directory / "data" / f"{stem}{relative.suffix.lower()}"
                cropped = source_image.crop((x, y, x + crop_width, y + crop_height))
                save_image(cropped, image_output)
                generated_images += 1
                if raw is not None:
                    label_output = image_output.with_suffix(".json")
                    label_data = prepare_labelme(
                        raw, image_output.name, crop_width, crop_height, shapes
                    )
                    try:
                        self._json_store.write(label_output, label_data)
                    except Exception:
                        image_output.unlink(missing_ok=True)
                        raise
                    generated_json += 1
            return generated_images, generated_json, skipped

    @staticmethod
    def _validate(config: CropConfig) -> None:
        if config.rows <= 0 or config.columns <= 0:
            raise ValueError("裁剪行数和列数必须大于 0")

    @staticmethod
    def _grid_windows(
        width: int, height: int, config: CropConfig
    ) -> list[tuple[int, int, int, int, int, int]]:
        if config.columns > width or config.rows > height:
            raise ValueError(
                f"裁剪网格 {config.rows}×{config.columns} 超过图片像素尺寸 {height}×{width}"
            )
        x_edges = [column * width // config.columns for column in range(config.columns + 1)]
        y_edges = [row * height // config.rows for row in range(config.rows + 1)]
        windows: list[tuple[int, int, int, int, int, int]] = []
        for row in range(config.rows):
            for column in range(config.columns):
                left, right = x_edges[column], x_edges[column + 1]
                top, bottom = y_edges[row], y_edges[row + 1]
                windows.append((row, column, left, top, right - left, bottom - top))
        return windows

    def _crop_shapes(
        self, shapes: Any, x: int, y: int, width: int, height: int
    ) -> list[dict[str, Any]]:
        if not isinstance(shapes, list):
            raise ValueError("shapes 必须是数组")
        result: list[dict[str, Any]] = []
        for raw_shape in shapes:
            if not isinstance(raw_shape, dict):
                raise ValueError("shape 必须是对象")
            points = raw_shape.get("points")
            if not isinstance(points, list):
                raise ValueError("shape.points 必须是数组")
            raw_shape_type = raw_shape.get("shape_type")
            shape_type = str(raw_shape_type).lower() if raw_shape_type else (
                "rectangle" if len(points) == 2 else "polygon"
            )
            shape = copy.deepcopy(raw_shape)
            if shape_type == "rectangle":
                x1, y1, x2, y2 = rectangle_bounds(points)
                left, top = max(x1, x), max(y1, y)
                right, bottom = min(x2, x + width), min(y2, y + height)
                if right <= left or bottom <= top:
                    continue
                translated = [[left - x, top - y], [right - x, bottom - y]]
                if len(points) >= 4:
                    translated = [
                        [left - x, top - y], [right - x, top - y],
                        [right - x, bottom - y], [left - x, bottom - y],
                    ]
                shape["points"] = translated
            elif shape_type == "polygon":
                clipped = self._clip_polygon(
                    [(float(point[0]), float(point[1])) for point in points],
                    x, y, x + width, y + height,
                )
                if len(clipped) < 3:
                    continue
                shape["points"] = [[px - x, py - y] for px, py in clipped]
            elif shape_type == "point":
                if len(points) != 1:
                    raise ValueError("point 标注必须包含一个坐标点")
                px, py = float(points[0][0]), float(points[0][1])
                if not (x <= px <= x + width and y <= py <= y + height):
                    continue
                shape["points"] = [[px - x, py - y]]
            elif shape_type in {"line", "linestrip"}:
                clipped = self._clip_line(
                    [(float(point[0]), float(point[1])) for point in points],
                    x, y, x + width, y + height,
                )
                if len(clipped) < 2:
                    continue
                shape["points"] = [[px - x, py - y] for px, py in clipped]
            else:
                raise ValueError(f"裁剪暂不支持 shape_type: {shape_type!r}")
            result.append(shape)
        return result

    @staticmethod
    def _clip_polygon(
        points: list[tuple[float, float]], left: float, top: float, right: float, bottom: float
    ) -> list[tuple[float, float]]:
        def clip(poly, inside, intersect):
            if not poly:
                return []
            output = []
            previous = poly[-1]
            for current in poly:
                if inside(current):
                    if not inside(previous):
                        output.append(intersect(previous, current))
                    output.append(current)
                elif inside(previous):
                    output.append(intersect(previous, current))
                previous = current
            return output

        polygon = points
        polygon = clip(polygon, lambda p: p[0] >= left, lambda a, b: (left, a[1] + (b[1]-a[1]) * (left-a[0]) / (b[0]-a[0])))
        polygon = clip(polygon, lambda p: p[0] <= right, lambda a, b: (right, a[1] + (b[1]-a[1]) * (right-a[0]) / (b[0]-a[0])))
        polygon = clip(polygon, lambda p: p[1] >= top, lambda a, b: (a[0] + (b[0]-a[0]) * (top-a[1]) / (b[1]-a[1]), top))
        polygon = clip(polygon, lambda p: p[1] <= bottom, lambda a, b: (a[0] + (b[0]-a[0]) * (bottom-a[1]) / (b[1]-a[1]), bottom))
        return polygon

    @classmethod
    def _clip_line(
        cls,
        points: list[tuple[float, float]],
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> list[tuple[float, float]]:
        if len(points) < 2:
            raise ValueError("line/linestrip 标注至少需要两个坐标点")
        result: list[tuple[float, float]] = []
        for start, end in zip(points, points[1:]):
            segment = cls._clip_segment(start, end, left, top, right, bottom)
            if segment is None:
                continue
            for point in segment:
                if not result or point != result[-1]:
                    result.append(point)
        return result

    @staticmethod
    def _clip_segment(
        start: tuple[float, float],
        end: tuple[float, float],
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> list[tuple[float, float]] | None:
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        low, high = 0.0, 1.0
        for p, q in (
            (-dx, x1 - left),
            (dx, right - x1),
            (-dy, y1 - top),
            (dy, bottom - y1),
        ):
            if p == 0:
                if q < 0:
                    return None
                continue
            ratio = q / p
            if p < 0:
                low = max(low, ratio)
            else:
                high = min(high, ratio)
            if low > high:
                return None
        return [
            (x1 + low * dx, y1 + low * dy),
            (x1 + high * dx, y1 + high * dy),
        ]
