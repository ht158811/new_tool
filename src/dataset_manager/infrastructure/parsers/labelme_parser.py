from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from dataset_manager.application.dto.parse_result import ParseIssue, ParseResult, Severity
from dataset_manager.application.ports.annotation_parser import AnnotationParser
from dataset_manager.domain.models.labelme import LabelMeDocument, LabelMeShape

ImageSizeProbe = Callable[[Path], tuple[int, int]]
SUPPORTED_SHAPES = {"rectangle", "polygon"}
ROOT_FIELDS = {
    "version", "flags", "shapes", "imagePath", "imageData", "imageHeight", "imageWidth"
}
SHAPE_FIELDS = {"label", "points", "group_id", "shape_type", "flags"}


class LabelMeParser(AnnotationParser):
    def __init__(self, image_size_probe: ImageSizeProbe | None = None) -> None:
        self._image_size_probe = image_size_probe

    def parse(self, annotation_path: Path, image_path: Path | None = None) -> ParseResult:
        try:
            with annotation_path.open("r", encoding="utf-8-sig") as stream:
                raw = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return ParseResult(None, [ParseIssue("json_read_failed", str(error), Severity.ERROR)])
        if not isinstance(raw, dict):
            return ParseResult(None, [ParseIssue("invalid_root", "JSON 根节点必须是对象", Severity.ERROR)])
        return self.parse_data(raw, image_path=image_path)

    def parse_data(self, raw: dict[str, Any], image_path: Path | None = None) -> ParseResult:
        issues: list[ParseIssue] = []
        raw_shapes = raw.get("shapes")
        if not isinstance(raw_shapes, list):
            return ParseResult(None, [ParseIssue("missing_shapes", "shapes 必须是数组", Severity.ERROR)])

        flags = raw.get("flags", {})
        if not isinstance(flags, dict):
            issues.append(ParseIssue("invalid_flags", "flags 必须是对象", Severity.ERROR, "flags"))
            flags = {}

        shapes: list[LabelMeShape] = []
        for index, item in enumerate(raw_shapes):
            shape = self._parse_shape(item, index, issues)
            if shape is not None:
                shapes.append(shape)
        if not raw_shapes:
            issues.append(ParseIssue("empty_annotation", "标注中没有 shape", Severity.INFO, "shapes"))

        height = self._optional_positive_int(raw.get("imageHeight"), "imageHeight", issues)
        width = self._optional_positive_int(raw.get("imageWidth"), "imageWidth", issues)
        image_path_value = raw.get("imagePath")
        if image_path_value is not None and not isinstance(image_path_value, str):
            issues.append(ParseIssue("invalid_image_path", "imagePath 必须是字符串", Severity.ERROR, "imagePath"))
            image_path_value = None

        document = LabelMeDocument(
            version=raw.get("version") if isinstance(raw.get("version"), str) else None,
            flags=flags,
            shapes=shapes,
            image_path=image_path_value,
            image_data=raw.get("imageData") if isinstance(raw.get("imageData"), str) else None,
            image_height=height,
            image_width=width,
            extra={key: value for key, value in raw.items() if key not in ROOT_FIELDS},
        )
        self._check_bounds(document, issues)
        self._check_image_size(document, image_path, issues)
        return ParseResult(document, issues)

    def _parse_shape(
        self, item: Any, index: int, issues: list[ParseIssue]
    ) -> LabelMeShape | None:
        location = f"shapes[{index}]"
        if not isinstance(item, dict):
            issues.append(ParseIssue("invalid_shape", "shape 必须是对象", Severity.ERROR, location))
            return None
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            issues.append(ParseIssue("invalid_label", "label 必须是非空字符串", Severity.ERROR, location))
            label = "" if label is None else str(label)
        points = item.get("points")
        parsed_points: list[list[float]] = []
        if not isinstance(points, list):
            issues.append(ParseIssue("invalid_points", "points 必须是数组", Severity.ERROR, location))
        else:
            for point_index, point in enumerate(points):
                if (
                    not isinstance(point, (list, tuple))
                    or len(point) != 2
                    or isinstance(point[0], bool)
                    or isinstance(point[1], bool)
                    or not isinstance(point[0], (int, float))
                    or not isinstance(point[1], (int, float))
                    or not math.isfinite(float(point[0]))
                    or not math.isfinite(float(point[1]))
                ):
                    issues.append(ParseIssue(
                        "invalid_point", "坐标点必须是两个有限数值", Severity.ERROR,
                        f"{location}.points[{point_index}]",
                    ))
                    continue
                parsed_points.append([float(point[0]), float(point[1])])
        shape_type = item.get("shape_type")
        if shape_type not in SUPPORTED_SHAPES:
            issues.append(ParseIssue(
                "unsupported_shape", f"暂不支持可视化的 shape_type: {shape_type!r}",
                Severity.WARNING, location,
            ))
        required_points = 2 if shape_type == "rectangle" else 3 if shape_type == "polygon" else 0
        if required_points and len(parsed_points) < required_points:
            issues.append(ParseIssue(
                "insufficient_points", f"{shape_type} 至少需要 {required_points} 个点",
                Severity.ERROR, location,
            ))
        shape_flags = item.get("flags", {})
        if not isinstance(shape_flags, dict):
            issues.append(ParseIssue("invalid_shape_flags", "shape.flags 必须是对象", Severity.ERROR, location))
            shape_flags = {}
        return LabelMeShape(
            label=label,
            points=parsed_points,
            group_id=item.get("group_id"),
            shape_type=shape_type if isinstance(shape_type, str) else None,
            flags=shape_flags,
            extra={key: value for key, value in item.items() if key not in SHAPE_FIELDS},
        )

    @staticmethod
    def _optional_positive_int(
        value: Any, field_name: str, issues: list[ParseIssue]
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            issues.append(ParseIssue(
                "invalid_dimension", f"{field_name} 必须是正整数", Severity.ERROR, field_name
            ))
            return None
        return value

    @staticmethod
    def _check_bounds(document: LabelMeDocument, issues: list[ParseIssue]) -> None:
        if document.image_width is None or document.image_height is None:
            return
        for shape_index, shape in enumerate(document.shapes):
            for point_index, (x, y) in enumerate(shape.points):
                if x < 0 or y < 0 or x > document.image_width or y > document.image_height:
                    issues.append(ParseIssue(
                        "point_out_of_bounds",
                        f"坐标 ({x}, {y}) 超出声明的图片尺寸",
                        Severity.ERROR,
                        f"shapes[{shape_index}].points[{point_index}]",
                    ))

    def _check_image_size(
        self, document: LabelMeDocument, image_path: Path | None, issues: list[ParseIssue]
    ) -> None:
        if image_path is None:
            if not document.image_path and not document.image_data:
                issues.append(ParseIssue(
                    "image_unresolved", "没有可用于关联图片的 imagePath/imageData", Severity.WARNING
                ))
            return
        if not image_path.is_file():
            issues.append(ParseIssue("image_missing", f"图片不存在: {image_path}", Severity.ERROR))
            return
        if self._image_size_probe is None:
            return
        try:
            actual_width, actual_height = self._image_size_probe(image_path)
        except Exception as error:
            issues.append(ParseIssue("image_read_failed", str(error), Severity.ERROR))
            return
        declared = (document.image_width, document.image_height)
        if None not in declared and declared != (actual_width, actual_height):
            issues.append(ParseIssue(
                "image_size_mismatch",
                f"声明尺寸 {declared[0]}x{declared[1]} 与实际尺寸 {actual_width}x{actual_height} 不一致",
                Severity.ERROR,
            ))

