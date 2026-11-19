from __future__ import annotations

from pathlib import Path

from dataset_manager.application.dto.parse_result import ParseResult
from dataset_manager.application.ports.annotation_parser import AnnotationParser
from dataset_manager.application.ports.annotation_writer import AnnotationWriter


class AnnotationService:
    def __init__(self, parser: AnnotationParser, writer: AnnotationWriter) -> None:
        self._parser = parser
        self._writer = writer

    def read(self, annotation_path: str | None, image_path: str) -> ParseResult | None:
        if annotation_path is None:
            return None
        return self._parser.parse(Path(annotation_path), Path(image_path))

    def update_shape_label(
        self, annotation_path: str, shape_index: int, label: str
    ) -> ParseResult:
        path = Path(annotation_path)
        self._writer.update_shape_label(path, shape_index, label)
        return self._parser.parse(path)

    def delete_shape(self, annotation_path: str, shape_index: int) -> ParseResult:
        path = Path(annotation_path)
        self._writer.delete_shape(path, shape_index)
        return self._parser.parse(path)

    def delete_annotation(self, annotation_path: str) -> None:
        self._writer.delete_annotation(Path(annotation_path))

    def update_rectangle_points(
        self, annotation_path: str, shape_index: int, points: list[list[float]]
    ) -> ParseResult:
        path = Path(annotation_path)
        self._writer.update_rectangle_points(path, shape_index, points)
        return self._parser.parse(path)

    def add_rectangle(
        self, annotation_path: str, label: str, points: list[list[float]]
    ) -> ParseResult:
        path = Path(annotation_path)
        self._writer.add_rectangle(path, label, points)
        return self._parser.parse(path)

    def create_annotation_with_rectangle(
        self,
        annotation_path: str,
        image_path: str,
        width: int,
        height: int,
        label: str,
        points: list[list[float]],
    ) -> ParseResult:
        path = Path(annotation_path)
        self._writer.create_annotation_with_rectangle(
            path, Path(image_path).name, width, height, label, points
        )
        return self._parser.parse(path, Path(image_path))

