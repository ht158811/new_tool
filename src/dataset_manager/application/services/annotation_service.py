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

