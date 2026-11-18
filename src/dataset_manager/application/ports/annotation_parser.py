from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dataset_manager.application.dto.parse_result import ParseResult


class AnnotationParser(ABC):
    @abstractmethod
    def parse(self, annotation_path: Path, image_path: Path | None = None) -> ParseResult:
        """Read and validate an annotation without modifying source files."""

