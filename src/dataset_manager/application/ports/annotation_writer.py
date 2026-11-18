from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AnnotationWriter(ABC):
    @abstractmethod
    def update_shape_label(self, annotation_path: Path, shape_index: int, label: str) -> None:
        """Atomically update one shape label while preserving all other raw fields."""

