from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AnnotationWriter(ABC):
    @abstractmethod
    def update_shape_label(self, annotation_path: Path, shape_index: int, label: str) -> None:
        """Atomically update one shape label while preserving all other raw fields."""

    @abstractmethod
    def delete_shape(self, annotation_path: Path, shape_index: int) -> None:
        """Atomically delete one shape while preserving the rest of the JSON."""

    @abstractmethod
    def delete_annotation(self, annotation_path: Path) -> None:
        """Delete one JSON annotation file without touching its image."""

    @abstractmethod
    def update_rectangle_points(
        self, annotation_path: Path, shape_index: int, points: list[list[float]]
    ) -> None:
        """Atomically move one rectangle while preserving its other fields."""

    @abstractmethod
    def add_rectangle(
        self, annotation_path: Path, label: str, points: list[list[float]]
    ) -> None:
        """Atomically append one LabelMe rectangle shape."""

