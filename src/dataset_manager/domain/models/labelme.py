from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LabelMeShape:
    label: str
    points: list[list[float]]
    group_id: int | str | None = None
    shape_type: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update(
            {
                "label": self.label,
                "points": self.points,
                "group_id": self.group_id,
                "shape_type": self.shape_type,
                "flags": self.flags,
            }
        )
        return result


@dataclass(slots=True)
class LabelMeDocument:
    version: str | None
    flags: dict[str, Any]
    shapes: list[LabelMeShape]
    image_path: str | None
    image_data: str | None
    image_height: int | None
    image_width: int | None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def category_names(self) -> set[str]:
        """Return normalized non-empty shape labels without inventing class IDs."""
        return {shape.label.strip() for shape in self.shapes if shape.label.strip()}

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update(
            {
                "version": self.version,
                "flags": self.flags,
                "shapes": [shape.to_dict() for shape in self.shapes],
                "imagePath": self.image_path,
                "imageData": self.image_data,
                "imageHeight": self.image_height,
                "imageWidth": self.image_width,
            }
        )
        return result

