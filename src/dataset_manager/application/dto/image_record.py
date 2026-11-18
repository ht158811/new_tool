from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ImageFilter:
    filename: str = ""
    category: str = ""
    min_width: int = 0
    max_width: int = 0
    min_height: int = 0
    max_height: int = 0


@dataclass(frozen=True, slots=True)
class BrowserImage:
    image_path: str
    annotation_path: str | None
    width: int | None
    height: int | None
    categories: tuple[str, ...] = field(default_factory=tuple)
    issue_count: int = 0
    annotation_valid: bool = False

