from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_json_copy(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"LabelMe JSON 根节点不是对象: {path}")
    return copy.deepcopy(data)


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def relative_output_path(image_path: str, dataset_root: str) -> Path:
    source = Path(image_path).resolve()
    try:
        relative = source.relative_to(Path(dataset_root).resolve())
    except ValueError:
        relative = Path(source.name)
    return relative


def flattened_output_stem(image_path: str, dataset_root: str) -> str:
    """Return a stable flat filename stem without recreating the source directory tree."""
    relative = relative_output_path(image_path, dataset_root)
    identity = relative.as_posix().casefold().encode("utf-8")
    short_id = hashlib.sha256(identity).hexdigest()[:10]
    # Keep names comfortably below the Windows component limit after operation suffixes.
    source_stem = relative.stem[:100]
    return f"{source_stem}_{short_id}"


def prepare_labelme(
    raw: dict[str, Any], image_name: str, width: int, height: int, shapes: list[dict[str, Any]]
) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    result["imagePath"] = image_name
    result["imageData"] = None
    result["imageWidth"] = width
    result["imageHeight"] = height
    result["shapes"] = shapes
    return result


def rectangle_bounds(points: list[Any]) -> tuple[float, float, float, float]:
    if len(points) < 2:
        raise ValueError("rectangle 至少需要两个点")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)
