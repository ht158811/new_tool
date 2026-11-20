from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dataset_manager.application.preprocessing.models import (
    LabelRenameConfig, ProcessingRecord,
)
from dataset_manager.infrastructure.preprocessing.common import (
    copy_image_exact, flattened_output_stem, load_json_copy, relative_output_path,
)
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class LabelRenameEngine:
    """Create a LabelMe copy with exact shape.label replacements."""

    def __init__(self, json_store: AtomicJsonStore | None = None) -> None:
        self._json_store = json_store or AtomicJsonStore()

    def process(
        self,
        record: ProcessingRecord,
        dataset_root: str,
        output_directory: Path,
        config: LabelRenameConfig,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> tuple[int, int, int]:
        if not record.annotation_valid or not record.annotation_path:
            return 0, 0, 1
        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("批量标签重命名任务已取消")
        raw = load_json_copy(record.annotation_path)
        shapes = raw.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError("LabelMe JSON 缺少有效的 shapes 数组")
        output_data = copy.deepcopy(raw)
        for shape in output_data["shapes"]:
            if not isinstance(shape, dict):
                raise ValueError("LabelMe shape 必须是对象")
            label = shape.get("label")
            if not isinstance(label, str):
                raise ValueError("LabelMe shape.label 必须是字符串")
            if label == config.source_label:
                shape["label"] = config.target_label

        relative = relative_output_path(record.image_path, dataset_root)
        stem = record.output_stem or flattened_output_stem(record.image_path, dataset_root)
        image_output = output_directory / "data" / f"{stem}{relative.suffix.lower()}"
        json_output = image_output.with_suffix(".json")
        copy_image_exact(record.image_path, image_output)
        output_data["imagePath"] = image_output.name
        output_data["imageData"] = None
        try:
            self._json_store.write(json_output, output_data)
        except Exception:
            image_output.unlink(missing_ok=True)
            raise
        return 1, 1, 0
