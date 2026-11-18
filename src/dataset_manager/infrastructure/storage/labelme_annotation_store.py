from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_manager.application.ports.annotation_writer import AnnotationWriter
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class LabelMeAnnotationStore(AnnotationWriter):
    def __init__(self, store: AtomicJsonStore | None = None) -> None:
        self._store = store or AtomicJsonStore()

    def update_shape_label(self, annotation_path: Path, shape_index: int, label: str) -> None:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("类别名称不能为空")
        data = self._store.read(annotation_path)
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError("LabelMe JSON 缺少有效的 shapes 数组")
        if shape_index < 0 or shape_index >= len(shapes):
            raise IndexError(f"shape 索引超出范围: {shape_index}")
        shape: Any = shapes[shape_index]
        if not isinstance(shape, dict):
            raise ValueError(f"shapes[{shape_index}] 不是对象")
        shape["label"] = clean_label
        self._store.write(annotation_path, data)

