from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from dataset_manager.application.preprocessing.models import (
    LabelRenameConfig, ProcessingRecord,
)
from dataset_manager.infrastructure.preprocessing.common import load_json_copy
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class LabelRenameEngine:
    """Atomically rename LabelMe shape labels in the source JSON files."""

    def __init__(self, json_store: AtomicJsonStore | None = None) -> None:
        self._json_store = json_store or AtomicJsonStore()

    def rename_in_place(
        self,
        records: list[ProcessingRecord],
        config: LabelRenameConfig,
        is_cancelled: Callable[[], bool] | None = None,
        on_validated: Callable[[int, str], None] | None = None,
    ) -> tuple[int, int]:
        changes: list[tuple[Path, bytes, dict[str, object]]] = []
        modified_shapes = 0
        seen_paths: set[Path] = set()
        for index, record in enumerate(records, start=1):
            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("批量标签重命名任务已取消，原 JSON 未修改")
            if not record.annotation_valid or not record.annotation_path:
                raise ValueError(f"缺少有效 LabelMe JSON: {record.image_path}")
            path = Path(record.annotation_path).resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            raw = load_json_copy(str(path))
            shapes = raw.get("shapes")
            if not isinstance(shapes, list):
                raise ValueError(f"LabelMe JSON 缺少有效的 shapes 数组: {path}")
            output_data = copy.deepcopy(raw)
            changed = 0
            for shape in output_data["shapes"]:
                if not isinstance(shape, dict):
                    raise ValueError(f"LabelMe shape 必须是对象: {path}")
                label = shape.get("label")
                if not isinstance(label, str):
                    raise ValueError(f"LabelMe shape.label 必须是字符串: {path}")
                if label == config.source_label:
                    shape["label"] = config.target_label
                    changed += 1
            if changed:
                changes.append((path, path.read_bytes(), output_data))
                modified_shapes += changed
            if on_validated is not None:
                on_validated(index, path.name)

        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("批量标签重命名任务已取消，原 JSON 未修改")

        committed: list[tuple[Path, bytes]] = []
        try:
            for path, original_bytes, output_data in changes:
                self._json_store.write(path, output_data)
                committed.append((path, original_bytes))
        except Exception:
            for path, original_bytes in reversed(committed):
                self._restore_bytes(path, original_bytes)
            raise
        return len(changes), modified_shapes

    @staticmethod
    def _restore_bytes(path: Path, content: bytes) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".rollback", delete=False,
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
