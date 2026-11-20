from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dataset_manager.application.preprocessing.models import (
    LabelRetentionMode, LabelSplitConfig, ProcessingRecord,
)
from dataset_manager.infrastructure.preprocessing.common import (
    copy_image_exact, flattened_output_stem, load_json_copy, relative_output_path,
)
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class LabelSplitEngine:
    def __init__(self, json_store: AtomicJsonStore | None = None) -> None:
        self._json_store = json_store or AtomicJsonStore()

    def process(
        self,
        record: ProcessingRecord,
        dataset_root: str,
        output_directory: Path,
        config: LabelSplitConfig,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> tuple[int, int, int]:
        self._validate(config)
        if not record.annotation_valid or not record.annotation_path:
            return 0, 0, 1
        raw = load_json_copy(record.annotation_path)
        if not isinstance(raw.get("shapes"), list):
            raise ValueError("LabelMe JSON 缺少有效的 shapes 数组")

        generated_images = generated_json = skipped = 0
        for side, enabled, retention, grouped in (
            (
                "before", config.create_before, config.before_retention,
                config.before_group_by_label,
            ),
            (
                "after", config.create_after, config.after_retention,
                config.after_group_by_label,
            ),
        ):
            if not enabled:
                continue
            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("标签分割任务已取消")
            groups = self._build_groups(raw, config.separator, retention, grouped)
            if not groups:
                skipped += 1
                continue
            for group_label, output_data in groups:
                if is_cancelled is not None and is_cancelled():
                    raise InterruptedError("标签分割任务已取消")
                target = output_directory / side
                if group_label is not None:
                    target /= self._safe_folder_name(group_label)
                image_count, json_count = self._write_pair(
                    record, dataset_root, target, side, group_label, output_data
                )
                generated_images += image_count
                generated_json += json_count
        return generated_images, generated_json, skipped

    @staticmethod
    def _validate(config: LabelSplitConfig) -> None:
        if not config.separator:
            raise ValueError("标签分隔符不能为空")
        if not config.create_before and not config.create_after:
            raise ValueError("前标签输出和后标签输出至少选择一项")
        if not isinstance(config.before_retention, LabelRetentionMode):
            raise ValueError("前标签输出的标签保留方式无效")
        if not isinstance(config.after_retention, LabelRetentionMode):
            raise ValueError("后标签输出的标签保留方式无效")

    def _build_groups(
        self,
        raw: dict[str, Any],
        separator: str,
        retention: LabelRetentionMode,
        grouped: bool,
    ) -> list[tuple[str | None, dict[str, Any]]]:
        selected: list[tuple[str, dict[str, Any]]] = []
        for shape in raw["shapes"]:
            if not isinstance(shape, dict) or not isinstance(shape.get("label"), str):
                raise ValueError("shape.label 必须是字符串")
            original = shape["label"].strip()
            before, after, _ = self._split_label(original, separator)
            transformed = copy.deepcopy(shape)
            transformed["label"] = {
                LabelRetentionMode.BEFORE: before,
                LabelRetentionMode.AFTER: after,
                LabelRetentionMode.BOTH: original,
            }[retention]
            selected.append((transformed["label"], transformed))
        if not selected:
            return []
        if not grouped:
            result = copy.deepcopy(raw)
            result["shapes"] = [shape for _, shape in selected]
            return [(None, result)]
        by_label: dict[str, list[dict[str, Any]]] = {}
        for label, shape in selected:
            by_label.setdefault(label, []).append(shape)
        results: list[tuple[str | None, dict[str, Any]]] = []
        for label in sorted(by_label, key=str.casefold):
            result = copy.deepcopy(raw)
            result["shapes"] = by_label[label]
            results.append((label, result))
        return results

    def _write_pair(
        self,
        record: ProcessingRecord,
        dataset_root: str,
        target: Path,
        side: str,
        group_label: str | None,
        data: dict[str, Any],
    ) -> tuple[int, int]:
        relative = relative_output_path(record.image_path, dataset_root)
        flat_stem = record.output_stem or flattened_output_stem(record.image_path, dataset_root)
        side_suffix = "pre" if side == "before" else "post"
        stem = f"{flat_stem}_{side_suffix}"
        image_output = target / f"{stem}{relative.suffix.lower()}"
        json_output = target / f"{stem}.json"
        copy_image_exact(record.image_path, image_output)
        output_data = copy.deepcopy(data)
        output_data["imagePath"] = image_output.name
        output_data["imageData"] = None
        try:
            self._json_store.write(json_output, output_data)
        except Exception:
            image_output.unlink(missing_ok=True)
            raise
        return 1, 1

    @staticmethod
    def _split_label(label: str, separator: str) -> tuple[str, str, bool]:
        if separator not in label:
            clean = label.strip()
            return clean, clean, False
        before, after = label.split(separator, 1)
        return before.strip(), after.strip(), True

    @staticmethod
    def _safe_folder_name(label: str) -> str:
        clean = label.strip()
        if not clean:
            return "_空标签"
        # Windows forbids these characters in directory names. Use a readable
        # underscore instead of exposing implementation-oriented %XXXX escapes.
        clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", clean)
        while clean.endswith((".", " ")):
            clean = clean[:-1] + "_"
        reserved = {"CON", "PRN", "AUX", "NUL"}
        reserved.update(f"COM{index}" for index in range(1, 10))
        reserved.update(f"LPT{index}" for index in range(1, 10))
        if clean.upper() in reserved:
            clean = f"_{clean}"
        if len(clean) > 240:
            raise ValueError(f"标签过长，无法作为 Windows 目录名: {label[:40]}…")
        return clean
