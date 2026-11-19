from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from dataset_manager.application.preprocessing.models import AugmentationConfig, ProcessingRecord
from dataset_manager.infrastructure.preprocessing.common import (
    flattened_output_stem, load_json_copy, prepare_labelme, relative_output_path,
    save_image,
)
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class AugmentationEngine:
    def __init__(self, json_store: AtomicJsonStore | None = None) -> None:
        self._json_store = json_store or AtomicJsonStore()

    def process(
        self,
        record: ProcessingRecord,
        dataset_root: str,
        output_directory: Path,
        config: AugmentationConfig,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> tuple[int, int, int]:
        enabled = config.enabled_operations()
        if not enabled:
            raise ValueError("至少选择一项数据增强")
        if config.outputs_per_image <= 0:
            raise ValueError("每张图片增强数量必须大于 0")
        if config.max_operations_per_output <= 0:
            raise ValueError("每个增强结果的最大组合操作数必须大于 0")
        relative = relative_output_path(record.image_path, dataset_root)
        flat_stem = flattened_output_stem(record.image_path, dataset_root)
        raw = load_json_copy(record.annotation_path) if record.annotation_valid and record.annotation_path else None
        stable_seed = int(hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(config.random_seed + stable_seed)
        selected = []
        for _ in range(config.outputs_per_image):
            pick_count = rng.randint(1, min(config.max_operations_per_output, len(enabled)))
            picked = set(rng.sample(enabled, pick_count))
            selected.append([operation for operation in enabled if operation in picked])
        generated_images = generated_json = 0
        with Image.open(record.image_path) as source:
            source.load()
            for index, operations in enumerate(selected, start=1):
                if is_cancelled is not None and is_cancelled():
                    raise InterruptedError("增强任务已取消")
                image = source.copy()
                shapes = copy.deepcopy(raw.get("shapes", [])) if raw else []
                current_width, current_height = source.width, source.height
                for operation in operations:
                    if is_cancelled is not None and is_cancelled():
                        raise InterruptedError("增强任务已取消")
                    if raw:
                        shapes = self._transform_shapes(
                            shapes, operation, current_width, current_height
                        )
                    image, new_width, new_height = self._apply_image(
                        image, operation, config,
                        config.random_seed + stable_seed + index,
                    )
                    current_width, current_height = new_width, new_height
                operation_name = "_".join(operations)
                stem = f"{flat_stem}_aug_{operation_name}_{index}"
                image_output = output_directory / "data" / f"{stem}{relative.suffix.lower()}"
                save_image(image, image_output)
                generated_images += 1
                if raw is not None:
                    label_output = image_output.with_suffix(".json")
                    label_data = prepare_labelme(
                        raw, image_output.name, new_width, new_height, shapes
                    )
                    try:
                        self._json_store.write(label_output, label_data)
                    except Exception:
                        image_output.unlink(missing_ok=True)
                        raise
                    generated_json += 1
        return generated_images, generated_json, 0

    @staticmethod
    def _apply_image(
        source: Image.Image, operation: str, config: AugmentationConfig, seed: int
    ) -> tuple[Image.Image, int, int]:
        if operation == "horizontal_flip":
            result = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif operation == "vertical_flip":
            result = source.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif operation == "rotate_90":
            result = source.transpose(Image.Transpose.ROTATE_90)
        elif operation == "rotate_180":
            result = source.transpose(Image.Transpose.ROTATE_180)
        elif operation == "rotate_270":
            result = source.transpose(Image.Transpose.ROTATE_270)
        elif operation == "brightness":
            result = ImageEnhance.Brightness(source).enhance(float(config.brightness))
        elif operation == "contrast":
            result = ImageEnhance.Contrast(source).enhance(float(config.contrast))
        elif operation == "saturation":
            result = ImageEnhance.Color(source.convert("RGB")).enhance(float(config.saturation))
        elif operation == "gaussian_blur":
            result = source.filter(ImageFilter.GaussianBlur(float(config.gaussian_blur)))
        elif operation == "gaussian_noise":
            array = np.asarray(source.convert("RGB"), dtype=np.float32)
            noise = np.random.default_rng(seed).normal(0, float(config.gaussian_noise), array.shape)
            result = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")
        else:
            raise ValueError(f"未知增强操作: {operation}")
        return result, result.width, result.height

    def _transform_shapes(
        self, shapes: Any, operation: str, width: int, height: int
    ) -> list[dict[str, Any]]:
        if not isinstance(shapes, list):
            raise ValueError("shapes 必须是数组")
        result = copy.deepcopy(shapes)
        if operation in {"brightness", "contrast", "saturation", "gaussian_blur", "gaussian_noise"}:
            return result
        for shape in result:
            if not isinstance(shape, dict) or not isinstance(shape.get("points"), list):
                raise ValueError("shape 或 points 格式无效")
            transformed = []
            for point in shape["points"]:
                x, y = float(point[0]), float(point[1])
                if operation == "horizontal_flip":
                    x, y = width - x, y
                elif operation == "vertical_flip":
                    x, y = x, height - y
                elif operation == "rotate_90":
                    x, y = y, width - x
                elif operation == "rotate_180":
                    x, y = width - x, height - y
                elif operation == "rotate_270":
                    x, y = height - y, x
                transformed.append([x, y])
            shape["points"] = transformed
        return result
