from __future__ import annotations

import os
import math
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from dataset_manager.application.model_testing.models import (
    ModelTestConfig, ModelTestProgress, ModelTestResult, ModelType,
)
from dataset_manager.application.model_testing.runner import ModelRunner
from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.application.ports.scan_cache_reader import ScanCacheReader
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class ModelTestService:
    """Run a model and commit generated LabelMe annotations beside source images."""

    def __init__(
        self,
        repository: ProjectRepository,
        cache_reader: ScanCacheReader,
        runner: ModelRunner,
        json_store: AtomicJsonStore | None = None,
    ) -> None:
        self._repository = repository
        self._cache_reader = cache_reader
        self._runner = runner
        self._json_store = json_store or AtomicJsonStore()

    def execute(
        self,
        config: ModelTestConfig,
        progress: Callable[[ModelTestProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ModelTestResult:
        _, records = self._validate(config)
        pending: list[tuple[Path, dict[str, Any]]] = []
        succeeded = failed = skipped = 0
        cancelled = False
        loaded = False
        failure_messages: list[str] = []
        repeated_failure: tuple[str, int] | None = None
        for index, record in enumerate(records, start=1):
            if is_cancelled is not None and is_cancelled():
                cancelled = True
                break
            image_path = str(Path(record["image_path"]).resolve())
            annotation_path = self._annotation_target(record, image_path)
            if annotation_path.exists() and not config.overwrite_existing:
                skipped += 1
            else:
                if not loaded:
                    try:
                        self._runner.load(config)
                    except Exception as error:
                        raise RuntimeError(
                            f"模型加载失败：{type(error).__name__}: {error}"
                        ) from error
                    loaded = True
                try:
                    prediction = self._runner.predict(image_path, config)
                    if is_cancelled is not None and is_cancelled():
                        cancelled = True
                        break
                    pending.append((
                        annotation_path,
                        self._to_labelme(image_path, prediction, config.model_type),
                    ))
                    succeeded += 1
                    repeated_failure = None
                except Exception as error:
                    failed += 1
                    signature = f"{type(error).__name__}: {error}"
                    failure_messages.append(f"{Path(image_path).name}: {signature}")
                    if repeated_failure is not None and repeated_failure[0] == signature:
                        repeated_failure = (signature, repeated_failure[1] + 1)
                    else:
                        repeated_failure = (signature, 1)
                    if succeeded == 0 and repeated_failure[1] >= 3:
                        raise RuntimeError(
                            "连续 3 张图片推理失败，任务已停止。原因："
                            f"{signature}"
                        ) from error
            if progress is not None:
                progress(ModelTestProgress(
                    index, len(records), succeeded, failed, skipped, image_path
                ))

        if cancelled or (is_cancelled is not None and is_cancelled()):
            return ModelTestResult(
                len(records), succeeded, failed, 0, skipped, cancelled=True,
                failure_messages=tuple(failure_messages[:20]),
            )
        if failed and succeeded == 0:
            raise RuntimeError(
                "所有待处理图片均推理失败。首个错误："
                + failure_messages[0]
            )
        self._commit_annotations(pending)
        return ModelTestResult(
            len(records), succeeded, failed, len(pending), skipped, cancelled=False,
            failure_messages=tuple(failure_messages[:20]),
        )

    def _validate(self, config: ModelTestConfig) -> tuple[Path, list[dict[str, Any]]]:
        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise ValueError("请选择存在的模型文件")
        if config.model_type not in {
            ModelType.YOLO_DETECTION, ModelType.YOLO_SEGMENTATION,
        }:
            raise ValueError("当前仅支持生成检测框或分割多边形 LabelMe 标注")
        if config.image_width < 32 or config.image_height < 32:
            raise ValueError("模型输入宽高不能小于 32")
        if config.image_width > 8192 or config.image_height > 8192:
            raise ValueError("模型输入宽高不能大于 8192")
        if not 0.0 <= config.confidence <= 1.0 or not 0.0 <= config.iou_threshold <= 1.0:
            raise ValueError("置信度和 IoU 阈值必须在 0 到 1 之间")
        if config.max_detections < 1:
            raise ValueError("最大目标数必须大于 0")
        dataset = self._repository.get(config.project_id).find_dataset(config.dataset_id)
        if dataset.active_scan_id is None:
            raise ValueError("数据集尚未完成扫描")
        cache = self._cache_reader.read(dataset.id, dataset.active_scan_id)
        raw_records = cache.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("扫描缓存 records 字段损坏")
        records = sorted(
            (
                record for record in raw_records
                if isinstance(record, dict)
                and isinstance(record.get("image_path"), str)
                and Path(record["image_path"]).is_file()
            ),
            key=lambda record: str(record["image_path"]).casefold(),
        )
        if not records:
            raise ValueError("当前数据集没有可测试的图片")
        return model_path, records

    @staticmethod
    def _annotation_target(record: dict[str, Any], image_path: str) -> Path:
        annotation_path = record.get("annotation_path")
        if isinstance(annotation_path, str) and annotation_path.strip():
            return Path(annotation_path).resolve()
        return Path(image_path).with_suffix(".json")

    @staticmethod
    def _to_labelme(
        image_path: str, prediction: dict[str, Any], model_type: ModelType
    ) -> dict[str, Any]:
        original_shape = prediction.get("original_shape")
        if (
            isinstance(original_shape, list)
            and len(original_shape) >= 2
            and all(isinstance(value, (int, float)) for value in original_shape[:2])
        ):
            height, width = int(original_shape[0]), int(original_shape[1])
        else:
            with Image.open(image_path) as image:
                width, height = image.size
        detections = prediction.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError("模型检测结果 detections 格式无效")
        segments = prediction.get("segments", [])
        if not isinstance(segments, list):
            segments = []
        shapes: list[dict[str, Any]] = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise ValueError("模型检测结果必须是对象")
            label = str(detection.get("class_name", "")).strip()
            if not label:
                raise ValueError("模型检测结果缺少类别名称")
            confidence = float(detection.get("confidence", 0.0))
            if not math.isfinite(confidence):
                confidence = 0.0
            confidence = min(max(confidence, 0.0), 1.0)
            polygon = None
            if (
                model_type is ModelType.YOLO_SEGMENTATION
                and index < len(segments)
                and isinstance(segments[index], (list, tuple))
                and len(segments[index]) >= 3
            ):
                polygon = ModelTestService._polygon_points(
                    segments[index], width, height
                )
            if polygon is not None:
                points = polygon
                shape_type = "polygon"
            else:
                box = detection.get("box_xyxy")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    raise ValueError("模型检测结果缺少 box_xyxy")
                coordinates = [float(value) for value in box]
                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError("模型检测框包含非有限坐标")
                x1, y1, x2, y2 = coordinates
                x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
                y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
                if x2 <= x1 or y2 <= y1:
                    continue
                points = [[x1, y1], [x2, y2]]
                shape_type = "rectangle"
            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
                "confidence": confidence,
            })
        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": Path(image_path).name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

    @staticmethod
    def _polygon_points(
        raw_points: list[Any] | tuple[Any, ...], width: int, height: int
    ) -> list[list[float]] | None:
        points: list[list[float]] = []
        try:
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    return None
                x, y = float(point[0]), float(point[1])
                if not math.isfinite(x) or not math.isfinite(y):
                    return None
                points.append([
                    min(max(x, 0.0), width),
                    min(max(y, 0.0), height),
                ])
        except (TypeError, ValueError):
            return None
        unique = {(round(x, 6), round(y, 6)) for x, y in points}
        return points if len(unique) >= 3 else None

    def _commit_annotations(self, pending: list[tuple[Path, dict[str, Any]]]) -> None:
        committed: list[tuple[Path, bytes | None]] = []
        try:
            for path, document in pending:
                original = path.read_bytes() if path.exists() else None
                self._json_store.write(path, document)
                committed.append((path, original))
        except Exception:
            for path, original in reversed(committed):
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    self._restore_bytes(path, original)
            raise

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
