from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dataset_manager.application.model_testing.models import (
    ModelTestConfig, ModelTestProgress, ModelTestResult,
)
from dataset_manager.application.model_testing.runner import ModelRunner
from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.application.ports.scan_cache_reader import ScanCacheReader
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class ModelTestService:
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
        model_path, output_path, image_paths = self._validate(config)
        model_info = self._runner.load(config)
        predictions: list[dict[str, object]] = []
        succeeded = 0
        failed = 0
        cancelled = False
        started_at = datetime.now(timezone.utc)
        for index, image_path in enumerate(image_paths, start=1):
            if is_cancelled is not None and is_cancelled():
                cancelled = True
                break
            try:
                prediction = self._runner.predict(image_path, config)
                predictions.append({"image_path": image_path, "status": "success", **prediction})
                succeeded += 1
            except Exception as error:  # keep one bad image from losing the report
                predictions.append({
                    "image_path": image_path,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                })
                failed += 1
            if progress is not None:
                progress(ModelTestProgress(index, len(image_paths), succeeded, failed, image_path))
        finished_at = datetime.now(timezone.utc)
        report = {
            "schema_version": 1,
            "status": "cancelled" if cancelled else "completed",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 6),
            "configuration": config.to_dict(),
            "model": {
                "path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "sha256": self._sha256(model_path),
                **model_info,
            },
            "summary": {
                "total_images": len(image_paths),
                "processed": succeeded + failed,
                "succeeded": succeeded,
                "failed": failed,
            },
            "predictions": predictions,
        }
        self._json_store.write(output_path, report)
        return ModelTestResult(str(output_path), len(image_paths), succeeded, failed, cancelled)

    def _validate(self, config: ModelTestConfig) -> tuple[Path, Path, list[str]]:
        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise ValueError("请选择存在的模型文件")
        if config.image_width < 32 or config.image_height < 32:
            raise ValueError("模型输入宽高不能小于 32")
        if not 0.0 <= config.confidence <= 1.0 or not 0.0 <= config.iou_threshold <= 1.0:
            raise ValueError("置信度和 IoU 阈值必须在 0 到 1 之间")
        output_path = Path(config.output_json).expanduser().resolve()
        if output_path.suffix.casefold() != ".json":
            raise ValueError("输出文件必须使用 .json 扩展名")
        if output_path.exists():
            raise FileExistsError(f"为防止覆盖，请选择新的输出文件：{output_path}")
        dataset = self._repository.get(config.project_id).find_dataset(config.dataset_id)
        if dataset.active_scan_id is None:
            raise ValueError("数据集尚未完成扫描")
        cache = self._cache_reader.read(dataset.id, dataset.active_scan_id)
        records = cache.get("records")
        if not isinstance(records, list):
            raise ValueError("扫描缓存 records 字段损坏")
        image_paths = sorted(
            str(Path(record["image_path"]).resolve())
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("image_path"), str)
            and Path(record["image_path"]).is_file()
        )
        if not image_paths:
            raise ValueError("当前数据集没有可测试的图片")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return model_path, output_path, image_paths

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
