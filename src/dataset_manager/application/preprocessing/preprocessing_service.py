from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.application.ports.scan_cache_reader import ScanCacheReader
from dataset_manager.application.preprocessing.models import (
    OperationType, ProcessingConfig, ProcessingProgress, ProcessingResult,
    ProcessingScope, ScopePreview,
)
from dataset_manager.application.preprocessing.processing_scope import (
    build_preview, records_from_cache, select_records,
)
from dataset_manager.infrastructure.preprocessing.augmentation_engine import AugmentationEngine
from dataset_manager.infrastructure.preprocessing.conversion_engine import ConversionEngine
from dataset_manager.infrastructure.preprocessing.crop_engine import CropEngine
from dataset_manager.infrastructure.preprocessing.label_split_engine import LabelSplitEngine
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore

ProgressCallback = Callable[[ProcessingProgress], None]
CancelCallback = Callable[[], bool]


class PreprocessingService:
    def __init__(
        self,
        repository: ProjectRepository,
        cache_reader: ScanCacheReader,
        crop_engine: CropEngine | None = None,
        augmentation_engine: AugmentationEngine | None = None,
        conversion_engine: ConversionEngine | None = None,
        label_split_engine: LabelSplitEngine | None = None,
        json_store: AtomicJsonStore | None = None,
    ) -> None:
        self._repository = repository
        self._cache = cache_reader
        self._crop = crop_engine or CropEngine()
        self._augmentation = augmentation_engine or AugmentationEngine()
        self._conversion = conversion_engine or ConversionEngine()
        self._label_split = label_split_engine or LabelSplitEngine()
        self._json_store = json_store or AtomicJsonStore()

    def preview(
        self, project_id: str, dataset_id: str, scope: ProcessingScope
    ) -> ScopePreview:
        _, dataset, records = self._context(project_id, dataset_id)
        if not dataset.active_scan_id:
            raise ValueError("数据集尚未完成完整扫描")
        return build_preview(records, scope)

    def execute(
        self,
        config: ProcessingConfig,
        progress: ProgressCallback | None = None,
        is_cancelled: CancelCallback | None = None,
    ) -> ProcessingResult:
        project, dataset, records = self._context(config.project_id, config.dataset_id)
        if (
            config.operation is OperationType.LABEL_SPLIT
            and config.scope is not ProcessingScope.VALID_ANNOTATIONS
        ):
            raise ValueError("标签分割仅支持具有有效 LabelMe JSON 的图片")
        selected = select_records(records, config.scope)
        if not config.output_root.strip():
            raise ValueError("必须指定输出目录")
        output_root = self._validate_output(Path(config.output_root), Path(dataset.root_path))
        final_directory = self._result_directory(
            output_root, config.operation, config.result_name
        )
        temporary = output_root / f".{final_directory.name}.{uuid4().hex}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
        cancelled = is_cancelled or (lambda: False)
        result = ProcessingResult(
            output_directory=str(final_directory),
            input_images=len(records),
            selected_images=len(selected),
            skipped=len(records) - len(selected),
        )
        try:
            if config.operation is OperationType.CONVERSION:
                self._execute_conversion(config, dataset, selected, temporary, result, progress, cancelled)
            else:
                self._execute_per_record(config, dataset, selected, temporary, result, progress, cancelled)
            if cancelled():
                result.cancelled = True
                raise _Cancelled
            manifest = {
                "schema_version": 1,
                "input_project_id": project.id,
                "input_project_name": project.name,
                "input_dataset_id": dataset.id,
                "input_dataset_name": dataset.name,
                "operation": config.operation.value,
                "scope": config.scope.value,
                "parameters": config.parameters(),
                "output_path": str(final_directory),
                "input_images": result.input_images,
                "selected_images": result.selected_images,
                "succeeded": result.succeeded,
                "failed": result.failed,
                "skipped": result.skipped,
                "generated_images": result.generated_images,
                "generated_annotations": result.generated_annotations,
                "warnings": result.warnings,
                "completed_at": datetime.now().astimezone().isoformat(),
                "complete": True,
            }
            self._json_store.write(temporary / "result_manifest.json", manifest)
            os.replace(temporary, final_directory)
            self._emit(progress, len(selected), len(selected), result, "完成", "处理完成")
            return result
        except _Cancelled:
            result.cancelled = True
            shutil.rmtree(temporary, ignore_errors=True)
            return result
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _context(self, project_id: str, dataset_id: str):
        project = self._repository.get(project_id)
        dataset = project.find_dataset(dataset_id)
        if not dataset.active_scan_id:
            raise ValueError("数据集尚未完成完整扫描，请先刷新数据集")
        cache = self._cache.read(dataset.id, dataset.active_scan_id)
        return project, dataset, records_from_cache(cache)

    def _execute_per_record(
        self, config, dataset, selected, temporary, result, progress, cancelled
    ) -> None:
        for index, record in enumerate(selected, start=1):
            if cancelled():
                raise _Cancelled
            try:
                if config.operation is OperationType.CROP:
                    if config.crop is None:
                        raise ValueError("缺少裁剪配置")
                    generated_images, generated_json, extra_skipped = self._crop.process(
                        record, dataset.root_path, temporary, config.crop, cancelled
                    )
                elif config.operation is OperationType.AUGMENTATION:
                    if config.augmentation is None:
                        raise ValueError("缺少增强配置")
                    generated_images, generated_json, extra_skipped = self._augmentation.process(
                        record, dataset.root_path, temporary, config.augmentation, cancelled
                    )
                elif config.operation is OperationType.LABEL_SPLIT:
                    if config.label_split is None:
                        raise ValueError("缺少标签分割配置")
                    generated_images, generated_json, extra_skipped = self._label_split.process(
                        record, dataset.root_path, temporary, config.label_split, cancelled
                    )
                else:
                    raise ValueError(f"未知处理类型: {config.operation}")
                result.succeeded += 1
                result.generated_images += generated_images
                result.generated_annotations += generated_json
                result.skipped += extra_skipped
                if record.has_json and not record.annotation_valid:
                    result.warnings.append(
                        f"无效 JSON 未同步处理: {Path(record.annotation_path).name}"
                    )
            except InterruptedError as error:
                raise _Cancelled from error
            except Exception as error:
                result.failed += 1
                result.warnings.append(
                    f"{Path(record.image_path).name}: {type(error).__name__}: {error}"
                )
            self._emit(
                progress, len(selected), index, result, Path(record.image_path).name, "处理中"
            )

    def _execute_conversion(
        self, config, dataset, selected, temporary, result, progress, cancelled
    ) -> None:
        if cancelled():
            raise _Cancelled
        if config.conversion is None:
            raise ValueError("缺少格式转换配置")
        def report(processed: int, current: str) -> None:
            result.succeeded = processed
            self._emit(progress, len(selected), processed, result, current, "处理中")

        try:
            generated_images, generated_annotations, warnings = self._conversion.process_all(
                selected,
                dataset.root_path,
                temporary,
                dataset.task_type,
                dataset.scan_summary.categories,
                config.conversion,
                cancelled,
                report,
            )
        except InterruptedError as error:
            raise _Cancelled from error
        result.generated_images = generated_images
        result.generated_annotations = generated_annotations
        result.succeeded = len(selected)
        result.warnings.extend(warnings)
        self._emit(progress, len(selected), len(selected), result, "格式转换", "处理中")

    @staticmethod
    def _validate_output(output_root: Path, source_root: Path) -> Path:
        if not str(output_root).strip():
            raise ValueError("必须指定输出目录")
        output = output_root.expanduser().resolve()
        source = source_root.expanduser().resolve()
        if output == source or output.is_relative_to(source):
            raise ValueError("输出目录不能是原始数据集目录或其子目录")
        output.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir=output, prefix=".write_test_", delete=True):
                pass
        except OSError as error:
            raise ValueError(f"输出目录不可写: {error}") from error
        return output

    @staticmethod
    def _next_result_directory(output_root: Path, operation: OperationType) -> Path:
        prefix = {
            OperationType.CROP: "Crop",
            OperationType.AUGMENTATION: "Augmentation",
            OperationType.LABEL_SPLIT: "LabelSplit",
            OperationType.CONVERSION: "Conversion",
        }[operation]
        date = datetime.now().strftime("%Y%m%d")
        for index in range(1, 10000):
            candidate = output_root / f"{prefix}_{date}_{index:03d}"
            if not candidate.exists():
                return candidate
        raise FileExistsError("无法分配新的结果目录名称")

    @classmethod
    def _result_directory(
        cls, output_root: Path, operation: OperationType, requested_name: str | None
    ) -> Path:
        if requested_name is None or not requested_name.strip():
            return cls._next_result_directory(output_root, operation)
        name = requested_name.strip()
        invalid_characters = set('<>:"/\\|?*')
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
        if (
            name in {".", ".."}
            or name != requested_name
            or name.endswith((".", " "))
            or any(character in invalid_characters or ord(character) < 32 for character in name)
            or name.split(".", 1)[0].upper() in reserved
        ):
            raise ValueError("结果目录名称包含 Windows 不允许的字符或保留名称")
        candidate = output_root / name
        if candidate.exists():
            raise FileExistsError(f"结果目录已存在，禁止覆盖: {candidate}")
        return candidate

    @staticmethod
    def _emit(progress, total, processed, result, current, status) -> None:
        if progress is not None:
            progress(ProcessingProgress(
                total=total,
                processed=processed,
                succeeded=result.succeeded,
                failed=result.failed,
                skipped=result.skipped,
                current_file=current,
                status=status,
            ))


class _Cancelled(Exception):
    pass
