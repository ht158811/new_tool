from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.application.ports.scan_cache_reader import ScanCacheReader
from dataset_manager.application.preprocessing.models import (
    OperationType, ProcessingConfig, ProcessingProgress, ProcessingResult,
    ProcessingRecord, ProcessingScope, ScopePreview,
)
from dataset_manager.application.preprocessing.processing_scope import (
    build_preview, records_from_cache, select_records,
)
from dataset_manager.infrastructure.preprocessing.augmentation_engine import AugmentationEngine
from dataset_manager.infrastructure.preprocessing.conversion_engine import ConversionEngine
from dataset_manager.infrastructure.preprocessing.crop_engine import CropEngine
from dataset_manager.infrastructure.preprocessing.label_split_engine import LabelSplitEngine
from dataset_manager.infrastructure.preprocessing.label_rename_engine import LabelRenameEngine
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
        label_rename_engine: LabelRenameEngine | None = None,
        json_store: AtomicJsonStore | None = None,
    ) -> None:
        self._repository = repository
        self._cache = cache_reader
        self._crop = crop_engine or CropEngine()
        self._augmentation = augmentation_engine or AugmentationEngine()
        self._conversion = conversion_engine or ConversionEngine()
        self._label_split = label_split_engine or LabelSplitEngine()
        self._label_rename = label_rename_engine or LabelRenameEngine()
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
            config.operation in {OperationType.LABEL_SPLIT, OperationType.LABEL_RENAME}
            and config.scope is not ProcessingScope.VALID_ANNOTATIONS
        ):
            raise ValueError("标签处理仅支持具有有效 LabelMe JSON 的图片")
        selected_records = select_records(records, config.scope)
        cancelled = is_cancelled or (lambda: False)
        if config.operation is OperationType.LABEL_RENAME:
            return self._execute_label_rename_in_place(
                config, dataset, records, selected_records, progress, cancelled
            )
        if config.operation is OperationType.DELETE_UNANNOTATED:
            if config.scope is not ProcessingScope.WITHOUT_JSON:
                raise ValueError("删除无 JSON 图片仅支持‘没有对应 JSON 的图片’范围")
            return self._execute_delete_unannotated_in_place(
                dataset, records, selected_records, progress, cancelled
            )
        selected = self._assign_output_stems(selected_records)
        if not config.output_root.strip():
            raise ValueError("必须指定输出目录")
        output_root = self._validate_output(Path(config.output_root), Path(dataset.root_path))
        final_directory = self._result_directory(
            output_root, config.operation, config.result_name
        )
        temporary = output_root / f".{final_directory.name}.{uuid4().hex}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
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
                "deleted_images": result.deleted_images,
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

    def _execute_label_rename_in_place(
        self, config, dataset, records, selected, progress, cancelled
    ) -> ProcessingResult:
        if config.label_rename is None:
            raise ValueError("缺少批量标签重命名配置")
        result = ProcessingResult(
            output_directory=dataset.root_path,
            input_images=len(records),
            selected_images=len(selected),
            skipped=len(records) - len(selected),
        )

        def validated(processed: int, current: str) -> None:
            self._emit(progress, len(selected), processed, result, current, "校验 JSON")

        try:
            modified_files, modified_shapes = self._label_rename.rename_in_place(
                selected, config.label_rename, cancelled, validated
            )
        except InterruptedError:
            result.cancelled = True
            return result
        result.succeeded = len(selected)
        result.generated_annotations = modified_files
        result.modified_shapes = modified_shapes
        if modified_shapes == 0:
            result.warnings.append(
                f"没有找到精确标签：{config.label_rename.source_label}"
            )
        self._emit(
            progress, len(selected), len(selected), result,
            f"修改 {modified_shapes} 个 shape", "原数据写入完成",
        )
        return result

    def _execute_delete_unannotated_in_place(
        self, dataset, records, selected, progress, cancelled
    ) -> ProcessingResult:
        """Transactionally remove images that still have no colocated LabelMe JSON."""
        root = Path(dataset.root_path).expanduser().resolve()
        result = ProcessingResult(
            output_directory=str(root),
            input_images=len(records),
            selected_images=len(selected),
            skipped=len(records) - len(selected),
        )
        staging = Path(tempfile.mkdtemp(prefix=".dataset_manager_delete_", dir=root))
        staged: list[tuple[Path, Path]] = []
        json_cache: dict[Path, list[Path]] = {}
        try:
            for index, record in enumerate(selected, start=1):
                if cancelled():
                    raise _Cancelled
                source_path = Path(record.image_path).expanduser()
                source_is_symlink = source_path.is_symlink()
                image = source_path.resolve()
                if (
                    not image.is_relative_to(root)
                    or not image.is_file()
                    or source_is_symlink
                    or self._has_corresponding_json(image, json_cache)
                ):
                    result.skipped += 1
                    result.warnings.append(
                        f"已跳过（文件不存在、超出目录或已有 JSON）：{image.name}"
                    )
                else:
                    staged_path = staging / f"{index:08d}{image.suffix}"
                    os.replace(image, staged_path)
                    staged.append((staged_path, image))
                    result.succeeded += 1
                self._emit(
                    progress, len(selected), index, result, image.name,
                    "复核并暂存待删除图片",
                )
            if cancelled():
                raise _Cancelled
            shutil.rmtree(staging)
            result.deleted_images = len(staged)
            self._emit(
                progress, len(selected), len(selected), result,
                f"删除 {result.deleted_images} 张图片", "原目录清理完成",
            )
            return result
        except _Cancelled:
            self._restore_staged_images(staged)
            shutil.rmtree(staging, ignore_errors=True)
            result.succeeded = 0
            result.deleted_images = 0
            result.cancelled = True
            return result
        except Exception:
            self._restore_staged_images(staged)
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _restore_staged_images(staged: list[tuple[Path, Path]]) -> None:
        errors: list[str] = []
        for staged_path, original in reversed(staged):
            try:
                if staged_path.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged_path, original)
            except OSError as error:
                errors.append(f"{original}: {error}")
        if errors:
            raise OSError("删除回滚失败：" + "; ".join(errors))

    @staticmethod
    def _has_corresponding_json(
        image: Path, cache: dict[Path, list[Path]]
    ) -> bool:
        parent = image.parent.resolve()
        annotations = cache.get(parent)
        if annotations is None:
            annotations = sorted(
                (
                    path for path in parent.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".json"
                ),
                key=lambda path: path.name.casefold(),
            )
            cache[parent] = annotations
        for annotation in annotations:
            if annotation.stem.casefold() == image.stem.casefold():
                return True
            try:
                with annotation.open("r", encoding="utf-8-sig") as stream:
                    document = json.load(stream)
                declared = document.get("imagePath") if isinstance(document, dict) else None
                if not isinstance(declared, str) or not declared.strip():
                    continue
                declared_path = Path(declared.replace("\\", "/"))
                if declared_path.name.casefold() == image.name.casefold():
                    return True
                candidate = (
                    declared_path if declared_path.is_absolute()
                    else annotation.parent / declared_path
                )
                if os.path.normcase(str(candidate.resolve())) == os.path.normcase(str(image)):
                    return True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return False

    @staticmethod
    def _assign_output_stems(records: list[ProcessingRecord]) -> list[ProcessingRecord]:
        """Preserve source names and add only a short numeric suffix on collisions."""
        groups: dict[str, list[ProcessingRecord]] = {}
        for record in records:
            base = Path(record.image_path).stem[:180]
            groups.setdefault(base.casefold(), []).append(record)
        reserved = set(groups)
        used: set[str] = set()
        names: dict[str, str] = {}
        for group in groups.values():
            ordered = sorted(
                group, key=lambda item: str(Path(item.image_path).resolve()).casefold()
            )
            base = Path(ordered[0].image_path).stem[:180]
            for index, record in enumerate(ordered):
                candidate = base
                if index:
                    suffix = 2
                    candidate = f"{base}_{suffix}"
                    while candidate.casefold() in reserved or candidate.casefold() in used:
                        suffix += 1
                        candidate = f"{base}_{suffix}"
                used.add(candidate.casefold())
                names[str(Path(record.image_path).resolve()).casefold()] = candidate
        return [
            replace(
                record,
                output_stem=names[str(Path(record.image_path).resolve()).casefold()],
            )
            for record in records
        ]

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
            OperationType.LABEL_RENAME: "LabelRename",
            OperationType.CONVERSION: "Conversion",
            OperationType.DELETE_UNANNOTATED: "DeleteUnannotated",
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
