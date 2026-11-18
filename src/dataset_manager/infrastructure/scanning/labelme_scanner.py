from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import os
from pathlib import Path
from uuid import uuid4

from dataset_manager.application.dto.scan_result import ScanIssue, ScanProgress, ScanRecord, ScanResult
from dataset_manager.application.dto.parse_result import ParseResult
from dataset_manager.application.errors import DatasetConfigurationError, ScanCancelledError
from dataset_manager.application.ports.dataset_scanner import CancelCallback, DatasetScanner, ProgressCallback
from dataset_manager.domain.models.labelme import LabelMeDocument
from dataset_manager.domain.models.project import Dataset, TaskType, utc_now_iso
from dataset_manager.infrastructure.images import pillow_image_size
from dataset_manager.infrastructure.parsers.labelme_parser import LabelMeParser

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class LabelMeDatasetScanner(DatasetScanner):
    def __init__(
        self,
        parser: LabelMeParser | None = None,
        image_size_probe: Callable[[Path], tuple[int, int]] = pillow_image_size,
    ) -> None:
        self._parser = parser or LabelMeParser()
        self._image_size_probe = image_size_probe

    def scan(
        self,
        dataset: Dataset,
        progress: ProgressCallback | None = None,
        is_cancelled: CancelCallback | None = None,
    ) -> ScanResult:
        image_root = Path(dataset.image_directory)
        annotation_root = Path(dataset.annotation_directory)
        self._validate_configuration(dataset, image_root, annotation_root)
        cancelled = is_cancelled or (lambda: False)
        self._raise_if_cancelled(cancelled)
        discovery_issues: list[ScanIssue] = []

        if image_root.resolve() == annotation_root.resolve():
            discovered = self._collect_files(
                image_root,
                lambda path: path.suffix.lower() in IMAGE_EXTENSIONS or path.suffix.lower() == ".json",
                cancelled,
                progress,
                discovery_issues,
            )
            images = [path for path in discovered if path.suffix.lower() in IMAGE_EXTENSIONS]
            annotations = [path for path in discovered if path.suffix.lower() == ".json"]
        else:
            images = self._collect_files(
                image_root,
                lambda path: path.suffix.lower() in IMAGE_EXTENSIONS,
                cancelled,
                progress,
                discovery_issues,
            )
            annotations = self._collect_files(
                annotation_root,
                lambda path: path.suffix.lower() == ".json",
                cancelled,
                progress,
                discovery_issues,
            )
        image_map = self._group_by_relative_stem(images, image_root)
        annotation_map = self._group_by_relative_stem(annotations, annotation_root)
        global_issues = discovery_issues + self._duplicate_issues(image_map, annotation_map)
        annotation_for_image, orphan_annotations, parse_cache, association_issues = self._associate(
            images, annotations, image_root, annotation_root, image_map, annotation_map
        )
        global_issues.extend(association_issues)
        if dataset.task_type is TaskType.IMAGE_CLASSIFICATION:
            global_issues.append(ScanIssue(
                "classification_mapping_required",
                "图像分类数据集尚未配置图像级类别来源，未从 shape.label 推断图片类别",
                "warning",
            ))
        records: list[ScanRecord] = []
        categories: set[str] = set()
        paired_count = 0
        processed = 0
        total = len(images) + len(orphan_annotations)

        for image_path in images:
            self._raise_if_cancelled(cancelled)
            key = self._relative_stem(image_path, image_root)
            candidates = annotation_map.get(key, [])
            annotation_path = annotation_for_image.get(image_path)
            record = self._scan_image(
                image_path,
                annotation_path,
                dataset.task_type,
                parse_cache.get(annotation_path) if annotation_path else None,
            )
            if len(image_map[key]) > 1:
                record.issues.append(ScanIssue(
                    "ambiguous_image_key", "多个图片具有相同的相对路径主键，未自动关联标注", "error",
                    str(image_path),
                ))
            elif len(candidates) > 1:
                record.issues.append(ScanIssue(
                    "ambiguous_annotation_key", "多个标注匹配该图片，未自动关联", "error",
                    str(image_path),
                ))
            elif annotation_path is None:
                record.issues.append(ScanIssue(
                    "missing_annotation", "图片缺少对应的 LabelMe JSON", "warning", str(image_path)
                ))
            if annotation_path is not None:
                paired_count += 1
            categories.update(record.categories)
            records.append(record)
            processed += 1
            self._emit_progress(progress, image_path, processed, total, global_issues, records)

        for annotation_path in sorted(orphan_annotations, key=lambda path: str(path).casefold()):
            self._raise_if_cancelled(cancelled)
            parse_result = parse_cache.get(annotation_path) or self._parser.parse(annotation_path)
            record_issues = [
                ScanIssue(
                    issue.code,
                    issue.message,
                    self._severity_text(issue.severity),
                    str(annotation_path),
                )
                for issue in parse_result.issues
            ]
            record_issues.insert(0, ScanIssue(
                "orphan_annotation", "标注文件没有对应图片", "warning", str(annotation_path)
            ))
            record_categories = sorted(
                self._task_categories(parse_result.document, dataset.task_type),
                key=str.casefold,
            ) if parse_result.document else []
            categories.update(record_categories)
            records.append(ScanRecord(
                image_path=None,
                annotation_path=str(annotation_path.resolve()),
                categories=record_categories,
                issues=record_issues,
                annotation_valid=parse_result.is_valid,
            ))
            processed += 1
            self._emit_progress(progress, annotation_path, processed, total, global_issues, records)

        self._raise_if_cancelled(cancelled)
        return ScanResult(
            scan_id=uuid4().hex,
            dataset_id=dataset.id,
            completed_at=utc_now_iso(),
            records=records,
            issues=global_issues,
            image_count=len(images),
            annotation_count=len(annotations),
            paired_count=paired_count,
            categories=sorted(categories, key=str.casefold),
        )

    def _scan_image(
        self,
        image_path: Path,
        annotation_path: Path | None,
        task_type: TaskType,
        cached_parse_result: ParseResult | None = None,
    ) -> ScanRecord:
        issues: list[ScanIssue] = []
        width: int | None = None
        height: int | None = None
        try:
            width, height = self._image_size_probe(image_path)
        except Exception as error:
            issues.append(ScanIssue("image_read_failed", str(error), "error", str(image_path)))

        categories: list[str] = []
        annotation_valid = False
        if annotation_path is not None:
            parse_result = cached_parse_result or self._parser.parse(
                annotation_path, image_path=image_path
            )
            issues.extend(
                ScanIssue(
                    issue.code,
                    issue.message,
                    self._severity_text(issue.severity),
                    str(annotation_path),
                )
                for issue in parse_result.issues
            )
            if parse_result.document is not None:
                annotation_valid = parse_result.is_valid
                categories = sorted(
                    self._task_categories(parse_result.document, task_type), key=str.casefold
                )
                declared_image_path = parse_result.document.image_path
                if declared_image_path:
                    declared_stem = Path(declared_image_path.replace("\\", "/")).stem.casefold()
                    if declared_stem != image_path.stem.casefold():
                        issues.append(ScanIssue(
                            "image_path_mismatch",
                            f"imagePath 指向 {declared_image_path!r}，与关联图片名称不一致",
                            "warning",
                            str(annotation_path),
                        ))
                declared = (parse_result.document.image_width, parse_result.document.image_height)
                if width is not None and height is not None and None not in declared:
                    if declared != (width, height):
                        issues.append(ScanIssue(
                            "image_size_mismatch",
                            f"标注尺寸 {declared[0]}x{declared[1]} 与图片尺寸 {width}x{height} 不一致",
                            "error",
                            str(annotation_path),
                        ))
        return ScanRecord(
            image_path=str(image_path.resolve()),
            annotation_path=str(annotation_path.resolve()) if annotation_path else None,
            width=width,
            height=height,
            categories=categories,
            issues=issues,
            annotation_valid=annotation_valid,
        )

    @staticmethod
    def _task_categories(document: LabelMeDocument, task_type: TaskType) -> set[str]:
        if task_type is TaskType.OBJECT_DETECTION:
            supported_type = "rectangle"
        elif task_type is TaskType.SEMANTIC_SEGMENTATION:
            supported_type = "polygon"
        else:
            return set()
        return {
            shape.label.strip()
            for shape in document.shapes
            if shape.shape_type == supported_type and shape.label.strip()
        }

    @staticmethod
    def _severity_text(severity: object) -> str:
        value = getattr(severity, "value", severity)
        return str(value)

    @staticmethod
    def _validate_configuration(dataset: Dataset, image_root: Path, annotation_root: Path) -> None:
        root = Path(dataset.root_path)
        if not root.is_dir():
            raise DatasetConfigurationError(f"数据集根目录不可访问: {root}")
        if not image_root.is_dir():
            raise DatasetConfigurationError(f"图片目录不可访问: {image_root}")
        if not annotation_root.is_dir():
            raise DatasetConfigurationError(f"标注目录不可访问: {annotation_root}")
        if dataset.data_format.casefold() != "labelme":
            raise DatasetConfigurationError(f"当前扫描器不支持格式: {dataset.data_format}")

    @classmethod
    def _collect_files(
        cls,
        root: Path,
        include: Callable[[Path], bool],
        is_cancelled: CancelCallback,
        progress: ProgressCallback | None,
        issues: list[ScanIssue],
    ) -> list[Path]:
        paths: list[Path] = []

        def report_walk_error(error: OSError) -> None:
            issues.append(ScanIssue(
                "directory_read_failed",
                str(error),
                "error",
                getattr(error, "filename", None),
            ))

        for current_directory, directory_names, file_names in os.walk(
            root, topdown=True, onerror=report_walk_error, followlinks=False
        ):
            cls._raise_if_cancelled(is_cancelled)
            directory_names.sort(key=str.casefold)
            if progress is not None:
                progress(ScanProgress(str(current_directory), 0, 0, 0))
            current = Path(current_directory)
            for file_name in sorted(file_names, key=str.casefold):
                cls._raise_if_cancelled(is_cancelled)
                path = current / file_name
                if include(path):
                    paths.append(path)
        return paths

    def _associate(
        self,
        images: list[Path],
        annotations: list[Path],
        image_root: Path,
        annotation_root: Path,
        image_map: dict[str, list[Path]],
        annotation_map: dict[str, list[Path]],
    ) -> tuple[dict[Path, Path], set[Path], dict[Path, ParseResult], list[ScanIssue]]:
        paired: dict[Path, Path] = {}
        used_annotations: set[Path] = set()
        parse_cache: dict[Path, ParseResult] = {}
        issues: list[ScanIssue] = []

        for key, image_candidates in image_map.items():
            annotation_candidates = annotation_map.get(key, [])
            if len(image_candidates) == 1 and len(annotation_candidates) == 1:
                paired[image_candidates[0]] = annotation_candidates[0]
                used_annotations.add(annotation_candidates[0])

        images_by_absolute = self._group_images_by_key(
            images, lambda path: str(path.resolve()).casefold()
        )
        images_by_parent_name = self._group_images_by_key(
            images,
            lambda path: f"{str(path.parent.resolve()).casefold()}|{path.name.casefold()}",
        )

        for annotation_path in annotations:
            if annotation_path in used_annotations:
                continue
            parse_result = self._parser.parse(annotation_path)
            parse_cache[annotation_path] = parse_result
            document = parse_result.document
            if document is None or not document.image_path:
                continue
            declared = document.image_path.replace("\\", "/")
            declared_path = Path(declared)
            candidate_paths = []
            if declared_path.is_absolute():
                candidate_paths.append(declared_path)
            else:
                candidate_paths.extend((annotation_path.parent / declared_path, image_root / declared_path))

            matches: list[Path] = []
            for candidate in candidate_paths:
                matches.extend(images_by_absolute.get(str(candidate.resolve()).casefold(), []))
            parent_name_key = (
                f"{str(annotation_path.parent.resolve()).casefold()}|{declared_path.name.casefold()}"
            )
            matches.extend(images_by_parent_name.get(parent_name_key, []))
            unique_matches = list(dict.fromkeys(matches))
            available_matches = [path for path in unique_matches if path not in paired]
            if len(available_matches) == 1:
                paired[available_matches[0]] = annotation_path
                used_annotations.add(annotation_path)
            elif len(available_matches) > 1:
                issues.append(ScanIssue(
                    "ambiguous_image_path",
                    f"imagePath 匹配到多个图片，未自动关联: {document.image_path}",
                    "error",
                    str(annotation_path),
                ))
            elif unique_matches:
                issues.append(ScanIssue(
                    "duplicate_annotation_target",
                    f"多个 JSON 指向同一图片: {document.image_path}",
                    "error",
                    str(annotation_path),
                ))
            else:
                issues.append(ScanIssue(
                    "image_path_target_missing",
                    f"imagePath 未找到图片: {document.image_path}",
                    "warning",
                    str(annotation_path),
                ))

        return paired, set(annotations) - used_annotations, parse_cache, issues

    @staticmethod
    def _group_images_by_key(
        images: list[Path], key_factory: Callable[[Path], str]
    ) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for image in images:
            grouped[key_factory(image)].append(image)
        return dict(grouped)

    @classmethod
    def _group_by_relative_stem(cls, paths: list[Path], root: Path) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            grouped[cls._relative_stem(path, root)].append(path)
        return dict(grouped)

    @staticmethod
    def _relative_stem(path: Path, root: Path) -> str:
        return path.relative_to(root).with_suffix("").as_posix().casefold()

    @staticmethod
    def _duplicate_issues(
        images: dict[str, list[Path]], annotations: dict[str, list[Path]]
    ) -> list[ScanIssue]:
        issues: list[ScanIssue] = []
        for key, paths in images.items():
            if len(paths) > 1:
                issues.append(ScanIssue(
                    "duplicate_image_key", f"图片关联主键重复: {key}", "error",
                    "; ".join(str(path) for path in paths),
                ))
        for key, paths in annotations.items():
            if len(paths) > 1:
                issues.append(ScanIssue(
                    "duplicate_annotation_key", f"标注关联主键重复: {key}", "error",
                    "; ".join(str(path) for path in paths),
                ))
        return issues

    @staticmethod
    def _raise_if_cancelled(is_cancelled: CancelCallback) -> None:
        if is_cancelled():
            raise ScanCancelledError("扫描已取消")

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        path: Path,
        processed: int,
        total: int,
        global_issues: list[ScanIssue],
        records: list[ScanRecord],
    ) -> None:
        if callback is not None:
            issue_count = len(global_issues) + sum(len(record.issues) for record in records)
            callback(ScanProgress(str(path), processed, total, issue_count))

