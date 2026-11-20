from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_manager.application.dto.image_record import BrowserImage, ImageFilter
from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.application.ports.scan_cache_reader import ScanCacheReader


class ImageService:
    def __init__(self, repository: ProjectRepository, cache_reader: ScanCacheReader) -> None:
        self._repository = repository
        self._cache_reader = cache_reader

    def list_images(
        self, project_id: str, dataset_id: str, query: ImageFilter | None = None
    ) -> list[BrowserImage]:
        dataset = self._repository.get(project_id).find_dataset(dataset_id)
        if dataset.active_scan_id is None:
            return []
        cache = self._cache_reader.read(dataset.id, dataset.active_scan_id)
        records = cache.get("records", [])
        if not isinstance(records, list):
            raise ValueError("扫描缓存 records 字段损坏")
        images = [self._from_record(record) for record in records if self._is_image_record(record)]
        selected = [image for image in images if self._matches(image, query or ImageFilter())]
        return sorted(selected, key=lambda image: Path(image.image_path).name.casefold())

    def categories(self, project_id: str, dataset_id: str) -> list[str]:
        dataset = self._repository.get(project_id).find_dataset(dataset_id)
        return list(dataset.scan_summary.categories)

    @staticmethod
    def _is_image_record(record: Any) -> bool:
        return isinstance(record, dict) and isinstance(record.get("image_path"), str)

    @staticmethod
    def _from_record(record: dict[str, Any]) -> BrowserImage:
        categories = record.get("categories", [])
        issues = record.get("issues", [])
        annotation_valid = record.get("annotation_valid")
        if not isinstance(annotation_valid, bool):
            annotation_valid = ImageService._infer_legacy_annotation_valid(record, issues)
        return BrowserImage(
            image_path=record["image_path"],
            annotation_path=record.get("annotation_path") if isinstance(record.get("annotation_path"), str) else None,
            width=record.get("width") if isinstance(record.get("width"), int) else None,
            height=record.get("height") if isinstance(record.get("height"), int) else None,
            categories=tuple(str(item) for item in categories if isinstance(item, str)),
            issue_count=len(issues) if isinstance(issues, list) else 0,
            annotation_valid=annotation_valid,
        )

    @staticmethod
    def _infer_legacy_annotation_valid(record: dict[str, Any], issues: Any) -> bool:
        """Support scan caches created before annotation_valid was stored."""
        if not isinstance(record.get("annotation_path"), str) or not isinstance(issues, list):
            return False
        return not any(
            isinstance(issue, dict) and issue.get("severity") == "error"
            for issue in issues
        )

    @staticmethod
    def _matches(image: BrowserImage, query: ImageFilter) -> bool:
        if query.filename and query.filename.casefold() not in Path(image.image_path).name.casefold():
            return False
        if query.category and query.category not in image.categories:
            return False
        if query.valid_annotations_only and not image.annotation_valid:
            return False
        if query.min_width and (image.width is None or image.width < query.min_width):
            return False
        if query.max_width and (image.width is None or image.width > query.max_width):
            return False
        if query.min_height and (image.height is None or image.height < query.min_height):
            return False
        if query.max_height and (image.height is None or image.height > query.max_height):
            return False
        return True

