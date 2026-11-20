from __future__ import annotations

import threading
from collections.abc import Callable

from dataset_manager.application.dto.scan_result import ScanProgress, ScanResult
from dataset_manager.application.errors import ScanAlreadyRunningError
from dataset_manager.application.ports.dataset_scanner import DatasetScanner
from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.domain.models.project import Dataset, ScanSummary, utc_now_iso
from dataset_manager.infrastructure.scanning.scan_cache import ScanCacheStore


class ScanService:
    def __init__(
        self,
        repository: ProjectRepository,
        scanner: DatasetScanner,
        cache_store: ScanCacheStore,
    ) -> None:
        self._repository = repository
        self._scanner = scanner
        self._cache_store = cache_store
        self._guard = threading.Lock()
        self._running: set[str] = set()

    def add_dataset(
        self,
        project_id: str,
        dataset: Dataset,
        progress: Callable[[ScanProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        insert_after_dataset_id: str | None = None,
    ) -> ScanResult:
        self._repository.get(project_id)
        return self._execute(
            dataset,
            lambda result: self._commit_new_dataset(
                project_id, dataset, result, insert_after_dataset_id
            ),
            progress,
            is_cancelled,
        )

    def refresh_dataset(
        self,
        project_id: str,
        dataset_id: str,
        progress: Callable[[ScanProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ScanResult:
        project = self._repository.get(project_id)
        dataset = project.find_dataset(dataset_id)
        return self._execute(
            dataset,
            lambda result: self._commit_refresh(project_id, dataset_id, result),
            progress,
            is_cancelled,
        )

    def is_running(self, dataset_id: str) -> bool:
        with self._guard:
            return dataset_id in self._running

    def _execute(
        self,
        dataset: Dataset,
        commit: Callable[[ScanResult], None],
        progress: Callable[[ScanProgress], None] | None,
        is_cancelled: Callable[[], bool] | None,
    ) -> ScanResult:
        self._acquire(dataset.id)
        try:
            result = self._scanner.scan(dataset, progress=progress, is_cancelled=is_cancelled)
            self._cache_store.commit(result)
            try:
                commit(result)
            except Exception:
                self._cache_store.delete(result.dataset_id, result.scan_id)
                raise
            return result
        finally:
            self._release(dataset.id)

    def _commit_new_dataset(
        self,
        project_id: str,
        dataset: Dataset,
        result: ScanResult,
        insert_after_dataset_id: str | None = None,
    ) -> None:
        project = self._repository.get(project_id)
        self._apply_result(dataset, result)
        insert_at = len(project.datasets)
        if insert_after_dataset_id:
            for index, existing in enumerate(project.datasets):
                if existing.id == insert_after_dataset_id:
                    insert_at = index + 1
                    break
        project.datasets.insert(insert_at, dataset)
        project.updated_at = utc_now_iso()
        self._repository.save(project)

    def _commit_refresh(self, project_id: str, dataset_id: str, result: ScanResult) -> None:
        project = self._repository.get(project_id)
        dataset = project.find_dataset(dataset_id)
        old_scan_id = dataset.active_scan_id
        self._apply_result(dataset, result)
        project.updated_at = utc_now_iso()
        self._repository.save(project)
        if old_scan_id and old_scan_id != result.scan_id:
            self._cache_store.delete(dataset_id, old_scan_id)

    @staticmethod
    def _apply_result(dataset: Dataset, result: ScanResult) -> None:
        dataset.active_scan_id = result.scan_id
        dataset.last_scanned_at = result.completed_at
        dataset.accessible = True
        dataset.scan_summary = ScanSummary(
            image_count=result.image_count,
            annotation_count=result.annotation_count,
            paired_count=result.paired_count,
            valid_annotation_image_count=result.valid_annotation_image_count,
            issue_count=result.issue_count,
            category_count=len(result.categories),
            categories=result.categories,
        )

    def _acquire(self, dataset_id: str) -> None:
        with self._guard:
            if dataset_id in self._running:
                raise ScanAlreadyRunningError("同一数据集正在扫描，不能重复提交")
            self._running.add(dataset_id)

    def _release(self, dataset_id: str) -> None:
        with self._guard:
            self._running.discard(dataset_id)

