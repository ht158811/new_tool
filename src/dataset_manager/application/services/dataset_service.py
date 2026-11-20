from __future__ import annotations

from dataclasses import dataclass

from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.application.ports.scan_cache_manager import ScanCacheManager
from dataset_manager.domain.models.project import Dataset, utc_now_iso


@dataclass(frozen=True, slots=True)
class DatasetDeletionResult:
    dataset: Dataset
    cache_cleanup_warning: str | None = None


class DatasetService:
    def __init__(
        self, repository: ProjectRepository, cache_manager: ScanCacheManager
    ) -> None:
        self._repository = repository
        self._cache_manager = cache_manager

    def delete_dataset_management_info(
        self, project_id: str, dataset_id: str
    ) -> DatasetDeletionResult:
        project = self._repository.get(project_id)
        dataset = project.find_dataset(dataset_id)
        project.datasets = [item for item in project.datasets if item.id != dataset_id]
        project.updated_at = utc_now_iso()
        self._repository.save(project)

        warning: str | None = None
        if dataset.active_scan_id:
            try:
                self._cache_manager.delete(dataset.id, dataset.active_scan_id)
            except OSError as error:
                warning = f"扫描缓存清理失败，可稍后手动清理: {error}"
        return DatasetDeletionResult(dataset, warning)

    def reorder_datasets(self, project_id: str, ordered_dataset_ids: list[str]) -> None:
        project = self._repository.get(project_id)
        existing_ids = [dataset.id for dataset in project.datasets]
        if (
            len(ordered_dataset_ids) != len(existing_ids)
            or len(set(ordered_dataset_ids)) != len(ordered_dataset_ids)
            or set(ordered_dataset_ids) != set(existing_ids)
        ):
            raise ValueError("数据集排序内容与项目中的数据集不一致")
        by_id = {dataset.id: dataset for dataset in project.datasets}
        project.datasets = [by_id[dataset_id] for dataset_id in ordered_dataset_ids]
        project.updated_at = utc_now_iso()
        self._repository.save(project)
