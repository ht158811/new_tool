from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from dataset_manager.application.dto.scan_result import ScanProgress, ScanResult
from dataset_manager.domain.models.project import Dataset

ProgressCallback = Callable[[ScanProgress], None]
CancelCallback = Callable[[], bool]


class DatasetScanner(ABC):
    @abstractmethod
    def scan(
        self,
        dataset: Dataset,
        progress: ProgressCallback | None = None,
        is_cancelled: CancelCallback | None = None,
    ) -> ScanResult:
        """Perform a complete read-only scan or raise ScanCancelledError."""

