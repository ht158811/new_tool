from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from dataset_manager.application.dto.scan_result import ScanProgress, ScanResult
from dataset_manager.application.errors import ScanCancelledError
from dataset_manager.application.services.scan_service import ScanService
from dataset_manager.domain.models.project import Dataset


class ScanWorker(QThread):
    progress_changed = Signal(int, int, int, str)
    scan_succeeded = Signal(object)
    scan_failed = Signal(str)
    scan_cancelled = Signal()

    def __init__(
        self,
        scan_service: ScanService,
        project_id: str,
        dataset: Dataset | None = None,
        dataset_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._scan_service = scan_service
        self._project_id = project_id
        self._dataset = dataset
        self._dataset_id = dataset_id
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if self._dataset is not None:
                result = self._scan_service.add_dataset(
                    self._project_id,
                    self._dataset,
                    progress=self._on_progress,
                    is_cancelled=self._cancel_event.is_set,
                )
            elif self._dataset_id is not None:
                result = self._scan_service.refresh_dataset(
                    self._project_id,
                    self._dataset_id,
                    progress=self._on_progress,
                    is_cancelled=self._cancel_event.is_set,
                )
            else:
                raise ValueError("扫描任务缺少数据集参数")
        except ScanCancelledError:
            self.scan_cancelled.emit()
        except Exception as error:
            self.scan_failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.scan_succeeded.emit(result)

    def _on_progress(self, progress: ScanProgress) -> None:
        self.progress_changed.emit(
            progress.processed,
            progress.total,
            progress.issue_count,
            progress.current_path,
        )

