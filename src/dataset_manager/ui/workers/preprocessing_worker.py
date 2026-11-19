from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from dataset_manager.application.preprocessing.models import ProcessingConfig, ProcessingProgress
from dataset_manager.application.preprocessing.preprocessing_service import PreprocessingService


class PreprocessingWorker(QThread):
    progress_changed = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(object)

    def __init__(
        self, service: PreprocessingService, config: ProcessingConfig, parent=None
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._config = config
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = self._service.execute(
                self._config,
                progress=self._progress,
                is_cancelled=self._cancel.is_set,
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        if result.cancelled:
            self.cancelled.emit(result)
        else:
            self.succeeded.emit(result)

    def _progress(self, progress: ProcessingProgress) -> None:
        self.progress_changed.emit(progress)
