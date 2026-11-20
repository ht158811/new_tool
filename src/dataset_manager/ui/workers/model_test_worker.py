from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from dataset_manager.application.model_testing.models import ModelTestConfig
from dataset_manager.application.model_testing.service import ModelTestService


class ModelTestWorker(QThread):
    progress_changed = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, service: ModelTestService, config: ModelTestConfig, parent=None) -> None:
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
                progress=self.progress_changed.emit,
                is_cancelled=self._cancel.is_set,
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.succeeded.emit(result)
