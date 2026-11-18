from __future__ import annotations

import sys

from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.infrastructure.paths import AppPaths
from dataset_manager.infrastructure.storage.project_repository import JsonProjectRepository


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        raise SystemExit("缺少 PySide6。请先运行: python -m pip install -e .") from error

    from dataset_manager.ui.main_window import MainWindow

    paths = AppPaths.discover()
    paths.ensure_directories()
    service = ProjectService(JsonProjectRepository(paths.projects))
    app = QApplication(sys.argv)
    app.setApplicationName("Dataset Manager")
    window = MainWindow(service)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

