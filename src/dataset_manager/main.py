from __future__ import annotations

import sys

from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.application.services.scan_service import ScanService
from dataset_manager.application.services.image_service import ImageService
from dataset_manager.application.services.annotation_service import AnnotationService
from dataset_manager.application.services.dataset_service import DatasetService
from dataset_manager.application.preprocessing.preprocessing_service import PreprocessingService
from dataset_manager.application.model_testing.service import ModelTestService
from dataset_manager.infrastructure.parsers.labelme_parser import LabelMeParser
from dataset_manager.infrastructure.paths import AppPaths
from dataset_manager.infrastructure.scanning.labelme_scanner import LabelMeDatasetScanner
from dataset_manager.infrastructure.scanning.scan_cache import ScanCacheStore
from dataset_manager.infrastructure.images import pillow_image_size
from dataset_manager.infrastructure.storage.project_repository import JsonProjectRepository
from dataset_manager.infrastructure.storage.labelme_annotation_store import LabelMeAnnotationStore
from dataset_manager.infrastructure.model_testing.ultralytics_runner import UltralyticsRunner


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        raise SystemExit("缺少 PySide6。请先运行: python -m pip install -e .") from error

    from dataset_manager.ui.main_window import MainWindow

    paths = AppPaths.discover()
    paths.ensure_directories()
    repository = JsonProjectRepository(paths.projects)
    service = ProjectService(repository)
    parser = LabelMeParser()
    scanner = LabelMeDatasetScanner(parser, pillow_image_size)
    cache_store = ScanCacheStore(paths.cache)
    scan_service = ScanService(repository, scanner, cache_store)
    image_service = ImageService(repository, cache_store)
    annotation_service = AnnotationService(parser, LabelMeAnnotationStore())
    app = QApplication(sys.argv)
    app.setApplicationName("Dataset Manager")
    dataset_service = DatasetService(repository, cache_store)
    preprocessing_service = PreprocessingService(
        repository, cache_store, source_scanner=scanner
    )
    model_test_service = ModelTestService(repository, cache_store, UltralyticsRunner())
    window = MainWindow(
        service, dataset_service, scan_service, image_service, annotation_service,
        preprocessing_service, model_test_service,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

