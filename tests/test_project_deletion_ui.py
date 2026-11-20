from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QMessageBox

from dataset_manager.application.services.annotation_service import AnnotationService
from dataset_manager.application.services.dataset_service import DatasetService
from dataset_manager.application.services.image_service import ImageService
from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.application.services.scan_service import ScanService
from dataset_manager.application.preprocessing.preprocessing_service import PreprocessingService
from dataset_manager.application.model_testing.service import ModelTestService
from dataset_manager.infrastructure.parsers.labelme_parser import LabelMeParser
from dataset_manager.infrastructure.scanning.labelme_scanner import LabelMeDatasetScanner
from dataset_manager.infrastructure.scanning.scan_cache import ScanCacheStore
from dataset_manager.infrastructure.storage.labelme_annotation_store import LabelMeAnnotationStore
from dataset_manager.infrastructure.storage.project_repository import JsonProjectRepository
from dataset_manager.ui.main_window import MainWindow


class ProjectDeletionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_confirmed_delete_removes_metadata_and_refreshes_ui(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = JsonProjectRepository(root / "Projects")
            project_service = ProjectService(repository)
            project = project_service.create_project("Delete me")
            parser = LabelMeParser()
            cache = ScanCacheStore(root / "Cache")
            window = MainWindow(
                project_service,
                DatasetService(repository, cache),
                ScanService(repository, LabelMeDatasetScanner(parser), cache),
                ImageService(repository, cache),
                AnnotationService(parser, LabelMeAnnotationStore()),
                PreprocessingService(repository, cache),
                ModelTestService(repository, cache, object()),
            )
            operation_list = window._preprocessing._operation_list
            self.assertEqual(
                operation_list.dragDropMode(),
                QAbstractItemView.DragDropMode.InternalMove,
            )
            rename_item = next(
                operation_list.item(row)
                for row in range(operation_list.count())
                if operation_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "label_rename"
            )
            operation_list.setCurrentItem(rename_item)
            self.assertFalse(window._preprocessing._output.isEnabled())
            moved = operation_list.takeItem(0)
            operation_list.insertItem(operation_list.count(), moved)
            self.assertIs(operation_list.item(operation_list.count() - 1), moved)
            self.assertTrue(window._home._delete_button.isEnabled())

            with (
                patch.object(
                    QMessageBox,
                    "question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
                patch.object(QMessageBox, "information") as information,
            ):
                window._delete_project(project.id)

            self.assertEqual(repository.list(), [])
            self.assertEqual(window._home._list.count(), 0)
            self.assertFalse(window._home._delete_button.isEnabled())
            information.assert_called_once()
            window.close()


if __name__ == "__main__":
    unittest.main()
