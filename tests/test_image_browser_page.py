from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFrame, QScrollArea, QSpinBox, QSplitter, QTextEdit

from dataset_manager.application.dto.image_record import BrowserImage
from dataset_manager.domain.models.project import Dataset, Project, ScanSummary, TaskType
from dataset_manager.ui.pages.image_browser_page import ImageBrowserPage


class _ProjectService:
    def __init__(self, project: Project) -> None:
        self.project = project

    def list_projects(self) -> list[Project]:
        return [self.project]

    def get_project(self, project_id: str) -> Project:
        assert project_id == self.project.id
        return self.project


class _ImageService:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.list_calls = 0

    def categories(self, project_id: str, dataset_id: str) -> list[str]:
        return ["scratch", "dent"]

    def list_images(self, project_id: str, dataset_id: str, query) -> list[BrowserImage]:
        self.list_calls += 1
        return [BrowserImage(self.image_path, None, 32, 24, (query.category,), 0, False)]


class _AnnotationService:
    def read(self, annotation_path: str | None, image_path: str):
        return None


class ImageBrowserPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_initially_empty_until_user_selects_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            Image.new("RGB", (32, 24), "white").save(image_path)
            project = Project.create("P")
            dataset = Dataset.create(
                "D", directory, directory, directory, TaskType.OBJECT_DETECTION
            )
            dataset.scan_summary = ScanSummary(categories=["scratch", "dent"], category_count=2)
            project.datasets.append(dataset)
            image_service = _ImageService(str(image_path))
            page = ImageBrowserPage(
                _ProjectService(project),
                image_service,
                _AnnotationService(),
            )

            self.assertEqual(page._list.count(), 0)
            self.assertEqual(page._result_label.text(), "请选择类别")
            self.assertEqual(image_service.list_calls, 0)
            self.assertEqual(page._category.currentText(), "请选择类别")
            self.assertIsNotNone(page.findChild(QScrollArea))
            self.assertIsInstance(page._details, QTextEdit)
            self.assertEqual(page._details.lineWrapMode(), QTextEdit.LineWrapMode.NoWrap)
            self.assertEqual(page.findChildren(QSpinBox), [])
            source_splitter = page.findChild(QSplitter, "sourceSplitter")
            content_splitter = page.findChild(QSplitter, "browserContentSplitter")
            left_splitter = page.findChild(QSplitter, "fileListInfoSplitter")
            info_panel = page.findChild(QFrame, "fileInfoPanel")
            preview_panel = page.findChild(QFrame, "card")
            self.assertIsNotNone(source_splitter)
            self.assertEqual(source_splitter.count(), 2)
            self.assertIsNotNone(content_splitter)
            self.assertIsNotNone(left_splitter)
            self.assertEqual(left_splitter.orientation(), Qt.Orientation.Vertical)
            self.assertEqual(left_splitter.count(), 2)
            self.assertIs(left_splitter.widget(0), page._list)
            self.assertIs(left_splitter.widget(1), info_panel)
            self.assertIsNotNone(info_panel)
            self.assertTrue(info_panel.isAncestorOf(page._details))
            self.assertFalse(preview_panel.isAncestorOf(page._details))

            page._category.setCurrentIndex(1)

            self.assertEqual(image_service.list_calls, 1)
            self.assertEqual(page._list.count(), 1)
            self.assertEqual(page._list.item(0).text(), "sample.jpg")
            self.assertTrue(page._list.item(0).icon().isNull())
            page.close()


if __name__ == "__main__":
    unittest.main()
