from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFrame, QInputDialog, QLabel, QMessageBox,
    QScrollArea, QSpinBox, QSplitter, QTextEdit,
)

from dataset_manager.application.dto.image_record import BrowserImage
from dataset_manager.application.services.annotation_service import AnnotationService
from dataset_manager.domain.models.project import Dataset, Project, ScanSummary, TaskType
from dataset_manager.infrastructure.parsers.labelme_parser import LabelMeParser
from dataset_manager.infrastructure.storage.labelme_annotation_store import LabelMeAnnotationStore
from dataset_manager.ui.pages.image_browser_page import ImageBrowserPage
from dataset_manager.ui.pages.home_page import HomePage


class _ProjectService:
    def __init__(self, project: Project) -> None:
        self.project = project

    def list_projects(self) -> list[Project]:
        return [self.project]

    def get_project(self, project_id: str) -> Project:
        assert project_id == self.project.id
        return self.project


class _ImageService:
    def __init__(self, image_path: str | list[str], annotation_path: str | None = None) -> None:
        self.image_paths = [image_path] if isinstance(image_path, str) else image_path
        self.annotation_path = annotation_path
        self.list_calls = 0
        self.last_category: str | None = None
        self.last_valid_annotations_only = False

    def categories(self, project_id: str, dataset_id: str) -> list[str]:
        return ["scratch", "dent"]

    def list_images(self, project_id: str, dataset_id: str, query) -> list[BrowserImage]:
        self.list_calls += 1
        self.last_category = query.category
        self.last_valid_annotations_only = query.valid_annotations_only
        return [
            BrowserImage(
                path, self.annotation_path, 32, 24, ("scratch",), 0,
                self.annotation_path is not None,
            )
            for path in self.image_paths
        ]


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
            self.assertEqual(page._category.itemText(1), "全部有效 JSON")
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
            self.assertEqual(image_service.last_category, "")
            self.assertTrue(image_service.last_valid_annotations_only)
            self.assertEqual(page._list.count(), 1)
            self.assertEqual(page._list.item(0).text(), "sample.jpg")
            self.assertTrue(page._list.item(0).icon().isNull())

            page._category.setCurrentIndex(2)
            self.assertEqual(image_service.last_category, "scratch")
            self.assertFalse(image_service.last_valid_annotations_only)
            page.close()

    def test_previous_and_next_buttons_browse_filtered_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                path = Path(directory) / name
                Image.new("RGB", (32, 24), "white").save(path)
                paths.append(str(path))
            project = Project.create("P")
            dataset = Dataset.create(
                "D", directory, directory, directory, TaskType.OBJECT_DETECTION
            )
            dataset.scan_summary = ScanSummary(categories=["scratch"], category_count=1)
            project.datasets.append(dataset)
            page = ImageBrowserPage(
                _ProjectService(project), _ImageService(paths), _AnnotationService()
            )

            page._category.setCurrentIndex(1)
            self.assertEqual(page._list.currentRow(), 0)
            self.assertEqual(page._position_label.text(), "当前图片：1 / 3")
            self.assertFalse(page._previous_button.isEnabled())
            self.assertTrue(page._next_button.isEnabled())

            page._next_button.click()
            self.assertEqual(page._list.currentRow(), 1)
            self.assertEqual(page._position_label.text(), "当前图片：2 / 3")
            page._next_shortcut.activated.emit()
            self.assertEqual(page._list.currentRow(), 2)
            self.assertFalse(page._next_button.isEnabled())
            page._previous_shortcut.activated.emit()
            self.assertEqual(page._list.currentRow(), 1)

        with patch.object(QApplication, "focusWidget", return_value=page._filename):
            page._next_shortcut.activated.emit()
            self.assertEqual(page._list.currentRow(), 1)
            page.show()
            page._filename.setFocus()
            QApplication.processEvents()
            self.assertFalse(page._next_shortcut.isEnabled())
            QTest.keyClicks(page._filename, "adqer")
            self.assertEqual(page._filename.text(), "adqer")
            self.assertEqual(page._list.currentRow(), 1)
            page.close()

    def test_delete_shape_and_json_actions_update_files_and_ui(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            annotation_path = Path(directory) / "sample.json"
            Image.new("RGB", (32, 24), "white").save(image_path)
            annotation_path.write_text(json.dumps({
                "shapes": [
                    {
                        "label": "box", "points": [[1, 1], [10, 10]],
                        "shape_type": "rectangle", "flags": {},
                    },
                    {
                        "label": "other", "points": [[12, 2], [20, 12]],
                        "shape_type": "rectangle", "flags": {},
                    },
                ],
                "imagePath": image_path.name,
                "imageData": None,
                "imageWidth": 32,
                "imageHeight": 24,
            }), encoding="utf-8")
            project = Project.create("P")
            dataset = Dataset.create(
                "D", directory, directory, directory, TaskType.OBJECT_DETECTION
            )
            dataset.scan_summary = ScanSummary(categories=["box"], category_count=1)
            project.datasets.append(dataset)
            page = ImageBrowserPage(
                _ProjectService(project),
                _ImageService(str(image_path), str(annotation_path)),
                AnnotationService(LabelMeParser(), LabelMeAnnotationStore()),
            )
            page._category.setCurrentIndex(1)
            self.assertTrue(page._add_rectangle_button.isEnabled())
            self.assertFalse(page._delete_shape_button.isEnabled())
            self.assertTrue(page._delete_json_button.isEnabled())

            page._move_rectangle(0, [[2.0, 2.0], [11.0, 11.0]])
            moved = json.loads(annotation_path.read_text(encoding="utf-8"))["shapes"]
            self.assertEqual(moved[0]["points"], [[2.0, 2.0], [11.0, 11.0]])

            with patch.object(QInputDialog, "getText", return_value=("added", True)):
                page._add_rectangle([[3.0, 12.0], [15.0, 22.0]])
            added = json.loads(annotation_path.read_text(encoding="utf-8"))["shapes"]
            self.assertEqual(added[-1]["label"], "added")
            self.assertEqual(added[-1]["shape_type"], "rectangle")

            page._select_rectangle(0)
            self.assertTrue(page._delete_shape_button.isEnabled())
            with patch.object(
                QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
            ):
                page._delete_shape_shortcut.activated.emit()
            remaining = json.loads(annotation_path.read_text(encoding="utf-8"))["shapes"]
            self.assertEqual([shape["label"] for shape in remaining], ["other", "added"])
            self.assertTrue(image_path.exists())

            page._show_boxes.setChecked(False)
            self.assertFalse(page._canvas._show_shape_boxes)
            page._show_labels.setChecked(False)
            self.assertFalse(page._canvas._show_labels)
            page._show_boxes.setChecked(True)
            page._add_rectangle_shortcut.activated.emit()
            self.assertTrue(page._add_rectangle_button.isChecked())
            page._add_rectangle_shortcut.activated.emit()
            self.assertFalse(page._add_rectangle_button.isChecked())

            with patch.object(
                QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
            ):
                page._delete_json_shortcut.activated.emit()
            self.assertFalse(annotation_path.exists())
            self.assertTrue(image_path.exists())
            self.assertFalse(page._delete_shape_button.isEnabled())
            self.assertFalse(page._delete_json_button.isEnabled())
            self.assertIsNone(page._list.currentItem().data(Qt.ItemDataRole.UserRole).annotation_path)
            page.close()

    def test_home_page_shows_valid_annotations_and_anomaly_wording(self) -> None:
        project = Project.create("P")
        dataset = Dataset.create(
            "D", "D:\\data", "D:\\data", "D:\\data", TaskType.OBJECT_DETECTION
        )
        dataset.scan_summary = ScanSummary(
            image_count=4,
            annotation_count=1,
            paired_count=1,
            valid_annotation_image_count=1,
            issue_count=3,
            category_count=1,
            categories=["scratch"],
        )
        project.datasets.append(dataset)
        page = HomePage(lambda: [project])

        value = page._valid_annotation_count.findChild(QLabel, "value")
        self.assertEqual(value.text(), "1")
        dataset_text = page._dataset_list.item(0).text()
        self.assertIn("1 张有效标注", dataset_text)
        self.assertIn("3 个异常项", dataset_text)
        self.assertTrue(page._delete_project_button.isEnabled())
        self.assertTrue(page._delete_dataset_button.isEnabled())
        project_delete = QSignalSpy(page.delete_requested)
        dataset_delete = QSignalSpy(page.delete_dataset_requested)
        page._delete_project_button.click()
        page._delete_dataset_button.click()
        self.assertEqual(project_delete.at(0), [project.id])
        self.assertEqual(dataset_delete.at(0), [project.id, dataset.id])
        page.close()

    def test_home_dataset_list_supports_internal_drag_order(self) -> None:
        project = Project.create("P")
        datasets = [
            Dataset.create(name, f"D:\\{name}", f"D:\\{name}", f"D:\\{name}",
                           TaskType.OBJECT_DETECTION)
            for name in ("A", "B", "C")
        ]
        project.datasets.extend(datasets)
        page = HomePage(lambda: [project])
        self.assertEqual(
            page._dataset_list.dragDropMode(),
            QAbstractItemView.DragDropMode.InternalMove,
        )
        moved = page._dataset_list.takeItem(0)
        page._dataset_list.insertItem(2, moved)
        order_spy = QSignalSpy(page.dataset_order_changed)
        page._persist_dataset_order()
        self.assertEqual(
            order_spy.at(0),
            [project.id, [datasets[1].id, datasets[2].id, datasets[0].id]],
        )
        page.close()


if __name__ == "__main__":
    unittest.main()
