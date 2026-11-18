from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from dataset_manager.ui.dialogs.add_dataset_dialog import AddDatasetDialog
from dataset_manager.domain.models.project import TaskType


class AddDatasetDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_single_root_is_used_for_images_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dialog = AddDatasetDialog()
            dialog._name.setText("Mixed files")
            dialog._line_edit(dialog._root).setText(str(root))

            dataset = dialog.build_dataset()

            self.assertEqual(dataset.root_path, str(root))
            self.assertEqual(dataset.image_directory, str(root))
            self.assertEqual(dataset.annotation_directory, str(root))
            self.assertIs(dataset.task_type, TaskType.OBJECT_DETECTION)
            dialog.close()

    def test_dataset_name_defaults_to_root_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dialog = AddDatasetDialog()
            dialog._line_edit(dialog._root).setText(str(root))

            dialog._validate_and_accept()
            dataset = dialog.build_dataset()

            self.assertEqual(dataset.name, root.name)
            dialog.close()


if __name__ == "__main__":
    unittest.main()
