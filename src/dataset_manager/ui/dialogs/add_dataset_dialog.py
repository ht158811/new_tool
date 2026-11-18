from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from dataset_manager.domain.models.project import Dataset, TaskType


class AddDatasetDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("添加 LabelMe 数据集")
        self.setMinimumWidth(620)
        root_layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(12)

        self._name = QLineEdit()
        self._root = self._directory_field("选择数据集根目录")
        self._task_type = QComboBox()
        self._task_type.addItem("目标检测", TaskType.OBJECT_DETECTION)
        self._task_type.addItem("图像分类", TaskType.IMAGE_CLASSIFICATION)
        self._task_type.addItem("语义分割", TaskType.SEMANTIC_SEGMENTATION)

        self._name.setPlaceholderText("可选，默认使用根文件夹名称")
        form.addRow("数据集名称", self._name)
        form.addRow("根目录（递归扫描）", self._root)
        form.addRow("任务类型", self._task_type)
        root_layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始扫描并添加")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

    def build_dataset(self) -> Dataset:
        root_path = Path(self._line_edit(self._root).text()).resolve()
        return Dataset.create(
            name=self._name.text(),
            root_path=str(root_path),
            image_directory=str(root_path),
            annotation_directory=str(root_path),
            task_type=self._task_type.currentData(),
        )

    def _directory_field(self, caption: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(caption)
        line_edit.setProperty("directoryInput", True)
        button = QPushButton("浏览…")
        button.clicked.connect(lambda: self._browse(line_edit, caption))
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return container

    def _browse(self, line_edit: QLineEdit, caption: str) -> None:
        initial = line_edit.text() or self._line_edit(self._root).text()
        selected = QFileDialog.getExistingDirectory(self, caption, initial)
        if not selected:
            return
        root = Path(selected).resolve()
        line_edit.setText(str(root))
        if not self._name.text().strip():
            self._name.setText(root.name)

    def _validate_and_accept(self) -> None:
        value = self._line_edit(self._root).text().strip()
        if not value or not Path(value).is_dir():
            QMessageBox.warning(self, "目录不可访问", f"根目录不存在或不可访问：\n{value}")
            return
        if not self._name.text().strip():
            self._name.setText(Path(value).resolve().name)
        self.accept()

    @staticmethod
    def _line_edit(container: QWidget) -> QLineEdit:
        line_edit = container.findChild(QLineEdit)
        if line_edit is None:
            raise RuntimeError("目录输入组件损坏")
        return line_edit

