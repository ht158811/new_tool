from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from dataset_manager.domain.models.project import Project


class HomePage(QWidget):
    create_requested = Signal()
    delete_requested = Signal(str)
    add_dataset_requested = Signal(str)
    refresh_dataset_requested = Signal(str, str)

    def __init__(self, project_loader: Callable[[], list[Project]]) -> None:
        super().__init__()
        self._project_loader = project_loader
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("项目首页")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel("管理本地视觉数据集，不修改原始文件")
        subtitle.setStyleSheet("color: #7b8491;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        create_button = QPushButton("＋ 创建项目")
        create_button.setObjectName("primary")
        create_button.clicked.connect(self.create_requested)
        header.addWidget(create_button)
        root.addLayout(header)

        cards = QHBoxLayout()
        self._project_count = self._card("项目", "0")
        self._dataset_count = self._card("数据集", "0")
        self._image_count = self._card("图片", "—")
        cards.addWidget(self._project_count)
        cards.addWidget(self._dataset_count)
        cards.addWidget(self._image_count)
        root.addLayout(cards)

        section = QLabel("项目列表")
        section.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(section)
        browser = QHBoxLayout()
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentItemChanged.connect(self._show_project_datasets)
        self._dataset_list = QListWidget()
        self._dataset_list.setAlternatingRowColors(True)
        browser.addWidget(self._list, 1)
        browser.addWidget(self._dataset_list, 2)
        root.addLayout(browser, 1)

        action_row = QHBoxLayout()
        add_dataset_button = QPushButton("＋ 向选中项目添加数据集")
        add_dataset_button.clicked.connect(self._emit_add_dataset)
        refresh_button = QPushButton("刷新选中数据集")
        refresh_button.clicked.connect(self._emit_refresh_dataset)
        delete_button = QPushButton("删除选中项目的管理信息")
        delete_button.clicked.connect(self._emit_delete)
        action_row.addStretch()
        action_row.addWidget(add_dataset_button)
        action_row.addWidget(refresh_button)
        action_row.addWidget(delete_button)
        root.addLayout(action_row)
        self.refresh()

    @staticmethod
    def _card(caption: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(100)
        layout = QVBoxLayout(card)
        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        layout.addWidget(value_label)
        layout.addWidget(QLabel(caption))
        return card

    def refresh(self) -> None:
        selected_project_id = self._selected_project_id()
        projects = self._project_loader()
        self._list.clear()
        self._dataset_list.clear()
        dataset_count = 0
        image_count = 0
        for project in projects:
            dataset_count += len(project.datasets)
            project_images = sum(dataset.scan_summary.image_count for dataset in project.datasets)
            image_count += project_images
            item = QListWidgetItem(
                f"{project.name}    ·    {len(project.datasets)} 个数据集    ·    {project_images} 张图片"
            )
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, project)
            self._list.addItem(item)
            if project.id == selected_project_id:
                self._list.setCurrentItem(item)
        self._set_card_value(self._project_count, str(len(projects)))
        self._set_card_value(self._dataset_count, str(dataset_count))
        self._set_card_value(self._image_count, str(image_count))
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)

    @staticmethod
    def _set_card_value(card: QFrame, value: str) -> None:
        label = card.findChild(QLabel, "value")
        if label is not None:
            label.setText(value)

    def _emit_delete(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self.delete_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))

    def _emit_add_dataset(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self.add_dataset_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))

    def _emit_refresh_dataset(self) -> None:
        project_id = self._selected_project_id()
        item = self._dataset_list.currentItem()
        if project_id and item is not None:
            self.refresh_dataset_requested.emit(
                project_id, str(item.data(Qt.ItemDataRole.UserRole))
            )

    def _show_project_datasets(self, current: QListWidgetItem | None) -> None:
        self._dataset_list.clear()
        if current is None:
            return
        project = current.data(Qt.ItemDataRole.UserRole + 1)
        for dataset in project.datasets:
            summary = dataset.scan_summary
            item = QListWidgetItem(
                f"{dataset.name}    ·    {summary.image_count} 张图片    ·    "
                f"{summary.category_count} 个类别    ·    {summary.issue_count} 个检查项\n"
                f"{Path(dataset.root_path)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, dataset.id)
            self._dataset_list.addItem(item)
        if self._dataset_list.count():
            self._dataset_list.setCurrentRow(0)

    def _selected_project_id(self) -> str | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

