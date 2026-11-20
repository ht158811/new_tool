from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from dataset_manager.domain.models.project import Project


class HomePage(QWidget):
    create_requested = Signal()
    delete_requested = Signal(str)
    delete_dataset_requested = Signal(str, str)
    add_dataset_requested = Signal(str)
    refresh_dataset_requested = Signal(str, str)
    dataset_selection_changed = Signal(str, str)
    dataset_order_changed = Signal(str, object)

    def __init__(self, project_loader: Callable[[], list[Project]]) -> None:
        super().__init__()
        self._project_loader = project_loader
        self._preferred_dataset_id: str | None = None
        self._populating_datasets = False
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
        self._valid_annotation_count = self._card("有效标注图片", "0")
        cards.addWidget(self._project_count)
        cards.addWidget(self._dataset_count)
        cards.addWidget(self._image_count)
        cards.addWidget(self._valid_annotation_count)
        root.addLayout(cards)

        section = QLabel("项目列表")
        section.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(section)
        browser = QHBoxLayout()
        project_column = QVBoxLayout()
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentItemChanged.connect(self._show_project_datasets)
        self._list.currentItemChanged.connect(self._update_action_buttons)
        self._dataset_list = QListWidget()
        self._dataset_list.setAlternatingRowColors(True)
        self._dataset_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._dataset_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._dataset_list.setDragEnabled(True)
        self._dataset_list.setAcceptDrops(True)
        self._dataset_list.setDropIndicatorShown(True)
        self._dataset_list.model().rowsMoved.connect(self._persist_dataset_order)
        self._dataset_list.currentItemChanged.connect(self._update_action_buttons)
        self._dataset_list.currentItemChanged.connect(self._emit_dataset_selection_changed)
        self._delete_project_button = QPushButton("删除选中项目的管理信息")
        self._delete_project_button.clicked.connect(self._emit_delete)
        project_column.addWidget(self._list, 1)
        project_column.addWidget(self._delete_project_button)
        dataset_column = QVBoxLayout()
        self._delete_dataset_button = QPushButton("删除选中数据集的管理信息")
        self._delete_dataset_button.clicked.connect(self._emit_delete_dataset)
        dataset_column.addWidget(self._dataset_list, 1)
        dataset_column.addWidget(self._delete_dataset_button)
        browser.addLayout(project_column, 1)
        browser.addLayout(dataset_column, 2)
        root.addLayout(browser, 1)

        action_row = QHBoxLayout()
        self._add_dataset_button = QPushButton("＋ 向选中项目添加数据集")
        self._add_dataset_button.clicked.connect(self._emit_add_dataset)
        self._refresh_button = QPushButton("刷新选中数据集")
        self._refresh_button.clicked.connect(self._emit_refresh_dataset)
        action_row.addStretch()
        action_row.addWidget(self._add_dataset_button)
        action_row.addWidget(self._refresh_button)
        root.addLayout(action_row)
        self._delete_button = self._delete_project_button
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

    def refresh(self, preferred_dataset_id: str | None = None) -> None:
        selected_project_id = self._selected_project_id()
        if preferred_dataset_id is None:
            _, preferred_dataset_id = self.selected_dataset_context()
        self._preferred_dataset_id = preferred_dataset_id
        projects = self._project_loader()
        self._list.clear()
        self._dataset_list.clear()
        dataset_count = 0
        image_count = 0
        valid_annotation_count = 0
        for project in projects:
            dataset_count += len(project.datasets)
            project_images = sum(dataset.scan_summary.image_count for dataset in project.datasets)
            project_valid_annotations = sum(
                dataset.scan_summary.valid_annotation_image_count
                for dataset in project.datasets
            )
            image_count += project_images
            valid_annotation_count += project_valid_annotations
            item = QListWidgetItem(
                f"{project.name}    ·    {len(project.datasets)} 个数据集    ·    "
                f"{project_images} 张图片    ·    {project_valid_annotations} 张有效标注"
            )
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, project)
            self._list.addItem(item)
            if project.id == selected_project_id:
                self._list.setCurrentItem(item)
        self._set_card_value(self._project_count, str(len(projects)))
        self._set_card_value(self._dataset_count, str(dataset_count))
        self._set_card_value(self._image_count, str(image_count))
        self._set_card_value(self._valid_annotation_count, str(valid_annotation_count))
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        self._update_action_buttons()

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

    def _emit_delete_dataset(self) -> None:
        project_id = self._selected_project_id()
        item = self._dataset_list.currentItem()
        if project_id and item is not None:
            self.delete_dataset_requested.emit(
                project_id, str(item.data(Qt.ItemDataRole.UserRole))
            )

    def _emit_dataset_selection_changed(self, *args) -> None:
        del args
        project_id = self._selected_project_id()
        item = self._dataset_list.currentItem()
        if project_id and item is not None:
            self.dataset_selection_changed.emit(
                project_id, str(item.data(Qt.ItemDataRole.UserRole))
            )

    def _show_project_datasets(self, current: QListWidgetItem | None) -> None:
        self._populating_datasets = True
        try:
            self._dataset_list.clear()
            if current is None:
                return
            project = current.data(Qt.ItemDataRole.UserRole + 1)
            preferred_item = None
            for dataset in project.datasets:
                summary = dataset.scan_summary
                item = QListWidgetItem(
                    f"{dataset.name}    ·    {summary.image_count} 张图片    ·    "
                    f"{summary.valid_annotation_image_count} 张有效标注    ·    "
                    f"{summary.category_count} 个类别    ·    {summary.issue_count} 个异常项\n"
                    f"{Path(dataset.root_path)}"
                )
                item.setData(Qt.ItemDataRole.UserRole, dataset.id)
                self._dataset_list.addItem(item)
                if dataset.id == self._preferred_dataset_id:
                    preferred_item = item
            if preferred_item is not None:
                self._dataset_list.setCurrentItem(preferred_item)
            elif self._dataset_list.count():
                self._dataset_list.setCurrentRow(0)
        finally:
            self._preferred_dataset_id = None
            self._populating_datasets = False

    def _persist_dataset_order(self, *args) -> None:
        del args
        if self._populating_datasets:
            return
        project_id = self._selected_project_id()
        if not project_id:
            return
        ordered_ids = [
            str(self._dataset_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self._dataset_list.count())
        ]
        project_item = self._list.currentItem()
        if project_item is not None:
            project = project_item.data(Qt.ItemDataRole.UserRole + 1)
            by_id = {dataset.id: dataset for dataset in project.datasets}
            if set(by_id) == set(ordered_ids):
                project.datasets = [by_id[dataset_id] for dataset_id in ordered_ids]
        self.dataset_order_changed.emit(project_id, ordered_ids)

    def _selected_project_id(self) -> str | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

    def selected_dataset_context(self) -> tuple[str | None, str | None]:
        project_id = self._selected_project_id()
        item = self._dataset_list.currentItem()
        dataset_id = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None
        return project_id, dataset_id

    def _update_action_buttons(self, *args) -> None:
        del args
        has_project = self._list.currentItem() is not None
        has_dataset = has_project and self._dataset_list.currentItem() is not None
        self._add_dataset_button.setEnabled(has_project)
        self._delete_project_button.setEnabled(has_project)
        self._refresh_button.setEnabled(has_dataset)
        self._delete_dataset_button.setEnabled(has_dataset)

