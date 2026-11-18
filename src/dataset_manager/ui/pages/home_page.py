from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from dataset_manager.domain.models.project import Project


class HomePage(QWidget):
    create_requested = Signal()
    delete_requested = Signal(str)

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
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        root.addWidget(self._list, 1)

        delete_button = QPushButton("删除选中项目的管理信息")
        delete_button.clicked.connect(self._emit_delete)
        root.addWidget(delete_button, alignment=Qt.AlignmentFlag.AlignRight)
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
        projects = self._project_loader()
        self._list.clear()
        dataset_count = 0
        for project in projects:
            dataset_count += len(project.datasets)
            item = QListWidgetItem(f"{project.name}    ·    {len(project.datasets)} 个数据集")
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            self._list.addItem(item)
        self._set_card_value(self._project_count, str(len(projects)))
        self._set_card_value(self._dataset_count, str(dataset_count))

    @staticmethod
    def _set_card_value(card: QFrame, value: str) -> None:
        label = card.findChild(QLabel, "value")
        if label is not None:
            label.setText(value)

    def _emit_delete(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self.delete_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))

