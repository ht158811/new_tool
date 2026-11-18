from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.ui.pages.home_page import HomePage
from dataset_manager.ui.themes import Theme, stylesheet


class MainWindow(QMainWindow):
    def __init__(self, project_service: ProjectService) -> None:
        super().__init__()
        self._project_service = project_service
        self._theme = Theme.LIGHT
        self.setWindowTitle("Dataset Manager")
        self.resize(1180, 760)
        self.setMinimumSize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 20, 16, 16)
        brand = QLabel("Dataset Manager")
        brand.setStyleSheet("font-size: 19px; font-weight: 700; padding: 8px;")
        side_layout.addWidget(brand)
        side_layout.addSpacing(18)

        self._stack = QStackedWidget()
        self._home = HomePage(self._project_service.list_projects)
        self._home.create_requested.connect(self._create_project)
        self._home.delete_requested.connect(self._delete_project)
        self._stack.addWidget(self._home)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        for index, name in enumerate(("项目首页", "图片浏览", "质量检查", "数据统计", "系统设置")):
            button = QPushButton(name)
            button.setCheckable(True)
            button_group.addButton(button)
            side_layout.addWidget(button)
            if index == 0:
                button.setChecked(True)
                button.clicked.connect(lambda: self._stack.setCurrentIndex(0))
            else:
                page = self._placeholder(name)
                self._stack.addWidget(page)
                button.clicked.connect(lambda checked=False, i=index: self._stack.setCurrentIndex(i))
        side_layout.addStretch()
        theme_button = QPushButton("切换深色 / 浅色")
        theme_button.clicked.connect(self._toggle_theme)
        side_layout.addWidget(theme_button)

        layout.addWidget(sidebar)
        layout.addWidget(self._stack, 1)
        self.setStyleSheet(stylesheet(self._theme))

    @staticmethod
    def _placeholder(name: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel(name)
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        note = QLabel("该功能将在后续阶段实现。")
        note.setStyleSheet("color: #7b8491;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note, alignment=Qt.AlignmentFlag.AlignCenter)
        return page

    def _toggle_theme(self) -> None:
        self._theme = Theme.DARK if self._theme is Theme.LIGHT else Theme.LIGHT
        self.setStyleSheet(stylesheet(self._theme))

    def _create_project(self) -> None:
        name, accepted = QInputDialog.getText(self, "创建项目", "项目名称：")
        if not accepted:
            return
        try:
            self._project_service.create_project(name)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "创建失败", str(error))
            return
        self._home.refresh()

    def _delete_project(self, project_id: str) -> None:
        answer = QMessageBox.warning(
            self,
            "确认删除",
            "只删除此项目的管理信息。外部原始数据集不会被删除。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is not QMessageBox.StandardButton.Yes:
            return
        try:
            self._project_service.delete_project_management_info(project_id)
        except (OSError, KeyError, ValueError) as error:
            QMessageBox.critical(self, "删除失败", str(error))
            return
        self._home.refresh()

