from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
    QProgressDialog,
)

from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.application.services.scan_service import ScanService
from dataset_manager.application.services.image_service import ImageService
from dataset_manager.application.services.annotation_service import AnnotationService
from dataset_manager.ui.dialogs.add_dataset_dialog import AddDatasetDialog
from dataset_manager.ui.pages.home_page import HomePage
from dataset_manager.ui.pages.image_browser_page import ImageBrowserPage
from dataset_manager.ui.themes import Theme, stylesheet
from dataset_manager.ui.workers.scan_worker import ScanWorker


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_service: ProjectService,
        scan_service: ScanService,
        image_service: ImageService,
        annotation_service: AnnotationService,
    ) -> None:
        super().__init__()
        self._project_service = project_service
        self._scan_service = scan_service
        self._scan_workers: set[ScanWorker] = set()
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
        self._home.add_dataset_requested.connect(self._add_dataset)
        self._home.refresh_dataset_requested.connect(self._refresh_dataset)
        self._stack.addWidget(self._home)
        self._image_browser = ImageBrowserPage(
            self._project_service, image_service, annotation_service
        )
        self._stack.addWidget(self._image_browser)

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
            elif index == 1:
                button.clicked.connect(lambda checked=False: self._stack.setCurrentIndex(1))
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
        self._image_browser.reload_sources()

    def _add_dataset(self, project_id: str) -> None:
        dialog = AddDatasetDialog(self)
        if dialog.exec() != AddDatasetDialog.DialogCode.Accepted:
            return
        try:
            dataset = dialog.build_dataset()
        except ValueError as error:
            QMessageBox.warning(self, "配置错误", str(error))
            return

        worker = ScanWorker(self._scan_service, project_id, dataset=dataset, parent=self)
        self._start_scan(worker, f"扫描数据集：{dataset.name}", first_add=True)

    def _refresh_dataset(self, project_id: str, dataset_id: str) -> None:
        try:
            dataset = self._project_service.get_project(project_id).find_dataset(dataset_id)
        except KeyError:
            QMessageBox.warning(self, "数据集不存在", "所选数据集已不存在，请刷新页面。")
            self._home.refresh()
            self._image_browser.reload_sources()
            return
        worker = ScanWorker(
            self._scan_service,
            project_id,
            dataset_id=dataset_id,
            parent=self,
        )
        self._start_scan(worker, f"刷新数据集：{dataset.name}", first_add=False)

    def _start_scan(self, worker: ScanWorker, title: str, first_add: bool) -> None:
        progress = QProgressDialog("正在准备扫描…", "取消扫描", 0, 0, self)
        progress.setWindowTitle(title)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.show()

        self._scan_workers.add(worker)
        progress.canceled.connect(worker.request_cancel)

        def update_progress(processed: int, total: int, issues: int, path: str) -> None:
            progress.setRange(0, max(total, 1))
            progress.setValue(processed)
            progress.setLabelText(
                f"已处理 {processed}/{total}，发现 {issues} 个异常\n{path}"
            )

        def succeeded(result: object) -> None:
            progress.close()
            self._home.refresh()
            self._image_browser.reload_sources()
            issue_count = getattr(result, "issue_count", 0)
            QMessageBox.information(
                self,
                "数据集扫描完成",
                f"完整扫描已提交。共发现 {issue_count} 个需检查项。",
            )

        def failed(message: str) -> None:
            progress.close()
            QMessageBox.critical(self, "扫描失败", message)

        def cancelled() -> None:
            progress.close()
            message = (
                "本次结果已丢弃，数据集未添加。"
                if first_add
                else "本次结果已丢弃，继续保留上一次完整扫描结果。"
            )
            QMessageBox.information(self, "扫描已取消", message)

        def cleanup() -> None:
            self._scan_workers.discard(worker)
            worker.deleteLater()

        worker.progress_changed.connect(update_progress)
        worker.scan_succeeded.connect(succeeded)
        worker.scan_failed.connect(failed)
        worker.scan_cancelled.connect(cancelled)
        worker.finished.connect(cleanup)
        worker.start()

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
        self._image_browser.reload_sources()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        for worker in tuple(self._scan_workers):
            worker.request_cancel()
            worker.wait(3000)
        super().closeEvent(event)

