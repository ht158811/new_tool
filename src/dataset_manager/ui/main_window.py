from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
    QProgressDialog,
)

from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.application.services.dataset_service import DatasetService
from dataset_manager.application.services.scan_service import ScanService
from dataset_manager.application.services.image_service import ImageService
from dataset_manager.application.services.annotation_service import AnnotationService
from dataset_manager.application.preprocessing.preprocessing_service import PreprocessingService
from dataset_manager.application.model_testing.service import ModelTestService
from dataset_manager.ui.dialogs.add_dataset_dialog import AddDatasetDialog
from dataset_manager.ui.pages.home_page import HomePage
from dataset_manager.ui.pages.image_browser_page import ImageBrowserPage
from dataset_manager.ui.pages.preprocessing_page import PreprocessingPage
from dataset_manager.ui.pages.model_test_page import ModelTestPage
from dataset_manager.ui.themes import Theme, stylesheet
from dataset_manager.ui.workers.scan_worker import ScanWorker


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_service: ProjectService,
        dataset_service: DatasetService,
        scan_service: ScanService,
        image_service: ImageService,
        annotation_service: AnnotationService,
        preprocessing_service: PreprocessingService,
        model_test_service: ModelTestService,
    ) -> None:
        super().__init__()
        self._project_service = project_service
        self._dataset_service = dataset_service
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
        self._home.delete_dataset_requested.connect(self._delete_dataset)
        self._home.add_dataset_requested.connect(self._add_dataset)
        self._home.refresh_dataset_requested.connect(self._refresh_dataset)
        self._home.dataset_order_changed.connect(self._reorder_datasets)
        self._stack.addWidget(self._home)
        self._image_browser = ImageBrowserPage(
            self._project_service, image_service, annotation_service
        )
        self._stack.addWidget(self._image_browser)
        self._preprocessing = PreprocessingPage(
            self._project_service, preprocessing_service
        )
        self._stack.addWidget(self._preprocessing)
        self._model_test = ModelTestPage(self._project_service, model_test_service)
        self._stack.addWidget(self._model_test)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        for index, name in enumerate(("项目首页", "图片浏览", "数据预处理", "模型测试", "系统设置")):
            button = QPushButton(name)
            button.setCheckable(True)
            button_group.addButton(button)
            side_layout.addWidget(button)
            if index == 0:
                button.setChecked(True)
                button.clicked.connect(lambda: self._stack.setCurrentIndex(0))
            elif index == 1:
                button.clicked.connect(lambda checked=False: self._stack.setCurrentIndex(1))
            elif index == 2:
                button.clicked.connect(self._open_preprocessing)
            elif index == 3:
                button.clicked.connect(self._open_model_test)
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

    def _open_preprocessing(self, checked: bool = False) -> None:
        del checked
        project_id, dataset_id = self._home.selected_dataset_context()
        self._preprocessing.set_dataset(project_id, dataset_id)
        self._stack.setCurrentIndex(2)

    def _open_model_test(self, checked: bool = False) -> None:
        del checked
        project_id, dataset_id = self._home.selected_dataset_context()
        self._model_test.set_dataset(project_id, dataset_id)
        self._stack.setCurrentIndex(3)

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
        self._model_test.reload_sources()

    def _add_dataset(self, project_id: str) -> None:
        _, insert_after_dataset_id = self._home.selected_dataset_context()
        dialog = AddDatasetDialog(self)
        if dialog.exec() != AddDatasetDialog.DialogCode.Accepted:
            return
        try:
            dataset = dialog.build_dataset()
        except ValueError as error:
            QMessageBox.warning(self, "配置错误", str(error))
            return

        worker = ScanWorker(
            self._scan_service,
            project_id,
            dataset=dataset,
            insert_after_dataset_id=insert_after_dataset_id,
            parent=self,
        )
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
            self._home.refresh(getattr(result, "dataset_id", None))
            self._image_browser.reload_sources()
            self._model_test.reload_sources()
            issue_count = getattr(result, "issue_count", 0)
            QMessageBox.information(
                self,
                "数据集扫描完成",
                f"完整扫描已提交。共发现 {issue_count} 个异常项。",
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

    def _reorder_datasets(self, project_id: str, ordered_ids: object) -> None:
        if not isinstance(ordered_ids, list):
            return
        _, selected_dataset_id = self._home.selected_dataset_context()
        try:
            self._dataset_service.reorder_datasets(
                project_id, [str(dataset_id) for dataset_id in ordered_ids]
            )
        except (OSError, KeyError, ValueError) as error:
            QMessageBox.warning(self, "数据集排序失败", str(error))
            self._home.refresh(selected_dataset_id)
        self._image_browser.reload_sources()
        self._model_test.reload_sources()

    def _delete_project(self, project_id: str) -> None:
        try:
            project = self._project_service.get_project(project_id)
        except KeyError:
            QMessageBox.warning(self, "项目不存在", "所选项目已不存在，项目列表将刷新。")
            self._home.refresh()
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"项目：{project.name}\n"
            f"包含数据集：{len(project.datasets)} 个\n\n"
            "将只删除此项目的管理信息。外部原始图片和 JSON 不会被删除。\n"
            "是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._project_service.delete_project_management_info(project_id)
        except (OSError, KeyError, ValueError) as error:
            QMessageBox.critical(self, "删除失败", str(error))
            return
        self._home.refresh()
        self._image_browser.reload_sources()
        self._model_test.reload_sources()
        QMessageBox.information(
            self,
            "删除完成",
            f"项目“{project.name}”的管理信息已删除。\n外部数据文件未修改。",
        )

    def _delete_dataset(self, project_id: str, dataset_id: str) -> None:
        if self._scan_service.is_running(dataset_id):
            QMessageBox.warning(self, "无法删除", "该数据集正在扫描，请等待扫描结束或先取消扫描。")
            return
        try:
            project = self._project_service.get_project(project_id)
            dataset = project.find_dataset(dataset_id)
        except KeyError:
            QMessageBox.warning(self, "数据集不存在", "所选数据集已不存在，页面将刷新。")
            self._home.refresh()
            self._image_browser.reload_sources()
            return
        answer = QMessageBox.question(
            self,
            "确认删除数据集管理信息",
            f"项目：{project.name}\n"
            f"数据集：{dataset.name}\n"
            f"路径：{dataset.root_path}\n\n"
            "只会从项目中移除此数据集并清理扫描缓存。\n"
            "外部图片和 JSON 不会被删除。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._dataset_service.delete_dataset_management_info(
                project_id, dataset_id
            )
        except (OSError, KeyError, ValueError) as error:
            QMessageBox.critical(self, "删除失败", str(error))
            return
        self._home.refresh()
        self._image_browser.reload_sources()
        self._model_test.reload_sources()
        message = f"数据集“{dataset.name}”的管理信息已删除。\n外部数据文件未修改。"
        if result.cache_cleanup_warning:
            message += f"\n\n{result.cache_cleanup_warning}"
        QMessageBox.information(self, "删除完成", message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self._preprocessing.shutdown()
        self._model_test.shutdown()
        for worker in tuple(self._scan_workers):
            worker.request_cancel()
            worker.wait(3000)
        super().closeEvent(event)

