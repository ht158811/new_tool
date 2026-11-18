from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QVBoxLayout, QWidget, QInputDialog,
)

from dataset_manager.application.dto.image_record import BrowserImage, ImageFilter
from dataset_manager.application.services.annotation_service import AnnotationService
from dataset_manager.application.services.image_service import ImageService
from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.ui.widgets.annotation_canvas import AnnotationCanvas
from dataset_manager.ui.widgets.file_info_view import FileInfoView


class ImageBrowserPage(QWidget):
    def __init__(
        self,
        project_service: ProjectService,
        image_service: ImageService,
        annotation_service: AnnotationService,
    ) -> None:
        super().__init__()
        self._project_service = project_service
        self._image_service = image_service
        self._annotation_service = annotation_service
        self._images: list[BrowserImage] = []
        self._current_image: BrowserImage | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)
        title = QLabel("图片浏览")
        title.setStyleSheet("font-size: 25px; font-weight: 700;")
        root.addWidget(title)

        self._source_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._source_splitter.setObjectName("sourceSplitter")
        self._source_splitter.setChildrenCollapsible(False)
        self._project_combo = QComboBox()
        self._dataset_combo = QComboBox()
        self._project_combo.currentIndexChanged.connect(self._reload_datasets)
        self._dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        project_source = QWidget()
        project_layout = QHBoxLayout(project_source)
        project_layout.setContentsMargins(0, 0, 8, 0)
        project_layout.addWidget(QLabel("项目"))
        project_layout.addWidget(self._project_combo, 1)
        dataset_source = QWidget()
        dataset_layout = QHBoxLayout(dataset_source)
        dataset_layout.setContentsMargins(8, 0, 0, 0)
        dataset_layout.addWidget(QLabel("数据集"))
        dataset_layout.addWidget(self._dataset_combo, 1)
        self._source_splitter.addWidget(project_source)
        self._source_splitter.addWidget(dataset_source)
        self._source_splitter.setStretchFactor(0, 1)
        self._source_splitter.setStretchFactor(1, 1)
        root.addWidget(self._source_splitter)

        self._info_panel = QFrame()
        self._info_panel.setObjectName("fileInfoPanel")
        info_layout = QVBoxLayout(self._info_panel)
        info_title = QLabel("文件与标注信息")
        info_title.setStyleSheet("font-weight: 700;")
        self._edit_status = QLabel("单击图片中的彩色标签名可修改类别名称")
        self._edit_status.setWordWrap(True)
        self._edit_status.setStyleSheet("color: #7b8491;")
        self._details = FileInfoView()
        self._details.setPlainText("选择图片后显示标注信息")
        info_layout.addWidget(info_title)
        info_layout.addWidget(self._edit_status)
        info_layout.addWidget(self._details, 1)

        filter_row = QHBoxLayout()
        self._filename = QLineEdit()
        self._filename.setPlaceholderText("按文件名筛选")
        self._category = QComboBox()
        self._category.addItem("请选择类别", None)
        self._category.currentIndexChanged.connect(self._category_selected)
        apply_button = QPushButton("应用筛选")
        apply_button.clicked.connect(self.refresh_images)
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self._reset_filters)
        filter_row.addWidget(self._filename, 2)
        filter_row.addWidget(self._category, 1)
        filter_row.addWidget(apply_button)
        filter_row.addWidget(reset_button)
        root.addLayout(filter_row)

        action_row = QHBoxLayout()
        self._result_label = QLabel("请选择类别")
        action_row.addWidget(self._result_label)
        valid_legend = QLabel("绿色文件名 = 有效 JSON")
        valid_legend.setStyleSheet("color: #22c55e; font-weight: 600;")
        action_row.addWidget(valid_legend)
        action_row.addStretch()
        self._overlay = QCheckBox("显示标注")
        self._overlay.setChecked(True)
        self._overlay.toggled.connect(self._toggle_overlay)
        action_row.addWidget(self._overlay)
        root.addLayout(action_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("browserContentSplitter")
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setObjectName("fileListInfoSplitter")
        left_splitter.setChildrenCollapsible(False)
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.ListMode)
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.currentItemChanged.connect(self._preview_item)
        left_splitter.addWidget(self._list)
        left_splitter.addWidget(self._info_panel)
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)
        splitter.addWidget(left_splitter)

        preview_panel = QFrame()
        preview_panel.setObjectName("card")
        preview_layout = QVBoxLayout(preview_panel)

        image_column = QWidget()
        image_layout = QVBoxLayout(image_column)
        image_layout.setContentsMargins(0, 0, 0, 0)
        zoom_row = QHBoxLayout()
        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(self._fit_image)
        zoom_out_button = QPushButton("－")
        zoom_out_button.clicked.connect(lambda: self._canvas.zoom_by(0.8))
        zoom_in_button = QPushButton("＋")
        zoom_in_button.clicked.connect(lambda: self._canvas.zoom_by(1.25))
        self._zoom_label = QLabel("100%")
        zoom_row.addWidget(fit_button)
        zoom_row.addWidget(zoom_out_button)
        zoom_row.addWidget(zoom_in_button)
        zoom_row.addWidget(self._zoom_label)
        zoom_row.addStretch()
        zoom_row.addWidget(QLabel("Ctrl + 滚轮缩放"))
        image_layout.addLayout(zoom_row)

        self._canvas = AnnotationCanvas()
        self._canvas.shape_label_clicked.connect(self._edit_shape_label)
        self._canvas.zoom_changed.connect(lambda percent: self._zoom_label.setText(f"{percent}%"))
        self._image_scroll = QScrollArea()
        self._image_scroll.setWidget(self._canvas)
        self._image_scroll.setWidgetResizable(False)
        self._image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_scroll.setMinimumHeight(300)
        self._image_scroll.viewport().installEventFilter(self)
        image_layout.addWidget(self._image_scroll, 1)
        preview_layout.addWidget(image_column, 1)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)
        self.reload_sources()

    def reload_sources(self) -> None:
        selected = self._project_combo.currentData()
        self._project_combo.blockSignals(True)
        self._project_combo.clear()
        for project in self._project_service.list_projects():
            self._project_combo.addItem(project.name, project.id)
        index = self._project_combo.findData(selected)
        self._project_combo.setCurrentIndex(index if index >= 0 else (0 if self._project_combo.count() else -1))
        self._project_combo.blockSignals(False)
        self._reload_datasets()

    def refresh_images(self) -> None:
        project_id = self._project_combo.currentData()
        dataset_id = self._dataset_combo.currentData()
        category = self._category.currentData()
        if not project_id or not dataset_id or not isinstance(category, str) or not category:
            self._images = []
            self._populate(waiting_for_category=True)
            return
        query = ImageFilter(
            filename=self._filename.text().strip(),
            category=category,
        )
        try:
            self._images = self._image_service.list_images(project_id, dataset_id, query)
        except (OSError, KeyError, ValueError) as error:
            self._images = []
            QMessageBox.warning(self, "无法读取扫描缓存", str(error))
        self._populate()

    def _reload_datasets(self) -> None:
        selected = self._dataset_combo.currentData()
        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        project_id = self._project_combo.currentData()
        if project_id:
            try:
                project = self._project_service.get_project(project_id)
            except KeyError:
                project = None
            if project is not None:
                for dataset in project.datasets:
                    self._dataset_combo.addItem(dataset.name, dataset.id)
                    self._dataset_combo.setItemData(
                        self._dataset_combo.count() - 1,
                        dataset.root_path,
                        Qt.ItemDataRole.ToolTipRole,
                    )
        index = self._dataset_combo.findData(selected)
        self._dataset_combo.setCurrentIndex(index if index >= 0 else (0 if self._dataset_combo.count() else -1))
        self._dataset_combo.blockSignals(False)
        self._dataset_changed()

    def _dataset_changed(self) -> None:
        self._category.blockSignals(True)
        self._category.clear()
        self._category.addItem("请选择类别", None)
        project_id = self._project_combo.currentData()
        dataset_id = self._dataset_combo.currentData()
        if project_id and dataset_id:
            try:
                for category in self._image_service.categories(project_id, dataset_id):
                    self._category.addItem(category, category)
            except (KeyError, ValueError):
                pass
        self._category.setCurrentIndex(0)
        self._category.blockSignals(False)
        self._images = []
        self._populate(waiting_for_category=True)

    def _populate(self, waiting_for_category: bool = False) -> None:
        self._list.clear()
        self._current_image = None
        self._canvas.clear()
        self._details.setPlainText("选择图片后显示标注信息")
        self._result_label.setText(
            "请选择类别" if waiting_for_category else f"{len(self._images)} 张图片"
        )
        for image in self._images:
            name = Path(image.image_path).name
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, image)
            if image.annotation_valid:
                item.setForeground(QBrush(QColor("#22c55e")))
            item.setToolTip(
                f"{image.image_path}\n类别：{', '.join(image.categories) or '无'}\n"
                f"检查项：{image.issue_count}"
            )
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _preview_item(self, current: QListWidgetItem | None) -> None:
        if current is None:
            self._current_image = None
            self._canvas.clear()
            return
        image = current.data(Qt.ItemDataRole.UserRole)
        self._current_image = image
        result = self._annotation_service.read(image.annotation_path, image.image_path)
        document = result.document if result is not None else None
        self._canvas.set_content(image.image_path, document)
        self._fit_image()
        shape_lines = []
        if document is not None:
            shape_lines = [
                f"{index + 1}. {shape.label} · {shape.shape_type} · {len(shape.points)} 点"
                for index, shape in enumerate(document.shapes)
            ]
        issue_count = len(result.issues) if result is not None else image.issue_count
        self._details.setPlainText(
            f"文件名：{Path(image.image_path).name}\n"
            f"图片路径：{image.image_path}\n"
            f"JSON 路径：{image.annotation_path or '无'}\n"
            f"尺寸：{image.width or '?'} × {image.height or '?'}\n"
            f"类别：{', '.join(image.categories) or '无'}\n检查项：{issue_count}\n"
            + ("标注：\n" + "\n".join(shape_lines) if shape_lines else "标注：无")
        )

    def _toggle_overlay(self, checked: bool) -> None:
        self._canvas.set_annotations_visible(checked)

    def _edit_shape_label(self, shape_index: int, current_label: str) -> None:
        image = self._current_image
        if image is None or image.annotation_path is None:
            return
        new_label, accepted = QInputDialog.getText(
            self,
            "修改标注类别",
            f"Shape #{shape_index + 1} 类别名称：",
            text=current_label,
        )
        clean_label = new_label.strip()
        if not accepted or not clean_label or clean_label == current_label:
            return
        try:
            self._annotation_service.update_shape_label(
                image.annotation_path, shape_index, clean_label
            )
        except (OSError, ValueError, IndexError) as error:
            QMessageBox.critical(self, "标注修改失败", str(error))
            return
        self._edit_status.setText(
            "已原子写入 JSON。类别筛选索引将在刷新数据集后更新。"
        )
        self._edit_status.setStyleSheet("color: #22c55e; font-weight: 600;")
        self._preview_item(self._list.currentItem())

    def _reset_filters(self) -> None:
        self._filename.clear()
        self._category.setCurrentIndex(0)
        self._images = []
        self._populate(waiting_for_category=True)

    def _category_selected(self, index: int) -> None:
        if index <= 0:
            self._images = []
            self._populate(waiting_for_category=True)
            return
        self.refresh_images()

    def _fit_image(self) -> None:
        self._canvas.fit_to(self._image_scroll.viewport().size())

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API name
        if (
            watched is self._image_scroll.viewport()
            and event.type() == QEvent.Type.Wheel
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._canvas.zoom_by(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
            return True
        return super().eventFilter(watched, event)


