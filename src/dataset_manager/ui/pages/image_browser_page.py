from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
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
        self._current_document = None
        self._selected_shape_index = -1

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
        self._edit_status = QLabel(
            "单击彩色标签名修改类别；拖动框内部移动，选中后拖动角点调整大小"
        )
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
        self._category.addItem("全部类别", "")
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
        self._add_rectangle_button = QPushButton("新增矩形框")
        self._add_rectangle_button.setObjectName("addRectangleButton")
        self._add_rectangle_button.setCheckable(True)
        self._add_rectangle_button.setEnabled(False)
        self._add_rectangle_button.toggled.connect(self._toggle_add_rectangle)
        action_row.addWidget(self._add_rectangle_button)
        self._delete_shape_button = QPushButton("删除选中矩形框")
        self._delete_shape_button.setObjectName("deleteShapeButton")
        self._delete_shape_button.clicked.connect(self._delete_current_shape)
        self._delete_shape_button.setEnabled(False)
        action_row.addWidget(self._delete_shape_button)
        self._delete_json_button = QPushButton("删除 JSON…")
        self._delete_json_button.setObjectName("deleteAnnotationButton")
        self._delete_json_button.clicked.connect(self._delete_current_annotation)
        self._delete_json_button.setEnabled(False)
        action_row.addWidget(self._delete_json_button)
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
        self._previous_button = QPushButton("← 上一张")
        self._previous_button.setObjectName("previousImageButton")
        self._previous_button.clicked.connect(lambda: self._navigate_image(-1))
        self._next_button = QPushButton("下一张 →")
        self._next_button.setObjectName("nextImageButton")
        self._next_button.clicked.connect(lambda: self._navigate_image(1))
        self._position_label = QLabel("0 / 0")
        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(self._fit_image)
        zoom_out_button = QPushButton("－")
        zoom_out_button.clicked.connect(lambda: self._zoom_at_viewport_center(0.8))
        zoom_in_button = QPushButton("＋")
        zoom_in_button.clicked.connect(lambda: self._zoom_at_viewport_center(1.25))
        self._zoom_label = QLabel("100%")
        zoom_row.addWidget(self._previous_button)
        zoom_row.addWidget(self._next_button)
        zoom_row.addWidget(self._position_label)
        zoom_row.addSpacing(12)
        zoom_row.addWidget(fit_button)
        zoom_row.addWidget(zoom_out_button)
        zoom_row.addWidget(zoom_in_button)
        zoom_row.addWidget(self._zoom_label)
        zoom_row.addStretch()
        zoom_row.addWidget(QLabel("Ctrl + 滚轮缩放"))
        image_layout.addLayout(zoom_row)

        self._canvas = AnnotationCanvas()
        self._canvas.shape_label_clicked.connect(self._edit_shape_label)
        self._canvas.rectangle_selected.connect(self._select_rectangle)
        self._canvas.rectangle_move_requested.connect(self._move_rectangle)
        self._canvas.rectangle_add_requested.connect(self._add_rectangle)
        self._canvas.zoom_changed.connect(lambda percent: self._zoom_label.setText(f"{percent}%"))
        self._delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._delete_current_shape)
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
        self._update_navigation()
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
        if (
            not project_id
            or not dataset_id
            or self._category.currentIndex() <= 0
            or not isinstance(category, str)
        ):
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
        self._category.addItem("全部类别", "")
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
        self._current_document = None
        self._selected_shape_index = -1
        self._canvas.clear()
        self._update_delete_actions()
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
                f"异常项：{image.issue_count}"
            )
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)
        else:
            self._update_navigation()

    def _preview_item(self, current: QListWidgetItem | None) -> None:
        self._update_navigation()
        if current is None:
            self._current_image = None
            self._current_document = None
            self._selected_shape_index = -1
            self._canvas.clear()
            self._update_delete_actions()
            return
        image = current.data(Qt.ItemDataRole.UserRole)
        self._current_image = image
        self._selected_shape_index = -1
        result = self._annotation_service.read(image.annotation_path, image.image_path)
        document = result.document if result is not None else None
        self._current_document = document
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
            f"类别：{', '.join(image.categories) or '无'}\n异常项：{issue_count}\n"
            + ("标注：\n" + "\n".join(shape_lines) if shape_lines else "标注：无")
        )
        self._update_delete_actions()

    def _toggle_overlay(self, checked: bool) -> None:
        self._canvas.set_annotations_visible(checked)

    def _toggle_add_rectangle(self, checked: bool) -> None:
        if checked and self._current_image is None:
            self._add_rectangle_button.setChecked(False)
            return
        if checked and not self._overlay.isChecked():
            self._overlay.setChecked(True)
        self._canvas.set_add_rectangle_mode(checked)
        self._add_rectangle_button.setText("退出新增模式" if checked else "新增矩形框")
        if checked:
            self._edit_status.setText("新增模式：在图片上按住左键拖动绘制矩形框。")
            self._edit_status.setStyleSheet("color: #38bdf8; font-weight: 600;")

    def _move_rectangle(self, shape_index: int, points: object) -> None:
        image = self._current_image
        if image is None or image.annotation_path is None or not isinstance(points, list):
            return
        try:
            self._annotation_service.update_rectangle_points(
                image.annotation_path, shape_index, points
            )
        except (OSError, ValueError, IndexError) as error:
            QMessageBox.critical(self, "移动矩形框失败", str(error))
            self._preview_item(self._list.currentItem())
            return
        self._edit_status.setText("矩形框坐标已原子写入 JSON。")
        self._edit_status.setStyleSheet("color: #22c55e; font-weight: 600;")
        self._preview_item(self._list.currentItem())

    def _add_rectangle(self, points: object) -> None:
        image = self._current_image
        if image is None or not isinstance(points, list):
            return
        label, accepted = QInputDialog.getText(self, "新增矩形框", "类别名称：")
        clean_label = label.strip()
        if not accepted or not clean_label:
            return
        try:
            if image.annotation_path is not None:
                self._annotation_service.add_rectangle(
                    image.annotation_path, clean_label, points
                )
            else:
                if image.width is None or image.height is None:
                    raise ValueError("缺少图片尺寸，无法创建 LabelMe JSON")
                annotation_path = str(Path(image.image_path).with_suffix(".json"))
                self._annotation_service.create_annotation_with_rectangle(
                    annotation_path,
                    image.image_path,
                    image.width,
                    image.height,
                    clean_label,
                    points,
                )
                updated = replace(
                    image,
                    annotation_path=annotation_path,
                    categories=tuple(sorted({*image.categories, clean_label})),
                    annotation_valid=True,
                )
                row = self._list.currentRow()
                item = self._list.currentItem()
                if 0 <= row < len(self._images):
                    self._images[row] = updated
                if item is not None:
                    item.setData(Qt.ItemDataRole.UserRole, updated)
                    item.setForeground(QBrush(QColor("#22c55e")))
                self._current_image = updated
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "新增矩形框失败", str(error))
            self._preview_item(self._list.currentItem())
            return
        self._edit_status.setText(
            "新矩形框已原子写入 JSON。刷新数据集后更新类别索引。"
        )
        self._edit_status.setStyleSheet("color: #22c55e; font-weight: 600;")
        self._preview_item(self._list.currentItem())

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

    def _select_rectangle(self, shape_index: int) -> None:
        document = self._current_document
        if (
            document is None
            or shape_index < 0
            or shape_index >= len(document.shapes)
            or document.shapes[shape_index].shape_type != "rectangle"
        ):
            self._selected_shape_index = -1
            self._update_delete_actions()
            return
        self._selected_shape_index = shape_index
        shape = document.shapes[shape_index]
        self._edit_status.setText(
            f"已选中矩形框 #{shape_index + 1}：{shape.label}。"
            "可拖动角点调整大小，按 Delete 或点击删除按钮删除。"
        )
        self._edit_status.setStyleSheet("color: #38bdf8; font-weight: 600;")
        self._update_delete_actions()

    def _delete_current_shape(self) -> None:
        image = self._current_image
        document = self._current_document
        if image is None or image.annotation_path is None or document is None:
            return
        shape_index = self._selected_shape_index
        if shape_index < 0 or shape_index >= len(document.shapes):
            return
        shape = document.shapes[shape_index]
        if shape.shape_type != "rectangle":
            return
        answer = QMessageBox.question(
            self,
            "确认删除标注框",
            f"将从 JSON 中删除第 {shape_index + 1} 个标注：{shape.label}\n"
            "图片文件不会被删除。此操作会立即写入 JSON，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._annotation_service.delete_shape(image.annotation_path, shape_index)
        except (OSError, ValueError, IndexError) as error:
            QMessageBox.critical(self, "删除标注框失败", str(error))
            return
        self._selected_shape_index = -1
        self._edit_status.setText("标注框已删除并原子写入 JSON。刷新数据集后更新类别索引。")
        self._edit_status.setStyleSheet("color: #22c55e; font-weight: 600;")
        self._preview_item(self._list.currentItem())

    def _delete_current_annotation(self) -> None:
        image = self._current_image
        if image is None or image.annotation_path is None:
            return
        annotation_path = Path(image.annotation_path)
        answer = QMessageBox.question(
            self,
            "确认删除 JSON 标注",
            f"即将永久删除：{annotation_path.name}\n\n"
            "对应图片不会被删除，但此 JSON 文件删除后无法通过软件恢复。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._annotation_service.delete_annotation(image.annotation_path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "删除 JSON 失败", str(error))
            return
        updated = replace(
            image,
            annotation_path=None,
            categories=(),
            annotation_valid=False,
        )
        row = self._list.currentRow()
        item = self._list.currentItem()
        if 0 <= row < len(self._images):
            self._images[row] = updated
        if item is not None:
            item.setData(Qt.ItemDataRole.UserRole, updated)
            item.setForeground(QBrush())
            item.setToolTip(f"{updated.image_path}\n类别：无\nJSON 已删除")
        self._current_image = updated
        self._edit_status.setText("JSON 标注文件已删除，图片保留。请刷新数据集更新扫描索引。")
        self._edit_status.setStyleSheet("color: #f59e0b; font-weight: 600;")
        self._preview_item(item)

    def _update_delete_actions(self) -> None:
        has_json = bool(
            self._current_image is not None and self._current_image.annotation_path
        )
        has_selected_rectangle = bool(
            has_json
            and self._current_document is not None
            and 0 <= self._selected_shape_index < len(self._current_document.shapes)
            and self._current_document.shapes[self._selected_shape_index].shape_type
            == "rectangle"
        )
        self._delete_shape_button.setEnabled(has_selected_rectangle)
        self._delete_json_button.setEnabled(has_json)
        can_add_rectangle = bool(
            self._current_image is not None
            and (not has_json or self._current_document is not None)
        )
        self._add_rectangle_button.setEnabled(can_add_rectangle)
        if not can_add_rectangle and self._add_rectangle_button.isChecked():
            self._add_rectangle_button.setChecked(False)

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

    def _navigate_image(self, offset: int) -> None:
        count = self._list.count()
        if count <= 0:
            return
        current = self._list.currentRow()
        if current < 0:
            current = 0 if offset >= 0 else count - 1
        target = max(0, min(count - 1, current + offset))
        if target != current or self._list.currentRow() < 0:
            self._list.setCurrentRow(target)
            self._list.scrollToItem(self._list.item(target))
        self._update_navigation()

    def _update_navigation(self) -> None:
        count = self._list.count()
        row = self._list.currentRow()
        has_current = 0 <= row < count
        self._previous_button.setEnabled(has_current and row > 0)
        self._next_button.setEnabled(has_current and row < count - 1)
        self._position_label.setText(f"{row + 1 if has_current else 0} / {count}")

    def _fit_image(self) -> None:
        self._canvas.fit_to(self._image_scroll.viewport().size())

    def _zoom_at_viewport_center(self, factor: float) -> None:
        self._zoom_at(factor, QPointF(self._image_scroll.viewport().rect().center()))

    def _zoom_at(self, factor: float, viewport_position: QPointF) -> None:
        old_width = self._canvas.width()
        old_height = self._canvas.height()
        if old_width <= 0 or old_height <= 0:
            return
        viewport = self._image_scroll.viewport()
        horizontal = self._image_scroll.horizontalScrollBar()
        vertical = self._image_scroll.verticalScrollBar()
        offset_x = max(0.0, (viewport.width() - old_width) / 2)
        offset_y = max(0.0, (viewport.height() - old_height) / 2)
        image_ratio_x = (
            horizontal.value() + viewport_position.x() - offset_x
        ) / old_width
        image_ratio_y = (
            vertical.value() + viewport_position.y() - offset_y
        ) / old_height
        image_ratio_x = max(0.0, min(1.0, image_ratio_x))
        image_ratio_y = max(0.0, min(1.0, image_ratio_y))

        self._canvas.zoom_by(factor)
        new_horizontal = round(image_ratio_x * self._canvas.width() - viewport_position.x())
        new_vertical = round(image_ratio_y * self._canvas.height() - viewport_position.y())

        def restore_anchor() -> None:
            horizontal.setValue(new_horizontal)
            vertical.setValue(new_vertical)

        restore_anchor()
        QTimer.singleShot(0, restore_anchor)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API name
        if (
            watched is self._image_scroll.viewport()
            and event.type() == QEvent.Type.Wheel
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._zoom_at(
                1.2 if event.angleDelta().y() > 0 else 1 / 1.2,
                event.position(),
            )
            event.accept()
            return True
        return super().eventFilter(watched, event)


