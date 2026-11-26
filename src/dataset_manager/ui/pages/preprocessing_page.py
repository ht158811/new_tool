from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout,
    QWidget,
)

from dataset_manager.application.preprocessing.models import (
    AugmentationConfig, ConversionConfig, ConversionTarget, CropConfig,
    LabelRenameConfig, LabelRetentionMode, LabelSplitConfig, OperationType, ProcessingConfig,
    ProcessingProgress, ProcessingScope,
)
from dataset_manager.application.preprocessing.preprocessing_service import PreprocessingService
from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.domain.models.project import TaskType
from dataset_manager.ui.workers.preprocessing_worker import PreprocessingWorker


class PreprocessingPage(QWidget):
    def __init__(
        self, project_service: ProjectService, preprocessing_service: PreprocessingService
    ) -> None:
        super().__init__()
        self._project_service = project_service
        self._service = preprocessing_service
        self._project_id: str | None = None
        self._dataset_id: str | None = None
        self._operation: OperationType | None = None
        self._worker: PreprocessingWorker | None = None
        self._dataset_ready = False
        self._default_source_root: str | None = None
        self._loaded_scan_id: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)
        title = QLabel("数据预处理")
        title.setStyleSheet("font-size: 25px; font-weight: 700;")
        root.addWidget(title)

        dataset_box = QGroupBox("当前处理数据集")
        dataset_grid = QGridLayout(dataset_box)
        self._dataset_info = QLabel("请先在项目首页选择数据集")
        self._dataset_info.setWordWrap(True)
        dataset_grid.addWidget(self._dataset_info, 0, 0, 1, 4)
        self._source_root = QLineEdit()
        self._source_root.setPlaceholderText("默认使用数据集根目录，也可浏览选择其子目录")
        self._source_root.editingFinished.connect(self._source_root_changed)
        self._source_browse = QPushButton("浏览…")
        self._source_browse.clicked.connect(self._browse_source_root)
        self._source_default = QPushButton("默认目录")
        self._source_default.clicked.connect(self._restore_default_source_root)
        dataset_grid.addWidget(QLabel("根目录"), 1, 0)
        dataset_grid.addWidget(self._source_root, 1, 1)
        dataset_grid.addWidget(self._source_browse, 1, 2)
        dataset_grid.addWidget(self._source_default, 1, 3)
        self._source_root.setEnabled(False)
        self._source_browse.setEnabled(False)
        self._source_default.setEnabled(False)
        root.addWidget(dataset_box)

        self._output_box = QGroupBox("输出目录（强制）")
        output_layout = QGridLayout(self._output_box)
        self._output = QLineEdit()
        self._output.setPlaceholderText("必须选择原始数据集之外的输出目录")
        self._output.textChanged.connect(self._update_enabled)
        self._output_browse = QPushButton("浏览…")
        self._output_browse.clicked.connect(self._browse_output)
        output_layout.addWidget(QLabel("输出根目录"), 0, 0)
        output_layout.addWidget(self._output, 0, 1)
        output_layout.addWidget(self._output_browse, 0, 2)
        self._result_name = QLineEdit()
        self._result_name.setPlaceholderText("可选；留空时自动生成带日期和序号的目录名")
        output_layout.addWidget(QLabel("结果目录名"), 1, 0)
        output_layout.addWidget(self._result_name, 1, 1, 1, 2)
        output_note = QLabel(
            "裁剪/增强：图片与同名 LabelMe JSON 平铺保存；标签分割：按前后规则输出；"
            "格式转换：按目标格式生成目录结构。直接复制的图片保持字节级 100% 一致；"
            "裁剪和增强等必须重新编码的图片使用最高质量或无损参数保存。"
        )
        output_note.setWordWrap(True)
        output_layout.addWidget(output_note, 2, 0, 1, 3)
        root.addWidget(self._output_box)

        scope_box = QGroupBox("处理范围")
        scope_layout = QVBoxLayout(scope_box)
        self._scope_group = QButtonGroup(self)
        self._scope_radios: dict[ProcessingScope, QRadioButton] = {}
        for scope, text in (
            (ProcessingScope.VALID_ANNOTATIONS, "仅处理具有有效 LabelMe JSON 的图片"),
            (ProcessingScope.ALL_IMAGES, "处理全部图片"),
            (ProcessingScope.WITHOUT_JSON, "仅处理没有对应 JSON 的图片"),
        ):
            radio = QRadioButton(text)
            self._scope_group.addButton(radio)
            self._scope_radios[scope] = radio
            radio.toggled.connect(self._scope_changed)
            scope_layout.addWidget(radio)
        self._scope_radios[ProcessingScope.VALID_ANNOTATIONS].setChecked(True)
        self._preview = QLabel("尚未选择数据集")
        self._preview.setStyleSheet("padding: 8px; color: #7b8491;")
        scope_layout.addWidget(self._preview)
        root.addWidget(scope_box)

        operation_box = QGroupBox("选择处理功能（仅执行当前选择）")
        operation_layout = QVBoxLayout(operation_box)
        operation_note = QLabel("单击选择功能；按住功能项可拖动调整显示顺序。")
        operation_note.setStyleSheet("color: #7b8491;")
        operation_layout.addWidget(operation_note)
        self._operation_list = QListWidget()
        self._operation_list.setFlow(QListWidget.Flow.LeftToRight)
        self._operation_list.setWrapping(False)
        self._operation_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._operation_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._operation_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._operation_list.setFixedHeight(54)
        for operation, text, index in (
            (OperationType.CROP, "数据裁剪", 1),
            (OperationType.AUGMENTATION, "数据增强", 2),
            (OperationType.LABEL_SPLIT, "标签分割", 3),
            (OperationType.LABEL_RENAME, "批量标签重命名", 4),
            (OperationType.CONVERSION, "格式转换", 5),
            (OperationType.DELETE_UNANNOTATED, "删除无 JSON 图片", 6),
        ):
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, operation.value)
            item.setData(Qt.ItemDataRole.UserRole + 1, index)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(item.sizeHint().expandedTo(QLabel(text).sizeHint() + QSize(30, 18)))
            self._operation_list.addItem(item)
        self._operation_list.currentItemChanged.connect(self._operation_item_selected)
        operation_layout.addWidget(self._operation_list)
        root.addWidget(operation_box)

        config_box = QGroupBox("当前处理配置")
        config_layout = QVBoxLayout(config_box)
        self._config_stack = QStackedWidget()
        instruction = QLabel("请选择一个处理功能。不会自动执行未选择的功能。")
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._config_stack.addWidget(instruction)
        self._config_stack.addWidget(self._build_crop_panel())
        self._config_stack.addWidget(self._build_augmentation_panel())
        self._config_stack.addWidget(self._build_label_split_panel())
        self._config_stack.addWidget(self._build_label_rename_panel())
        self._config_stack.addWidget(self._build_conversion_panel())
        self._config_stack.addWidget(self._build_delete_unannotated_panel())
        config_layout.addWidget(self._config_stack)
        root.addWidget(config_box)

        actions = QHBoxLayout()
        self._start = QPushButton("开始处理")
        self._start.setObjectName("primary")
        self._start.clicked.connect(self._start_processing)
        self._cancel = QPushButton("取消任务")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self._cancel_processing)
        actions.addWidget(self._start)
        actions.addWidget(self._cancel)
        actions.addStretch()
        root.addLayout(actions)

        progress_box = QGroupBox("处理进度")
        progress_layout = QVBoxLayout(progress_box)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_stats = QLabel("已处理：0  成功：0  失败：0  跳过：0")
        self._status = QLabel("当前状态：等待处理")
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._progress_stats)
        progress_layout.addWidget(self._status)
        root.addWidget(progress_box)
        root.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self._update_enabled()

    def set_dataset(self, project_id: str | None, dataset_id: str | None) -> None:
        if not project_id or not dataset_id:
            self._project_id, self._dataset_id = project_id, dataset_id
            self._dataset_ready = False
            self._default_source_root = None
            self._loaded_scan_id = None
            self._source_root.clear()
            self._source_root.setEnabled(False)
            self._source_browse.setEnabled(False)
            self._source_default.setEnabled(False)
            self._dataset_info.setText("请先返回项目首页选择一个数据集。")
            self._preview.setText("尚未选择数据集")
            self._update_enabled()
            return
        try:
            project = self._project_service.get_project(project_id)
            dataset = project.find_dataset(dataset_id)
        except KeyError:
            self.set_dataset(None, None)
            return
        if (
            self._project_id == project_id
            and self._dataset_id == dataset_id
            and self._loaded_scan_id == dataset.active_scan_id
            and self._source_root.text().strip()
        ):
            self._dataset_ready = bool(dataset.active_scan_id)
            self._update_enabled()
            return
        self._project_id, self._dataset_id = project_id, dataset_id
        task_names = {
            TaskType.OBJECT_DETECTION: "目标检测",
            TaskType.IMAGE_CLASSIFICATION: "图像分类",
            TaskType.SEMANTIC_SEGMENTATION: "语义分割",
        }
        scan_status = "已完成" if dataset.active_scan_id else "未完成"
        self._dataset_ready = bool(dataset.active_scan_id)
        self._loaded_scan_id = dataset.active_scan_id
        self._default_source_root = str(Path(dataset.root_path).resolve())
        self._source_root.setText(self._default_source_root)
        self._source_root.setEnabled(True)
        self._source_browse.setEnabled(True)
        self._source_default.setEnabled(True)
        self._dataset_info.setText(
            f"项目名称：{project.name}    数据集名称：{dataset.name}\n"
            f"任务类型：{task_names[dataset.task_type]}    数据格式：LabelMe JSON    "
            f"扫描状态：{scan_status}\n"
            f"图片：{dataset.scan_summary.image_count}    "
            f"标注：{dataset.scan_summary.annotation_count}    "
            f"有效标注图片：{dataset.scan_summary.valid_annotation_image_count}"
        )
        self._refresh_preview()
        self._update_enabled()

    def _scope_changed(self, checked: bool) -> None:
        if checked:
            self._refresh_preview()

    def _browse_source_root(self) -> None:
        initial = self._source_root.text().strip() or self._default_source_root or ""
        selected = QFileDialog.getExistingDirectory(self, "选择处理根目录", initial)
        if not selected:
            return
        resolved = Path(selected).resolve()
        default = Path(self._default_source_root).resolve() if self._default_source_root else None
        is_dataset_path = (
            default is not None
            and (resolved == default or resolved.is_relative_to(default))
        )
        is_sibling = default is not None and resolved.parent == default.parent
        if not is_dataset_path and not is_sibling:
            QMessageBox.warning(
                self, "目录范围无效",
                "只能选择当前数据集根目录、其子目录或同级目录。",
            )
            return
        self._source_root.setText(str(resolved))
        self._source_root_changed()

    def _restore_default_source_root(self) -> None:
        if self._default_source_root:
            self._source_root.setText(self._default_source_root)
            self._source_root_changed()

    def _source_root_changed(self) -> None:
        self._refresh_preview()
        self._update_enabled()

    def _build_crop_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self._crop_rows = self._spin(1, 1000, 2)
        self._crop_columns = self._spin(1, 1000, 3)
        self._keep_empty = QCheckBox("保留没有目标标注的裁剪图片")
        explanation = QLabel(
            "按原图尺寸均匀划分网格；不能整除的像素自动分配到相邻分块，"
            "确保整张图片无遗漏、无重叠。"
        )
        explanation.setWordWrap(True)
        form.addRow("行数", self._crop_rows)
        form.addRow("列数", self._crop_columns)
        form.addRow("空裁剪", self._keep_empty)
        form.addRow(explanation)
        return panel

    def _build_augmentation_panel(self) -> QWidget:
        panel = QWidget()
        grid = QGridLayout(panel)
        self._aug_checks: dict[str, QCheckBox] = {}
        for row, (key, text) in enumerate((
            ("horizontal_flip", "水平翻转"), ("vertical_flip", "垂直翻转"),
            ("rotate_90", "旋转 90°"), ("rotate_180", "旋转 180°"),
            ("rotate_270", "旋转 270°"),
        )):
            check = QCheckBox(text)
            self._aug_checks[key] = check
            grid.addWidget(check, row, 0)
        self._photo_controls: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for row, (key, text, default, minimum, maximum) in enumerate((
            ("brightness", "亮度", 1.2, 0.1, 3.0),
            ("contrast", "对比度", 1.2, 0.1, 3.0),
            ("saturation", "饱和度", 1.2, 0.1, 3.0),
            ("gaussian_blur", "高斯模糊半径", 1.0, 0.1, 20.0),
            ("gaussian_noise", "高斯噪声标准差", 10.0, 0.1, 100.0),
        )):
            check, spin = QCheckBox(text), QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(default)
            spin.setDecimals(2)
            self._photo_controls[key] = (check, spin)
            grid.addWidget(check, row, 1)
            grid.addWidget(spin, row, 2)
        self._aug_count = self._spin(1, 100, 1)
        self._aug_combo_count = self._spin(1, 10, 1)
        self._aug_seed = self._spin(0, 2_147_483_647, 42)
        grid.addWidget(QLabel("每张图片增强数量"), 6, 0)
        grid.addWidget(self._aug_count, 6, 1)
        grid.addWidget(QLabel("每个结果最多组合操作"), 7, 0)
        grid.addWidget(self._aug_combo_count, 7, 1)
        grid.addWidget(QLabel("随机种子"), 8, 0)
        grid.addWidget(self._aug_seed, 8, 1)
        return panel

    def _build_conversion_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self._conversion_target = QComboBox()
        self._conversion_target.addItem("YOLO TXT", ConversionTarget.YOLO)
        self._conversion_target.addItem("COCO JSON", ConversionTarget.COCO)
        self._conversion_target.addItem("Pascal VOC XML", ConversionTarget.VOC)
        form.addRow("目标格式", self._conversion_target)
        ratio_row = QHBoxLayout()
        self._conversion_validation_ratio = QDoubleSpinBox()
        self._conversion_validation_ratio.setRange(0.0, 99.0)
        self._conversion_validation_ratio.setDecimals(1)
        self._conversion_validation_ratio.setSuffix("%")
        self._conversion_validation_ratio.setValue(20.0)
        self._conversion_test_ratio = QDoubleSpinBox()
        self._conversion_test_ratio.setRange(0.0, 99.0)
        self._conversion_test_ratio.setDecimals(1)
        self._conversion_test_ratio.setSuffix("%")
        self._conversion_test_ratio.setValue(10.0)
        ratio_row.addWidget(QLabel("验证集"))
        ratio_row.addWidget(self._conversion_validation_ratio)
        ratio_row.addWidget(QLabel("测试集"))
        ratio_row.addWidget(self._conversion_test_ratio)
        form.addRow("数据集划分", ratio_row)
        self._conversion_train_ratio = QLabel()
        form.addRow("训练集比例", self._conversion_train_ratio)
        self._conversion_validation_ratio.valueChanged.connect(
            self._update_conversion_train_ratio
        )
        self._conversion_test_ratio.valueChanged.connect(
            self._update_conversion_train_ratio
        )
        self._conversion_split_seed = self._spin(0, 2_147_483_647, 42)
        form.addRow("划分随机种子", self._conversion_split_seed)
        self._update_conversion_train_ratio()
        note = QLabel(
            "训练集比例自动计算。使用相同随机种子可重复得到相同划分；"
            "比例都设为 0 时保持原有未划分目录结构。当前实现目标检测转换。"
        )
        note.setWordWrap(True)
        form.addRow(note)
        return panel

    def _update_conversion_train_ratio(self) -> None:
        ratio = (
            100.0 - self._conversion_validation_ratio.value()
            - self._conversion_test_ratio.value()
        )
        self._conversion_train_ratio.setText(f"{ratio:.1f}%")
        color = "#c62828" if ratio <= 0 else "#2e7d32"
        self._conversion_train_ratio.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _build_delete_unannotated_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        warning = QLabel(
            "直接删除当前数据集原目录及其子目录中没有对应 LabelMe JSON 的图片。"
            "执行前会再次检查同名 JSON 和 JSON 的 imagePath 关联；操作完成后请重新扫描数据集。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #c62828; font-weight: 600;")
        form.addRow(warning)
        return panel

    def _build_label_rename_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self._rename_source_label = QLineEdit()
        self._rename_source_label.setPlaceholderText("输入当前 LabelMe 标签名，例如 defect")
        self._rename_target_label = QLineEdit()
        self._rename_target_label.setPlaceholderText("输入修改后的标签名，例如 scratch")
        form.addRow("要修改的标签名", self._rename_source_label)
        form.addRow("修改后的标签名", self._rename_target_label)
        note = QLabel(
            "仅精确修改原始 LabelMe JSON 的 shapes[].label，其他字段和图片保持不变。"
            "全部目标 JSON 会先校验，写入失败时回滚已修改文件。"
        )
        note.setWordWrap(True)
        form.addRow(note)
        return panel

    def _build_label_split_panel(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        self._split_separator = QLineEdit(";")
        self._split_separator.setPlaceholderText("例如 ; 或 :")
        layout.addWidget(QLabel("标签分隔符"), 0, 0)
        layout.addWidget(self._split_separator, 0, 1, 1, 3)

        layout.addWidget(QLabel("输出数据"), 1, 0)
        layout.addWidget(QLabel("标签保留方式"), 1, 1)
        layout.addWidget(QLabel("按保留标签建立目录"), 1, 2)

        self._split_before = QCheckBox("生成前标签输出")
        self._split_before.setChecked(True)
        self._split_before_retention = self._retention_combo(LabelRetentionMode.BEFORE)
        self._split_before_group = QCheckBox("目录分组")
        layout.addWidget(self._split_before, 2, 0)
        layout.addWidget(self._split_before_retention, 2, 1)
        layout.addWidget(self._split_before_group, 2, 2)

        self._split_after = QCheckBox("生成后标签输出")
        self._split_after.setChecked(True)
        self._split_after_retention = self._retention_combo(LabelRetentionMode.AFTER)
        self._split_after_group = QCheckBox("目录分组")
        self._split_after_group.setChecked(True)
        layout.addWidget(self._split_after, 3, 0)
        layout.addWidget(self._split_after_retention, 3, 1)
        layout.addWidget(self._split_after_group, 3, 2)

        no_separator = QLabel("无分隔符标签：前后输出均保留原标签")
        layout.addWidget(no_separator, 4, 0, 1, 3)

        note = QLabel(
            "前、后两套输出可独立配置。例如前输出选择“前标签保留”，后输出选择“后标签保留”；"
            "也可在任一输出中选择“前后标签同时保留”，保留原始完整标签。启用目录分组后，"
            "按该输出实际保留的标签建目录，每份 JSON 只保留该目录对应的 shapes。"
        )
        note.setWordWrap(True)
        layout.addWidget(note, 5, 0, 1, 3)
        return panel

    @staticmethod
    def _retention_combo(default: LabelRetentionMode) -> QComboBox:
        combo = QComboBox()
        combo.addItem("前标签保留", LabelRetentionMode.BEFORE)
        combo.addItem("后标签保留", LabelRetentionMode.AFTER)
        combo.addItem("前后标签同时保留", LabelRetentionMode.BOTH)
        combo.setCurrentIndex(combo.findData(default))
        return combo

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _operation_item_selected(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        del previous
        if current is None:
            return
        operation = OperationType(current.data(Qt.ItemDataRole.UserRole))
        index = int(current.data(Qt.ItemDataRole.UserRole + 1))
        self._select_operation(operation, index)

    def _select_operation(self, operation: OperationType, index: int) -> None:
        self._operation = operation
        self._config_stack.setCurrentIndex(index)
        in_place = operation in {
            OperationType.LABEL_RENAME, OperationType.DELETE_UNANNOTATED,
        }
        self._output_box.setTitle(
            "输出目录（当前功能直接操作原数据，无需填写）"
            if in_place else "输出目录（强制）"
        )
        self._output.setEnabled(not in_place)
        self._output_browse.setEnabled(not in_place)
        self._result_name.setEnabled(not in_place)
        self._output.setPlaceholderText(
            "当前功能直接操作原数据"
            if in_place else "必须选择原始数据集之外的输出目录"
        )
        label_operation = operation in {
            OperationType.LABEL_SPLIT, OperationType.LABEL_RENAME,
        }
        for scope, radio in self._scope_radios.items():
            allowed = (
                scope is ProcessingScope.VALID_ANNOTATIONS if label_operation
                else scope is ProcessingScope.WITHOUT_JSON
                if operation is OperationType.DELETE_UNANNOTATED
                else True
            )
            radio.setEnabled(allowed)
        if label_operation:
            self._scope_radios[ProcessingScope.VALID_ANNOTATIONS].setChecked(True)
        elif operation is OperationType.DELETE_UNANNOTATED:
            self._scope_radios[ProcessingScope.WITHOUT_JSON].setChecked(True)
        self._update_enabled()

    def _selected_scope(self) -> ProcessingScope:
        for scope, radio in self._scope_radios.items():
            if radio.isChecked():
                return scope
        return ProcessingScope.VALID_ANNOTATIONS

    def _refresh_preview(self) -> None:
        if not self._project_id or not self._dataset_id:
            return
        try:
            preview = self._service.preview(
                self._project_id,
                self._dataset_id,
                self._selected_scope(),
                self._source_root.text().strip(),
            )
        except (OSError, KeyError, ValueError) as error:
            self._preview.setText(str(error))
            return
        self._preview.setText(
            f"图片总数：{preview.total_images}    有效 JSON：{preview.valid_json}    "
            f"无 JSON：{preview.without_json}    无效 JSON：{preview.invalid_json}\n"
            f"本次预计处理：{preview.selected_total}    "
            f"有效标注：{preview.selected_valid}    无标注：{preview.selected_without_json}    "
            f"无效标注仅处理图片：{preview.selected_invalid_json}    跳过：{preview.skipped}"
        )

    def _browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择预处理输出根目录", self._output.text())
        if selected:
            self._output.setText(str(Path(selected).resolve()))

    def _build_config(self) -> ProcessingConfig:
        if not self._project_id or not self._dataset_id:
            raise ValueError("未选择数据集")
        if self._operation is None:
            raise ValueError("请选择处理功能")
        common = dict(
            project_id=self._project_id,
            dataset_id=self._dataset_id,
            output_root=self._output.text().strip(),
            scope=self._selected_scope(),
            operation=self._operation,
            result_name=self._result_name.text(),
            source_root=self._source_root.text().strip(),
        )
        if self._operation is OperationType.CROP:
            return ProcessingConfig(**common, crop=CropConfig(
                rows=self._crop_rows.value(),
                columns=self._crop_columns.value(),
                keep_empty=self._keep_empty.isChecked(),
            ))
        if self._operation is OperationType.AUGMENTATION:
            values = {
                key: spin.value() if check.isChecked() else None
                for key, (check, spin) in self._photo_controls.items()
            }
            return ProcessingConfig(**common, augmentation=AugmentationConfig(
                **{key: check.isChecked() for key, check in self._aug_checks.items()},
                **values,
                outputs_per_image=self._aug_count.value(),
                max_operations_per_output=self._aug_combo_count.value(),
                random_seed=self._aug_seed.value(),
            ))
        if self._operation is OperationType.LABEL_SPLIT:
            separator = self._split_separator.text()
            if not separator:
                raise ValueError("请输入标签分隔符")
            if not self._split_before.isChecked() and not self._split_after.isChecked():
                raise ValueError("前标签输出和后标签输出至少选择一项")
            return ProcessingConfig(**common, label_split=LabelSplitConfig(
                separator=separator,
                create_before=self._split_before.isChecked(),
                create_after=self._split_after.isChecked(),
                before_retention=LabelRetentionMode(
                    self._split_before_retention.currentData()
                ),
                after_retention=LabelRetentionMode(
                    self._split_after_retention.currentData()
                ),
                before_group_by_label=self._split_before_group.isChecked(),
                after_group_by_label=self._split_after_group.isChecked(),
            ))
        if self._operation is OperationType.LABEL_RENAME:
            return ProcessingConfig(**common, label_rename=LabelRenameConfig(
                source_label=self._rename_source_label.text(),
                target_label=self._rename_target_label.text(),
            ))
        if self._operation is OperationType.CONVERSION:
            return ProcessingConfig(**common, conversion=ConversionConfig(
                target=self._conversion_target.currentData(),
                validation_ratio=self._conversion_validation_ratio.value() / 100.0,
                test_ratio=self._conversion_test_ratio.value() / 100.0,
                split_seed=self._conversion_split_seed.value(),
            ))
        return ProcessingConfig(**common)

    def _start_processing(self) -> None:
        try:
            config = self._build_config()
            preview = self._service.preview(
                config.project_id, config.dataset_id, config.scope, config.source_root
            )
        except (OSError, KeyError, ValueError) as error:
            QMessageBox.warning(self, "无法开始处理", str(error))
            return
        in_place = config.operation in {
            OperationType.LABEL_RENAME, OperationType.DELETE_UNANNOTATED,
        }
        if not in_place and not config.output_root:
            QMessageBox.warning(self, "缺少输出目录", "请手动选择输出目录。")
            return
        if in_place:
            try:
                dataset = self._project_service.get_project(
                    config.project_id
                ).find_dataset(config.dataset_id)
            except KeyError as error:
                QMessageBox.warning(self, "数据集不存在", str(error))
                return
            if config.operation is OperationType.LABEL_RENAME:
                confirmation = (
                    f"数据集：{dataset.name}\n处理根目录：{config.source_root}\n"
                    f"精确标签：{config.label_rename.source_label} → "
                    f"{config.label_rename.target_label}\n"
                    f"预计检查：{preview.selected_total} 张图片\n\n"
                    "此操作会直接原子写回原始 LabelMe JSON，不创建输出副本。"
                    "全部 JSON 会先完成校验，是否继续？"
                )
                title = "确认直接修改原始 JSON"
            else:
                confirmation = (
                    f"数据集：{dataset.name}\n处理根目录：{config.source_root}\n"
                    f"预计删除：{preview.selected_total} 张无 JSON 图片\n\n"
                    "此操作会在原文件夹中永久删除图片，不创建输出副本。"
                    "执行时会再次复核 JSON 关联，是否继续？"
                )
                title = "确认永久删除无 JSON 图片"
        else:
            confirmation = (
                f"数据集 ID：{config.dataset_id}\n处理功能：{config.operation.value}\n"
                f"处理根目录：{config.source_root}\n"
                f"预计处理：{preview.selected_total} 张图片\n"
                f"输出根目录：{config.output_root}\n\n是否继续？"
            )
            if config.operation is OperationType.CONVERSION:
                train_ratio = (
                    1.0 - config.conversion.validation_ratio
                    - config.conversion.test_ratio
                )
                confirmation = confirmation.replace(
                    "\n\n是否继续？",
                    f"\n训练/验证/测试：{train_ratio:.1%} / "
                    f"{config.conversion.validation_ratio:.1%} / "
                    f"{config.conversion.test_ratio:.1%}\n\n是否继续？",
                )
            title = "确认开始预处理"
        answer = QMessageBox.question(
            self, title, confirmation,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        worker = PreprocessingWorker(self._service, config, self)
        self._worker = worker
        worker.progress_changed.connect(self._on_progress)
        worker.succeeded.connect(self._on_success)
        worker.failed.connect(self._on_failure)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._worker_finished)
        self._start.setEnabled(False)
        self._cancel.setEnabled(True)
        self._status.setText("当前状态：准备处理")
        worker.start()

    def _on_progress(self, progress: ProcessingProgress) -> None:
        percent = round(progress.processed * 100 / progress.total) if progress.total else 0
        self._progress_bar.setValue(percent)
        self._progress_stats.setText(
            f"已处理：{progress.processed}/{progress.total}  成功：{progress.succeeded}  "
            f"失败：{progress.failed}  跳过：{progress.skipped}"
        )
        self._status.setText(f"当前状态：{progress.status} — {progress.current_file}")

    def _on_success(self, result) -> None:
        self._progress_bar.setValue(100)
        self._status.setText(f"当前状态：完成 — {result.output_directory}")
        if self._operation is OperationType.LABEL_RENAME:
            QMessageBox.information(
                self, "批量标签重命名完成",
                f"检查 JSON：{result.selected_images}\n"
                f"修改 JSON：{result.generated_annotations}\n"
                f"修改 shape：{result.modified_shapes}\n\n"
                "原始图片未修改。请返回项目首页刷新数据集，以更新类别索引。",
            )
            return
        if self._operation is OperationType.DELETE_UNANNOTATED:
            QMessageBox.information(
                self, "无 JSON 图片清理完成",
                f"检查图片：{result.selected_images}\n"
                f"删除图片：{result.deleted_images}\n"
                f"跳过：{result.skipped}\n\n"
                "请返回项目首页重新扫描数据集，以刷新文件列表和统计。",
            )
            return
        QMessageBox.information(
            self, "数据预处理完成",
            f"输入图片：{result.input_images}\n符合范围：{result.selected_images}\n"
            f"成功：{result.succeeded}\n失败：{result.failed}\n跳过：{result.skipped}\n"
            f"生成图片：{result.generated_images}\n生成标注：{result.generated_annotations}\n\n"
            f"输出目录：\n{result.output_directory}",
        )

    def _on_failure(self, message: str) -> None:
        self._status.setText(f"当前状态：失败 — {message}")
        QMessageBox.critical(self, "预处理失败", message)

    def _on_cancelled(self, result) -> None:
        if self._operation in {
            OperationType.LABEL_RENAME, OperationType.DELETE_UNANNOTATED,
        }:
            self._status.setText(
                "当前状态：已取消，原始 JSON 未修改"
                if self._operation is OperationType.LABEL_RENAME
                else "当前状态：已取消，暂存图片已恢复"
            )
            message = (
                "原始 JSON 未修改。" if self._operation is OperationType.LABEL_RENAME
                else "删除已取消，暂存图片已恢复。"
            )
            QMessageBox.information(self, "任务已取消", message)
        else:
            self._status.setText("当前状态：已取消，临时输出已清理")
            QMessageBox.information(self, "任务已取消", "原始数据未修改，本次临时输出已清理。")

    def _cancel_processing(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self._status.setText("当前状态：正在取消…")

    def _worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self._cancel.setEnabled(False)
        self._update_enabled()
        if worker is not None:
            worker.deleteLater()

    def _update_enabled(self) -> None:
        output_ready = bool(
            self._operation in {
                OperationType.LABEL_RENAME, OperationType.DELETE_UNANNOTATED,
            }
            or self._output.text().strip()
        )
        ready = bool(
            self._project_id
            and self._dataset_id
            and self._dataset_ready
            and self._source_root.text().strip()
            and self._operation
            and output_ready
        )
        self._start.setEnabled(ready and self._worker is None)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self._worker.wait(5000)
