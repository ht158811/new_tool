from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from dataset_manager.application.preprocessing.models import (
    AugmentationConfig, ConversionConfig, ConversionTarget, CropConfig,
    LabelSplitConfig, NoSeparatorRule, OperationType, ProcessingConfig,
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
        dataset_grid.addWidget(self._dataset_info, 0, 0)
        root.addWidget(dataset_box)

        output_box = QGroupBox("输出目录（强制）")
        output_layout = QGridLayout(output_box)
        self._output = QLineEdit()
        self._output.setPlaceholderText("必须选择原始数据集之外的输出目录")
        self._output.textChanged.connect(self._update_enabled)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse_output)
        output_layout.addWidget(QLabel("输出根目录"), 0, 0)
        output_layout.addWidget(self._output, 0, 1)
        output_layout.addWidget(browse, 0, 2)
        self._result_name = QLineEdit()
        self._result_name.setPlaceholderText("可选；留空时自动生成带日期和序号的目录名")
        output_layout.addWidget(QLabel("结果目录名"), 1, 0)
        output_layout.addWidget(self._result_name, 1, 1, 1, 2)
        output_note = QLabel(
            "裁剪/增强：图片与同名 LabelMe JSON 平铺保存；标签分割：按前后规则输出；"
            "格式转换：按目标格式生成目录结构。"
        )
        output_note.setWordWrap(True)
        output_layout.addWidget(output_note, 2, 0, 1, 3)
        root.addWidget(output_box)

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
            radio.toggled.connect(self._refresh_preview)
            scope_layout.addWidget(radio)
        self._scope_radios[ProcessingScope.VALID_ANNOTATIONS].setChecked(True)
        self._preview = QLabel("尚未选择数据集")
        self._preview.setStyleSheet("padding: 8px; color: #7b8491;")
        scope_layout.addWidget(self._preview)
        root.addWidget(scope_box)

        operation_box = QGroupBox("选择处理功能（仅执行当前选择）")
        operation_layout = QHBoxLayout(operation_box)
        self._operation_group = QButtonGroup(self)
        self._operation_group.setExclusive(True)
        for operation, text, index in (
            (OperationType.CROP, "数据裁剪", 1),
            (OperationType.AUGMENTATION, "数据增强", 2),
            (OperationType.LABEL_SPLIT, "标签分割", 3),
            (OperationType.CONVERSION, "格式转换", 4),
        ):
            button = QPushButton(text)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, op=operation, i=index: self._select_operation(op, i))
            self._operation_group.addButton(button)
            operation_layout.addWidget(button)
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
        self._config_stack.addWidget(self._build_conversion_panel())
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
        self._project_id, self._dataset_id = project_id, dataset_id
        self._dataset_ready = False
        if not project_id or not dataset_id:
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
        task_names = {
            TaskType.OBJECT_DETECTION: "目标检测",
            TaskType.IMAGE_CLASSIFICATION: "图像分类",
            TaskType.SEMANTIC_SEGMENTATION: "语义分割",
        }
        scan_status = "已完成" if dataset.active_scan_id else "未完成"
        self._dataset_ready = bool(dataset.active_scan_id)
        self._dataset_info.setText(
            f"项目名称：{project.name}    数据集名称：{dataset.name}\n"
            f"任务类型：{task_names[dataset.task_type]}    数据格式：LabelMe JSON    "
            f"扫描状态：{scan_status}\n"
            f"根目录：{dataset.root_path}\n"
            f"图片：{dataset.scan_summary.image_count}    "
            f"标注：{dataset.scan_summary.annotation_count}    "
            f"有效标注图片：{dataset.scan_summary.valid_annotation_image_count}"
        )
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
        note = QLabel("当前实现目标检测转换。分类映射和语义分割目标格式尚待确认。")
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

        layout.addWidget(QLabel("输出内容"), 1, 0)
        layout.addWidget(QLabel("无分隔符标签"), 1, 1)
        layout.addWidget(QLabel("按前段标签建上级目录"), 1, 2)

        self._split_before = QCheckBox("前段数据")
        self._split_before.setChecked(True)
        self._split_before_rule = self._no_separator_combo()
        self._split_before_group = QCheckBox("启用")
        layout.addWidget(self._split_before, 2, 0)
        layout.addWidget(self._split_before_rule, 2, 1)
        layout.addWidget(self._split_before_group, 2, 2)

        self._split_after = QCheckBox("后段数据")
        self._split_after.setChecked(True)
        self._split_after_rule = self._no_separator_combo()
        self._split_after_group = QCheckBox("启用（参考脚本默认）")
        self._split_after_group.setChecked(True)
        layout.addWidget(self._split_after, 3, 0)
        layout.addWidget(self._split_after_rule, 3, 1)
        layout.addWidget(self._split_after_group, 3, 2)

        note = QLabel(
            "示例：car;damaged → 前段标签 car，后段标签 damaged。"
            "启用目录分组时，每个目录只保留相同前段标签的 shapes，图片与 JSON 同目录。"
        )
        note.setWordWrap(True)
        layout.addWidget(note, 4, 0, 1, 4)
        return panel

    @staticmethod
    def _no_separator_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItem("两侧保留原标签", NoSeparatorRule.KEEP)
        combo.addItem("标签置空", NoSeparatorRule.EMPTY)
        combo.addItem("跳过该 shape", NoSeparatorRule.SKIP)
        return combo

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _select_operation(self, operation: OperationType, index: int) -> None:
        self._operation = operation
        self._config_stack.setCurrentIndex(index)
        label_split = operation is OperationType.LABEL_SPLIT
        for scope, radio in self._scope_radios.items():
            radio.setEnabled(not label_split or scope is ProcessingScope.VALID_ANNOTATIONS)
        if label_split:
            self._scope_radios[ProcessingScope.VALID_ANNOTATIONS].setChecked(True)
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
                self._project_id, self._dataset_id, self._selected_scope()
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
                raise ValueError("前段数据和后段数据至少选择一项")
            return ProcessingConfig(**common, label_split=LabelSplitConfig(
                separator=separator,
                create_before=self._split_before.isChecked(),
                create_after=self._split_after.isChecked(),
                before_no_separator=self._split_before_rule.currentData(),
                after_no_separator=self._split_after_rule.currentData(),
                before_group_by_label=self._split_before_group.isChecked(),
                after_group_by_label=self._split_after_group.isChecked(),
            ))
        return ProcessingConfig(**common, conversion=ConversionConfig(
            self._conversion_target.currentData()
        ))

    def _start_processing(self) -> None:
        try:
            config = self._build_config()
            preview = self._service.preview(config.project_id, config.dataset_id, config.scope)
        except (OSError, KeyError, ValueError) as error:
            QMessageBox.warning(self, "无法开始处理", str(error))
            return
        if not config.output_root:
            QMessageBox.warning(self, "缺少输出目录", "请手动选择输出目录。")
            return
        answer = QMessageBox.question(
            self, "确认开始预处理",
            f"数据集 ID：{config.dataset_id}\n处理功能：{config.operation.value}\n"
            f"预计处理：{preview.selected_total} 张图片\n输出根目录：{config.output_root}\n\n是否继续？",
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
        ready = bool(
            self._project_id
            and self._dataset_id
            and self._dataset_ready
            and self._operation
            and self._output.text().strip()
        )
        self._start.setEnabled(ready and self._worker is None)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self._worker.wait(5000)
