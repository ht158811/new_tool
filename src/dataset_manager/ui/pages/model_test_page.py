from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QTextEdit, QVBoxLayout, QWidget,
)

from dataset_manager.application.model_testing.models import ModelTestConfig, ModelType
from dataset_manager.application.model_testing.service import ModelTestService
from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.ui.workers.model_test_worker import ModelTestWorker


class ModelTestPage(QWidget):
    def __init__(self, project_service: ProjectService, service: ModelTestService) -> None:
        super().__init__()
        self._project_service = project_service
        self._service = service
        self._worker: ModelTestWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)
        title = QLabel("模型测试")
        title.setStyleSheet("font-size: 25px; font-weight: 700;")
        note = QLabel("使用当前扫描数据集逐图推理，测试结果和参数输出为可查阅 JSON。")
        note.setStyleSheet("color: #7b8491;")
        root.addWidget(title)
        root.addWidget(note)

        card = QFrame()
        card.setObjectName("card")
        form = QFormLayout(card)
        self._project = QComboBox()
        self._dataset = QComboBox()
        self._project.currentIndexChanged.connect(self._reload_datasets)
        form.addRow("项目", self._project)
        form.addRow("数据集", self._dataset)

        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("选择 .pt 或 Ultralytics 支持的模型文件")
        model_row = QHBoxLayout()
        model_row.addWidget(self._model_path, 1)
        model_browse = QPushButton("浏览")
        model_browse.clicked.connect(self._browse_model)
        model_row.addWidget(model_browse)
        form.addRow("模型路径", model_row)

        self._model_type = QComboBox()
        self._model_type.addItem("YOLO 目标检测", ModelType.YOLO_DETECTION)
        self._model_type.addItem("YOLO 图像分类", ModelType.YOLO_CLASSIFICATION)
        self._model_type.addItem("YOLO 实例分割", ModelType.YOLO_SEGMENTATION)
        form.addRow("模型类型", self._model_type)

        size_row = QHBoxLayout()
        self._width = self._dimension_spin()
        self._height = self._dimension_spin()
        size_row.addWidget(QLabel("宽"))
        size_row.addWidget(self._width)
        size_row.addWidget(QLabel("高"))
        size_row.addWidget(self._height)
        form.addRow("模型输入尺寸", size_row)

        parameters = QHBoxLayout()
        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.0, 1.0)
        self._confidence.setDecimals(3)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(0.25)
        self._iou = QDoubleSpinBox()
        self._iou.setRange(0.0, 1.0)
        self._iou.setDecimals(3)
        self._iou.setSingleStep(0.05)
        self._iou.setValue(0.7)
        self._max_detections = QSpinBox()
        self._max_detections.setRange(1, 10000)
        self._max_detections.setValue(300)
        parameters.addWidget(QLabel("置信度"))
        parameters.addWidget(self._confidence)
        parameters.addWidget(QLabel("IoU"))
        parameters.addWidget(self._iou)
        parameters.addWidget(QLabel("最大目标数"))
        parameters.addWidget(self._max_detections)
        form.addRow("推理参数", parameters)

        self._device = QLineEdit()
        self._device.setPlaceholderText("留空=自动；也可输入 cpu、0、0,1")
        form.addRow("运行设备", self._device)

        self._output_path = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_path, 1)
        output_browse = QPushButton("选择")
        output_browse.clicked.connect(self._browse_output)
        output_row.addWidget(output_browse)
        form.addRow("结果 JSON", output_row)
        root.addWidget(card)

        action_row = QHBoxLayout()
        self._start = QPushButton("开始模型测试")
        self._start.clicked.connect(self._start_test)
        self._cancel = QPushButton("取消")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self._cancel_test)
        action_row.addWidget(self._start)
        action_row.addWidget(self._cancel)
        action_row.addStretch()
        root.addLayout(action_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._status = QLabel("等待测试")
        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._report.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._report.setPlaceholderText("测试完成后在此预览结果 JSON")
        root.addWidget(self._progress)
        root.addWidget(self._status)
        root.addWidget(self._report, 1)
        self.reload_sources()

    @staticmethod
    def _dimension_spin() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(32, 8192)
        spin.setSingleStep(32)
        spin.setValue(640)
        return spin

    def reload_sources(self) -> None:
        selected = self._project.currentData()
        self._project.blockSignals(True)
        self._project.clear()
        for project in self._project_service.list_projects():
            self._project.addItem(project.name, project.id)
        index = self._project.findData(selected)
        self._project.setCurrentIndex(index if index >= 0 else (0 if self._project.count() else -1))
        self._project.blockSignals(False)
        self._reload_datasets()

    def set_dataset(self, project_id: str | None, dataset_id: str | None) -> None:
        self.reload_sources()
        if project_id:
            project_index = self._project.findData(project_id)
            if project_index >= 0:
                self._project.setCurrentIndex(project_index)
        if dataset_id:
            dataset_index = self._dataset.findData(dataset_id)
            if dataset_index >= 0:
                self._dataset.setCurrentIndex(dataset_index)

    def _reload_datasets(self) -> None:
        selected = self._dataset.currentData()
        self._dataset.clear()
        project_id = self._project.currentData()
        if project_id:
            try:
                project = self._project_service.get_project(project_id)
            except KeyError:
                project = None
            if project is not None:
                for dataset in project.datasets:
                    suffix = "" if dataset.active_scan_id else "（未扫描）"
                    self._dataset.addItem(dataset.name + suffix, dataset.id)
        index = self._dataset.findData(selected)
        self._dataset.setCurrentIndex(index if index >= 0 else (0 if self._dataset.count() else -1))

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", self._model_path.text(),
            "模型文件 (*.pt *.onnx *.engine *.torchscript);;所有文件 (*)",
        )
        if path:
            self._model_path.setText(path)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "保存测试结果", self._output_path.text() or "model_test_result.json",
            "JSON 文件 (*.json)",
        )
        if path:
            self._output_path.setText(path if path.casefold().endswith(".json") else path + ".json")

    def _build_config(self) -> ModelTestConfig:
        project_id = self._project.currentData()
        dataset_id = self._dataset.currentData()
        model_type = self._model_type.currentData()
        if not project_id or not dataset_id:
            raise ValueError("请选择项目和已扫描的数据集")
        if not isinstance(model_type, ModelType):
            model_type = ModelType(model_type)
        return ModelTestConfig(
            project_id=project_id,
            dataset_id=dataset_id,
            model_path=self._model_path.text().strip(),
            model_type=model_type,
            image_width=self._width.value(),
            image_height=self._height.value(),
            confidence=self._confidence.value(),
            iou_threshold=self._iou.value(),
            max_detections=self._max_detections.value(),
            device=self._device.text().strip(),
            output_json=self._output_path.text().strip(),
        )

    def _start_test(self) -> None:
        try:
            config = self._build_config()
        except ValueError as error:
            QMessageBox.warning(self, "配置错误", str(error))
            return
        self._worker = ModelTestWorker(self._service, config, self)
        self._worker.progress_changed.connect(self._update_progress)
        self._worker.succeeded.connect(self._completed)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._worker_finished)
        self._set_running(True)
        self._status.setText("正在加载模型…")
        self._report.clear()
        self._worker.start()

    def _update_progress(self, progress) -> None:
        self._progress.setRange(0, max(progress.total, 1))
        self._progress.setValue(progress.processed)
        self._status.setText(
            f"已处理 {progress.processed}/{progress.total}，成功 {progress.succeeded}，"
            f"失败 {progress.failed}：{Path(progress.current_image).name}"
        )

    def _completed(self, result) -> None:
        status = "已取消，已保存部分结果" if result.cancelled else "模型测试完成"
        self._status.setText(
            f"{status}：成功 {result.succeeded}，失败 {result.failed}；{result.output_json}"
        )
        try:
            data = json.loads(Path(result.output_json).read_text(encoding="utf-8"))
            self._report.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        except (OSError, ValueError) as error:
            self._report.setPlainText(f"结果已生成，但预览失败：{error}")

    def _failed(self, message: str) -> None:
        self._status.setText("模型测试失败")
        QMessageBox.critical(self, "模型测试失败", message)

    def _worker_finished(self) -> None:
        self._set_running(False)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _set_running(self, running: bool) -> None:
        self._start.setEnabled(not running)
        self._cancel.setEnabled(running)

    def _cancel_test(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self._status.setText("正在取消，将在当前图片完成后停止…")

    def shutdown(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(5000)
